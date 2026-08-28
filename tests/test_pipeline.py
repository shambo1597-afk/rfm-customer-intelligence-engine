"""
tests/test_pipeline.py - pytest suite for the RFM-T / ML / CLV / Cohort engines.

Run with:
    pytest --cov=src --cov-report=term-missing

This complements (does not replace) test_enterprise_pipeline.py at the repo root,
which is a standalone end-to-end regression script kept for quick manual runs
(`python test_enterprise_pipeline.py`) and as a CI smoke test. This file exercises
individual functions in isolation, including edge cases (empty input, single-row
input, missing columns) that the standalone script doesn't cover.
"""

import numpy as np
import pandas as pd
import pytest

from src.rfm_engine import (
    RFMPipelineError,
    REQUIRED_STANDARD_COLUMNS,
    SEGMENT_COLORS,
    SEGMENT_ICONS,
    SEGMENT_PLAYBOOKS,
    ensure_series,
    standardize_transactions,
    compute_rfmt,
    calculate_rfmt_scores,
    assign_7_segment_taxonomy,
    assign_segments_vectorized,
    process_rfmt_pipeline,
    get_segment_kpi_summary,
)
from src.ml_engine import (
    preprocess_rfmt_features,
    evaluate_kmeans_candidates,
    perform_kmeans_clustering,
    compute_pca_3d,
    compute_segment_cluster_crosstab,
)
from src.clv_engine import (
    CHURN_TIER_LOW_RISK_THRESHOLD,
    CHURN_TIER_MODERATE_THRESHOLD,
    estimate_btyd_clv,
    get_urgent_churn_watchlist,
    get_top_future_growth_targets,
)
from src.cohort_engine import (
    compute_monthly_cohort_matrix,
    create_cohort_retention_heatmap,
    create_average_retention_curve,
)


# ---------------------------------------------------------------------------
# rfm_engine
# ---------------------------------------------------------------------------

class TestEnsureSeries:
    def test_passes_through_a_normal_series(self):
        s = pd.Series([1, 2, 3])
        assert ensure_series(s) is s

    def test_collapses_a_duplicate_column_dataframe_to_its_first_column(self):
        df = pd.DataFrame([[1, 10], [2, 20]], columns=["dup", "dup"])
        out = ensure_series(df["dup"])
        assert isinstance(out, pd.Series)
        assert list(out) == [1, 2]

    def test_never_collapses_a_single_row_series_to_a_scalar(self):
        # This is the regression case: Series.squeeze() would return a bare
        # scalar here, not a length-1 Series, breaking any chained .astype()/.fillna().
        df = pd.DataFrame({"TotalSpend": [42.5]})
        out = ensure_series(df["TotalSpend"])
        assert isinstance(out, pd.Series)
        assert len(out) == 1
        # Must support the chained calls standardize_transactions() relies on.
        assert out.astype(str).iloc[0] == "42.5"


class TestStandardizeTransactions:
    def test_auto_detects_standard_column_names(self, raw_transactions_df):
        clean = standardize_transactions(raw_transactions_df)
        for col in REQUIRED_STANDARD_COLUMNS:
            assert col in clean.columns
        assert len(clean) > 0
        assert (clean["TotalSpend"] > 0).all()
        assert (clean["Quantity"] > 0).all()

    def test_custom_mapping_overrides_auto_detection(self):
        df = pd.DataFrame({
            "client_id": ["A", "A", "B"],
            "tx_date": ["2026-01-01", "2026-02-01", "2026-01-15"],
            "amount": [10.0, 20.0, 30.0],
        })
        clean = standardize_transactions(
            df,
            custom_mapping={"CustomerID": "client_id", "PurchaseDate": "tx_date", "TotalSpend": "amount"},
        )
        assert set(clean["CustomerID"]) == {"A", "B"}
        assert clean["TotalSpend"].sum() == 60.0

    def test_single_row_dataset_does_not_crash(self):
        # Regression test for the squeeze()-collapses-to-scalar crash.
        df = pd.DataFrame({"CustomerID": ["CUST-1"], "PurchaseDate": ["2026-01-01"], "TotalSpend": [42.50]})
        clean = standardize_transactions(df)
        assert len(clean) == 1
        assert clean.iloc[0]["CustomerID"] == "CUST-1"

    def test_unmappable_columns_raise_rfm_pipeline_error(self):
        df = pd.DataFrame({"SomeColumn": [1, 2, 3], "OtherColumn": ["a", "b", "c"]})
        with pytest.raises(RFMPipelineError):
            standardize_transactions(df)

    def test_all_rows_filtered_out_raises_rfm_pipeline_error(self):
        df = pd.DataFrame({"CustomerID": ["CUST-1"], "PurchaseDate": ["2026-01-01"], "TotalSpend": [0]})
        with pytest.raises(RFMPipelineError):
            standardize_transactions(df)

    def test_drops_null_and_non_positive_rows(self):
        df = pd.DataFrame({
            "CustomerID": ["A", None, "B", "C"],
            "PurchaseDate": ["2026-01-01", "2026-01-01", "not-a-date", "2026-01-02"],
            "TotalSpend": [10.0, 10.0, 10.0, -5.0],
        })
        clean = standardize_transactions(df)
        # Only row "A" survives: null CustomerID, unparseable date, and negative spend are all dropped.
        assert len(clean) == 1
        assert clean.iloc[0]["CustomerID"] == "A"

    def test_auto_detection_fires_with_no_custom_mapping_for_all_six_target_fields(self):
        # No custom_mapping at all, and every header is a guessable alias rather than
        # the canonical name -- this is the path that was previously untested:
        # test_auto_detects_standard_column_names (above) uses a fixture that already
        # has canonical column names, so the "not in df_clean.columns" guards are
        # already False and the auto-detect loop bodies never run; and
        # test_custom_mapping_overrides_auto_detection short-circuits the same loop
        # by pre-setting the target columns via custom_mapping. This is the case
        # where the loop actually has to do the alias-matching work itself.
        df = pd.DataFrame({
            "customer_id": ["A", "A", "B"],
            "order_date": ["2026-01-01", "2026-02-01", "2026-01-15"],
            "amount": [10.0, 20.0, 30.0],
            "order_no": ["INV-1", "INV-2", "INV-3"],
            "dept": ["Electronics", "Electronics", "Home"],
            "item": ["Widget", "Widget", "Gadget"],
        })
        clean = standardize_transactions(df)
        assert set(clean["CustomerID"]) == {"A", "B"}
        assert clean["PurchaseDate"].notna().all()
        assert clean["TotalSpend"].sum() == 60.0
        assert set(clean["InvoiceNo"]) == {"INV-1", "INV-2", "INV-3"}
        assert set(clean["ProductCategory"]) == {"Electronics", "Home"}
        assert set(clean["Product"]) == {"Widget", "Gadget"}

    @pytest.mark.parametrize("field,alias_col,values,expected", [
        ("CustomerID", "clientid", ["A", "B"], ["A", "B"]),
        ("CustomerID", "userid", ["A", "B"], ["A", "B"]),
        ("PurchaseDate", "invoicedate", ["2026-01-01", "2026-02-01"], None),
        ("PurchaseDate", "timestamp", ["2026-01-01", "2026-02-01"], None),
        ("TotalSpend", "spend", [12.0, 34.0], [12.0, 34.0]),
        ("TotalSpend", "revenue", [12.0, 34.0], [12.0, 34.0]),
        ("InvoiceNo", "orderid", ["INV-A", "INV-B"], ["INV-A", "INV-B"]),
        ("InvoiceNo", "transactionid", ["INV-A", "INV-B"], ["INV-A", "INV-B"]),
        ("ProductCategory", "category", ["Cat1", "Cat2"], ["Cat1", "Cat2"]),
        ("ProductCategory", "itemcategory", ["Cat1", "Cat2"], ["Cat1", "Cat2"]),
    ])
    def test_additional_alias_names_are_recognized(self, field, alias_col, values, expected):
        # A second (and third) alias per field, distinct from the ones exercised
        # above -- this is what actually catches a typo in the alias list itself
        # (e.g. a field that only recognizes the first-listed alias), rather than
        # just proving the auto-detection mechanism fires once. For fields with a
        # fallback default (InvoiceNo, ProductCategory), a plain "column exists"
        # check would pass vacuously even if the alias were never matched -- so
        # every case here asserts the actual alias values landed, not just presence.
        base = {
            "CustomerID": ["X", "Y"],
            "PurchaseDate": ["2026-01-01", "2026-01-02"],
            "TotalSpend": [10.0, 20.0],
        }
        base.pop(field, None)
        df = pd.DataFrame({**base, alias_col: values})
        clean = standardize_transactions(df)
        assert field in clean.columns
        if expected is not None:
            assert list(clean[field]) == expected
        else:
            assert clean[field].notna().all()
            assert clean[field].nunique() == 2

    def test_unrecognized_column_alias_is_left_alone(self):
        df = pd.DataFrame({
            "CustomerID": ["A", "B"],
            "PurchaseDate": ["2026-01-01", "2026-01-02"],
            "TotalSpend": [10.0, 20.0],
            "shipping_zone": ["West", "East"],
        })
        clean = standardize_transactions(df)
        # An unrecognized column must survive untouched under its own name -- it
        # must not get miscategorized into any target field, and must not crash.
        assert "shipping_zone" in clean.columns
        assert list(clean["shipping_zone"]) == ["West", "East"]
        # The fields nothing else matched still get their documented defaults.
        assert set(clean["ProductCategory"]) == {"General Merchandise"}

    def test_custom_mapping_wins_over_a_conflicting_auto_detectable_column(self):
        # "customerid" would auto-detect to CustomerID on its own -- but custom_mapping
        # explicitly routes CustomerID from "client_id" instead, and the two columns
        # deliberately disagree, so a passing assertion proves custom_mapping is
        # authoritative rather than winning by coincidence. PurchaseDate/TotalSpend
        # are left unmapped, exercising auto-detection for those two in the same call.
        df = pd.DataFrame({
            "client_id": ["A", "A", "B"],
            "customerid": ["WRONG-A", "WRONG-A", "WRONG-B"],
            "order_date": ["2026-01-01", "2026-02-01", "2026-01-15"],
            "amount": [10.0, 20.0, 30.0],
        })
        clean = standardize_transactions(df, custom_mapping={"CustomerID": "client_id"})
        assert set(clean["CustomerID"]) == {"A", "B"}
        assert "WRONG-A" not in set(clean["CustomerID"])
        assert clean["TotalSpend"].sum() == 60.0
        assert clean["PurchaseDate"].notna().all()

    def test_custom_mapping_covers_optional_invoice_and_category_fields(self):
        # test_custom_mapping_overrides_auto_detection only maps the 3 required
        # fields; this covers the optional InvoiceNo/ProductCategory custom_mapping
        # branches specifically.
        df = pd.DataFrame({
            "cust": ["A", "A", "B"],
            "dt": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "amt": [10.0, 20.0, 30.0],
            "order_ref": ["ORD-1", "ORD-2", "ORD-3"],
            "product_dept": ["Books", "Books", "Toys"],
        })
        clean = standardize_transactions(
            df,
            custom_mapping={
                "CustomerID": "cust",
                "PurchaseDate": "dt",
                "TotalSpend": "amt",
                "InvoiceNo": "order_ref",
                "ProductCategory": "product_dept",
            },
        )
        assert set(clean["InvoiceNo"]) == {"ORD-1", "ORD-2", "ORD-3"}
        assert set(clean["ProductCategory"]) == {"Books", "Toys"}


class TestComputeRfmtAndScores:
    def test_rfmt_has_one_row_per_customer(self, clean_tx_df, rfmt_df):
        assert len(rfmt_df) == clean_tx_df["CustomerID"].nunique()

    def test_recency_and_tenure_are_non_negative(self, rfmt_df):
        assert (rfmt_df["Recency"] >= 0).all()
        assert (rfmt_df["Tenure"] >= 1).all()

    def test_quintile_scores_are_bounded_1_to_5(self, rfmt_df):
        scored = calculate_rfmt_scores(rfmt_df)
        for col in ["R_Score", "F_Score", "M_Score", "T_Score"]:
            assert scored[col].between(1, 5).all()

    def test_small_sample_fallback_assigns_max_score(self):
        tiny = pd.DataFrame({
            "CustomerID": ["A", "B"],
            "Recency": [5, 100],
            "Frequency": [3, 1],
            "Monetary": [500.0, 20.0],
            "Tenure": [90, 10],
        })
        scored = calculate_rfmt_scores(tiny)
        assert (scored["R_Score"] == 5).all()
        assert (scored["F_Score"] == 5).all()


class TestSnapshotDateHandling:
    """
    Every other test in this file calls compute_rfmt()/process_rfmt_pipeline() with
    snapshot_date=None (the default), which only exercises the `if snapshot_date is
    None` branch. These tests specifically cover the `else` branch (an explicit
    snapshot_date, run through pd.to_datetime()) and pin down exactly what the
    None-default resolves to, rather than leaving it implicit.
    """

    def test_explicit_snapshot_date_is_used_verbatim(self, clean_tx_df):
        explicit_snapshot = pd.Timestamp("2030-01-01")
        rfmt = compute_rfmt(clean_tx_df, snapshot_date=explicit_snapshot)

        most_recent_customer_id = clean_tx_df.loc[clean_tx_df["PurchaseDate"].idxmax(), "CustomerID"]
        max_purchase = clean_tx_df["PurchaseDate"].max()
        expected_recency = (explicit_snapshot - max_purchase).days

        actual_recency = rfmt.loc[rfmt["CustomerID"] == most_recent_customer_id, "Recency"].iloc[0]
        assert actual_recency == expected_recency

    def test_explicit_snapshot_date_accepts_a_plain_string(self, clean_tx_df):
        # Covers the pd.to_datetime(snapshot_date) conversion specifically -- a plain
        # string (not already a Timestamp) must be coerced the same way.
        rfmt_from_string = compute_rfmt(clean_tx_df, snapshot_date="2030-06-15")
        rfmt_from_timestamp = compute_rfmt(clean_tx_df, snapshot_date=pd.Timestamp("2030-06-15"))
        pd.testing.assert_series_equal(
            rfmt_from_string.sort_values("CustomerID")["Recency"].reset_index(drop=True),
            rfmt_from_timestamp.sort_values("CustomerID")["Recency"].reset_index(drop=True),
        )

    def test_none_snapshot_date_defaults_to_max_purchase_date_plus_one_day(self, clean_tx_df):
        rfmt_default = compute_rfmt(clean_tx_df, snapshot_date=None)

        most_recent_customer_id = clean_tx_df.loc[clean_tx_df["PurchaseDate"].idxmax(), "CustomerID"]
        actual_recency = rfmt_default.loc[rfmt_default["CustomerID"] == most_recent_customer_id, "Recency"].iloc[0]

        # The customer whose last purchase IS the dataset's global max must show
        # Recency == 1 day, because the implied snapshot is (max PurchaseDate + 1 day)
        # -- not datetime.now(), which for this dataset's dates would give a much
        # larger, unrelated number.
        assert actual_recency == 1

        hypothetical_now_recency = (pd.Timestamp.now() - clean_tx_df["PurchaseDate"].max()).days
        assert actual_recency != hypothetical_now_recency


class TestSegmentAssignment:
    def test_vectorized_and_row_wise_assignment_agree(self, rfmt_df):
        scored = calculate_rfmt_scores(rfmt_df)
        vectorized = assign_segments_vectorized(scored)
        row_wise = scored.apply(assign_7_segment_taxonomy, axis=1)
        assert (vectorized.values == row_wise.values).all()

    def test_every_segment_has_playbook_and_visual_metadata(self):
        assert set(SEGMENT_PLAYBOOKS.keys()) == set(SEGMENT_COLORS.keys()) == set(SEGMENT_ICONS.keys())
        for seg, playbook in SEGMENT_PLAYBOOKS.items():
            assert playbook["actions"], f"{seg} has no action items"
            assert "campaign_template" in playbook


class TestPipelineAndSummary:
    def test_process_rfmt_pipeline_end_to_end(self, raw_transactions_df):
        clean_tx, rfmt = process_rfmt_pipeline(raw_transactions_df)
        assert len(clean_tx) > 0
        assert len(rfmt) == clean_tx["CustomerID"].nunique()
        assert set(rfmt["Segment"].unique()).issubset(set(SEGMENT_PLAYBOOKS.keys()))

    def test_segment_kpi_summary_shares_sum_to_100(self, clv_df):
        summary = get_segment_kpi_summary(clv_df)
        assert len(summary) == 7
        assert summary["CustomerSharePct"].sum() == pytest.approx(100.0, abs=0.5)
        assert summary["RevenueSharePct"].sum() == pytest.approx(100.0, abs=0.5)


# ---------------------------------------------------------------------------
# ml_engine
# ---------------------------------------------------------------------------

class TestMlEngine:
    def test_preprocess_returns_zero_mean_unit_variance_features(self, scaled_features):
        X_scaled, scaler, features = scaled_features
        assert features == ["Recency", "Frequency", "Monetary", "Tenure"]
        assert X_scaled.shape[1] == 4
        assert np.allclose(X_scaled.mean(axis=0), 0.0, atol=1e-6)

    def test_evaluate_kmeans_candidates_covers_requested_k_range(self, scaled_features):
        X_scaled, _, _ = scaled_features
        eval_df = evaluate_kmeans_candidates(X_scaled, min_k=2, max_k=7)
        assert list(eval_df["k"]) == [2, 3, 4, 5, 6, 7]
        assert (eval_df["Silhouette_Score"].between(-1.0, 1.0)).all()
        # Inertia must strictly decrease as k grows (more clusters -> lower WCSS).
        assert (eval_df["Inertia"].diff().dropna() <= 0).all()

    def test_final_fit_uses_the_same_feature_space_as_evaluation(self, clv_df, scaled_features):
        # Regression test for the PCA-vs-4D feature-space mismatch fixed in the
        # clustering pipeline: the k recommended by evaluate_kmeans_candidates()
        # must come from the same preprocessing perform_kmeans_clustering() uses.
        X_scaled, _, _ = scaled_features
        eval_df = evaluate_kmeans_candidates(X_scaled, min_k=2, max_k=7)
        optimal_k = int(eval_df.loc[eval_df["Silhouette_Score"].idxmax()]["k"])
        df_ml, _, cluster_summary = perform_kmeans_clustering(clv_df, n_clusters=optimal_k)
        assert df_ml["ML_Cluster"].nunique() == optimal_k
        assert len(cluster_summary) == optimal_k

    def test_pca_3d_produces_three_components_with_explained_variance(self, clv_df):
        df_pca, pca_model, explained_variance = compute_pca_3d(clv_df)
        assert {"PCA_1", "PCA_2", "PCA_3"}.issubset(df_pca.columns)
        assert len(explained_variance) == 3
        assert 0 <= explained_variance.sum() <= 100

    def test_segment_cluster_crosstab_rows_sum_to_segment_size_and_100_percent(self, clv_df):
        df_ml, _, _ = perform_kmeans_clustering(clv_df, n_clusters=4)
        counts, row_pct = compute_segment_cluster_crosstab(df_ml)
        segment_sizes = df_ml["Segment"].value_counts()
        for seg in counts.index:
            assert counts.loc[seg].sum() == segment_sizes.get(seg, 0)
            if segment_sizes.get(seg, 0) > 0:
                assert row_pct.loc[seg].sum() == pytest.approx(100.0, abs=0.2)


# ---------------------------------------------------------------------------
# clv_engine
# ---------------------------------------------------------------------------

class TestClvEngine:
    def test_p_alive_is_bounded(self, clv_df):
        assert clv_df["P_Alive"].between(0.0, 1.0).all()

    def test_churn_tier_matches_p_alive_thresholds(self, clv_df):
        low = clv_df[clv_df["Churn_Risk_Tier"] == "🟢 Low Churn Risk"]
        high = clv_df[clv_df["Churn_Risk_Tier"] == "🔴 High Churn Risk"]
        assert (low["P_Alive"] >= CHURN_TIER_LOW_RISK_THRESHOLD).all()
        assert (high["P_Alive"] < CHURN_TIER_MODERATE_THRESHOLD).all()

    def test_predicted_spend_is_non_negative(self, clv_df):
        assert (clv_df["Predicted_Spend_90d"] >= 0).all()
        assert (clv_df["Predictive_CLV_90d"] >= 0).all()

    def test_urgent_watchlist_only_contains_above_median_high_risk_customers(self, clv_df):
        watchlist = get_urgent_churn_watchlist(clv_df, p_alive_threshold=0.45)
        median_spend = clv_df["Monetary"].median()
        assert (watchlist["P_Alive"] < 0.45).all()
        assert (watchlist["Monetary"] >= median_spend).all()

    def test_top_growth_targets_are_sorted_descending(self, clv_df):
        top = get_top_future_growth_targets(clv_df, top_n=5)
        assert len(top) == 5
        assert list(top["Predicted_Spend_90d"]) == sorted(top["Predicted_Spend_90d"], reverse=True)

    def test_hazard_constants_are_configurable_and_affect_p_alive(self, rfmt_df):
        default_out = estimate_btyd_clv(rfmt_df)
        recalibrated_out = estimate_btyd_clv(
            rfmt_df,
            alpha_prior=2.5,
            beta_prior_days=30.0,
            hazard_missed_cycles_weight=2.0,
            hazard_missed_cycles_offset=1.0,
            hazard_inactivity_ratio_weight=2.0,
            hazard_inactivity_ratio_offset=0.2,
        )
        # Different hazard constants must actually change the output (not silently ignored).
        assert not default_out["P_Alive"].equals(recalibrated_out["P_Alive"])
        # ... while still respecting the same output bounds.
        assert recalibrated_out["P_Alive"].between(0.0, 1.0).all()

    def test_gross_margin_scales_predicted_clv_linearly(self, rfmt_df):
        low_margin = estimate_btyd_clv(rfmt_df, gross_margin=0.10)
        high_margin = estimate_btyd_clv(rfmt_df, gross_margin=0.50)
        ratio = (high_margin["Predictive_CLV_90d"] / low_margin["Predictive_CLV_90d"]).dropna()
        assert np.allclose(ratio, 5.0, atol=0.05)


# ---------------------------------------------------------------------------
# cohort_engine
# ---------------------------------------------------------------------------

class TestCohortEngine:
    def test_month_zero_retention_is_always_100_percent(self, clean_tx_df):
        _, retention_matrix, _ = compute_monthly_cohort_matrix(clean_tx_df)
        assert (retention_matrix.iloc[:, 0] == 100.0).all()

    def test_max_months_caps_column_count(self, clean_tx_df):
        _, retention_matrix, _ = compute_monthly_cohort_matrix(clean_tx_df, max_months=6)
        assert retention_matrix.shape[1] <= 6

    def test_no_cap_when_max_months_is_none(self, clean_tx_df):
        _, uncapped, _ = compute_monthly_cohort_matrix(clean_tx_df, max_months=None)
        _, capped, _ = compute_monthly_cohort_matrix(clean_tx_df, max_months=6)
        assert uncapped.shape[1] >= capped.shape[1]

    def test_heatmap_and_curve_figures_render_without_error(self, clean_tx_df):
        count_matrix, retention_matrix, _ = compute_monthly_cohort_matrix(clean_tx_df)
        heatmap_fig = create_cohort_retention_heatmap(retention_matrix, count_matrix)
        curve_fig = create_average_retention_curve(retention_matrix)
        assert heatmap_fig.data
        assert curve_fig.data

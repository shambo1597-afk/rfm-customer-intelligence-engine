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

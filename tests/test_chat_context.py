"""
tests/test_chat_context.py - pytest suite for src/chat_context.py's context-blob
builder. No LLM calls happen anywhere in this file -- this module only builds a
structured dict/text blob from already-computed pipeline outputs; there is
nothing here to mock.
"""

import pandas as pd
import pytest

from src.chat_context import (
    build_account_context_blob,
    build_context_text,
    build_methodology_context,
    GROWTH_TARGETS_TOP_N,
)
from src.digest_engine import _build_aggregate_stats


@pytest.fixture(scope="module")
def context_blob(rfmt_df, clv_df, segment_summary, cohort_retention_matrix, crosstab_counts):
    return build_account_context_blob(rfmt_df, clv_df, segment_summary, cohort_retention_matrix, crosstab_counts)


@pytest.fixture(scope="module")
def context_text(context_blob):
    return build_context_text(context_blob)


class TestBlobStructure:
    def test_returns_the_documented_top_level_keys(self, context_blob):
        assert set(context_blob.keys()) == {
            "aggregate_stats", "segment_breakdown", "churn_watchlist",
            "growth_targets", "cohort_retention_by_month", "segment_cluster_crosstab",
            "methodology",
        }

    def test_aggregate_stats_matches_digest_engines_own_function_exactly(
        self, rfmt_df, clv_df, segment_summary, context_blob
    ):
        # Reuse, not re-derivation: this is the SAME dict digest_engine.py's
        # _build_aggregate_stats() would produce for the same inputs.
        assert context_blob["aggregate_stats"] == _build_aggregate_stats(rfmt_df, clv_df, segment_summary)

    def test_segment_breakdown_has_one_entry_per_segment_row(self, segment_summary, context_blob):
        assert len(context_blob["segment_breakdown"]) == len(segment_summary)
        for entry in context_blob["segment_breakdown"]:
            assert set(entry.keys()) == {
                "segment", "customer_count", "customer_share_pct", "total_revenue",
                "revenue_share_pct", "avg_recency_days", "avg_frequency", "avg_tenure_days",
            }

    def test_churn_watchlist_is_aggregate_only(self, context_blob):
        watchlist = context_blob["churn_watchlist"]
        assert set(watchlist.keys()) == {"count", "total_value", "avg_p_alive_pct", "segment_composition"}
        assert isinstance(watchlist["count"], int)
        assert isinstance(watchlist["segment_composition"], dict)

    def test_growth_targets_is_aggregate_only_and_respects_top_n(self, clv_df, context_blob):
        growth = context_blob["growth_targets"]
        assert set(growth.keys()) == {"count", "total_predicted_90d_value", "segment_composition"}
        assert growth["count"] <= min(GROWTH_TARGETS_TOP_N, len(clv_df))

    def test_cohort_retention_by_month_is_a_flat_month_to_pct_dict(self, context_blob):
        cohort = context_blob["cohort_retention_by_month"]
        assert isinstance(cohort, dict)
        for month, pct in cohort.items():
            assert isinstance(month, int)
            assert isinstance(pct, float)
            assert 0.0 <= pct <= 100.0

    def test_segment_cluster_crosstab_reduces_to_dominant_cluster_per_segment(self, context_blob):
        crosstab = context_blob["segment_cluster_crosstab"]
        assert isinstance(crosstab, dict)
        for segment, info in crosstab.items():
            assert set(info.keys()) == {"dominant_ml_cluster", "dominant_cluster_share_pct"}
            assert 0.0 <= info["dominant_cluster_share_pct"] <= 100.0


class TestGracefulDegradationOnMissingOptionalInputs:
    def test_none_cohort_matrix_degrades_to_empty_summary_not_an_error(
        self, rfmt_df, clv_df, segment_summary, crosstab_counts
    ):
        blob = build_account_context_blob(rfmt_df, clv_df, segment_summary, None, crosstab_counts)
        assert blob["cohort_retention_by_month"] == {}

    def test_empty_cohort_matrix_degrades_to_empty_summary_not_an_error(
        self, rfmt_df, clv_df, segment_summary, crosstab_counts
    ):
        blob = build_account_context_blob(rfmt_df, clv_df, segment_summary, pd.DataFrame(), crosstab_counts)
        assert blob["cohort_retention_by_month"] == {}

    def test_none_crosstab_degrades_to_empty_summary_not_an_error(
        self, rfmt_df, clv_df, segment_summary, cohort_retention_matrix
    ):
        blob = build_account_context_blob(rfmt_df, clv_df, segment_summary, cohort_retention_matrix, None)
        assert blob["segment_cluster_crosstab"] == {}

    def test_empty_crosstab_degrades_to_empty_summary_not_an_error(
        self, rfmt_df, clv_df, segment_summary, cohort_retention_matrix
    ):
        blob = build_account_context_blob(rfmt_df, clv_df, segment_summary, cohort_retention_matrix, pd.DataFrame())
        assert blob["segment_cluster_crosstab"] == {}

    def test_both_optional_inputs_missing_still_builds_a_usable_blob(
        self, rfmt_df, clv_df, segment_summary
    ):
        blob = build_account_context_blob(rfmt_df, clv_df, segment_summary, None, None)
        assert blob["aggregate_stats"]["total_customers"] == len(rfmt_df)
        text = build_context_text(blob)
        assert "no cohort retention data available" in text
        assert "no ML clustering data available" in text


class TestContextTextContainsOnlyAggregateData:
    """Mirrors tests/test_digest_engine.py::test_prompt_never_contains_a_raw_customer_id
    exactly -- same guarantee, same test shape, applied to this module's serialized
    context blob instead of digest_engine.py's prompt."""

    def test_context_text_never_contains_a_raw_customer_id(self, rfmt_df, context_text):
        for customer_id in rfmt_df["CustomerID"].astype(str):
            assert customer_id not in context_text

    def test_context_text_contains_the_documented_aggregate_totals(self, context_blob, context_text):
        stats = context_blob["aggregate_stats"]
        assert f"{stats['total_customers']:,}" in context_text
        assert f"{stats['pct_at_risk']:.1f}%" in context_text
        for seg in context_blob["segment_breakdown"]:
            assert seg["segment"] in context_text

    def test_context_text_stays_a_reasonable_size(self, context_text):
        # "Keep the blob a reasonable size" per the task spec -- a loose upper
        # bound (not a tight one) just to catch an accidental future regression
        # toward per-row serialization, not to constrain normal growth.
        assert len(context_text) < 8000


class TestEmptyOptionalSlicesDegradeToEmptyDicts:
    def test_zero_churn_watchlist_and_growth_targets_yield_empty_composition_dicts(
        self, rfmt_df, clv_df, segment_summary, cohort_retention_matrix, crosstab_counts
    ):
        # A churn-free, zero-forecast account has an empty watchlist/growth-targets
        # slice -- _segment_value_counts_dict()'s empty-input branch.
        clv_df_no_risk = clv_df.copy()
        clv_df_no_risk["P_Alive"] = 0.99  # nobody crosses the watchlist threshold
        clv_df_no_risk["Predicted_Spend_90d"] = 0.0
        blob = build_account_context_blob(
            rfmt_df, clv_df_no_risk, segment_summary, cohort_retention_matrix, crosstab_counts
        )
        assert blob["churn_watchlist"]["segment_composition"] == {}
        assert blob["churn_watchlist"]["count"] == 0

    def test_crosstab_row_that_sums_to_zero_is_skipped_not_included(
        self, rfmt_df, clv_df, segment_summary, cohort_retention_matrix
    ):
        # A Segment category with zero customers in the crosstab (e.g. an unused
        # pandas Categorical level) must be skipped, not divide-by-zero or appear
        # with a nonsensical 0/0 share.
        zero_row_crosstab = pd.DataFrame(
            {"Cluster 1": [0, 5], "Cluster 2": [0, 2]},
            index=pd.Index(["Empty Segment", "Champions"], name="Segment"),
        )
        blob = build_account_context_blob(
            rfmt_df, clv_df, segment_summary, cohort_retention_matrix, zero_row_crosstab
        )
        assert "Empty Segment" not in blob["segment_cluster_crosstab"]
        assert "Champions" in blob["segment_cluster_crosstab"]


class TestBuildAccountContextBlobDoesNotMutateInputs:
    def test_segment_summary_dataframe_is_not_mutated(self, rfmt_df, clv_df, segment_summary, cohort_retention_matrix, crosstab_counts):
        before = segment_summary.copy(deep=True)
        build_account_context_blob(rfmt_df, clv_df, segment_summary, cohort_retention_matrix, crosstab_counts)
        pd.testing.assert_frame_equal(before, segment_summary)


class TestMethodologyContextCrossReferencesReadme:
    """
    The actual point of this test class: build_methodology_context() must
    contain the SAME facts/numbers as the specific README.md sections it
    claims to source from -- not just "a methodology section exists" but
    "the honest, specific claims a professor could fact-check are actually
    in there." The expected values below were copied by hand from README.md
    at the time this test was written (not derived from the function under
    test) -- if a real recalibration changes the backtest numbers, THIS test
    should fail until README.md and this file are updated together.
    """

    @pytest.fixture(scope="module")
    def methodology(self):
        return build_methodology_context()

    def test_returns_the_documented_top_level_keys(self, methodology):
        assert set(methodology.keys()) == {
            "rfm_segment_methodology", "clv_model_caveats",
            "ml_clustering_scope", "backtest_results",
        }

    def test_segment_precedence_order_matches_readme_rule_precedence_section(self, methodology):
        # Source: README "Rule precedence" under "Enterprise 7-Segment
        # Taxonomy & Marketing Playbooks" -- the exact first-match-wins order.
        assert methodology["rfm_segment_methodology"]["precedence_order"] == [
            "New Customers", "Champions", "Can't Lose Them", "At-Risk VIPs",
            "Potential Growth", "Loyalists", "Hibernating",
        ]

    def test_clv_model_is_described_as_heuristic_not_a_fitted_probabilistic_model(self, methodology):
        # Source: README "5. Heuristic Churn-Hazard Model (BTYD-Inspired)".
        # Do not soften this into "a BTYD model" or similar.
        model_type = methodology["clv_model_caveats"]["model_type"].lower()
        assert "heuristic" in model_type
        assert "not a fitted" in model_type or "not fitted" in model_type
        assert "no likelihood" in model_type or "no posterior" in model_type

    def test_clv_constants_are_described_as_manually_tuned_not_data_derived(self, methodology):
        # Source: README "Configuring the Churn Hazard Model".
        caveat = methodology["clv_model_caveats"]["constants_are_manually_tuned"].lower()
        assert "manually-tuned" in caveat or "manually tuned" in caveat
        assert "not a value derived from real churn outcomes" in caveat or "not derived from real" in caveat

    def test_ml_clustering_is_described_as_validation_only_never_modulating_segment(self, methodology):
        # Source: README "ML Cluster vs. Segment Agreement: What the
        # Clustering Step Is Actually For" -- the specific claim a professor
        # is likely to probe: does clustering change which segment a
        # customer is in? The README says no, explicitly -- so must this.
        scope = methodology["ml_clustering_scope"]["validation_only"]
        assert "does NOT modulate the Segment label" in scope
        assert "Urgent Churn Watchlist" in scope
        assert "deliberate scope boundary" in scope

    def test_synthetic_backtest_numbers_match_readme_exactly(self, methodology):
        # Source: README "Results on the synthetic dataset (ecommerce_
        # transactions.csv, 384 backtested customers)" table + churn-flag line.
        synth = methodology["backtest_results"]["synthetic_dataset"]
        assert synth["n_backtested_customers"] == 384
        assert synth["model_mae_usd"] == 752.61
        assert synth["model_rmse_usd"] == 1172.77
        assert synth["baseline_trailing_90d_mae_usd"] == 912.36
        assert synth["baseline_population_mean_mae_usd"] == 966.68
        assert synth["model_beats_both_baselines_on_mae"] is True
        assert synth["churn_flag_precision_pct"] == 74.6
        assert synth["churn_flag_recall_pct"] == 47.5

    def test_real_dataset_backtest_numbers_match_readme_exactly(self, methodology):
        # Source: README "Results on the real UCI Online Retail dataset
        # (real_online_retail.csv.gz, 3,370 backtested customers)" table +
        # churn-flag line. This is the single most important fact in this
        # whole file to get right: it must say the model LOSES here.
        real = methodology["backtest_results"]["real_uci_dataset"]
        assert real["n_backtested_customers"] == 3370
        assert real["model_mae_usd"] == 682.05
        assert real["model_rmse_usd"] == 4232.38
        assert real["baseline_trailing_90d_mae_usd"] == 657.56
        assert real["baseline_trailing_90d_rmse_usd"] == 4054.82
        assert real["model_beats_both_baselines_on_mae"] is False
        assert real["churn_flag_precision_pct"] == 48.4
        assert real["churn_flag_recall_pct"] == 10.3

    def test_honest_bottom_line_states_the_model_loses_on_real_data_without_spin(self, methodology):
        # Source: README "Bottom line" paragraph. Assert on the actual claim,
        # not just that some text is present -- a vague or softened
        # restatement would defeat the entire point of this section.
        bottom_line = methodology["backtest_results"]["honest_bottom_line"]
        assert "does NOT beat" in bottom_line
        assert "trailing-spend baseline wins" in bottom_line or "trailing spend baseline wins" in bottom_line.replace("-", " ")
        assert "10.3%" in bottom_line


class TestContextTextIncludesMethodologySection:
    """Confirms build_context_text() actually serializes the methodology
    section (not just that build_account_context_blob() carries it in the
    dict) -- this is what the model actually sees in the system prompt."""

    def test_methodology_section_header_is_present(self, context_text):
        assert "MODEL METHODOLOGY & KNOWN LIMITATIONS" in context_text

    def test_segment_precedence_rules_appear_in_order(self, context_text):
        idx_new = context_text.index("New Customers is checked first")
        idx_champions = context_text.index("Champions is checked next")
        idx_cant_lose = context_text.index("Can't Lose Them is checked before At-Risk VIPs")
        idx_hibernating = context_text.index("Hibernating is the default")
        assert idx_new < idx_champions < idx_cant_lose < idx_hibernating

    def test_clv_heuristic_caveat_appears(self, context_text):
        assert "NOT a fitted probabilistic model" in context_text

    def test_ml_clustering_validation_only_scope_appears(self, context_text):
        assert "does NOT modulate the Segment label" in context_text

    def test_real_dataset_backtest_numbers_appear(self, context_text):
        assert "$682.05" in context_text  # real-dataset model MAE
        assert "$657.56" in context_text  # real-dataset trailing-90d baseline MAE (the winner)
        assert "10.3%" in context_text    # real-dataset churn-flag recall

    def test_synthetic_dataset_backtest_numbers_appear(self, context_text):
        assert "$752.61" in context_text  # synthetic-dataset model MAE
        assert "74.6%" in context_text    # synthetic-dataset churn-flag precision

    def test_honest_bottom_line_appears_verbatim_in_spirit(self, context_text):
        assert "does NOT beat the simplest baseline" in context_text

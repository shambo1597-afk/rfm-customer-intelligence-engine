"""
src/chat_context.py - Builds the precomputed, aggregate-only context blob for the
optional Chat Q&A feature (src/chat_engine.py).

This module does NOT call any LLM and makes NO network calls. It assembles a
structured dict from output the pipeline has ALREADY COMPUTED (RFM-T, CLV/churn,
cohort retention, ML clustering) -- no new pandas computation happens here, and
nothing here ever reads a raw per-customer row into the output. Built ONCE per
batch run (same cadence as the existing digest_engine.py summary), then reused
for every question asked in a chat session -- never rebuilt per-question, and
the chat engine never queries these DataFrames directly mid-conversation.

--- Why this exists as a separate blob-building step, not live tool-calling ---

The person deploying this explicitly does NOT want an open-ended chat agent that
calls back into src/rfm_engine.py / src/clv_engine.py mid-conversation to answer
questions (function/tool-calling). That would turn a bounded, once-per-batch-run
LLM cost into an unbounded one, and risks the model inventing plausible-sounding
numbers instead of reporting only what the pipeline actually computed. Building
one static, aggregate-only context blob up front -- and instructing the model
(src/chat_engine.py's system prompt) to answer ONLY from it -- keeps both
properties intact: the model can genuinely say "I don't know" for anything
outside the blob, and there's a single point (this file) where the PII posture
is enforced and testable, exactly like _build_prompt()/_build_aggregate_stats()
in digest_engine.py.

--- PII posture (matches digest_engine.py exactly) ---

Every field in the returned dict is an aggregate: a count, a sum, a mean, a
percentage, or a segment/cluster NAME (not an instance identifier). No
CustomerID, no individual transaction row, no per-customer loop of any kind
ever reaches the returned dict or its serialized text form. See
tests/test_chat_context.py's PII-exclusion test, which mirrors
tests/test_digest_engine.py::test_prompt_never_contains_a_raw_customer_id.

--- What's in the blob, and what's deliberately left aggregate-only ---

- _build_aggregate_stats() output, reused directly from digest_engine.py rather
  than re-derived here (same numbers, one source of truth).
- Segment KPI breakdown (from src/rfm_engine.py's get_segment_kpi_summary output,
  passed in already computed -- segment-level rows only, never per-customer).
- Churn watchlist: size, total value, average P(Alive), and segment composition
  of src/clv_engine.py's get_urgent_churn_watchlist() output -- counts and sums,
  never the individual watchlist rows themselves.
- Growth targets: size, total predicted 90-day value, and segment composition of
  get_top_future_growth_targets() output -- same aggregate framing as the
  watchlist, not individual target-customer rows. (The task spec that
  introduced this file left surfacing a few anonymized top-line examples --
  e.g. "top target segment X, worth $Y" -- open to a follow-up decision;
  absent an explicit confirmation, this defaults to aggregate-only, which is
  what's implemented here.)
- Cohort retention: average retention percentage by month-since-acquisition
  across ALL cohorts (a single row, not the full per-cohort-period grid) --
  keeps the blob compact per the task's "reasonable size" instruction.
- Segment x ML_Cluster crosstab: reduced to each segment's single dominant
  ML_Cluster and that cluster's share of the segment -- not the full grid --
  same "keep it compact" reasoning as the cohort summary above.
- Methodology & known limitations (build_methodology_context()): static content
  about the PIPELINE'S OWN DESIGN (segment rule precedence, the CLV model's
  heuristic/not-fitted nature, the ML clustering's validation-only scope, and
  the actual out-of-sample backtest numbers) -- not per-account data, computed
  once, effectively free. Added so the chatbot can answer methodology/
  confidence/limitations questions ("how confident is this CLV number," "does
  ML clustering change a customer's segment") accurately instead of only being
  able to restate a raw number with no context to reason about it. Every fact
  in it is pulled from an already-written, already-verified README section --
  see the source-section comment next to each fact in
  build_methodology_context() below. Nothing here is invented.
"""

from src.digest_engine import _build_aggregate_stats
from src.clv_engine import get_urgent_churn_watchlist, get_top_future_growth_targets

# How many growth targets get_top_future_growth_targets() considers before this
# module reduces them to an aggregate count/sum/composition -- mirrors that
# function's own top_n default; named here so it's an intentional choice, not a
# silently-inherited default.
GROWTH_TARGETS_TOP_N = 10


def _segment_value_counts_dict(df, column: str = "Segment") -> dict:
    """
    Aggregate-only segment composition of a DataFrame slice (e.g. the churn
    watchlist or growth-target rows) -- a {segment_name: count} dict, never the
    underlying rows. Returns {} for an empty/missing-column input rather than
    raising, since both the watchlist and growth-target slices can legitimately
    be empty (a healthy account may have no urgent churn risk).
    """
    if df is None or df.empty or column not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df[column].value_counts().items()}


def build_methodology_context() -> dict:
    """
    Static methodology/limitations content about the PIPELINE'S OWN DESIGN --
    not per-account data, takes no DataFrame inputs, and needs no computation
    (it's a fixed dict of facts already written and verified in README.md).
    Effectively free relative to the rest of the blob; still built once per
    batch run alongside everything else in this file, for a single call site
    and a single place to keep this in sync with the README.

    Every fact below is pulled from an EXISTING, already-verified README
    section -- see the comment citing the source section next to each one.
    Nothing here is a new caveat, threshold, or number invented for this
    function; where a number appears, it is the exact figure the README
    reports (paraphrasing prose for prompt brevity is fine per the task that
    added this function, but numbers are copied verbatim).
    """
    rfm_segment_methodology = {
        # Source: README "Rule precedence" under "Enterprise 7-Segment
        # Taxonomy & Marketing Playbooks" -- assign_segments_vectorized()
        # (src/rfm_engine.py) resolves overlapping rule matches via
        # numpy.select, first-match-wins, in this exact order (not the
        # value-tier display order of the segment table).
        "precedence_order": [
            "New Customers", "Champions", "Can't Lose Them", "At-Risk VIPs",
            "Potential Growth", "Loyalists", "Hibernating",
        ],
        "precedence_rules": [
            "New Customers is checked first and deliberately overrides any other match, "
            "so a customer's very first purchase isn't misclassified as an established "
            "Champion just because it happened to be large.",
            "Champions is checked next.",
            "Can't Lose Them is checked before At-Risk VIPs because Recency-score 1 "
            "(completely dormant) is a stricter, more urgent subset of At-Risk VIPs' "
            "Recency-score <= 2 condition -- e.g. a completely dormant former big spender "
            "is deterministically Can't Lose Them, never At-Risk VIPs.",
            "At-Risk VIPs is checked next.",
            "Potential Growth is checked next.",
            "Loyalists is checked next.",
            "Hibernating is the default -- every customer matching none of the above rules.",
        ],
    }

    clv_model_caveats = {
        # Source: README "5. Heuristic Churn-Hazard Model (BTYD-Inspired) CLV
        # & Churn Radar" -- states this explicitly, do not soften it.
        "model_type": (
            "A heuristic, rule-based hazard function inspired by the shape of "
            "continuous-time Buy-Till-You-Die (BTYD/BG-NBD) models -- NOT a fitted "
            "probabilistic model in the statistical sense: no likelihood is "
            "maximized, no posterior is estimated."
        ),
        # Source: README "Configuring the Churn Hazard Model".
        "constants_are_manually_tuned": (
            "Every constant in the formula (the lambda-smoothing alpha/beta priors "
            "and the four churn-hazard weight/offset terms) is a manually-tuned "
            "default calibrated to look reasonable on this project's SYNTHETIC "
            "dataset -- not a value derived from real churn outcomes. They are "
            "exposed as function parameters on estimate_btyd_clv() specifically so "
            "a real deployment can recalibrate them against its own known-outcome "
            "data instead of trusting these defaults."
        ),
    }

    ml_clustering_scope = {
        # Source: README "ML Cluster vs. Segment Agreement: What the
        # Clustering Step Is Actually For".
        "independence": (
            "K-Means clustering and the 7-segment rule taxonomy are computed "
            "completely independently -- the rules have no knowledge of the "
            "clustering, and K-Means has no knowledge of the rules."
        ),
        "validation_only": (
            "The Segment x ML_Cluster crosstab is a validation/diagnostic tool for "
            "the analyst -- it answers whether the unsupervised clustering "
            "corroborates the hand-picked segment thresholds, or suggests one needs "
            "revisiting. It does NOT modulate the Segment label, the Urgent Churn "
            "Watchlist's ranking, or any other customer-facing output anywhere in "
            "this platform. This is a deliberate scope boundary for this version, "
            "not an unfinished feature."
        ),
    }

    backtest_results = {
        # Source: README "Model Validation & Backtest Results" -- "Method" paragraph.
        "method": (
            "backtest_clv.py picks a cutoff date 90 days before the last transaction "
            "in the dataset, computes RFM-T and the CLV forecast using only "
            "transactions on or before that cutoff, then compares the forecast "
            "against what those customers ACTUALLY did in the following 90 days -- "
            "a genuine temporal train/test split, not a check against the same data "
            "the model was fit on. Checked against two naive baselines computed the "
            "same out-of-sample way: each customer repeats their own trailing "
            "90-day spend, and every customer gets the population-average trailing "
            "spend."
        ),
        # Source: README "Results on the synthetic dataset (ecommerce_transactions.csv,
        # 384 backtested customers)" -- figures copied verbatim.
        "synthetic_dataset": {
            "n_backtested_customers": 384,
            "model_mae_usd": 752.61,
            "model_rmse_usd": 1172.77,
            "baseline_trailing_90d_mae_usd": 912.36,
            "baseline_trailing_90d_rmse_usd": 1485.98,
            "baseline_population_mean_mae_usd": 966.68,
            "baseline_population_mean_rmse_usd": 1265.96,
            "model_beats_both_baselines_on_mae": True,
            "churn_flag_precision_pct": 74.6,
            "churn_flag_recall_pct": 47.5,
        },
        # Source: README "Results on the real UCI Online Retail dataset
        # (real_online_retail.csv.gz, 3,370 backtested customers)" -- figures
        # copied verbatim, INCLUDING the honest negative finding.
        "real_uci_dataset": {
            "n_backtested_customers": 3370,
            "model_mae_usd": 682.05,
            "model_rmse_usd": 4232.38,
            "baseline_trailing_90d_mae_usd": 657.56,
            "baseline_trailing_90d_rmse_usd": 4054.82,
            "baseline_population_mean_mae_usd": 910.68,
            "baseline_population_mean_rmse_usd": 5015.35,
            "model_beats_both_baselines_on_mae": False,
            "churn_flag_precision_pct": 48.4,
            "churn_flag_recall_pct": 10.3,
        },
        # Source: README "Bottom line" paragraph -- the single most important
        # fact to get right here, per the task that added this function:
        # state it accurately and without spin, never soften it.
        "honest_bottom_line": (
            "On REAL transaction data, the model does NOT beat the simplest "
            "baseline (predicting each customer repeats their own trailing 90-day "
            "spend) on either MAE or RMSE -- it is close, but the trailing-spend "
            "baseline wins. Treat the 90-day spend forecast and the churn flag as a "
            "cheap, directional prioritization signal, not a precise forecast: on "
            "real data it misses roughly 9 out of 10 customers who actually go "
            "quiet (10.3% recall)."
        ),
    }

    return {
        "rfm_segment_methodology": rfm_segment_methodology,
        "clv_model_caveats": clv_model_caveats,
        "ml_clustering_scope": ml_clustering_scope,
        "backtest_results": backtest_results,
    }


def build_account_context_blob(
    rfmt_df, clv_df, segment_summary, cohort_matrix, crosstab_counts
) -> dict:
    """
    Assembles the structured, aggregate-only context dict for one account, from
    already-computed pipeline outputs. Called ONCE per batch run (see
    src/chat_context.py's module docstring) -- never re-derives anything
    pandas-side that the pipeline hasn't already computed.

    Parameters (all already-computed pipeline outputs, read-only here):
    - rfmt_df, clv_df, segment_summary: same objects generate_account_digest()
      in src/digest_engine.py already consumes -- see src/rfm_engine.py,
      src/clv_engine.py.
    - cohort_matrix: the retention-percentage DataFrame from
      src/cohort_engine.py's compute_monthly_cohort_matrix() (its second
      return value) -- rows are acquisition cohorts, columns are months-since-
      acquisition, values are retention %. May be None/empty (e.g. a dataset
      too small or too recent to form cohorts) -- degrades to an empty summary
      rather than raising.
    - crosstab_counts: the raw-counts DataFrame from src/ml_engine.py's
      compute_segment_cluster_crosstab() (its first return value) -- Segment
      (rows) x ML_Cluster (columns) customer counts. Also optional/nullable.

    Returns a dict of aggregate-only sub-sections (see module docstring for
    exactly what each contains and why nothing here is per-customer):
      aggregate_stats, segment_breakdown, churn_watchlist, growth_targets,
      cohort_retention_by_month, segment_cluster_crosstab, methodology.
    """
    aggregate_stats = _build_aggregate_stats(rfmt_df, clv_df, segment_summary)

    segment_breakdown = []
    if segment_summary is not None and not segment_summary.empty:
        for _, row in segment_summary.iterrows():
            segment_breakdown.append({
                "segment": str(row["Segment"]),
                "customer_count": int(row["CustomerCount"]),
                "customer_share_pct": float(row["CustomerSharePct"]) if "CustomerSharePct" in segment_summary.columns else 0.0,
                "total_revenue": float(row["TotalRevenue"]) if "TotalRevenue" in segment_summary.columns else 0.0,
                "revenue_share_pct": float(row["RevenueSharePct"]) if "RevenueSharePct" in segment_summary.columns else 0.0,
                "avg_recency_days": float(row["AvgRecency"]) if "AvgRecency" in segment_summary.columns else 0.0,
                "avg_frequency": float(row["AvgFrequency"]) if "AvgFrequency" in segment_summary.columns else 0.0,
                "avg_tenure_days": float(row["AvgTenure"]) if "AvgTenure" in segment_summary.columns else 0.0,
            })

    watchlist_df = get_urgent_churn_watchlist(clv_df)
    churn_watchlist = {
        "count": int(len(watchlist_df)),
        "total_value": float(watchlist_df["Monetary"].sum()) if not watchlist_df.empty else 0.0,
        "avg_p_alive_pct": float(watchlist_df["P_Alive_Pct"].mean()) if not watchlist_df.empty else 0.0,
        "segment_composition": _segment_value_counts_dict(watchlist_df),
    }

    growth_df = get_top_future_growth_targets(clv_df, top_n=GROWTH_TARGETS_TOP_N)
    growth_targets = {
        "count": int(len(growth_df)),
        "total_predicted_90d_value": float(growth_df["Predicted_Spend_90d"].sum()) if not growth_df.empty else 0.0,
        "segment_composition": _segment_value_counts_dict(growth_df),
    }

    cohort_retention_by_month = {}
    if cohort_matrix is not None and not cohort_matrix.empty:
        mean_retention = cohort_matrix.mean(axis=0, numeric_only=True).round(1)
        cohort_retention_by_month = {int(month): float(pct) for month, pct in mean_retention.items()}

    segment_cluster_crosstab = {}
    if crosstab_counts is not None and not crosstab_counts.empty:
        for segment in crosstab_counts.index:
            row = crosstab_counts.loc[segment]
            total = int(row.sum())
            if total == 0:
                continue
            segment_cluster_crosstab[str(segment)] = {
                "dominant_ml_cluster": str(row.idxmax()),
                "dominant_cluster_share_pct": round(float(row.max()) / total * 100.0, 1),
            }

    return {
        "aggregate_stats": aggregate_stats,
        "segment_breakdown": segment_breakdown,
        "churn_watchlist": churn_watchlist,
        "growth_targets": growth_targets,
        "cohort_retention_by_month": cohort_retention_by_month,
        "segment_cluster_crosstab": segment_cluster_crosstab,
        "methodology": build_methodology_context(),
    }


def build_context_text(blob: dict) -> str:
    """
    Serializes build_account_context_blob()'s dict into a compact text block for
    the chat system prompt (src/chat_engine.py) -- same spirit as
    digest_engine.py's _build_prompt(), and the same guarantee: every value
    written out here comes from the blob dict, which is itself 100% aggregate
    (see module docstring) -- no CustomerID, no per-row data of any kind.
    """
    stats = blob["aggregate_stats"]

    segment_lines = "\n".join(
        f"- {seg['segment']}: {seg['customer_count']:,} customers ({seg['customer_share_pct']:.1f}% of total), "
        f"${seg['total_revenue']:,.2f} revenue ({seg['revenue_share_pct']:.1f}% of total), "
        f"avg recency {seg['avg_recency_days']:.0f} days, avg {seg['avg_frequency']:.1f} orders, "
        f"avg tenure {seg['avg_tenure_days']:.0f} days"
        for seg in blob["segment_breakdown"]
    ) or "- (no segment data available)"

    watchlist = blob["churn_watchlist"]
    watchlist_composition = ", ".join(
        f"{seg}: {count}" for seg, count in watchlist["segment_composition"].items()
    ) or "none"

    growth = blob["growth_targets"]
    growth_composition = ", ".join(
        f"{seg}: {count}" for seg, count in growth["segment_composition"].items()
    ) or "none"

    cohort_lines = "\n".join(
        f"- Month {month}: {pct:.1f}% average retention"
        for month, pct in sorted(blob["cohort_retention_by_month"].items())
    ) or "- (no cohort retention data available)"

    crosstab_lines = "\n".join(
        f"- {segment}: {info['dominant_ml_cluster']} ({info['dominant_cluster_share_pct']:.1f}% of segment)"
        for segment, info in blob["segment_cluster_crosstab"].items()
    ) or "- (no ML clustering data available)"

    methodology = blob["methodology"]
    rfm_meth = methodology["rfm_segment_methodology"]
    precedence_lines = "\n".join(
        f"{i}. {rule}" for i, rule in enumerate(rfm_meth["precedence_rules"], start=1)
    )
    clv_caveats = methodology["clv_model_caveats"]
    ml_scope = methodology["ml_clustering_scope"]
    bt = methodology["backtest_results"]
    synth = bt["synthetic_dataset"]
    real = bt["real_uci_dataset"]

    methodology_text = (
        "MODEL METHODOLOGY & KNOWN LIMITATIONS (use this section for questions about "
        "HOW the scoring/segmentation/CLV works, its accuracy, or its limitations -- "
        "not about this account's specific numbers)\n\n"
        f"Segment assignment order ({', '.join(rfm_meth['precedence_order'])}) is "
        "first-match-wins, not the display order of the segment table:\n"
        f"{precedence_lines}\n\n"
        f"CLV/churn model: {clv_caveats['model_type']} {clv_caveats['constants_are_manually_tuned']}\n\n"
        f"ML clustering scope: {ml_scope['independence']} {ml_scope['validation_only']}\n\n"
        f"Out-of-sample backtest results ({bt['method']}):\n"
        f"- Synthetic dataset ({synth['n_backtested_customers']:,} backtested customers): "
        f"model MAE ${synth['model_mae_usd']:,.2f} / RMSE ${synth['model_rmse_usd']:,.2f}, "
        f"vs. trailing-90-day baseline MAE ${synth['baseline_trailing_90d_mae_usd']:,.2f} / "
        f"RMSE ${synth['baseline_trailing_90d_rmse_usd']:,.2f} and population-mean baseline "
        f"MAE ${synth['baseline_population_mean_mae_usd']:,.2f} / RMSE ${synth['baseline_population_mean_rmse_usd']:,.2f} "
        f"-- the model beats both baselines on MAE here. Churn flag: "
        f"{synth['churn_flag_precision_pct']:.1f}% precision, {synth['churn_flag_recall_pct']:.1f}% recall.\n"
        f"- Real UCI Online Retail dataset ({real['n_backtested_customers']:,} backtested customers): "
        f"model MAE ${real['model_mae_usd']:,.2f} / RMSE ${real['model_rmse_usd']:,.2f}, "
        f"vs. trailing-90-day baseline MAE ${real['baseline_trailing_90d_mae_usd']:,.2f} / "
        f"RMSE ${real['baseline_trailing_90d_rmse_usd']:,.2f} and population-mean baseline "
        f"MAE ${real['baseline_population_mean_mae_usd']:,.2f} / RMSE ${real['baseline_population_mean_rmse_usd']:,.2f} "
        f"-- the model does NOT beat the trailing-90-day baseline here. Churn flag: "
        f"{real['churn_flag_precision_pct']:.1f}% precision, {real['churn_flag_recall_pct']:.1f}% recall.\n\n"
        f"{bt['honest_bottom_line']}\n"
    )

    return (
        "ACCOUNT-WIDE AGGREGATE STATISTICS\n"
        f"Total customers: {stats['total_customers']:,}\n"
        f"Total historical revenue: ${stats['total_historical_revenue']:,.2f}\n"
        f"Predicted next-90-day revenue: ${stats['total_predicted_90d_revenue']:,.2f}\n"
        f"Customers at high churn risk: {stats['pct_at_risk']:.1f}%\n\n"
        "SEGMENT BREAKDOWN (RFM-T rule-based taxonomy)\n"
        f"{segment_lines}\n\n"
        "CHURN WATCHLIST (high-spend accounts in critical P(Alive) decay)\n"
        f"Count: {watchlist['count']:,}\n"
        f"Total value at risk: ${watchlist['total_value']:,.2f}\n"
        f"Average P(Alive): {watchlist['avg_p_alive_pct']:.1f}%\n"
        f"Segment composition: {watchlist_composition}\n\n"
        "TOP 90-DAY GROWTH TARGETS\n"
        f"Count: {growth['count']:,}\n"
        f"Total predicted 90-day value: ${growth['total_predicted_90d_value']:,.2f}\n"
        f"Segment composition: {growth_composition}\n\n"
        "COHORT RETENTION (average across all acquisition cohorts, by month since acquisition)\n"
        f"{cohort_lines}\n\n"
        "SEGMENT x ML CLUSTER AGREEMENT (independent unsupervised validation of the segment rules)\n"
        f"{crosstab_lines}\n\n"
        f"{methodology_text}"
    )

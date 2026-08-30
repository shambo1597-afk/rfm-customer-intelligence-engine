"""
src/roi_advisor.py - Optional AI Budget Advisor for the "What-If" Campaign ROI
Simulator (app.py Tab 5, the same tab README calls "What-If Campaign ROI
Simulation Framework").

app.py's Tab 5 already lets an analyst simulate ONE segment's campaign ROI at
a time via sliders (target segment, audience reach, budget, conversion rate,
gross margin) -- all deterministic math, zero API calls, unchanged by this
module. This file adds a second, optional layer on top: run that SAME
deterministic math across EVERY segment at once (splitting one total budget
across them), then have an LLM explain the resulting comparison and recommend
an allocation. The LLM never computes a number -- it only reads and reasons
about a table this module has already built with plain arithmetic.

--- Two things this deliberately does NOT do ---

- No live tool-calling. get_roi_recommendation() is a single one-shot call
  against a fixed table already computed by simulate_all_segment_allocations()
  -- the model cannot ask this codebase to recompute anything mid-answer, the
  same architectural boundary src/chat_context.py/chat_engine.py already
  established for account Q&A.
- No AI-generated campaign copy. src/rfm_engine.py's own SEGMENT_PLAYBOOKS
  already ships a static campaign-copy template per segment (subject/
  headline/body/cta, shown in app.py Tab 5's "Campaign Copy Blueprint");
  generating NEW marketing copy via an LLM was explicitly out of scope for
  the task that added this module and is not attempted here.

--- Formulas: one source of truth, not duplicated a second time ---

simulate_campaign_roi() below is the EXACT math app.py's Tab 5 slider-driven
simulator uses (README's "What-If Campaign ROI Simulation Framework" section
documents the same formulas). That math used to live inline in app.py; it now
lives here instead, and app.py's Tab 5 calls this function directly rather
than recomputing the same five lines of arithmetic a second time.
simulate_all_segment_allocations() below calls the identical function once
per segment -- so there is exactly one place these formulas are defined,
regardless of whether they're being used for the single-segment interactive
simulator or the multi-segment advisor comparison.

--- Providers, model, and error handling: reused, not re-derived ---

Exactly like src/chat_engine.py, this module does NOT define its own
provider-selection logic, model constants, provider-client construction, or
exception-handling pattern -- it imports and reuses src/digest_engine.py's
_resolve_provider() / _call_groq() / _call_anthropic(). See
src/chat_engine.py's own module docstring for the fuller rationale (the same
one applies here verbatim): duplicating that logic a third time was a real,
already-experienced maintainability risk in this codebase, not a
hypothetical one.

--- Cost model: one call per "Get Recommendation" click, not per rerun ---

Like Chat Q&A, this is usage-based rather than the AI Digest's fixed
one-call-per-batch-run bound -- but narrower in practice, since a budget
question is typically asked once or twice per campaign-planning session, not
5-20 times the way a free-form chat conversation runs. app.py caches the call
via st.cache_data (keyed on the allocations table, budget, question, and
keys) for the same reason the digest and chat-context blob are cached:
Streamlit reruns the whole script on every widget interaction elsewhere in
the app, and without caching, an unrelated slider move in another tab would
silently re-trigger a fresh API call.
"""

import pandas as pd

from src.rfm_engine import get_segment_kpi_summary
from src.digest_engine import (
    anthropic,
    groq,
    _resolve_provider,
    _call_groq,
    _call_anthropic,
    GROQ_REASONING_TOKEN_HEADROOM,
)

# Mirrors app.py Tab 5's own slider defaults (`value=8.5` for the conversion-
# rate slider, `value=40` for the gross-margin slider) exactly -- a single
# source of truth so simulate_all_segment_allocations()'s "no explicit
# per-segment override" default can never silently drift from the
# interactive simulator's own default. app.py imports these same constants
# for its slider `value=` arguments instead of repeating the literals a
# second time (the same "named constant, not a silently-inherited default"
# discipline src/chat_context.py already applies to its own row caps).
DEFAULT_CONV_RATE_PCT = 8.5
DEFAULT_GROSS_MARGIN_PCT = 40

# Advisor answers get a bit more room than the digest's 300-token cap (a
# recommendation that compares several segments' figures needs more than one
# short paragraph) but stay well short of chat's 500-token cap -- this is a
# single one-shot answer, not a back-and-forth conversation. This is the
# visible-answer target passed to Anthropic unchanged; see
# GROQ_ROI_ADVISOR_MAX_OUTPUT_TOKENS below for what's actually sent to Groq.
ROI_ADVISOR_MAX_OUTPUT_TOKENS = 400
# The value actually passed as max_tokens on the Groq call specifically --
# ROI_ADVISOR_MAX_OUTPUT_TOKENS's own visible-answer target (400) is
# unchanged; only Groq's call gets the extra reasoning headroom (see
# GROQ_REASONING_TOKEN_HEADROOM's own comment in digest_engine.py for the
# reasoning-token-truncation incident this fixes).
GROQ_ROI_ADVISOR_MAX_OUTPUT_TOKENS = ROI_ADVISOR_MAX_OUTPUT_TOKENS + GROQ_REASONING_TOKEN_HEADROOM

_UNAVAILABLE_PREFIX = "**[Budget advisor temporarily unavailable"


def simulate_campaign_roi(
    audience_size: float,
    avg_segment_aov: float,
    campaign_budget: float,
    conv_rate_pct: float,
    gross_margin_pct: float,
) -> dict:
    """
    THE single source of truth for this platform's campaign-ROI formulas --
    see the module docstring's "Formulas" section and README's "What-If
    Campaign ROI Simulation Framework" for the documented math. Every input
    is already a plain number the caller has computed or chosen; this
    function does no DataFrame/data lookup of its own, so it is trivially
    testable against hand-calculated values (see tests/test_roi_advisor.py).

    Returns a dict with projected_conversions, projected_gross_revenue,
    projected_gross_profit, net_incremental_profit, campaign_roi_pct, and
    cost_per_acquisition -- exactly the figures app.py's Tab 5 already
    displays for a single segment.
    """
    projected_conversions = audience_size * (conv_rate_pct / 100.0)
    projected_gross_revenue = projected_conversions * avg_segment_aov
    projected_gross_profit = projected_gross_revenue * (gross_margin_pct / 100.0)
    net_incremental_profit = projected_gross_profit - campaign_budget
    campaign_roi_pct = (net_incremental_profit / max(campaign_budget, 1)) * 100.0
    cost_per_acquisition = campaign_budget / max(projected_conversions, 1)
    return {
        "projected_conversions": projected_conversions,
        "projected_gross_revenue": projected_gross_revenue,
        "projected_gross_profit": projected_gross_profit,
        "net_incremental_profit": net_incremental_profit,
        "campaign_roi_pct": campaign_roi_pct,
        "cost_per_acquisition": cost_per_acquisition,
    }


def simulate_all_segment_allocations(
    rfmt_df: pd.DataFrame,
    clv_df: pd.DataFrame,
    total_budget: float,
    conv_rate_by_segment: dict = None,
) -> pd.DataFrame:
    """
    Runs simulate_campaign_roi() -- the SAME formulas app.py's Tab 5
    slider-driven simulator uses -- across EVERY segment present in
    rfmt_df, one row per segment, so a budget question can compare
    allocations across segments instead of only the one segment the sliders
    currently point at.

    Parameters:
    - rfmt_df: used to enumerate segments and their population counts, via
      get_segment_kpi_summary() (src/rfm_engine.py) -- the SAME segment
      aggregation src/digest_engine.py and src/chat_context.py already reuse
      elsewhere, not re-derived a third time here.
    - clv_df: used for each segment's actual customer rows, exactly like
      app.py's Tab 5 (`clv_df[clv_df["Segment"] == segment]`) -- same
      AvgOrderValue mean, same 150.0 fallback app.py's Tab 5 already uses for
      a segment with zero customers in this slice.
    - total_budget: split proportionally to segment size by default (a
      segment's share of total customers x total_budget) -- NOT an equal
      split, and NOT re-derived per segment from a reach-percentage slider.
      Audience size for every segment here is that segment's FULL population
      (100% reach): unlike the single-segment simulator, there is no
      per-segment reach input in the advisor -- the question this function
      answers is "how should I split my budget across segments," not "how
      deep should I reach into any one segment."
    - conv_rate_by_segment: optional {segment: conversion_rate_pct} override
      for one or more segments. Any segment not present in this dict, and
      every segment when the dict itself is None, uses
      DEFAULT_CONV_RATE_PCT -- app.py Tab 5's own slider default (8.5%), not
      a new, invented per-segment assumption (per the task that added this
      module: segment-specific conversion-rate assumptions are a separate
      decision, not something to guess at here).

    Returns one row per segment (segment order matches
    get_segment_kpi_summary()'s own groupby order) with columns: segment,
    customer_count, allocated_budget, avg_segment_aov,
    projected_conversions, projected_gross_profit, net_incremental_profit,
    campaign_roi_pct, cost_per_acquisition. Values are left unrounded here
    (raw floats) -- rounding/formatting happens only at display time
    (build_roi_advisor_context() below, or app.py's own st.dataframe
    formatting), the same division of concerns src/chat_context.py already
    uses between its blob-building and text-serialization functions.
    """
    conv_rate_by_segment = conv_rate_by_segment or {}
    segment_summary = get_segment_kpi_summary(rfmt_df)
    total_customers = int(segment_summary["CustomerCount"].sum())

    rows = []
    for _, seg_row in segment_summary.iterrows():
        segment = str(seg_row["Segment"])
        customer_count = int(seg_row["CustomerCount"])

        seg_cust_df = clv_df[clv_df["Segment"] == segment]
        avg_segment_aov = (
            float(seg_cust_df["AvgOrderValue"].mean()) if len(seg_cust_df) > 0 else 150.0
        )

        share = (customer_count / total_customers) if total_customers > 0 else 0.0
        allocated_budget = total_budget * share
        conv_rate_pct = conv_rate_by_segment.get(segment, DEFAULT_CONV_RATE_PCT)

        sim = simulate_campaign_roi(
            audience_size=customer_count,
            avg_segment_aov=avg_segment_aov,
            campaign_budget=allocated_budget,
            conv_rate_pct=conv_rate_pct,
            gross_margin_pct=DEFAULT_GROSS_MARGIN_PCT,
        )

        rows.append({
            "segment": segment,
            "customer_count": customer_count,
            "allocated_budget": allocated_budget,
            "avg_segment_aov": avg_segment_aov,
            "projected_conversions": sim["projected_conversions"],
            "projected_gross_profit": sim["projected_gross_profit"],
            "net_incremental_profit": sim["net_incremental_profit"],
            "campaign_roi_pct": sim["campaign_roi_pct"],
            "cost_per_acquisition": sim["cost_per_acquisition"],
        })

    return pd.DataFrame(rows)


def build_roi_advisor_context(allocations_df: pd.DataFrame, total_budget: float) -> str:
    """
    Serializes simulate_all_segment_allocations()'s per-segment comparison
    DataFrame into a compact text block for the LLM prompt -- mirrors
    src/chat_context.py's build_context_text() serialization style (one
    line per row, plain f-string formatting). Purely a text rendering of
    numbers ALREADY computed by simulate_campaign_roi() -- no additional
    computation happens here or in the LLM call that reads this text.
    """
    if allocations_df is None or allocations_df.empty:
        return f"Total campaign budget: ${total_budget:,.2f}\n\n(no segment allocation data available)\n"

    lines = "\n".join(
        f"- {row['segment']}: {row['customer_count']:,} customers, allocated budget "
        f"${row['allocated_budget']:,.2f}, avg segment AOV ${row['avg_segment_aov']:,.2f}, "
        f"projected conversions {row['projected_conversions']:.1f}, projected gross profit "
        f"${row['projected_gross_profit']:,.2f}, net incremental profit "
        f"${row['net_incremental_profit']:,.2f} ({row['campaign_roi_pct']:+.1f}% ROI), "
        f"cost per acquisition ${row['cost_per_acquisition']:,.2f}"
        for _, row in allocations_df.iterrows()
    )

    return (
        f"Total campaign budget: ${total_budget:,.2f}, split proportionally to segment size "
        "by default across the segments below.\n\n"
        "PER-SEGMENT ALLOCATION COMPARISON:\n"
        f"{lines}\n"
    )


ROI_ADVISOR_SYSTEM_PROMPT_TEMPLATE = (
    "You are a marketing budget advisor. Below is a table comparing a proposed "
    "campaign's projected return on investment across every customer segment -- "
    "ALREADY COMPUTED by this platform's deterministic ROI simulator, every "
    "number in it is real. Using ONLY the numbers in this table, recommend how "
    "the analyst should allocate their budget, and explain your reasoning by "
    "comparing the segments' actual figures (e.g. \"Champions has the highest "
    "projected ROI% per dollar, but At-Risk VIPs has higher absolute net "
    "incremental profit at this budget level -- the trade-off is X\"). Never "
    "invent or recompute a number not already in the table below -- you may "
    "compare, rank, and reason about the given figures, but do not calculate a "
    "new ROI, conversion count, or profit figure yourself. If the analyst's "
    "question asks for something this table cannot answer (a segment not "
    "listed, a hypothetical not reflected in these numbers), say so plainly "
    "rather than guessing.\n\n"
    "Formatting: never use LaTeX or math notation of any kind -- do not wrap "
    "anything in single or double dollar signs. Write currency plainly, e.g. "
    "\"$1,234.56\".\n\n"
    "{context_blob}\n"
)


def _advisor_unavailable(reason: str) -> str:
    """
    Deterministic, clearly-labeled unavailability message -- mirrors
    src/chat_engine.py's _chat_unavailable() and src/digest_engine.py's
    _fallback_digest() in spirit: never crash, always return something
    clearly distinguishable from a real recommendation.
    """
    return f"{_UNAVAILABLE_PREFIX} — {reason}.]** Please try again, or contact support if this persists."


def get_roi_recommendation(
    question: str,
    allocations_df: pd.DataFrame,
    total_budget: float,
    anthropic_api_key: str = None,
    groq_api_key: str = None,
    provider_override: str = None,
) -> str:
    """
    Recommends a budget allocation for ONE question, using ONLY
    allocations_df (already computed by simulate_all_segment_allocations())
    plus total_budget for context -- see the module docstring for the full
    provider/cost-model rationale.

    Parameters:
    - question: the analyst's natural-language goal (e.g. "maximize total
      ROI%", "maximize absolute profit", or a free-form budget question).
    - allocations_df, total_budget: simulate_all_segment_allocations()'s
      output and the budget it was run against -- serialized into the
      prompt via build_roi_advisor_context(), never handed to the model as
      a raw DataFrame.
    - anthropic_api_key, groq_api_key, provider_override: identical contract
      to generate_account_digest()/answer_account_question() --
      _resolve_provider() (imported, not re-derived) makes the same
      Groq-preferred-by-default / Anthropic-fallback / "none" decision.

    Returns the recommendation text. NEVER raises: on any failure (no key
    configured, invalid key, rate limit, network error, empty response,
    anything unanticipated), returns a clearly-labeled "advisor temporarily
    unavailable" message instead, using the SAME distinct-reason-per-
    failure-mode strings _call_groq()/_call_anthropic() already produce for
    every other feature in this codebase -- nothing new invented here.
    """
    provider = _resolve_provider(anthropic_api_key, groq_api_key, override=provider_override)

    if provider == "none":
        return _advisor_unavailable("no GROQ_API_KEY or ANTHROPIC_API_KEY configured")

    context_text = build_roi_advisor_context(allocations_df, total_budget)
    system_prompt = ROI_ADVISOR_SYSTEM_PROMPT_TEMPLATE.format(context_blob=context_text)

    if provider == "groq":
        if groq is None:
            return _advisor_unavailable("groq package not installed")
        # Shared call/error-handling logic -- see _call_groq()'s own
        # docstring in src/digest_engine.py (also used by
        # generate_account_digest() and src/chat_engine.py's
        # answer_account_question()). Groq's OpenAI-compatible API has no
        # separate top-level `system` field, so the system prompt is just a
        # leading {"role": "system", ...} entry, same shape chat_engine.py
        # already uses.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        text, failure_reason = _call_groq(
            messages=messages,
            api_key=groq_api_key,
            max_tokens=GROQ_ROI_ADVISOR_MAX_OUTPUT_TOKENS,
        )
    else:
        # provider == "anthropic"
        if anthropic is None:
            return _advisor_unavailable("anthropic package not installed")
        messages = [{"role": "user", "content": question}]
        text, failure_reason = _call_anthropic(
            messages=messages,
            api_key=anthropic_api_key,
            max_tokens=ROI_ADVISOR_MAX_OUTPUT_TOKENS,
            system=system_prompt,
        )

    return text if text else _advisor_unavailable(failure_reason)

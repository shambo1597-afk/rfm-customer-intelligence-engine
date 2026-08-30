"""
src/digest_engine.py - Optional per-account AI Executive Summary.

Generates ONE natural-language executive-summary paragraph per account, after the
batch RFM-T/CLV scoring run completes, built entirely from already-computed
AGGREGATE output (segment counts, % at-risk, total forecasted revenue, top
segments) -- never raw per-customer rows. This keeps the prompt small and keeps
customer PII (IDs, names, individual transaction history) out of the API call
entirely, REGARDLESS of which provider below ends up handling the request.

This is a narrative WRAPPER on existing output, not a new inference step: no
per-record LLM calls are made anywhere in this pipeline. RFM-T, K-Means, PCA, and
the CLV/churn model all remain 100% baked-in/zero-API, exactly as before this
feature existed -- see README "AI Executive Summary (Optional)".

--- Two providers, Groq preferred by default ---

_resolve_provider() picks which provider actually gets called: Groq when a
Groq key is configured (the default -- see cost model below), Anthropic as the
fallback, or an explicit override via provider_override / the DIGEST_PROVIDER
secret or env var. Both providers read the exact same aggregate-only prompt from
_build_prompt() -- the PII-safety guarantee above doesn't change based on which
one is selected.

--- Provider swap, 2026-08-29: Gemini removed, replaced with Groq ---

Gemini was this project's original default provider. Live testing after the
request-timeout fix (see this project's git history for that incident) still
found Gemini's Interactions API timing out intermittently -- roughly 50-75% of
repeated calls in manual reproduction, no official Google incident confirmed --
a reliability problem with the provider itself, not this codebase. Decision:
replace Gemini as the default provider with Groq (free tier, no credit card,
OpenAI-compatible API, runs on independent LPU hardware with no shared infra
with Google -- so its failure modes are uncorrelated with whatever was causing
Gemini's timeouts). Gemini support is REMOVED here, not deprecated-in-place --
the person decided against maintaining three providers. Anthropic remains the
configured fallback, structurally unchanged by this swap.

--- Cost model this was designed against ---

This design's whole point is captured here so the reasoning survives in the code,
not just in a slide, and doesn't get silently violated by a well-intentioned
"let's personalize it per customer" change later.

  Per-end-customer digest (REJECTED): one LLM call per end-customer, per batch run.
    At an 8-account pilot averaging ~450 end-customers/account, one batch run/day:
        8 accounts * 450 customers = 3,600 calls/day = ~108,000 calls/month.
    Even at a cheap per-1M-token rate with a tiny ~150-in/~60-out-token prompt,
    that's real money at this call volume, and per-request overhead (rate
    limiting, latency, retries) at 100K+ calls/month pushes realistic pilot cost
    toward ~$208/month (Anthropic Haiku pricing) -- for a summary granularity
    (one paragraph per end-customer) nobody on the account actually reads end to end.

  Per-account digest (THIS DESIGN): one LLM call per ACCOUNT, per batch run.
        8 accounts * 1 call/day = 8 calls/day = ~240 calls/month.
    At Anthropic Haiku's per-call token cost, that's under $1/month at pilot
    volume. Groq's free tier (the default -- see "Provider details" below for
    the specific model, rate limits, and a caveat on how those numbers were
    sourced) costs $0 at this call volume, with wide headroom before any
    free-tier ceiling matters. Either way, roughly two orders of magnitude
    cheaper than the rejected design, for materially the same executive-facing
    value: a marketer wants one summary of their whole book of business per batch
    run, not one summary per customer.

--- Provider details ---

Groq (default, GROQ_MODEL_ID below): openai/gpt-oss-120b.

  Model-choice note, 2026-08-29 (read before ever changing GROQ_MODEL_ID): the
  task that introduced Groq support specified "llama-3.3-70b-versatile" as the
  model to use, sourced (per that task's own wording) from Groq's quickstart
  docs "as of today". Live verification here found otherwise: multiple,
  independently-worded WebSearch queries converge on Groq having announced
  deprecation of llama-3.3-70b-versatile (and llama-3.1-8b-instant) on
  2026-06-17, with "requests to this model no longer being served by
  August 2026" on the free/developer tier -- and today's date is 2026-08-29,
  past that cutoff. This is EXACTLY the failure mode this project's Gemini
  incident (git history) already burned time on once: a model name that was
  correct when written down can be stale by the time the code actually runs.
  GROQ_MODEL_ID was therefore set to openai/gpt-oss-120b instead -- Groq's own
  stated migration target for llama-3.3-70b-versatile (confirmed across
  multiple separately-worded searches, not one source), and independently
  confirmed to be a real, currently-recognized, free-tier model both via
  WebSearch and by finding it listed as a valid Literal option in the pinned
  groq SDK's own chat.completions.create() type stub (direct package
  introspection, not assumed).

  Direct-source verification caveat: console.groq.com and groq.com are BOTH
  unreachable from this development environment (network egress blocks --
  the same restriction that blocked ai.google.dev during this project's
  earlier Gemini-pricing investigation). Every claim in this section is
  therefore sourced via web-search result summaries citing third-party
  aggregator pages, not fetched directly from Groq's own docs. Re-verify
  against console.groq.com/docs/models and console.groq.com/docs/rate-limits
  directly (from a network that can reach them) before treating any number
  below as durably accurate, and update this note once re-verified.

  No rolling "-latest" alias mechanism: unlike Gemini (which offered
  gemini-flash-lite-latest specifically to dodge this exact pinned-model-decay
  problem), Groq's model names are plain pinned strings -- no alias
  indirection was found in any source checked here. GROQ_MODEL_ID therefore
  carries the same structural staleness risk that caused both the original
  Gemini incident and the substitution just described -- periodically
  re-check console.groq.com/docs/models regardless of whether anything here
  is currently failing.

  Rate limits (free tier, openai/gpt-oss-120b, via WebSearch -- see the
  direct-source verification caveat above): 30 requests/minute, 1,000
  requests/day, 8,000 tokens/minute, 200,000 tokens/day. Whichever ceiling is
  hit first triggers a 429 -- caught below and routed to the same fallback
  template as every other failure mode, never raised. At this project's pilot
  volume (8 accounts * 1 digest call/day = 8 calls/day), none of these are a
  meaningfully binding constraint.

  Reasoning-token headroom, 2026-08-29 (see GROQ_REASONING_TOKEN_HEADROOM's
  own comment below for the full incident and rationale): raising each
  feature's *_MAX_OUTPUT_TOKENS constant raises the worst-case token ceiling
  per call, not the typical actual usage -- Groq bills/quotas by tokens
  ACTUALLY generated, and GROQ_REASONING_EFFORT="low" is specifically chosen
  to keep real reasoning usage well under the new, larger ceiling for these
  short, simple prompts. The honest risk this doesn't fully rule out (flagged
  explicitly, not glossed over): a reasoning model given more room CAN, in
  rare cases, choose to reason longer rather than stopping early, so a larger
  ceiling is not a strict no-op on typical cost/latency the way it would be
  for a non-reasoning model. This could not be empirically verified against a
  real key in this environment (see the same constant's comment) -- if a real
  deployment sees materially higher latency or token-quota pressure after
  this change, that is the first thing to re-measure.

  DATA-USAGE DISCLOSURE: per WebSearch-sourced summaries of Groq's own
  documentation (again, not independently fetched -- see the verification
  caveat above), Groq does NOT use customer inputs/outputs to train or
  fine-tune models without explicit customer consent, on the free tier or any
  tier -- Groq positions itself as an inference provider rather than a
  foundation-model developer, so this policy is described as account-wide,
  not tier-gated. Inference requests are reportedly not retained by default,
  with narrow exceptions (troubleshooting a platform error, investigating
  suspected abuse) retained up to 30 days. This reads as materially more
  favorable than Gemini's confirmed free-tier policy was ("Used to improve
  our products: Yes") -- but it rests on secondary sources this environment
  could not verify directly, so treat it as a strong signal to confirm, not
  a settled guarantee, before relying on it for a real deployment's
  compliance posture.

  Retry behavior: confirmed by reading the installed groq package's own
  _base_client.py source directly (not assumed) -- the client automatically
  retries (default max_retries=2) on HTTP 408 (request timeout), 409 (lock
  timeout), 429 (rate limit), and any 5xx server error, honoring the server's
  `x-should-retry` response header when present. This is materially more
  robust out of the box than Gemini's Interactions API was, which needed a
  hand-rolled per-call timeout to avoid hanging indefinitely on the exact
  same class of transient failure. GROQ_REQUEST_TIMEOUT_SECONDS below is
  still set as an outer safety net -- this codebase's own hard-won lesson
  from that Gemini incident is to never trust a provider's default to bound
  worst-case latency without an explicit timeout of this codebase's own
  choosing -- but it is a safety net on top of a provider that already
  retries sensibly, not a first line of defense against a known-broken retry
  story the way it was for Gemini.

  Error handling: `client.chat.completions.create()` raises the public,
  top-level `groq.*` exception hierarchy (GroqError -> APIError ->
  APIConnectionError -> APITimeoutError, and APIError -> APIStatusError ->
  AuthenticationError / RateLimitError / etc.) -- confirmed via direct
  introspection of the pinned groq package (dir(groq), each class's
  __bases__ chain), not assumed from docs. This hierarchy is structurally
  identical in shape to anthropic's own (both are Stainless-generated,
  OpenAI-API-compatible SDKs), which is why _call_groq() below mirrors
  _call_anthropic() almost line for line.

Anthropic (fallback, MODEL_ID below): Claude Haiku 4.5, $1.00 / $5.00 per 1M
input/output tokens -- the "small/cheap model" tier, not free, but Anthropic's
standard paid-API terms do not use submitted content to train models.

Do not swap either MODEL_ID/GROQ_MODEL_ID for a larger/costlier model without
re-deriving the cost comparison above.
"""

import os

try:
    import streamlit as st
except ImportError:  # pragma: no cover - streamlit is a hard dependency of the app,
    st = None          # but this module should still be importable/testable without it.

try:
    import anthropic
except ImportError:  # pragma: no cover - exercised by the "package not installed" fallback path.
    anthropic = None

try:
    import groq
    # Unlike Gemini's google-genai SDK, every exception class groq raises for
    # this call is public and top-level (groq.APIError, groq.APIStatusError,
    # groq.AuthenticationError, etc.) -- confirmed via direct introspection of
    # the pinned package (dir(groq)), not assumed. No private-submodule import
    # workaround is needed here the way one was for Gemini.
except ImportError:  # pragma: no cover - exercised by the "package not installed" fallback path.
    groq = None

# Small/cheap model by design -- see the module docstring's cost model above.
MODEL_ID = "claude-haiku-4-5"
# Short paragraph only; also bounds worst-case per-call cost.
MAX_TOKENS = 300
REQUEST_TIMEOUT_SECONDS = 20.0

# Groq: default provider -- see the module docstring's "Provider details" and
# "Model-choice note, 2026-08-29" above for why this is openai/gpt-oss-120b
# rather than the llama-3.3-70b-versatile the task that introduced Groq
# support originally specified (confirmed deprecated on Groq's free/developer
# tier as of this date -- do not revert to it without re-checking
# console.groq.com/docs/models first). Groq has no rolling "-latest" alias
# mechanism like Gemini's, so this pinned name carries the same staleness
# risk that caused both the Gemini incident and this immediate substitution --
# periodically re-verify regardless.
GROQ_MODEL_ID = "openai/gpt-oss-120b"

# --- Reasoning-token headroom, 2026-08-29 (read before changing any of the
# four *_MAX_OUTPUT_TOKENS constants below or removing GROQ_REASONING_EFFORT) ---
#
# openai/gpt-oss-120b is a REASONING model: before producing the visible
# answer, it generates a hidden chain-of-thought (returned in the response
# message's own `reasoning` field, confirmed via direct introspection of the
# installed groq package's type stubs -- groq/types/chat/chat_completion_
# message.py) that is billed against, and truncated by, the SAME max_tokens
# budget as the visible content. Live testing found this: with the digest's
# 300-token budget applied to a cohort-narration call (150 tokens at the
# time), the reasoning text alone consumed nearly the entire budget, leaving
# either an empty or mid-sentence-truncated visible answer -- confirmed via
# finish_reason == "length" on the raw response. This is a structural risk
# for EVERY one-shot Groq call in this codebase, not just cohort narration --
# every one of generate_account_digest(), answer_account_question(),
# get_roi_recommendation(), and narrate_cohort_pattern() calls the same
# shared _call_groq() below and was equally exposed; each one's own
# *_MAX_OUTPUT_TOKENS constant (GROQ_MAX_OUTPUT_TOKENS here,
# CHAT_MAX_OUTPUT_TOKENS in chat_engine.py, ROI_ADVISOR_MAX_OUTPUT_TOKENS in
# roi_advisor.py, COHORT_NARRATION_MAX_OUTPUT_TOKENS in cohort_narration.py)
# is a SEPARATE constant, not literally one shared value the way the task
# that surfaced this bug initially assumed -- verified by reading all four
# files directly, not assumed. Each one is now defined as its own original
# visible-content target PLUS this shared GROQ_REASONING_TOKEN_HEADROOM, so
# the "how much extra room does reasoning need" number lives in exactly one
# place, reused by all four, while each feature's own visible-length intent
# stays legible at its own definition site.
#
# GROQ_REASONING_TOKEN_HEADROOM's value, honestly: this environment's network
# egress policy blocks console.groq.com and api.groq.com outright (the same
# restriction noted throughout this file's Groq-related sourcing caveats),
# and no GROQ_API_KEY is available here either -- so the empirical
# methodology this fix was supposed to follow (reproduce this project's real
# prompts against the raw SDK with a generous ceiling, inspect finish_reason
# and actual token usage, pick a value with headroom above the largest
# OBSERVED combined reasoning+visible usage) could not be carried out. 1500
# is a reasoned, deliberately generous estimate instead, informed by: (a) the
# live failure above, where a 150-token budget was insufficient by a wide
# margin; (b) OpenAI's own gpt-oss-120b/20b model card, which states
# reasoning-token counts "can vary from hundreds to tens of thousands
# depending on task complexity" (sourced via WebSearch -- console.groq.com/
# huggingface.co were not directly fetchable from this environment either);
# and (c) GROQ_REASONING_EFFORT below, set to "low" specifically to keep
# actual reasoning usage toward the low end of that range for these
# genuinely simple, short, single-purpose prompts (a one-paragraph digest, a
# short Q&A answer, a budget recommendation over a small table, a one-to-two
# sentence narration -- none of which need "deep and detailed analysis").
# This is NOT a substitute for the real empirical verification the task
# asked for -- re-run that exact methodology against a real key outside this
# environment's network restrictions, inspect the real finish_reason/usage
# numbers it reports, and tune this constant down (or up) from 1500 once
# real data exists, rather than trusting this estimate indefinitely.
GROQ_REASONING_TOKEN_HEADROOM = 1500

# 'low', 'medium' (Groq's own default), or 'high' -- confirmed supported for
# specifically openai/gpt-oss-20b and openai/gpt-oss-120b via WebSearch of
# Groq's own docs (console.groq.com/docs/reasoning was not directly
# fetchable -- see the sourcing caveat above) AND, independently, via direct
# introspection of the installed groq package's own request-parameter type
# stub (groq/types/chat/completion_create_params.py), whose reasoning_effort
# docstring names openai/gpt-oss-120b explicitly as supporting these three
# values -- not assumed from either source alone. Set to "low" ("fast
# responses... best for straightforward questions" per Groq's own
# documentation) rather than the "medium" default specifically so actual
# reasoning-token usage stays toward the low end of the range
# GROQ_REASONING_TOKEN_HEADROOM budgets for -- every prompt this codebase
# sends via Groq is a short, single-purpose business-text task, never a
# multi-step problem that would benefit from "high"'s deeper analysis.
GROQ_REASONING_EFFORT = "low"

# Mirrors MAX_TOKENS's role for Anthropic: short paragraph only, bounds
# worst-case cost, PLUS GROQ_REASONING_TOKEN_HEADROOM above for this
# reasoning model's hidden chain-of-thought (see that constant's comment for
# the full rationale and its honesty caveat about not being empirically
# verified in this environment). Passed as max_tokens on the
# chat.completions.create() call below (OpenAI-compatible API shape). Was
# 300 (visible-content-only) before this fix; the actual visible-content
# target is still ~300 tokens, unchanged -- only the reasoning headroom is
# new.
GROQ_MAX_OUTPUT_TOKENS = 300 + GROQ_REASONING_TOKEN_HEADROOM
# Outer safety-net timeout for client.chat.completions.create() -- passed via
# client.with_options(timeout=...) below, mirroring REQUEST_TIMEOUT_SECONDS's
# Anthropic mechanism exactly (both SDKs support the same with_options()
# pattern). Unlike GEMINI_REQUEST_TIMEOUT_SECONDS before it, this is NOT
# compensating for a known-broken client -- confirmed by reading the pinned
# groq package's own _base_client.py source, the client already retries
# automatically (max_retries=2 default) on 408/409/429/5xx. This constant is
# still set explicitly, though, per this codebase's own hard-won Gemini-hang
# lesson: never trust a provider's default to bound worst-case latency
# without an explicit timeout of this codebase's own choosing.
GROQ_REQUEST_TIMEOUT_SECONDS = 20.0

# How many of the largest segments to name in the prompt (keeps the prompt compact).
SEGMENTS_SHOWN_IN_PROMPT = 3

# Must match src/clv_engine.py's Churn_Risk_Tier label for the high-risk bucket.
HIGH_CHURN_RISK_TIER = "🔴 High Churn Risk"

_VALID_PROVIDERS = ("groq", "anthropic")


def get_anthropic_api_key():
    """
    Resolves an Anthropic API key from st.secrets first, then the ANTHROPIC_API_KEY
    environment variable. Never hardcoded, never logged. Returns None if neither is
    configured -- callers should treat that as "this provider unavailable", not as
    an error.
    """
    api_key = None
    if st is not None:
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY")
        except Exception:
            # st.secrets raises when no secrets.toml exists at all in some Streamlit
            # versions -- that's a normal "not configured" state here, not an error.
            pass
    return api_key or os.environ.get("ANTHROPIC_API_KEY")


def get_groq_api_key():
    """
    Resolves a Groq API key from st.secrets first, then the GROQ_API_KEY
    environment variable. Never hardcoded, never logged. Returns None if neither is
    configured -- callers should treat that as "this provider unavailable", not as
    an error. Mirrors get_anthropic_api_key() exactly.
    """
    api_key = None
    if st is not None:
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            pass
    return api_key or os.environ.get("GROQ_API_KEY")


def get_digest_provider_override():
    """
    Resolves an optional provider override from st.secrets first, then the
    DIGEST_PROVIDER environment variable. Returns "groq", "anthropic", or None.
    An unset value, or any value other than those two, returns None -- a typo'd
    override is treated the same as no override (falls through to
    _resolve_provider()'s normal preference order) rather than raising.
    """
    override = None
    if st is not None:
        try:
            override = st.secrets.get("DIGEST_PROVIDER")
        except Exception:
            pass
    override = override or os.environ.get("DIGEST_PROVIDER")
    override = override.strip().lower() if override else None
    return override if override in _VALID_PROVIDERS else None


def _resolve_provider(anthropic_key, groq_key, override=None) -> str:
    """
    Decides which provider generate_account_digest() should call. Never raises.
    Returns "groq", "anthropic", or "none" (neither key configured -- triggers
    the fallback template).

    - If `override` names a provider ("groq" or "anthropic") AND that provider's
      key is present, that provider wins outright.
    - Otherwise (no override, or the override names a provider whose key is
      absent): prefer Groq when a Groq key is present -- it's both this
      deployment's desired default and the genuinely free option (see module
      docstring) -- else fall back to Anthropic if that key is present, else
      "none".
    """
    if override in _VALID_PROVIDERS:
        if override == "groq" and groq_key:
            return "groq"
        if override == "anthropic" and anthropic_key:
            return "anthropic"
        # Override named a provider whose key is missing -- fall through to the
        # normal preference order below instead of returning "none" outright.

    if groq_key:
        return "groq"
    if anthropic_key:
        return "anthropic"
    return "none"


def _build_aggregate_stats(rfmt_df, clv_df, segment_summary) -> dict:
    """
    Extracts ONLY account-level aggregate numbers -- no CustomerID, no per-row data
    of any kind. This dict is both what gets serialized into the LLM prompt
    (_build_prompt) and what powers the no-key fallback template (_fallback_digest),
    so the two paths can never silently drift out of sync with each other.
    """
    total_customers = int(len(rfmt_df))
    total_historical_revenue = float(clv_df["Monetary"].sum()) if "Monetary" in clv_df.columns else 0.0
    total_predicted_90d_revenue = (
        float(clv_df["Predicted_Spend_90d"].sum()) if "Predicted_Spend_90d" in clv_df.columns else 0.0
    )

    if "Churn_Risk_Tier" in clv_df.columns and total_customers > 0:
        pct_at_risk = float((clv_df["Churn_Risk_Tier"] == HIGH_CHURN_RISK_TIER).sum()) / total_customers * 100.0
    else:
        pct_at_risk = 0.0

    top_segments = []
    if segment_summary is not None and not segment_summary.empty and "CustomerCount" in segment_summary.columns:
        ranked = segment_summary.sort_values("CustomerCount", ascending=False).head(SEGMENTS_SHOWN_IN_PROMPT)
        for _, row in ranked.iterrows():
            top_segments.append({
                "segment": str(row["Segment"]),
                "customer_count": int(row["CustomerCount"]),
                "customer_share_pct": float(row["CustomerSharePct"]) if "CustomerSharePct" in ranked.columns else 0.0,
            })

    return {
        "total_customers": total_customers,
        "total_historical_revenue": total_historical_revenue,
        "total_predicted_90d_revenue": total_predicted_90d_revenue,
        "pct_at_risk": pct_at_risk,
        "top_segments": top_segments,
    }


def _build_prompt(stats: dict) -> str:
    """
    Serializes the aggregate stats dict into a compact prompt. Deliberately contains
    ONLY the fields produced by _build_aggregate_stats() above -- no CustomerID, no
    raw transaction rows, no per-customer loop of any kind -- so no customer PII
    ever reaches the API call.
    """
    segment_lines = "\n".join(
        f"- {seg['segment']}: {seg['customer_count']:,} customers ({seg['customer_share_pct']:.1f}% of total)"
        for seg in stats["top_segments"]
    ) or "- (no segment data available)"

    return (
        "You are writing a short executive summary for a business owner reviewing "
        "their customer intelligence dashboard. Using ONLY the aggregate statistics "
        "below, write ONE concise paragraph (3-5 sentences) covering overall "
        "customer base health, the churn-risk situation, and the 90-day revenue "
        "outlook. Do not invent numbers not given below, and do not mention "
        "individual customers -- only account-wide totals.\n\n"
        f"Total customers: {stats['total_customers']:,}\n"
        f"Total historical revenue: ${stats['total_historical_revenue']:,.2f}\n"
        f"Predicted next-90-day revenue: ${stats['total_predicted_90d_revenue']:,.2f}\n"
        f"Customers at high churn risk: {stats['pct_at_risk']:.1f}%\n"
        f"Top segments by size:\n{segment_lines}\n"
    )


def _fallback_digest(stats: dict, reason: str) -> str:
    """
    Deterministic, template-based summary used whenever no API key is configured or
    the API call fails for any reason -- the feature must never break the app or
    leave the user with nothing. Clearly labeled so it's never mistaken for the
    AI-generated version.
    """
    top_seg_text = stats["top_segments"][0]["segment"] if stats["top_segments"] else "no clearly dominant segment"
    return (
        f"**[Template Summary — {reason}]** This account has {stats['total_customers']:,} customers "
        f"with ${stats['total_historical_revenue']:,.2f} in historical revenue and a 90-day forecast "
        f"of ${stats['total_predicted_90d_revenue']:,.2f}. {stats['pct_at_risk']:.1f}% of customers are "
        f"currently flagged as high churn risk. The largest customer segment is {top_seg_text}. "
        f"Configure GROQ_API_KEY (free) or ANTHROPIC_API_KEY to enable the AI-generated version of this summary."
    )


def _call_groq(
    messages: list,
    api_key: str,
    max_tokens: int,
) -> tuple:
    """
    Shared Groq call + error-classification logic for both
    generate_account_digest() below (a one-shot `messages=[{"role": "user", ...}]`
    list) and src/chat_engine.py's answer_account_question() (a `messages` list
    that additionally carries a leading {"role": "system", ...} entry plus prior
    conversation_history turns -- Groq's OpenAI-compatible API has no separate
    top-level `system` field the way Anthropic's does, so a system prompt is
    just another entry in this same list rather than a distinct parameter).
    Extracted here, in digest_engine.py, and imported BY chat_engine.py --
    never the reverse -- because chat_engine.py already depends on
    digest_engine.py for GROQ_MODEL_ID, GROQ_REQUEST_TIMEOUT_SECONDS, etc.;
    keeping the dependency one-directional avoids a circular import.

    Mirrors _call_anthropic() below almost line for line -- see the module
    docstring's "Error handling" note for why (both SDKs are Stainless-
    generated and OpenAI-API-compatible, so their exception hierarchies and
    client shapes are structurally identical). This function owns ONLY the
    provider call and its error classification; it does NOT decide whether
    Groq should be called at all (_resolve_provider() still does that in each
    caller, before it even builds `messages`) and does NOT do any
    caller-specific bookkeeping -- conversation_history mutation is
    chat_engine.py's own concern, not this function's (see
    answer_account_question()).

    Returns a (response_text, failure_reason) tuple: (text, None) on success,
    (None, reason) on failure -- `reason` is one of "Groq free-tier rate
    limit reached", "invalid Groq API key", "Groq API request timed out",
    "Groq API error", "empty response from Groq", "Groq response was
    truncated before completion -- try a shorter question or increase the
    token budget", "groq package not installed", or "unexpected error
    calling Groq".

    The "truncated" reason (see GROQ_REASONING_TOKEN_HEADROOM's comment
    above for the underlying cause) is checked BEFORE the "empty" one and
    wins whenever the API itself reports finish_reason == "length" -- this
    single check covers both observed shapes of the same underlying failure:
    reasoning consuming the ENTIRE token budget (visible content ends up
    empty) and reasoning consuming MOST of it (visible content is short and
    cut off mid-sentence). Conflating either of those with a genuinely empty
    response (e.g. finish_reason == "stop" with no content for some other
    reason) would misdirect a future debugging session toward "something is
    fundamentally broken" instead of "raise the token budget" -- exactly
    what happened live before this fix existed.

    Never raises -- both callers still guarantee "this optional feature never
    crashes the app" by construction (every exception the SDK is documented,
    and confirmed, to raise for this call is classified below), not by
    re-wrapping this call in yet another try/except.
    """
    if groq is None:
        # Defense-in-depth only, and what makes this function independently
        # testable for this failure mode (patch src.digest_engine.groq
        # directly, since that's the name this function itself reads). Both
        # callers already check `groq is None` themselves FIRST, before ever
        # calling this function -- that check can't move here instead,
        # because each caller's own existing tests patch THAT module's own
        # `groq` name (e.g. src.chat_engine.groq), which -- being a
        # `from src.digest_engine import groq` value-import -- is an
        # independent name binding from this module's `groq`, not an alias
        # of it. Rebinding one doesn't rebind the other, so the "package
        # missing" check has to stay caller-side to keep matching each
        # existing test's patch target.
        return None, "groq package not installed"
    try:
        client = groq.Groq(api_key=api_key)
        response = client.with_options(timeout=GROQ_REQUEST_TIMEOUT_SECONDS).chat.completions.create(
            model=GROQ_MODEL_ID,
            max_tokens=max_tokens,
            reasoning_effort=GROQ_REASONING_EFFORT,
            messages=messages,
        )
        # getattr(..., default=None), not direct attribute access: the real
        # groq.types.chat.chat_completion.Choice model always carries this
        # field (confirmed via direct package introspection -- it's a
        # required, non-Optional Literal["stop", "length", "tool_calls",
        # "function_call"]), so this default is purely defensive against a
        # test double that doesn't set it, never a real-response fallback.
        finish_reason = getattr(response.choices[0], "finish_reason", None)
        if finish_reason == "length":
            return None, (
                "Groq response was truncated before completion -- try a "
                "shorter question or increase the token budget"
            )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            return None, "empty response from Groq"
        return text, None
    except groq.AuthenticationError:
        return None, "invalid Groq API key"
    except groq.RateLimitError:
        # Expected at higher account counts on the free tier -- not a bug.
        return None, "Groq free-tier rate limit reached"
    except groq.APITimeoutError:
        # Raised when the per-call timeout above (GROQ_REQUEST_TIMEOUT_SECONDS,
        # via client.with_options()) is hit -- confirmed by reading the pinned
        # groq package's own _base_client.py source: it wraps
        # httpx.TimeoutException into groq.APITimeoutError, a subclass of
        # APIConnectionError, itself a subclass of APIError. Caught BEFORE the
        # broader groq.APIStatusError/APIConnectionError below with its own
        # distinct reason -- a timeout is a genuinely different situation from
        # a rate limit or an invalid key and should never be reported as
        # either, the same lesson this codebase already learned the hard way
        # with Gemini.
        return None, "Groq API request timed out"
    except groq.APIStatusError:
        # Broad base class for every other non-2xx response (a real, distinct
        # class per status code in this SDK -- BadRequestError, NotFoundError,
        # InternalServerError, etc. -- confirmed via dir(groq)/__bases__, not
        # assumed) not already distinguished above.
        return None, "Groq API error"
    except groq.APIConnectionError:
        # Network-level failure with no HTTP response at all (DNS, connection
        # refused, ...) -- NOT an APIStatusError subclass, so this is a
        # separate branch. APITimeoutError above is itself an
        # APIConnectionError subclass, but is caught first, so it never
        # reaches here.
        return None, "could not reach the Groq API"
    except Exception:
        # Last-resort safety net -- this optional feature must never crash the
        # app -- but by this point every error shape the SDK is documented (and
        # confirmed) to raise for this call has already been handled above, so
        # reaching here means something genuinely unanticipated happened.
        return None, "unexpected error calling Groq"


def _call_anthropic(
    messages: list,
    api_key: str,
    max_tokens: int,
    system: str = None,
) -> tuple:
    """
    Shared Anthropic call + error-classification logic for both
    generate_account_digest() below (system=None -- the digest never passed a
    separate system prompt even before this extraction, just a single
    one-shot user message) and src/chat_engine.py's answer_account_question()
    (a real system prompt plus the full conversation_history as `messages`).
    Mirrors _call_groq() above exactly in spirit and in why it lives here
    (see that function's docstring) -- extracted because the two callers'
    Anthropic branches were otherwise identical modulo how `messages`/`system`
    get built, which is now just a parameter each caller supplies. Unlike
    Groq's OpenAI-compatible shape, Anthropic's API genuinely has a separate
    top-level `system` field, which is why this function (and only this one)
    still takes `system` as its own parameter rather than folding it into
    `messages`.

    Returns a (response_text, failure_reason) tuple: (text, None) on success,
    (None, reason) on failure -- `reason` is one of the exact strings both
    callers already used before this extraction ("invalid Anthropic API key",
    "Anthropic API rate limited", "Anthropic API error", "could not reach the
    Anthropic API", "empty response from the model", "anthropic package not
    installed") so neither caller's existing fallback-reason tests needed to
    change. The one exception is the final generic-Exception catch-all, whose
    exact phrasing differed cosmetically between the two callers before this
    extraction ("...generating the AI summary" vs "...answering the
    question") -- both existing tests only assert the substring "unexpected
    error" is present, never the full phrase, so a single shared phrase here
    ("unexpected error calling Anthropic") satisfies both unchanged.

    Never raises, for the same reason _call_groq() doesn't.
    """
    if anthropic is None:
        # Defense-in-depth only -- see _call_groq()'s identical note on why
        # the "package missing" check has to stay caller-side too (each
        # caller's own existing test patches that module's own `anthropic`
        # name, an independent binding from this module's).
        return None, "anthropic package not installed"
    try:
        client = anthropic.Anthropic(api_key=api_key)
        create_kwargs = {"model": MODEL_ID, "max_tokens": max_tokens, "messages": messages}
        if system is not None:
            create_kwargs["system"] = system
        response = client.with_options(timeout=REQUEST_TIMEOUT_SECONDS).messages.create(**create_kwargs)
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        if not text:
            return None, "empty response from the model"
        return text, None
    except anthropic.AuthenticationError:
        return None, "invalid Anthropic API key"
    except anthropic.RateLimitError:
        return None, "Anthropic API rate limited"
    except anthropic.APIStatusError:
        return None, "Anthropic API error"
    except anthropic.APIConnectionError:
        return None, "could not reach the Anthropic API"
    except Exception:
        # Last-resort safety net -- this optional feature must never crash the app.
        return None, "unexpected error calling Anthropic"


def generate_account_digest(
    rfmt_df, clv_df, segment_summary,
    anthropic_api_key: str = None,
    groq_api_key: str = None,
    provider_override: str = None,
) -> str:
    """
    Generates ONE natural-language executive summary paragraph for an account, from
    already-computed aggregate output only -- see the module docstring for the full
    cost-model rationale (one call per account per batch run, never per end-customer)
    and why Groq is the default provider.

    Parameters:
    - rfmt_df, clv_df, segment_summary: the already-computed pipeline outputs for
      this account (see src/rfm_engine.py, src/clv_engine.py). Only aggregate
      statistics are ever read from them -- never raw per-customer rows, regardless
      of which provider ends up handling the request.
    - anthropic_api_key, groq_api_key: provider API keys (or None/falsy if that
      provider isn't configured). _resolve_provider() decides which one actually
      gets called: Groq is preferred by default when its key is present (see
      module docstring), Anthropic is the fallback.
    - provider_override: force "groq" or "anthropic" when that provider's key is
      available (falls through to the normal preference order otherwise). Typically
      sourced from get_digest_provider_override() (the DIGEST_PROVIDER secret/env
      var), but passed explicitly here to keep this function itself side-effect-free
      and independently testable.
    - If no provider resolves (neither key configured), or the call to whichever
      provider was selected fails for ANY reason (network error, rate limit,
      invalid key, empty response, ...), this function ALWAYS returns a
      clearly-labeled fallback string built from the same aggregate stats via an
      f-string template -- it never raises, so the feature degrades gracefully and
      the app never breaks without a key.
    """
    stats = _build_aggregate_stats(rfmt_df, clv_df, segment_summary)
    provider = _resolve_provider(anthropic_api_key, groq_api_key, override=provider_override)

    if provider == "none":
        return _fallback_digest(stats, reason="no GROQ_API_KEY or ANTHROPIC_API_KEY configured")

    prompt = _build_prompt(stats)

    if provider == "groq":
        if groq is None:
            return _fallback_digest(stats, reason="groq package not installed")
        # Shared call/error-handling logic -- see _call_groq()'s own
        # docstring for why this is extracted (also used by
        # src/chat_engine.py's answer_account_question()) and why the
        # "package missing" check above stays here rather than moving inside
        # it. The digest is a one-shot prompt: no system role in `messages`,
        # no multi-turn chaining.
        text, failure_reason = _call_groq(
            messages=[{"role": "user", "content": prompt}],
            api_key=groq_api_key,
            max_tokens=GROQ_MAX_OUTPUT_TOKENS,
        )
        return text if text else _fallback_digest(stats, reason=failure_reason)

    # provider == "anthropic"
    if anthropic is None:
        return _fallback_digest(stats, reason="anthropic package not installed")
    # Shared call/error-handling logic -- see _call_anthropic()'s own
    # docstring (also used by src/chat_engine.py's answer_account_question()).
    # The digest never passed a separate system prompt even before this
    # extraction -- just a single one-shot user message -- so system stays
    # at its default (None).
    text, failure_reason = _call_anthropic(
        messages=[{"role": "user", "content": prompt}],
        api_key=anthropic_api_key,
        max_tokens=MAX_TOKENS,
    )
    return text if text else _fallback_digest(stats, reason=failure_reason)

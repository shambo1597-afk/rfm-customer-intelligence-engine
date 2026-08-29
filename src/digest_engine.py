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

--- Two providers, Gemini preferred by default ---

_resolve_provider() picks which provider actually gets called: Gemini when a
Gemini key is configured (the default -- see cost model below), Anthropic as the
fallback, or an explicit override via provider_override / the DIGEST_PROVIDER
secret or env var. Both providers read the exact same aggregate-only prompt from
_build_prompt() -- the PII-safety guarantee above doesn't change based on which
one is selected.

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
    volume. On Gemini 2.5 Flash-Lite's free tier (the default -- see below), it's
    genuinely $0, not just cheap. Either way, roughly two orders of magnitude
    cheaper than the rejected design, for materially the same executive-facing
    value: a marketer wants one summary of their whole book of business per batch
    run, not one summary per customer.

--- Provider details ---

Gemini (default, GEMINI_MODEL_ID below): gemini-2.5-flash-lite is genuinely FREE
of charge on Google's free tier at this call volume (as of 2026, per
ai.google.dev/gemini-api/docs/pricing) -- not just cheap. Its paid-tier rate
($0.10 / $0.40 per 1M input/output tokens) is also far below Anthropic Haiku's
($1.00 / $5.00). Do not swap GEMINI_MODEL_ID for a Pro model -- Pro models
require billing to be configured and are not on the free tier.

  DATA-USAGE DISCLOSURE (read before deploying on the free tier): per Google's
  own pricing page, content sent to the Gemini API on the FREE TIER **is used to
  improve Google's products** ("Used to improve our products: Yes"). This is a
  real, material difference from Anthropic's standard paid API terms (not used
  to train models) and is stated here deliberately rather than left for someone
  to discover later. It does not weaken the PII-safety guarantee above -- only
  aggregate account-level stats are ever sent, never raw customer rows -- but it
  is a genuine data-handling tradeoff a deployer should weigh, not a footnote.

  Rate limits are per-project and shown live in Google AI Studio, not a fixed
  public table. Hitting the free-tier per-minute/per-day/spend ceiling produces a
  standard `429 RESOURCE_EXHAUSTED` error (ai.google.dev/gemini-api/docs/
  rate-limits). That is an EXPECTED operating condition of this design at higher
  account counts, not a bug -- it is caught below and routed to the same
  fallback template as every other failure mode, never raised.

  Error handling note: `client.interactions.create()` (the Interactions API this
  module calls) raises `google.genai._gaos.lib.compat_errors.APIStatusError` /
  `APIError` -- a private-submodule hierarchy, and NOT `google.genai.errors.
  ClientError`/`ServerError`, which is only raised by the older `client.models.
  generate_content()` call path this module deliberately does not use. This was
  confirmed by reproducing a real invalid-key call against the pinned SDK
  version, not assumed from documentation -- see requirements.txt's version
  bound. That reproduction also showed Google's real invalid-key response is
  `400 BadRequestError` with `'API_KEY_INVALID'` in the body, not 401/403 as
  HTTP convention might suggest -- both are checked below.

Anthropic (fallback, MODEL_ID below): Claude Haiku 4.5, $1.00 / $5.00 per 1M
input/output tokens -- the "small/cheap model" tier, not free, but Anthropic's
standard paid-API terms do not use submitted content to train models.

Do not swap either MODEL_ID/GEMINI_MODEL_ID for a larger/costlier model without
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
    from google import genai
    # NOTE: deliberately NOT `google.genai.errors` (ClientError/ServerError). That
    # public module is the exception hierarchy for the OLDER `client.models.
    # generate_content()` call path -- it is never raised by `client.interactions.
    # create()`, the API this module actually calls. Confirmed by reproducing a
    # real invalid-key call against the pinned google-genai version (see
    # requirements.txt): the exception that comes back is
    # `google.genai._gaos.lib.compat_errors.BadRequestError`, an entirely
    # different, unrelated class hierarchy. `_gaos` is a private/internal
    # submodule (leading underscore) with no public alias for these classes as of
    # this SDK version -- there is currently no other way to name them. This is
    # exactly the kind of import that can silently break on an SDK upgrade, which
    # is why requirements.txt pins google-genai to a bounded range rather than an
    # open floor, and why every except clause below falls through to a safe
    # generic fallback (never a crash) if this shape changes again.
    from google.genai._gaos.lib import compat_errors as genai_errors
except ImportError:  # pragma: no cover - exercised by the "package not installed"/API-shape-changed fallback path.
    genai = None
    genai_errors = None

# Small/cheap model by design -- see the module docstring's cost model above.
MODEL_ID = "claude-haiku-4-5"
# Short paragraph only; also bounds worst-case per-call cost.
MAX_TOKENS = 300
REQUEST_TIMEOUT_SECONDS = 20.0

# Gemini: free-tier default -- see the module docstring's "Provider details" above.
GEMINI_MODEL_ID = "gemini-2.5-flash-lite"
# Mirrors MAX_TOKENS's role for Anthropic: short paragraph only, bounds worst-case
# cost. Passed as generation_config={"max_output_tokens": ...} on the
# interactions.create() call below (google.genai.types.GenerationConfig's field).
GEMINI_MAX_OUTPUT_TOKENS = 300

# How many of the largest segments to name in the prompt (keeps the prompt compact).
SEGMENTS_SHOWN_IN_PROMPT = 3

# Must match src/clv_engine.py's Churn_Risk_Tier label for the high-risk bucket.
HIGH_CHURN_RISK_TIER = "🔴 High Churn Risk"

_VALID_PROVIDERS = ("gemini", "anthropic")


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


def get_gemini_api_key():
    """
    Resolves a Gemini API key from st.secrets first, then the GEMINI_API_KEY
    environment variable. Never hardcoded, never logged. Returns None if neither is
    configured -- callers should treat that as "this provider unavailable", not as
    an error. Mirrors get_anthropic_api_key() exactly.
    """
    api_key = None
    if st is not None:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            pass
    return api_key or os.environ.get("GEMINI_API_KEY")


def get_digest_provider_override():
    """
    Resolves an optional provider override from st.secrets first, then the
    DIGEST_PROVIDER environment variable. Returns "gemini", "anthropic", or None.
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


def _resolve_provider(anthropic_key, gemini_key, override=None) -> str:
    """
    Decides which provider generate_account_digest() should call. Never raises.
    Returns "gemini", "anthropic", or "none" (neither key configured -- triggers
    the fallback template).

    - If `override` names a provider ("gemini" or "anthropic") AND that provider's
      key is present, that provider wins outright.
    - Otherwise (no override, or the override names a provider whose key is
      absent): prefer Gemini when a Gemini key is present -- it's both this
      deployment's desired default and the genuinely free option (see module
      docstring) -- else fall back to Anthropic if that key is present, else
      "none".
    """
    if override in _VALID_PROVIDERS:
        if override == "gemini" and gemini_key:
            return "gemini"
        if override == "anthropic" and anthropic_key:
            return "anthropic"
        # Override named a provider whose key is missing -- fall through to the
        # normal preference order below instead of returning "none" outright.

    if gemini_key:
        return "gemini"
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
        f"Configure GEMINI_API_KEY (free) or ANTHROPIC_API_KEY to enable the AI-generated version of this summary."
    )


def generate_account_digest(
    rfmt_df, clv_df, segment_summary,
    anthropic_api_key: str = None,
    gemini_api_key: str = None,
    provider_override: str = None,
) -> str:
    """
    Generates ONE natural-language executive summary paragraph for an account, from
    already-computed aggregate output only -- see the module docstring for the full
    cost-model rationale (one call per account per batch run, never per end-customer)
    and why Gemini is the default provider.

    Parameters:
    - rfmt_df, clv_df, segment_summary: the already-computed pipeline outputs for
      this account (see src/rfm_engine.py, src/clv_engine.py). Only aggregate
      statistics are ever read from them -- never raw per-customer rows, regardless
      of which provider ends up handling the request.
    - anthropic_api_key, gemini_api_key: provider API keys (or None/falsy if that
      provider isn't configured). _resolve_provider() decides which one actually
      gets called: Gemini is preferred by default when its key is present (see
      module docstring), Anthropic is the fallback.
    - provider_override: force "gemini" or "anthropic" when that provider's key is
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
    provider = _resolve_provider(anthropic_api_key, gemini_api_key, override=provider_override)

    if provider == "none":
        return _fallback_digest(stats, reason="no GEMINI_API_KEY or ANTHROPIC_API_KEY configured")

    prompt = _build_prompt(stats)

    if provider == "gemini":
        if genai is None:
            return _fallback_digest(stats, reason="google-genai package not installed")
        try:
            client = genai.Client(api_key=gemini_api_key)
            interaction = client.interactions.create(
                model=GEMINI_MODEL_ID,
                input=prompt,
                generation_config={"max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS},
            )
            text = (interaction.output_text or "").strip()
            return text if text else _fallback_digest(stats, reason="empty response from Gemini")
        except genai_errors.APIStatusError as e:
            # `.status_code` is a real, documented int attribute on this SDK's
            # APIStatusError (confirmed by direct reproduction against the pinned
            # google-genai version, not assumed -- see requirements.txt). Note
            # Google's actual behavior here is NOT what HTTP-status convention
            # would suggest: an invalid API key comes back as `400 BadRequestError`
            # with `'reason': 'API_KEY_INVALID'` in the body, not 401/403 --
            # confirmed the same way. Both are checked so a real invalid-key error
            # is still recognized however a given deployment's Google Cloud
            # project happens to report it.
            status_code = getattr(e, "status_code", None)
            message = str(e)
            if status_code == 429 or "RESOURCE_EXHAUSTED" in message:
                # Expected at higher account counts on the free tier -- not a bug.
                return _fallback_digest(stats, reason="Gemini free-tier rate limit reached")
            if status_code in (401, 403) or "API_KEY_INVALID" in message:
                return _fallback_digest(stats, reason="invalid Gemini API key")
            return _fallback_digest(stats, reason="Gemini API error")
        except genai_errors.APIError:
            # Broad base class for everything else the SDK raises for this call
            # (connection/timeout errors, and any future APIStatusError subclass
            # not distinguished above) -- still a diagnosable "the SDK reported an
            # API error", not a truly unanticipated exception. Catching the base
            # class here (rather than only the specific subclasses above) means a
            # future error variant this code doesn't yet know about still gets a
            # meaningful reason instead of silently falling through to the
            # generic catch-all below, which is exactly the bug being fixed.
            return _fallback_digest(stats, reason="Gemini API error")
        except Exception:
            # Last-resort safety net -- this optional feature must never crash the
            # app -- but by this point every error shape the SDK is documented (and
            # confirmed) to raise for this call has already been handled above, so
            # reaching here means something genuinely unanticipated happened.
            return _fallback_digest(stats, reason="unexpected error generating the AI summary via Gemini")

    # provider == "anthropic" -- unchanged from the original Anthropic-only implementation.
    if anthropic is None:
        return _fallback_digest(stats, reason="anthropic package not installed")
    try:
        client = anthropic.Anthropic(api_key=anthropic_api_key)
        response = client.with_options(timeout=REQUEST_TIMEOUT_SECONDS).messages.create(
            model=MODEL_ID,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        return text if text else _fallback_digest(stats, reason="empty response from the model")
    except anthropic.AuthenticationError:
        return _fallback_digest(stats, reason="invalid Anthropic API key")
    except anthropic.RateLimitError:
        return _fallback_digest(stats, reason="Anthropic API rate limited")
    except anthropic.APIStatusError:
        return _fallback_digest(stats, reason="Anthropic API error")
    except anthropic.APIConnectionError:
        return _fallback_digest(stats, reason="could not reach the Anthropic API")
    except Exception:
        # Last-resort safety net -- this optional feature must never crash the app.
        return _fallback_digest(stats, reason="unexpected error generating the AI summary")

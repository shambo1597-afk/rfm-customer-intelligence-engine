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
    volume. Gemini's Flash-Lite tier (the default -- see "Provider details"
    below for the specific model and a caveat on re-verifying its current
    pricing/free-tier status) has historically been the free-tier-eligible
    option at this call volume. Either way, roughly two orders of magnitude
    cheaper than the rejected design, for materially the same executive-facing
    value: a marketer wants one summary of their whole book of business per batch
    run, not one summary per customer.

--- Provider details ---

Gemini (default, GEMINI_MODEL_ID below): gemini-flash-lite-latest -- Google's
rolling alias for the current recommended Flash-Lite-tier model, chosen
DELIBERATELY over pinning a specific dated model name (e.g.
gemini-3.5-flash-lite, which this constant was briefly set to) after the
incident below: a pinned dated model gets silently deprecated out from under
this code with no warning and no code change to point at, whereas the
"-latest" alias is Google's own mechanism for always resolving to whatever
Flash-Lite-tier model is currently recommended, without this file needing an
update each time. The accepted tradeoff is the mirror image of pinning: the
model actually answering a call can change (and with it, potentially,
behavior or pricing) without a corresponding change in this codebase's git
history -- there is no changelog signal here when Google repoints the alias.
Do not swap GEMINI_MODEL_ID for a Pro model -- Pro models require billing to
be configured and are not expected to be free-tier eligible.

  Pricing/free-tier status NOT independently re-verified for whatever specific
  model the alias currently resolves to -- ai.google.dev/gemini-api/docs/
  pricing was unreachable from this environment (network egress blocks that
  domain) when this and the preceding fix were made, so no specific $/1M-token
  or "genuinely free" claim is stated here without verification -- doing so
  would silently rot the same way the original deprecation-date assumption
  did (see incident note below). What IS confirmed (live API calls against a
  real account, not docs, both before and after switching to the alias): the
  alias is listed via client.models.list(), it resolves to a Flash-Lite-tier
  model (same lineage as every prior default), and real
  generate_account_digest() calls against it succeed and return real
  generated text. Confirm current pricing/free-tier status in Google AI
  Studio or the live pricing page before treating this as a $0 guarantee for
  a new deployment, and update this note (and the README's cost-model
  section, which cites the same caveat) once re-verified.

  Model deprecation incident, 2026-08-29: gemini-2.5-flash-lite (and, tested
  for comparison, gemini-2.5-flash) started returning `404 NotFoundError` --
  "This model ... is no longer available to new users" -- confirmed via a live
  call with a real API key/project, even though both models still appeared in
  that same project's client.models.list() output with generateContent listed
  as a supported action (i.e. the model-listing endpoint does not reliably
  reflect per-model callability -- do not trust it alone). Both
  client.interactions.create() AND the older client.models.generate_content()
  failed identically (same 404, same message) for the dead models, and both
  succeeded identically for the working candidates tested (including
  gemini-3.5-flash-lite and gemini-flash-lite-latest) -- ruling out, for this
  account, an earlier hypothesis that the Interactions API specifically was
  broken for Flash-tier models. This was a straightforward model deprecation,
  not an API-method bug, so no API-method change was made here (still
  client.interactions.create() below, unchanged). GEMINI_MODEL_ID was
  initially set to the specific gemini-3.5-flash-lite as the immediate fix,
  then changed to the gemini-flash-lite-latest rolling alias once it became
  clear that pinning a dated model name is exactly what caused this incident
  in the first place, and that dated model is itself expected to be
  deprecated in turn. Re-verify the alias still resolves to a working model
  via client.models.list() periodically regardless -- an alias is not
  immunity from Google discontinuing the entire Flash-Lite tier, only from
  needing a code change every time the *dated* model underneath it rotates.

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

  Request-hangs-indefinitely note, 2026-08-29: confirmed via live testing that
  `client.interactions.create()` calls made with no timeout configured can hang
  forever -- no response, no exception, no timeout -- roughly 1 in 4-5 calls in
  a small hand-run sample (NOT a precise measured rate, just what a handful of
  manual reproduction attempts showed). This is a known, currently-open bug in
  the google-genai SDK itself, not something specific to this codebase --
  see googleapis/python-genai#1893 ("Requests hang indefinitely") and #911
  ("Setting timeout in genai.Client() does not work"). Per #911, the
  client-construction-time `genai.Client(http_options=HttpOptions(timeout=...))`
  path is documented as unreliable/broken by Google's own SDK maintainers --
  DO NOT rely on it. The fix confirmed to actually work (reproduced directly:
  repeated calls with `timeout=8.0` produced real `APITimeoutError` raises at
  ~8.0s each instead of hanging) is the PER-CALL `timeout=` keyword argument on
  `interactions.create()` itself, below (GEMINI_REQUEST_TIMEOUT_SECONDS). This
  is a genuinely different code path from the broken client-level
  `http_options` -- do not "simplify" it back to that pattern.
  `genai_errors.APITimeoutError` (a subclass of `APIConnectionError`, itself a
  subclass of `APIError` -- confirmed by reading the installed SDK's
  `compat_errors` module, not assumed) is raised when the per-call timeout is
  hit, and is caught below with its own distinct fallback reason so a genuine
  timeout is never conflated with a rate limit or an invalid key.

  This is a disclosed limitation, not something today's fix eliminates: making
  a hung call fail fast after GEMINI_REQUEST_TIMEOUT_SECONDS instead of hanging
  the UI forever does not make the underlying SDK flakiness go away. A real
  user may see the "Gemini API request timed out" fallback message more often
  than the rate-limit or invalid-key fallbacks above -- that is expected given
  the open upstream bug, not a regression in this code.

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

# Gemini: default provider -- a rolling alias, not a dated model, deliberately
# -- see the module docstring's "Provider details" and "Model deprecation
# incident, 2026-08-29" above for why (a pinned dated model, most recently
# gemini-2.5-flash-lite then briefly gemini-3.5-flash-lite, is exactly what
# broke here; the "-latest" alias always resolves to Google's current
# recommended Flash-Lite-tier model without needing a code change).
GEMINI_MODEL_ID = "gemini-flash-lite-latest"
# Mirrors MAX_TOKENS's role for Anthropic: short paragraph only, bounds worst-case
# cost. Passed as generation_config={"max_output_tokens": ...} on the
# interactions.create() call below (google.genai.types.GenerationConfig's field).
GEMINI_MAX_OUTPUT_TOKENS = 300
# Request-level timeout for client.interactions.create() -- see the module
# docstring's "Request-hangs-indefinitely note" above (googleapis/python-genai
# #1893, #911): without this, a call can hang forever with no exception. 20s is
# generous enough for a normal, slower-than-usual real response to still
# succeed, short enough that a hung request fails fast rather than blocking the
# Streamlit UI indefinitely. MUST be passed as the PER-CALL `timeout=` kwarg on
# interactions.create() itself (as done below) -- NOT via
# genai.Client(http_options=HttpOptions(timeout=...)) at client-construction
# time, which #911 documents as unreliable/broken in this SDK. Mirrors
# REQUEST_TIMEOUT_SECONDS's role for Anthropic, kept as a separate constant
# rather than reused because it is passed through an entirely different
# mechanism (a per-call kwarg here vs. client.with_options(timeout=...) there).
GEMINI_REQUEST_TIMEOUT_SECONDS = 20.0

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


def _call_gemini(
    prompt_or_input: str,
    system_instruction: str,
    api_key: str,
    max_output_tokens: int,
    previous_interaction_id: str = None,
) -> tuple:
    """
    Shared Gemini call + error-classification logic for both
    generate_account_digest() below (a one-shot prompt: system_instruction=None,
    no previous_interaction_id) and src/chat_engine.py's answer_account_question()
    (a real system prompt, plus previous_interaction_id for multi-turn
    chaining). Extracted here, in digest_engine.py, and imported BY
    chat_engine.py -- never the reverse -- because chat_engine.py already
    depends on digest_engine.py for GEMINI_MODEL_ID, GEMINI_REQUEST_TIMEOUT_SECONDS,
    etc.; keeping the dependency one-directional avoids a circular import.

    Why this exists: before this extraction, generate_account_digest() and
    answer_account_question() each independently re-implemented the same
    client construction, the same interactions.create() call shape, and the
    same exception-handling chain -- confirmed duplication that already
    caused near-misses (the model-ID fix and the request-timeout fix each had
    to be hand-applied to both files separately, with real risk of the two
    drifting out of sync). This function owns ONLY the provider call and its
    error classification; it does NOT decide whether Gemini should be called
    at all (_resolve_provider() still does that in each caller, before it
    even builds a prompt/system_prompt) and does NOT do any caller-specific
    bookkeeping -- conversation_history mutation is chat_engine.py's own
    concern, not this function's (see answer_account_question()).

    Returns a (response_text, interaction_id, failure_reason) tuple:
    - Success: (text, the real interaction id if the response object carries
      one else None, None).
    - Failure (empty response, or any error class below): (None, None,
      reason) -- `reason` is one of the exact strings both callers already
      used before this extraction ("Gemini free-tier rate limit reached",
      "invalid Gemini API key", "Gemini API request timed out", "Gemini API
      error", "empty response from Gemini", "google-genai package not
      installed") so neither caller's existing fallback-reason tests needed
      to change.

    Never raises -- both callers still guarantee "this optional feature never
    crashes the app" by construction (every exception the SDK is documented,
    and confirmed, to raise for this call is classified below), not by
    re-wrapping this call in yet another try/except.
    """
    if genai is None:
        # Defense-in-depth only, and what makes this function independently
        # testable for this failure mode (patch src.digest_engine.genai
        # directly, since that's the name this function itself reads). Both
        # callers already check `genai is None` themselves FIRST, before ever
        # calling this function -- that check can't move here instead,
        # because each caller's own existing tests patch THAT module's own
        # `genai` name (e.g. src.chat_engine.genai), which -- being a
        # `from src.digest_engine import genai` value-import -- is an
        # independent name binding from this module's `genai`, not an alias
        # of it. Rebinding one doesn't rebind the other, so the "package
        # missing" check has to stay caller-side to keep matching each
        # existing test's patch target.
        return None, None, "google-genai package not installed"
    try:
        client = genai.Client(api_key=api_key)
        create_kwargs = {
            "model": GEMINI_MODEL_ID,
            "input": prompt_or_input,
            "generation_config": {"max_output_tokens": max_output_tokens},
            # Per-call timeout, NOT client-level http_options -- see
            # GEMINI_REQUEST_TIMEOUT_SECONDS above (googleapis/python-genai
            # #1893, #911): the client-level path is confirmed unreliable/
            # broken for this SDK; this is the one confirmed to actually work.
            "timeout": GEMINI_REQUEST_TIMEOUT_SECONDS,
        }
        if system_instruction is not None:
            create_kwargs["system_instruction"] = system_instruction
        if previous_interaction_id:
            create_kwargs["previous_interaction_id"] = previous_interaction_id

        interaction = client.interactions.create(**create_kwargs)
        text = (interaction.output_text or "").strip()
        if not text:
            return None, None, "empty response from Gemini"
        return text, getattr(interaction, "id", None), None
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
            return None, None, "Gemini free-tier rate limit reached"
        if status_code in (401, 403) or "API_KEY_INVALID" in message:
            return None, None, "invalid Gemini API key"
        return None, None, "Gemini API error"
    except genai_errors.APITimeoutError:
        # Raised when the per-call `timeout=GEMINI_REQUEST_TIMEOUT_SECONDS`
        # above is hit -- see the module docstring's "Request-hangs-
        # indefinitely note" (googleapis/python-genai#1893, #911). Caught
        # BEFORE the broad genai_errors.APIError below (APITimeoutError is a
        # subclass of APIConnectionError, itself a subclass of APIError, so
        # it would otherwise be swallowed by that branch's generic message)
        # with its own distinct reason -- a timeout is a genuinely different
        # situation from a rate limit or an invalid key and should never be
        # reported as either.
        return None, None, "Gemini API request timed out"
    except genai_errors.APIError:
        # Broad base class for everything else the SDK raises for this call
        # (connection errors, and any future APIStatusError subclass not
        # distinguished above) -- still a diagnosable "the SDK reported an
        # API error", not a truly unanticipated exception. Catching the base
        # class here (rather than only the specific subclasses above) means a
        # future error variant this code doesn't yet know about still gets a
        # meaningful reason instead of silently falling through to the
        # generic catch-all below, which is exactly the bug being fixed.
        return None, None, "Gemini API error"
    except Exception:
        # Last-resort safety net -- this optional feature must never crash the
        # app -- but by this point every error shape the SDK is documented (and
        # confirmed) to raise for this call has already been handled above, so
        # reaching here means something genuinely unanticipated happened.
        return None, None, "unexpected error calling Gemini"


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
    Mirrors _call_gemini() above exactly in spirit and in why it lives here
    (see that function's docstring) -- extracted because the two callers'
    Anthropic branches were otherwise identical modulo how `messages`/`system`
    get built, which is now just a parameter each caller supplies.

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

    Never raises, for the same reason _call_gemini() doesn't.
    """
    if anthropic is None:
        # Defense-in-depth only -- see _call_gemini()'s identical note on why
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
        # Shared call/error-handling logic -- see _call_gemini()'s own
        # docstring for why this is extracted (also used by
        # src/chat_engine.py's answer_account_question()) and why the
        # "package missing" check above stays here rather than moving inside
        # it. The digest is a one-shot prompt: no system instruction, no
        # multi-turn chaining, so system_instruction=None and
        # previous_interaction_id is left at its default (None).
        text, _interaction_id, failure_reason = _call_gemini(
            prompt_or_input=prompt,
            system_instruction=None,
            api_key=gemini_api_key,
            max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
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

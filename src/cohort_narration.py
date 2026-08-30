"""
src/cohort_narration.py - Optional AI narration of a single, deterministically-
identified cohort-retention pattern, shown in app.py Tab 1 below the existing
Monthly Acquisition Cohort Retention heatmap.

--- Two steps, deliberately kept separate ---

1. src/cohort_engine.py's find_notable_cohort_pattern() -- a PLAIN pandas/
   numpy statistical scan (no LLM call, no network) of the retention matrix
   that's already computed and already displayed as a heatmap. It picks the
   single cohort-month cell that deviates most from its peers via a
   documented, testable z-score definition -- see that function's docstring.
   This step is real information on its own: which cohort, which month, and
   by how much, all computed deterministically, with zero API cost and zero
   dependency on whether any LLM key is configured.
2. narrate_cohort_pattern() (this file) -- takes ONLY that already-identified
   finding (a small dict: cohort, month, the numbers) and asks an LLM to
   explain it in one or two plain-language sentences for a non-technical
   reader. It is never handed the raw retention matrix, and it is never
   asked to find the pattern itself -- asking a model to eyeball an entire
   matrix and guess which cell "looks interesting" would be expensive,
   nondeterministic, and untestable, exactly what step 1 exists to avoid.

Because step 1 is real information by itself, app.py shows it unconditionally
whenever this optional feature's checkbox is on -- the LLM narration in step 2
is additive polish (easier to read at a glance), never the only source of the
finding. This mirrors the same principle src/chat_context.py and
src/digest_engine.py already established: the deterministic engines remain
100% baked-in, and an LLM is only ever a narrator over output already
computed without it.

--- Providers, model, and error handling: reused, not re-derived ---

Exactly like src/chat_engine.py and src/roi_advisor.py, this module does NOT
define its own provider-selection logic, model constants, provider-client
construction, or exception-handling pattern -- it imports and reuses
src/digest_engine.py's _resolve_provider() / _call_groq() / _call_anthropic().
See src/chat_engine.py's own module docstring for the fuller rationale (the
same one applies here verbatim): duplicating that logic a fourth time was
never seriously considered.

--- Cost model ---

One LLM call per batch run at most, same cadence and same reasoning as the AI
Digest (src/digest_engine.py) -- there is exactly one "most notable" finding
per dataset (find_notable_cohort_pattern() returns a single dict, not a
list), so this can never scale per-customer or per-cohort the way a rejected
per-record design would. app.py caches the call via st.cache_data for the
same reason the digest and chat-context blob are cached: Streamlit reruns the
whole script on every widget interaction elsewhere in the app.
"""

from src.digest_engine import (
    anthropic,
    groq,
    _resolve_provider,
    _call_groq,
    _call_anthropic,
    GROQ_REASONING_TOKEN_HEADROOM,
)

# One or two sentences only -- smaller than even the digest's 300-token cap,
# since this is explaining a single already-identified number, not
# summarizing a whole account. This is the visible-answer target passed to
# Anthropic unchanged; see GROQ_COHORT_NARRATION_MAX_OUTPUT_TOKENS below for
# what's actually sent to Groq.
#
# THIS is the exact constant whose original value (150, with no reasoning
# headroom at all) caused the live truncation/empty-response bug this fix
# addresses -- see GROQ_REASONING_TOKEN_HEADROOM's own comment in
# digest_engine.py for the full incident. 150 tokens left essentially no
# room for openai/gpt-oss-120b's hidden reasoning at all, so this was the
# single most exposed call site of the four.
COHORT_NARRATION_MAX_OUTPUT_TOKENS = 150
# The value actually passed as max_tokens on the Groq call specifically --
# COHORT_NARRATION_MAX_OUTPUT_TOKENS's own visible-answer target (150) is
# unchanged; only Groq's call gets the extra reasoning headroom.
GROQ_COHORT_NARRATION_MAX_OUTPUT_TOKENS = COHORT_NARRATION_MAX_OUTPUT_TOKENS + GROQ_REASONING_TOKEN_HEADROOM

_UNAVAILABLE_PREFIX = "**[Cohort narration temporarily unavailable"


def _narration_unavailable(reason: str) -> str:
    """
    Deterministic, clearly-labeled unavailability message -- mirrors
    src/chat_engine.py's _chat_unavailable() and src/roi_advisor.py's
    _advisor_unavailable() in spirit. Note app.py does NOT actually surface
    this string to the user in the "no key configured" case -- see that
    module's Tab 1 wiring, which shows the raw deterministic finding alone
    instead of this fallback text when no key is configured, per the task
    that added this feature ("the raw numeric finding alone if not [a key
    is configured]"). This function still returns a real, non-empty string
    for that case (rather than None) so it remains a normal, always-callable
    function independent of any particular caller's UI choice about when to
    invoke it -- callers that DO want this text (or tests exercising this
    function directly) get it.
    """
    return f"{_UNAVAILABLE_PREFIX} — {reason}.]** Please try again, or contact support if this persists."


def _build_cohort_narration_prompt(pattern: dict) -> str:
    """
    Serializes find_notable_cohort_pattern()'s already-computed finding into
    a compact one-shot prompt -- mirrors src/digest_engine.py's
    _build_prompt() in spirit: every value here comes directly from
    `pattern`, nothing derived independently, no raw matrix or per-customer
    data ever included.
    """
    return (
        "You are explaining a single statistical finding from a customer-retention "
        "dashboard to a non-technical marketing reader. Using ONLY the numbers "
        "below -- already computed deterministically, not by you -- write ONE to "
        "TWO short, plain-language sentences explaining what this finding means "
        "in practical terms. Do not invent any numbers, other cohorts, or a "
        "definite cause not given below; you may suggest it MAY be worth "
        "investigating (e.g. a change in that month's acquisition channel or "
        "promotion), phrased as a possibility, never as a stated fact, since "
        "nothing here actually says why.\n\n"
        f"Cohort: {pattern['cohort']} (customers first acquired that month)\n"
        f"Month since acquisition: {pattern['month_index']}\n"
        f"This cohort's retention at that month: {pattern['retention_pct']:.1f}%\n"
        f"Average retention for other cohorts at that same month: {pattern['column_mean_pct']:.1f}%\n"
        f"Deviation: {pattern['deviation_pct_points']:+.1f} percentage points "
        f"({pattern['direction']}, z-score {pattern['z_score']:+.2f})\n"
    )


def narrate_cohort_pattern(
    pattern: dict,
    anthropic_api_key: str = None,
    groq_api_key: str = None,
    provider_override: str = None,
) -> str:
    """
    Explains ONE already-identified cohort pattern (from
    src/cohort_engine.py's find_notable_cohort_pattern() -- never raw matrix
    data) in one or two plain-language sentences.

    Parameters:
    - pattern: the dict find_notable_cohort_pattern() returns. An empty
      dict/falsy value (that function's own "nothing notable found" signal)
      short-circuits here without resolving a provider or building a
      prompt at all -- there is nothing to narrate.
    - anthropic_api_key, groq_api_key, provider_override: identical contract
      to every other optional AI feature in this codebase --
      _resolve_provider() (imported, not re-derived) makes the same
      Groq-preferred-by-default / Anthropic-fallback / "none" decision.

    Returns the narration text. NEVER raises: on any failure (no pattern to
    narrate, no key configured, invalid key, rate limit, network error,
    empty response, anything unanticipated), returns a clearly-labeled
    "temporarily unavailable" message instead -- this optional feature must
    never crash the app, exactly like every other AI feature here.
    """
    if not pattern:
        return _narration_unavailable("no notable cohort pattern to narrate")

    provider = _resolve_provider(anthropic_api_key, groq_api_key, override=provider_override)
    if provider == "none":
        return _narration_unavailable("no GROQ_API_KEY or ANTHROPIC_API_KEY configured")

    prompt = _build_cohort_narration_prompt(pattern)

    if provider == "groq":
        if groq is None:
            return _narration_unavailable("groq package not installed")
        # Shared call/error-handling logic -- see _call_groq()'s own
        # docstring in src/digest_engine.py. One-shot prompt: no system
        # role, no multi-turn chaining, same shape generate_account_digest()
        # already uses for its Groq branch.
        text, failure_reason = _call_groq(
            messages=[{"role": "user", "content": prompt}],
            api_key=groq_api_key,
            max_tokens=GROQ_COHORT_NARRATION_MAX_OUTPUT_TOKENS,
        )
    else:
        # provider == "anthropic"
        if anthropic is None:
            return _narration_unavailable("anthropic package not installed")
        text, failure_reason = _call_anthropic(
            messages=[{"role": "user", "content": prompt}],
            api_key=anthropic_api_key,
            max_tokens=COHORT_NARRATION_MAX_OUTPUT_TOKENS,
        )

    return text if text else _narration_unavailable(failure_reason)

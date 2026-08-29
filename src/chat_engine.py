"""
src/chat_engine.py - Optional Chat Q&A over a precomputed context blob.

Lets a user ask natural-language questions about ONE account's customer
intelligence, answered ONLY from a context blob built once per batch run by
src/chat_context.py -- never by live-querying rfmt_df/clv_df, never by calling
back into src/rfm_engine.py / src/clv_engine.py mid-conversation, never with its
own pandas computation. If a question asks for something the blob doesn't
contain (a hypothetical/what-if, a customer not in the watchlist/growth-target
lists, anything needing live computation), the model is instructed to say so
plainly rather than invent a plausible-sounding answer -- see
CHAT_SYSTEM_PROMPT_TEMPLATE below.

--- Why this is NOT open-ended chat with live tool-calling ---

This is an ADDITIONAL optional feature alongside the existing per-account
digest (src/digest_engine.py), not a replacement for it, and deliberately not
a general-purpose agent: no tool/function-calling back into the pipeline
mid-conversation, and no new pandas computation happens inside a chat turn.
Most of the context blob is aggregate-only; by an explicit, informed decision
(see src/chat_context.py's module docstring "PII posture" section), individual
CustomerID-level rows from exactly two already-prioritized tables -- the churn
watchlist and growth targets -- now reach the blob and this model, because the
two datasets bundled with this repo carry no real personal-privacy exposure.
That exception is narrow and does NOT extend to real customer data -- see the
same module's load-bearing caveat about src/shopify_ingest.py before ever
reusing this design against a live store. The reasoning otherwise mirrors
digest_engine.py's cost-model rationale almost exactly, with one important
difference in shape -- see the cost model section below.

--- Providers, model, and error handling: reused, not re-derived ---

This module deliberately does NOT define its own provider-selection logic,
model constants, provider-client construction, or exception-handling pattern
-- it imports and reuses digest_engine.py's:
  - _resolve_provider() for the Groq-preferred-by-default / Anthropic-
    fallback decision (identical semantics to the digest feature).
  - _call_groq() / _call_anthropic() for the ENTIRE provider call: client
    construction, the chat.completions.create()/messages.create() call
    shape, and the full exception-handling chain for each provider -- not
    just the same constants/exception classes reused in parallel, the
    literal same function bodies. This was a real, confirmed maintainability
    risk before this consolidation existed: generate_account_digest()
    (digest_engine.py) and answer_account_question() (below) each
    independently re-implemented nearly identical provider call logic, and
    it already caused near-misses -- fixes each had to be hand-applied to
    both files separately, with real risk of the two drifting out of sync.
    _call_groq()/_call_anthropic() own ONLY the provider call and its error
    classification (see their docstrings in digest_engine.py for the exact
    return shape) -- caller-specific bookkeeping (building the system
    prompt, assembling `messages` from conversation_history, and mutating
    conversation_history) stays here in answer_account_question(), NOT in
    the shared functions.
  - GROQ_MODEL_ID / MODEL_ID / GROQ_REQUEST_TIMEOUT_SECONDS -- re-exported
    here (imported but not directly called with -- _call_groq()/
    _call_anthropic() already read digest_engine.py's own copies internally)
    purely so a future model or timeout swap in digest_engine.py is
    verifiably visible from this module's own namespace too, never re-guessed
    or re-pinned here, so the two features can never silently drift onto
    different models.

--- Multi-turn conversation ---

A real chat needs conversation context across turns, unlike the digest
feature's one-shot call. Both providers now use the SAME stateless pattern --
a plain `messages` list of {"role", "content"} turns, resent in full on every
call (neither Groq's nor Anthropic's chat API keeps server-side conversation
state to chain against) -- so there is exactly one multi-turn mechanism here,
not two. (This wasn't always true: Gemini, this project's original default
provider before it was replaced with Groq, used a different, stateful
`previous_interaction_id`-chaining mechanism that required its own
bookkeeping key in every history entry. That asymmetry is gone along with
Gemini -- see digest_engine.py's module docstring "Provider swap" note.)

`conversation_history` is a plain list of {"role", "content"} turn dicts this
function both READS (to reconstruct prior context) and MUTATES IN PLACE
(appending this turn's question and answer before returning) -- the caller
(app.py) passes the same list object across turns within a Streamlit session
(st.session_state) rather than reassembling it. For Anthropic, this list IS
the `messages` payload (plus the new question appended), with the system
prompt passed separately via Anthropic's own `system=` field. For Groq
(OpenAI-compatible: no separate `system` field), a fresh {"role": "system",
...} entry is prepended ahead of `conversation_history` on every call instead
-- never stored back into conversation_history itself, so the stored history
stays provider-agnostic and reusable regardless of which provider ends up
resolved on any given turn (keys can change between turns).

--- Cost model: usage-based, NOT the digest's fixed-per-batch-run bound ---

State this plainly rather than pretend chat inherits the digest's tight bound:
this feature is ONE LLM CALL PER QUESTION ASKED, not one call per account per
batch run. That makes its cost usage-based rather than fixed -- a fundamentally
different shape from digest_engine.py's cost model, not a variant of it. The
practical bound: a chat session used by one analyst reviewing one account in
one sitting asks maybe 5-20 questions, each a small prompt (the context blob is
kept compact -- see src/chat_context.py) against a short answer (capped by
CHAT_MAX_OUTPUT_TOKENS below). On Groq's free tier (see digest_engine.py's
module docstring for the current rate-limit numbers and the caveat on how
they were sourced -- the same caveat applies here, unchanged), that's still
comfortably within the daily/per-minute ceilings for a single analyst's
sitting. But unlike the digest feature, there is no hard ceiling on how many
questions a session can ask -- this module does NOT implement per-session
budgeting/rate-limiting of its own; it relies on each provider's own
account-level rate limits (the same 429 handling digest_engine.py already
has) as the practical backstop. This is a scope decision, not an oversight:
per-session cost tracking was explicitly out of scope for this feature (see
the task that introduced this module).
"""

from src.digest_engine import (
    anthropic,
    groq,
    _resolve_provider,
    _call_groq,
    _call_anthropic,
    MODEL_ID,
    GROQ_MODEL_ID,
    GROQ_REQUEST_TIMEOUT_SECONDS,
)

# Chat answers get more room than the digest's 300-token cap (a single
# executive-summary paragraph) -- a real answer to an analyst's question may
# reasonably run a bit longer, but this still bounds worst-case per-call cost
# and keeps answers focused rather than open-ended.
CHAT_MAX_OUTPUT_TOKENS = 500

_UNAVAILABLE_PREFIX = "**[Chat temporarily unavailable"


def escape_markdown_dollar_signs(text: str) -> str:
    r"""
    Escapes every literal '$' in `text` as '\$' so Streamlit's markdown
    renderer never interprets it as an inline-LaTeX delimiter.

    THIS is the actual fix for the LaTeX-rendering bug (confirmed via a real
    screenshot: an answer containing a dollar amount and "P(Alive)" rendered
    as "716,276.02andanaverageP(Alive)$ of 10.1%" in the Streamlit UI). Root
    cause, confirmed by reading the installed Streamlit package's own code,
    not assumed: `st.markdown()`'s body is passed through unmodified (its
    `clean_text()` helper -- streamlit/string_util.py -- only dedents/strips
    whitespace, never touches '$'), and Streamlit's client-side markdown
    renderer is built on remark-math with `singleDollarTextMath` enabled by
    default (confirmed by locating that exact option, defaulted true, inside
    the installed package's static JS bundle,
    streamlit/static/static/js/StreamlitMarkdown.*.js) -- so ANY pair of bare
    '$' characters anywhere in the rendered text gets treated as an opening/
    closing inline-math span, with ordinary spaces inside that span collapsed
    by the math renderer. Two dollar amounts in the same answer (or one
    dollar amount plus the model wrapping "P(Alive)" in $ signs out of its
    own math-notation habit, as it evidently did in the screenshotted case)
    is enough to accidentally form such a pair -- this is a structural risk
    of ANY answer containing more than one bare '$', not a rare edge case.

    Backslash-escaping '$' is a standard CommonMark punctuation escape ('$'
    is one of the ASCII punctuation characters eligible for it per the
    CommonMark spec) that remark/micromark (what Streamlit's renderer is
    built on) honors as a literal character, never as a math delimiter --
    this is what actually neutralizes the bug, independent of anything the
    model itself writes. There is no Streamlit-side parameter to disable
    single-dollar math interpretation from the Python API (`st.markdown()`
    has no such kwarg -- confirmed by reading its signature/docstring in the
    installed package; `singleDollarTextMath` is only reachable from the
    frontend bundle, not exposed to callers), so escaping the text itself
    before it reaches `st.markdown()` is the only fix available at this
    layer. Applied unconditionally, not just "outside code blocks" -- this
    chat feature is a business data assistant, not a coding tool, so there is
    no legitimate use case here for an unescaped '$' (real LaTeX/currency-as-
    math) to preserve.

    CHAT_SYSTEM_PROMPT_TEMPLATE below separately instructs the model not to
    use LaTeX/math notation -- that reduces how often this situation even
    arises, but does NOT by itself fix the bug (the model doesn't control
    Streamlit's renderer, and a well-behaved model writing two independent
    dollar amounts in one answer would still trigger it). This function is
    the actual, guaranteed fix; the prompt instruction is a supplementary
    risk-reduction measure only.
    """
    return text.replace("$", "\\$")


CHAT_SYSTEM_PROMPT_TEMPLATE = (
    "You are a data assistant answering questions about ONE business account's "
    "customer intelligence dashboard, for the analyst reviewing it. Answer ONLY "
    "using the CONTEXT DATA below -- it is the complete, already-computed "
    "summary for this account. Never invent numbers not present in it, and "
    "never speculate about data it doesn't contain.\n\n"
    "Individual customers: the CHURN WATCHLIST and TOP 90-DAY GROWTH TARGETS "
    "sections each include an \"Individual accounts\" list with real "
    "CustomerIDs and their data (spend, recency, frequency, segment, and "
    "either P(Alive)/risk tier or predicted 90-day spend) -- these ARE "
    "legitimately available to reference. When asked a \"which/who\" question "
    "(e.g. \"which 5 customers should I focus on\", \"who are my top churn "
    "risks\"), answer with the actual CustomerIDs and figures from those "
    "lists -- do not refuse or fall back to a vague aggregate answer when the "
    "specific data is right there in front of you. But stay scoped to exactly "
    "those two lists: if asked about a CustomerID that does NOT appear in "
    "either \"Individual accounts\" list, say plainly that customer isn't in "
    "the current watchlist/growth-target data -- do not fabricate details "
    "about them, and do not imply access to the full customer base (this "
    "context contains individual rows for ONLY those two prioritized lists, "
    "not every customer on the account). Also never invent or imply access to "
    "any customer detail beyond what those lists actually contain -- no names, "
    "emails, addresses, or raw transaction history; CustomerID plus the listed "
    "fields is the full extent of what's available per customer.\n\n"
    "If a question asks for something the CONTEXT DATA cannot answer -- a "
    "hypothetical or what-if scenario (e.g. \"what if I raised prices 10%\"), "
    "a customer not in either individual-accounts list, or anything requiring "
    "computation beyond what is already summarized below -- say plainly that "
    "the current dashboard data can't answer that, rather than guessing or "
    "fabricating a plausible-sounding response. A short, honest \"I can't "
    "answer that from the current data\" is always preferable to a fabricated "
    "answer. If a question is about HOW the scoring/segmentation/CLV model "
    "works, its accuracy, or its limitations -- rather than about this "
    "account's specific numbers -- answer using the MODEL METHODOLOGY & KNOWN "
    "LIMITATIONS section of the CONTEXT DATA, and be exactly as candid about "
    "limitations (the CLV model being a heuristic, not a fitted probabilistic "
    "one; the real-data backtest not beating the naive baseline; the ML "
    "clustering being validation-only and never changing a customer's "
    "segment) as that section itself is -- never soften or spin a documented "
    "limitation into something more flattering than what the context data "
    "actually says.\n\n"
    "Formatting: never use LaTeX or math notation of any kind -- do not wrap "
    "anything in single or double dollar signs (e.g. do not write \"$P(Alive)$\" "
    "or \"$$...$$\") and do not use \\frac, \\sum, or similar math markup. "
    "Write currency plainly, e.g. \"$1,234.56\", and write \"P(Alive)\" as "
    "plain text with no surrounding dollar signs or other math delimiters. "
    "This is a business data assistant, not a math tool -- there is never a "
    "legitimate reason to use math notation here, and doing so breaks how "
    "your answer renders for the analyst.\n\n"
    "CONTEXT DATA:\n{context_blob}\n"
)


def _chat_unavailable(reason: str) -> str:
    """
    Deterministic, clearly-labeled unavailability message -- mirrors
    digest_engine.py's _fallback_digest() in spirit (never crash, always
    return something clearly distinguishable from a real answer), but chat has
    no aggregate stats to render into a template summary the way the digest
    does, so this is a short, honest status message instead.
    """
    return f"{_UNAVAILABLE_PREFIX} — {reason}.]** Please try again, or contact support if this persists."


def answer_account_question(
    question: str,
    context_blob: str,
    conversation_history: list,
    anthropic_api_key: str = None,
    groq_api_key: str = None,
    provider_override: str = None,
) -> str:
    """
    Answers ONE question about an account, using ONLY context_blob (built once
    per batch run by src/chat_context.py's build_context_text()) plus prior
    turns in conversation_history for multi-turn context -- see the module
    docstring for the full provider/cost-model rationale.

    Parameters:
    - question: the analyst's natural-language question for this turn.
    - context_blob: the serialized context text (see src/chat_context.py --
      mostly aggregate, plus individual watchlist/growth-target rows by
      informed decision) -- built ONCE per batch run by the caller, not
      rebuilt per question.
    - conversation_history: prior turns as a plain list of {"role", "content"}
      dicts (role is "user" or "assistant"). READ for multi-turn context AND
      MUTATED IN PLACE -- this call's question and answer are appended before
      returning, so the SAME list object should be passed across turns within
      a session (see the module docstring's "Multi-turn conversation" section
      for exactly what gets appended and why).
    - anthropic_api_key, groq_api_key, provider_override: identical contract
      to generate_account_digest() in src/digest_engine.py -- _resolve_provider()
      (imported, not re-derived) makes the same Groq-preferred-by-default /
      Anthropic-fallback / "none" decision.

    Returns the answer text. NEVER raises: on any failure (no key configured,
    invalid key, rate limit, network error, empty response, anything
    unanticipated), returns a clearly-labeled "chat temporarily unavailable"
    message instead -- this optional feature must never crash the app, exactly
    like the digest feature's fallback guarantee. Unlike the digest feature,
    a failed turn is NOT recorded into conversation_history (nothing useful
    for the model to reference next turn), so the next question still has a
    clean chain from the last SUCCESSFUL turn.
    """
    conversation_history = conversation_history if conversation_history is not None else []
    provider = _resolve_provider(anthropic_api_key, groq_api_key, override=provider_override)

    if provider == "none":
        return _chat_unavailable("no GROQ_API_KEY or ANTHROPIC_API_KEY configured")

    system_prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(context_blob=context_blob)

    # Shared call/error-handling logic for whichever provider resolves -- see
    # _call_groq()/_call_anthropic()'s own docstrings in digest_engine.py
    # (also used by generate_account_digest()). Building `messages` from
    # conversation_history plus this turn's question is chat-specific and
    # stays here, same as the system-prompt handling below (a fresh {"role":
    # "system", ...} entry for Groq's OpenAI-compatible shape, Anthropic's
    # own separate `system=` field for Anthropic) -- neither shared function
    # does any of this bookkeeping itself.
    if provider == "groq":
        if groq is None:
            return _chat_unavailable("groq package not installed")
        messages = [{"role": "system", "content": system_prompt}] + conversation_history + [
            {"role": "user", "content": question}
        ]
        text, failure_reason = _call_groq(
            messages=messages,
            api_key=groq_api_key,
            max_tokens=CHAT_MAX_OUTPUT_TOKENS,
        )
    else:
        # provider == "anthropic"
        if anthropic is None:
            return _chat_unavailable("anthropic package not installed")
        messages = conversation_history + [{"role": "user", "content": question}]
        text, failure_reason = _call_anthropic(
            messages=messages,
            api_key=anthropic_api_key,
            max_tokens=CHAT_MAX_OUTPUT_TOKENS,
            system=system_prompt,
        )

    if failure_reason:
        return _chat_unavailable(failure_reason)

    conversation_history.append({"role": "user", "content": question})
    conversation_history.append({"role": "assistant", "content": text})
    return text

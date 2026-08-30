"""
tests/test_chat_engine.py - pytest suite for the optional Chat Q&A feature.

No real network calls are made anywhere in this file -- both provider clients
are always mocked (unittest.mock), mirroring tests/test_digest_engine.py's
approach exactly (same mocking shape, same real-exception-class helper).

One honest limitation, stated plainly rather than glossed over: this file
CANNOT unit-test that the model actually refuses to hallucinate an answer to
an out-of-scope question -- that's a property of the real model's behavior
against a real API call, not something a mocked response can demonstrate (a
mock just returns whatever string the test tells it to). What IS tested here
is that the SYSTEM PROMPT actually sent to the provider instructs this
behavior explicitly -- see TestSystemPromptInstructsScopeLimitation below.
The real-model behavior itself was verified manually against a live API key
per this task's verification step (see the PR/commit description), not here.
"""

import re
import types
from unittest.mock import MagicMock, patch

import groq
import httpx
import pytest

from src.chat_context import build_account_context_blob, build_context_text
from src.chat_engine import (
    answer_account_question,
    escape_markdown_dollar_signs,
    CHAT_MAX_OUTPUT_TOKENS,
    GROQ_CHAT_MAX_OUTPUT_TOKENS,
    CHAT_SYSTEM_PROMPT_TEMPLATE,
    _chat_unavailable,
)
from src.digest_engine import MODEL_ID, GROQ_MODEL_ID, GROQ_REQUEST_TIMEOUT_SECONDS


def _make_groq_api_error(status_code: int, message: str = "error"):
    """Identical helper to tests/test_digest_engine.py's -- builds a real groq
    exception via the REAL groq.Groq client's own _make_status_error()
    dispatch method (confirmed via direct introspection of the pinned groq
    package, groq/_client.py -- not a guess). Duplicated here (not imported)
    since test files in this repo are self-contained."""
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status_code=status_code, request=request)
    client = groq.Groq(api_key="dummy-key-for-error-construction")
    return client._make_status_error(message, body=None, response=response)


def _mock_groq_response(text="An answer."):
    """Builds a fake groq ChatCompletion-shaped response -- response.choices[0].message.content
    is the real path (confirmed via direct introspection of groq.types.chat), mirroring
    tests/test_digest_engine.py's identical helper."""
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text))]
    )


@pytest.fixture(scope="module")
def context_text(rfmt_df, clv_df, segment_summary, cohort_retention_matrix, crosstab_counts):
    blob = build_account_context_blob(rfmt_df, clv_df, segment_summary, cohort_retention_matrix, crosstab_counts)
    return build_context_text(blob)


class TestNamedConstants:
    def test_chat_max_output_tokens_is_a_positive_int_constant(self):
        assert isinstance(CHAT_MAX_OUTPUT_TOKENS, int) and CHAT_MAX_OUTPUT_TOKENS > 0

    def test_chat_allows_more_room_than_the_digests_short_paragraph_cap(self):
        # The task spec is explicit: chat answers need more room than the
        # digest's 300-token cap, not the same short cap re-used verbatim.
        from src.digest_engine import MAX_TOKENS as DIGEST_MAX_TOKENS
        assert CHAT_MAX_OUTPUT_TOKENS > DIGEST_MAX_TOKENS

    def test_reuses_digest_engines_model_ids_rather_than_redefining_them(self):
        # Importing straight from src.chat_engine's own namespace confirms these
        # are literally the same names digest_engine.py exports -- not a
        # separately re-guessed model string that could silently drift.
        import src.chat_engine as chat_engine_module
        assert chat_engine_module.GROQ_MODEL_ID is GROQ_MODEL_ID
        assert chat_engine_module.MODEL_ID is MODEL_ID
        assert chat_engine_module.GROQ_REQUEST_TIMEOUT_SECONDS is GROQ_REQUEST_TIMEOUT_SECONDS


class TestSystemPromptInstructsScopeLimitation:
    """See this file's module docstring for why this is the honest substitute
    for unit-testing actual hallucination avoidance."""

    def test_system_prompt_instructs_answering_only_from_context(self):
        prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(context_blob="dummy context")
        assert "ONLY" in prompt
        assert "CONTEXT DATA" in prompt

    def test_system_prompt_instructs_declining_out_of_scope_questions_honestly(self):
        prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(context_blob="dummy context")
        assert "what if" in prompt.lower()
        assert "can't answer" in prompt.lower() or "cannot answer" in prompt.lower()
        assert "fabricat" in prompt.lower()

    def test_system_prompt_embeds_the_actual_context_blob_verbatim(self, context_text):
        prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(context_blob=context_text)
        assert context_text in prompt

    def test_system_prompt_permits_referencing_watchlist_and_growth_target_customer_ids(self):
        # Per the individual-row exposure change (see src/chat_context.py):
        # the model must be told these two lists' CustomerIDs are legitimately
        # available, so it doesn't over-apply the old blanket "no individual
        # customers" refusal to data that's now actually in front of it.
        prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(context_blob="dummy context")
        assert "CHURN WATCHLIST" in prompt
        assert "GROWTH TARGETS" in prompt
        assert "legitimately available" in prompt
        assert "which" in prompt.lower() and "who" in prompt.lower()

    def test_system_prompt_still_scopes_refusal_to_customers_outside_the_two_lists(self):
        prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(context_blob="dummy context")
        assert "does NOT appear" in prompt
        assert "not every customer on the account" in prompt.lower() or "not in the current watchlist" in prompt.lower()

    def test_system_prompt_instructs_avoiding_latex_and_math_notation(self):
        # This is the supplementary risk-reduction half of the LaTeX-rendering
        # bug fix (see escape_markdown_dollar_signs()'s docstring for the
        # actual, guaranteed fix) -- reduces how often the model's own
        # phrasing triggers Streamlit's inline-math parsing, though it can't
        # eliminate the risk on its own (the model doesn't control the
        # renderer, and two independent dollar amounts in one answer would
        # still form an accidental math-delimiter pair regardless of intent).
        prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(context_blob="dummy context")
        assert "latex" in prompt.lower()
        assert "P(Alive)" in prompt
        assert "$1,234.56" in prompt


class TestLatexRenderingBugFix:
    """
    Regression coverage for the LaTeX-rendering bug: a Chat Q&A answer
    containing a dollar amount and "P(Alive)" rendered as
    "716,276.02andanaverageP(Alive)$ of 10.1%" in Streamlit's markdown output
    (confirmed via a real screenshot). Root cause and the actual fix are
    documented in escape_markdown_dollar_signs()'s own docstring.
    """

    def test_every_bare_dollar_sign_is_escaped(self):
        raw = "Total historical value: $716,276.02 and an average P(Alive) of 10.1%"
        escaped = escape_markdown_dollar_signs(raw)
        assert "\\$716,276.02" in escaped
        # No bare, unescaped '$' should survive anywhere in the output.
        assert re.search(r"(?<!\\)\$", escaped) is None

    def test_reproduces_the_exact_screenshotted_failure_pattern_and_fixes_it(self):
        # Two dollar amounts (or, as here, a dollar amount plus the model's
        # own habit of wrapping "P(Alive)" in $ signs) form an accidental
        # matched pair that remark-math treats as one inline-math span -- the
        # exact mechanism behind the screenshotted bug. Confirms escaping
        # neutralizes it regardless of how many bare '$' characters are present.
        raw = "a total historical value of $716,276.02 and an average $P(Alive)$ of 10.1%"
        escaped = escape_markdown_dollar_signs(raw)
        assert re.search(r"(?<!\\)\$", escaped) is None
        assert escaped.count("\\$") == raw.count("$")

    def test_text_with_no_dollar_signs_is_unchanged(self):
        raw = "This account has 450 customers across 7 segments."
        assert escape_markdown_dollar_signs(raw) == raw

    def test_already_escaped_dollar_sign_is_not_double_escaped_incorrectly(self):
        # Not an expected real-world input (the model is instructed not to
        # produce LaTeX at all), but confirms the function is a pure,
        # predictable character-level transform rather than something that
        # could mangle already-backslashed text in a surprising way.
        raw = "literal backslash-dollar: \\$5"
        escaped = escape_markdown_dollar_signs(raw)
        # Every '$' (regardless of what precedes it) gets its own new
        # backslash prepended -- simple, unconditional, and idempotent-safe
        # in the sense that no unescaped '$' can ever remain.
        assert re.search(r"(?<!\\)\$", escaped) is None

    def test_escaped_text_survives_streamlits_own_markdown_element_construction(self):
        """
        Real regression test through Streamlit's own markdown element
        construction (streamlit.testing.v1.AppTest) -- not a mock -- confirming
        the escaped text reaches st.markdown() with its backslash-escapes
        intact end to end. Streamlit's Python-side clean_text() only dedents/
        strips (confirmed by reading streamlit/string_util.py), so this proves
        escaping survives that path unmodified.

        What this test CANNOT verify (no headless browser/JS engine available
        in this environment): that the browser's remark-math parser actually
        renders a backslash-escaped '$' as a literal character rather than a
        math delimiter. That half is standard CommonMark punctuation-escape
        behavior (confirmed by reading the CommonMark spec and locating
        remark-math's singleDollarTextMath option, defaulted true, in the
        installed Streamlit package's own static JS bundle -- see
        escape_markdown_dollar_signs()'s docstring), not re-verified by an
        actual browser here.
        """
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_function(_latex_bug_probe_script).run()
        rendered = at.markdown[0].value
        assert "\\$716,276.02" in rendered
        assert re.search(r"(?<!\\)\$", rendered) is None


def _latex_bug_probe_script():
    """Standalone Streamlit script for TestLatexRenderingBugFix's AppTest
    regression test -- must be a real, source-file-backed top-level function
    (streamlit.testing.v1.AppTest.from_function reads it via
    inspect.getsourcelines(), which fails on anything not backed by a real
    file, e.g. code executed via exec()/-c), so this lives at module level
    rather than nested inside the test method."""
    import streamlit as st

    from src.chat_engine import escape_markdown_dollar_signs

    raw = "Total historical value: $716,276.02 and an average P(Alive) of 10.1%"
    st.markdown(escape_markdown_dollar_signs(raw))


class TestFallbackPathNoKeys:
    def test_no_keys_returns_unavailable_message_without_any_network_call(self, context_text):
        history = []
        with patch("src.chat_engine.anthropic.Anthropic") as mock_anthropic_cls, \
             patch("src.chat_engine.groq.Groq") as mock_groq_cls:
            result = answer_account_question("What's my churn risk?", context_text, history)
            mock_anthropic_cls.assert_not_called()
            mock_groq_cls.assert_not_called()
        assert "temporarily unavailable" in result
        # A failed turn is not recorded -- see the function's docstring.
        assert history == []

    def test_unavailable_message_is_clearly_labeled_and_never_empty(self):
        result = _chat_unavailable("some reason")
        assert isinstance(result, str) and result
        assert "some reason" in result


class TestSuccessfulGroqCall:
    def test_successful_call_returns_model_text_and_records_the_turn(self, context_text):
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.return_value = _mock_groq_response(
            "Your churn watchlist has 3 accounts."
        )

        history = []
        with patch("src.chat_engine.groq.Groq", return_value=mock_client):
            result = answer_account_question(
                "What's my churn risk?", context_text, history, groq_api_key="fake-groq-key"
            )

        assert result == "Your churn watchlist has 3 accounts."
        assert "temporarily unavailable" not in result
        # Plain {"role", "content"} turns only -- no provider-specific
        # bookkeeping key (Gemini's interaction-id chaining, and the key that
        # went with it, is gone entirely -- see src/chat_engine.py's module
        # docstring "Provider swap" note).
        assert history == [
            {"role": "user", "content": "What's my churn risk?"},
            {"role": "assistant", "content": "Your churn watchlist has 3 accounts."},
        ]

    def test_call_uses_the_digest_engines_groq_model_and_chat_max_output_tokens(self, context_text):
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.return_value = _mock_groq_response()

        with patch("src.chat_engine.groq.Groq", return_value=mock_client):
            answer_account_question("A question?", context_text, [], groq_api_key="fake-groq-key")

        _, kwargs = mock_client.with_options.return_value.chat.completions.create.call_args
        assert kwargs["model"] == GROQ_MODEL_ID
        # The Groq call gets CHAT_MAX_OUTPUT_TOKENS PLUS reasoning headroom
        # (GROQ_CHAT_MAX_OUTPUT_TOKENS), not the bare visible-answer target --
        # see GROQ_REASONING_TOKEN_HEADROOM's comment in digest_engine.py.
        assert kwargs["max_tokens"] == GROQ_CHAT_MAX_OUTPUT_TOKENS
        # Last message is this turn's question.
        assert kwargs["messages"][-1] == {"role": "user", "content": "A question?"}

    def test_call_passes_the_timeout_via_with_options_not_a_per_call_kwarg(self, context_text):
        # Mirrors Anthropic's own with_options(timeout=...) mechanism exactly
        # (both SDKs support it) -- NOT Gemini's old per-call `timeout=` kwarg
        # pattern, which no longer applies anywhere in this codebase.
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.return_value = _mock_groq_response()

        with patch("src.chat_engine.groq.Groq", return_value=mock_client):
            answer_account_question("A question?", context_text, [], groq_api_key="fake-groq-key")

        mock_client.with_options.assert_called_once_with(timeout=GROQ_REQUEST_TIMEOUT_SECONDS)

    def test_system_prompt_carries_the_context_blob_as_the_first_message(self, context_text):
        # Groq's OpenAI-compatible API has no separate top-level `system`
        # field (unlike Anthropic's) -- the system prompt is the first entry
        # in `messages` instead.
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.return_value = _mock_groq_response()

        with patch("src.chat_engine.groq.Groq", return_value=mock_client):
            answer_account_question("A question?", context_text, [], groq_api_key="fake-groq-key")

        _, kwargs = mock_client.with_options.return_value.chat.completions.create.call_args
        first_message = kwargs["messages"][0]
        assert first_message["role"] == "system"
        assert context_text in first_message["content"]

    def test_empty_output_text_falls_back(self, context_text):
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.return_value = _mock_groq_response("   ")

        history = []
        with patch("src.chat_engine.groq.Groq", return_value=mock_client):
            result = answer_account_question("A question?", context_text, history, groq_api_key="fake-groq-key")

        assert "temporarily unavailable" in result
        assert "empty response from Groq" in result
        assert history == []  # failed turn not recorded


class TestGroqMultiTurnChaining:
    """
    Confirms conversation_history is genuinely threaded into the Groq call --
    the multi-turn requirement from the task spec, now served by Groq's
    plain, stateless messages-list pattern (system prompt + full history +
    new question, resent in full every turn) rather than Gemini's old
    stateful previous_interaction_id chaining, which is gone along with
    Gemini itself (see src/chat_engine.py's module docstring "Multi-turn
    conversation" section).
    """

    def test_second_turn_includes_the_full_conversation_history_in_messages(self, context_text):
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.side_effect = [
            _mock_groq_response("First answer."),
            _mock_groq_response("Second answer."),
        ]

        history = []
        with patch("src.chat_engine.groq.Groq", return_value=mock_client):
            answer_account_question("First question?", context_text, history, groq_api_key="fake-groq-key")
            answer_account_question("Second question?", context_text, history, groq_api_key="fake-groq-key")

        assert mock_client.with_options.return_value.chat.completions.create.call_count == 2
        _, second_call_kwargs = mock_client.with_options.return_value.chat.completions.create.call_args_list[1]
        # system + first Q + first A + second Q -- the model genuinely has
        # the prior turn's content available, not just a chained opaque id.
        assert second_call_kwargs["messages"][1:] == [
            {"role": "user", "content": "First question?"},
            {"role": "assistant", "content": "First answer."},
            {"role": "user", "content": "Second question?"},
        ]
        assert context_text in second_call_kwargs["messages"][0]["content"]
        # History now carries both turns, in order, as plain {"role", "content"} dicts.
        assert history == [
            {"role": "user", "content": "First question?"},
            {"role": "assistant", "content": "First answer."},
            {"role": "user", "content": "Second question?"},
            {"role": "assistant", "content": "Second answer."},
        ]


class TestGroqFailuresDegradeGracefully:
    def test_missing_key_falls_back_without_any_network_call(self, context_text):
        with patch("src.chat_engine.groq.Groq") as mock_client_cls:
            result = answer_account_question("A question?", context_text, [], groq_api_key=None)
            mock_client_cls.assert_not_called()
        assert "temporarily unavailable" in result

    def test_groq_package_missing_falls_back(self, context_text):
        with patch("src.chat_engine.groq", None):
            result = answer_account_question("A question?", context_text, [], groq_api_key="fake-groq-key")
        assert "temporarily unavailable" in result
        assert "not installed" in result

    def test_rate_limit_falls_back_with_specific_reason(self, context_text):
        exc = _make_groq_api_error(429, "quota exceeded")
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.side_effect = exc

        history = []
        with patch("src.chat_engine.groq.Groq", return_value=mock_client):
            result = answer_account_question("A question?", context_text, history, groq_api_key="fake-groq-key")

        assert "temporarily unavailable" in result
        assert "Groq free-tier rate limit reached" in result
        assert history == []

    def test_invalid_key_falls_back_with_specific_reason(self, context_text):
        # Standard HTTP 401 for bad credentials -- no Gemini-style body-shape
        # sniffing needed here (see digest_engine.py's equivalent test).
        exc = _make_groq_api_error(401, "bad credentials")
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.side_effect = exc

        with patch("src.chat_engine.groq.Groq", return_value=mock_client):
            result = answer_account_question("A question?", context_text, [], groq_api_key="bad-key")

        assert "temporarily unavailable" in result
        assert "invalid Groq API key" in result

    def test_generic_client_error_falls_back_with_generic_reason(self, context_text):
        exc = _make_groq_api_error(404, "not found")
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.side_effect = exc

        with patch("src.chat_engine.groq.Groq", return_value=mock_client):
            result = answer_account_question("A question?", context_text, [], groq_api_key="fake-groq-key")

        assert "temporarily unavailable" in result
        assert "Groq API error" in result

    def test_timeout_falls_back_with_its_own_distinct_reason_never_hangs_the_test(self, context_text):
        # A hung client.chat.completions.create() call must never hang the
        # caller. Mocking a raised APITimeoutError (rather than actually
        # sleeping) proves the except clause routes to its own distinct
        # reason -- if this test itself takes anywhere near
        # GROQ_REQUEST_TIMEOUT_SECONDS to run, something is badly wrong.
        exc = groq.APITimeoutError(request=httpx.Request(
            "POST", "https://api.groq.com/openai/v1/chat/completions"
        ))
        assert isinstance(exc, groq.APIError)
        assert not isinstance(exc, groq.APIStatusError)
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.side_effect = exc

        history = []
        with patch("src.chat_engine.groq.Groq", return_value=mock_client):
            result = answer_account_question("A question?", context_text, history, groq_api_key="fake-groq-key")

        assert "temporarily unavailable" in result
        assert "Groq API request timed out" in result
        assert "rate limit" not in result
        assert "invalid Groq API key" not in result
        assert result.count("Groq API error") == 0
        assert history == []  # failed turn not recorded

    def test_connection_error_is_caught_by_the_broad_apierror_base_class(self, context_text):
        # APIConnectionError has no HTTP response/status_code (network-level
        # failure) -- NOT an APIStatusError subclass, so this specifically
        # exercises the broader `except groq.APIConnectionError` branch,
        # mirroring tests/test_digest_engine.py's equivalent test exactly.
        exc = groq.APIConnectionError(message="connection failed", request=httpx.Request(
            "POST", "https://api.groq.com/openai/v1/chat/completions"
        ))
        assert isinstance(exc, groq.APIError)
        assert not isinstance(exc, groq.APIStatusError)
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.side_effect = exc

        with patch("src.chat_engine.groq.Groq", return_value=mock_client):
            result = answer_account_question("A question?", context_text, [], groq_api_key="fake-groq-key")

        assert "temporarily unavailable" in result
        assert "could not reach the Groq API" in result
        assert "unexpected error" not in result

    def test_generic_exception_falls_back_instead_of_raising(self, context_text):
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.side_effect = RuntimeError("boom")

        history = []
        with patch("src.chat_engine.groq.Groq", return_value=mock_client):
            result = answer_account_question("A question?", context_text, history, groq_api_key="fake-groq-key")

        assert "temporarily unavailable" in result
        assert "unexpected error" in result
        assert history == []


class TestSuccessfulAnthropicCall:
    def _mock_response(self, text="Your top segment is Champions."):
        block = types.SimpleNamespace(type="text", text=text)
        return types.SimpleNamespace(content=[block])

    def test_successful_call_returns_model_text_and_records_the_turn(self, context_text):
        mock_client = MagicMock()
        mock_client.with_options.return_value.messages.create.return_value = self._mock_response()

        history = []
        with patch("src.chat_engine.anthropic.Anthropic", return_value=mock_client):
            result = answer_account_question(
                "What's my top segment?", context_text, history, anthropic_api_key="sk-ant-fake-key"
            )

        assert result == "Your top segment is Champions."
        assert history == [
            {"role": "user", "content": "What's my top segment?"},
            {"role": "assistant", "content": "Your top segment is Champions."},
        ]

    def test_call_uses_the_digest_engines_anthropic_model_and_chat_max_output_tokens(self, context_text):
        mock_client = MagicMock()
        mock_client.with_options.return_value.messages.create.return_value = self._mock_response()

        with patch("src.chat_engine.anthropic.Anthropic", return_value=mock_client):
            answer_account_question("A question?", context_text, [], anthropic_api_key="sk-ant-fake-key")

        _, kwargs = mock_client.with_options.return_value.messages.create.call_args
        assert kwargs["model"] == MODEL_ID
        assert kwargs["max_tokens"] == CHAT_MAX_OUTPUT_TOKENS
        assert context_text in kwargs["system"]

    def test_prior_turns_are_passed_through_as_the_messages_list_not_dropped(self, context_text):
        mock_client = MagicMock()
        mock_client.with_options.return_value.messages.create.return_value = self._mock_response(text="Second answer.")

        history = [
            {"role": "user", "content": "First question?"},
            {"role": "assistant", "content": "First answer."},
        ]
        with patch("src.chat_engine.anthropic.Anthropic", return_value=mock_client):
            answer_account_question("Second question?", context_text, history, anthropic_api_key="sk-ant-fake-key")

        _, kwargs = mock_client.with_options.return_value.messages.create.call_args
        assert kwargs["messages"] == [
            {"role": "user", "content": "First question?"},
            {"role": "assistant", "content": "First answer."},
            {"role": "user", "content": "Second question?"},
        ]

    def test_empty_model_response_falls_back(self, context_text):
        mock_client = MagicMock()
        mock_client.with_options.return_value.messages.create.return_value = self._mock_response(text="   ")

        history = []
        with patch("src.chat_engine.anthropic.Anthropic", return_value=mock_client):
            result = answer_account_question("A question?", context_text, history, anthropic_api_key="sk-ant-fake-key")

        assert "temporarily unavailable" in result
        assert history == []


class TestAnthropicFailuresDegradeGracefully:
    def test_anthropic_package_missing_falls_back(self, context_text):
        with patch("src.chat_engine.anthropic", None):
            result = answer_account_question("A question?", context_text, [], anthropic_api_key="sk-ant-fake-key")
        assert "temporarily unavailable" in result
        assert "not installed" in result

    @pytest.mark.parametrize("exception_name,expected_reason_snippet", [
        ("AuthenticationError", "invalid Anthropic API key"),
        ("RateLimitError", "rate limited"),
        ("APIStatusError", "Anthropic API error"),
        ("APIConnectionError", "could not reach"),
    ])
    def test_each_typed_anthropic_exception_falls_back_with_its_own_reason(
        self, context_text, exception_name, expected_reason_snippet
    ):
        import anthropic as real_anthropic

        exc_cls = getattr(real_anthropic, exception_name)
        exc_instance = exc_cls.__new__(exc_cls)

        mock_client = MagicMock()
        mock_client.with_options.return_value.messages.create.side_effect = exc_instance

        with patch("src.chat_engine.anthropic.Anthropic", return_value=mock_client):
            result = answer_account_question("A question?", context_text, [], anthropic_api_key="sk-ant-fake-key")

        assert "temporarily unavailable" in result
        assert expected_reason_snippet in result

    def test_generic_exception_falls_back_instead_of_raising(self, context_text):
        mock_client = MagicMock()
        mock_client.with_options.return_value.messages.create.side_effect = RuntimeError("boom")

        with patch("src.chat_engine.anthropic.Anthropic", return_value=mock_client):
            result = answer_account_question("A question?", context_text, [], anthropic_api_key="sk-ant-fake-key")

        assert "temporarily unavailable" in result


class TestTargetConfiguration:
    def test_groq_only_configuration_calls_groq_not_anthropic(self, context_text):
        mock_groq_client = MagicMock()
        mock_groq_client.with_options.return_value.chat.completions.create.return_value = _mock_groq_response(
            "Groq-generated answer."
        )

        with patch("src.chat_engine.groq.Groq", return_value=mock_groq_client) as mock_groq_cls, \
             patch("src.chat_engine.anthropic.Anthropic") as mock_anthropic_cls:
            result = answer_account_question(
                "A question?", context_text, [],
                anthropic_api_key=None, groq_api_key="fake-groq-key",
            )

        assert result == "Groq-generated answer."
        mock_groq_cls.assert_called_once()
        mock_anthropic_cls.assert_not_called()

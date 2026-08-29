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

import types
from unittest.mock import MagicMock, patch

import httpx
import pytest
from google.genai._gaos.lib import compat_errors as genai_errors

from src.chat_context import build_account_context_blob, build_context_text
from src.chat_engine import (
    answer_account_question,
    CHAT_MAX_OUTPUT_TOKENS,
    CHAT_SYSTEM_PROMPT_TEMPLATE,
    _chat_unavailable,
    _last_gemini_interaction_id,
    _anthropic_messages_from_history,
)
from src.digest_engine import MODEL_ID, GEMINI_MODEL_ID


def _make_gemini_api_error(status_code: int, message: str, reason: str = "", body: dict = None):
    """Identical helper to tests/test_digest_engine.py's -- builds a real
    genai_errors.APIStatusError subclass via the SDK's own status-code-to-class
    factory. See that file's version for the full rationale; duplicated here
    (not imported) since test files in this repo are self-contained."""
    request = httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/interactions")
    response = httpx.Response(status_code=status_code, request=request)
    if body is None:
        body = {"error": {"code": status_code, "message": message, "status": reason}}
    return genai_errors.APIError.generate(
        status_code=status_code,
        body=body,
        message=f"Error code: {status_code} - [{body}]",
        response=response,
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
        assert chat_engine_module.GEMINI_MODEL_ID is GEMINI_MODEL_ID
        assert chat_engine_module.MODEL_ID is MODEL_ID


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


class TestFallbackPathNoKeys:
    def test_no_keys_returns_unavailable_message_without_any_network_call(self, context_text):
        history = []
        with patch("src.chat_engine.anthropic.Anthropic") as mock_anthropic_cls, \
             patch("src.chat_engine.genai.Client") as mock_genai_cls:
            result = answer_account_question("What's my churn risk?", context_text, history)
            mock_anthropic_cls.assert_not_called()
            mock_genai_cls.assert_not_called()
        assert "temporarily unavailable" in result
        # A failed turn is not recorded -- see the function's docstring.
        assert history == []

    def test_unavailable_message_is_clearly_labeled_and_never_empty(self):
        result = _chat_unavailable("some reason")
        assert isinstance(result, str) and result
        assert "some reason" in result


class TestSuccessfulGeminiCall:
    def test_successful_call_returns_model_text_and_records_the_turn(self, context_text):
        mock_interaction = types.SimpleNamespace(output_text="Your churn watchlist has 3 accounts.", id="interaction-1")
        mock_client = MagicMock()
        mock_client.interactions.create.return_value = mock_interaction

        history = []
        with patch("src.chat_engine.genai.Client", return_value=mock_client):
            result = answer_account_question(
                "What's my churn risk?", context_text, history, gemini_api_key="fake-gemini-key"
            )

        assert result == "Your churn watchlist has 3 accounts."
        assert "temporarily unavailable" not in result
        assert history == [
            {"role": "user", "content": "What's my churn risk?"},
            {"role": "assistant", "content": "Your churn watchlist has 3 accounts.", "gemini_interaction_id": "interaction-1"},
        ]

    def test_call_uses_the_digest_engines_gemini_model_and_chat_max_output_tokens(self, context_text):
        mock_interaction = types.SimpleNamespace(output_text="An answer.", id="interaction-2")
        mock_client = MagicMock()
        mock_client.interactions.create.return_value = mock_interaction

        with patch("src.chat_engine.genai.Client", return_value=mock_client):
            answer_account_question("A question?", context_text, [], gemini_api_key="fake-gemini-key")

        _, kwargs = mock_client.interactions.create.call_args
        assert kwargs["model"] == GEMINI_MODEL_ID
        assert kwargs["generation_config"]["max_output_tokens"] == CHAT_MAX_OUTPUT_TOKENS
        assert kwargs["input"] == "A question?"
        assert "previous_interaction_id" not in kwargs  # first turn -- no prior interaction to chain from

    def test_system_instruction_carries_the_context_blob(self, context_text):
        mock_interaction = types.SimpleNamespace(output_text="An answer.", id="interaction-3")
        mock_client = MagicMock()
        mock_client.interactions.create.return_value = mock_interaction

        with patch("src.chat_engine.genai.Client", return_value=mock_client):
            answer_account_question("A question?", context_text, [], gemini_api_key="fake-gemini-key")

        _, kwargs = mock_client.interactions.create.call_args
        assert context_text in kwargs["system_instruction"]

    def test_empty_output_text_falls_back(self, context_text):
        mock_interaction = types.SimpleNamespace(output_text="   ", id="interaction-4")
        mock_client = MagicMock()
        mock_client.interactions.create.return_value = mock_interaction

        history = []
        with patch("src.chat_engine.genai.Client", return_value=mock_client):
            result = answer_account_question("A question?", context_text, history, gemini_api_key="fake-gemini-key")

        assert "temporarily unavailable" in result
        assert "empty response from Gemini" in result
        assert history == []  # failed turn not recorded


class TestGeminiMultiTurnChaining:
    """Confirms conversation_history is actually passed through to the provider
    call (via previous_interaction_id), not dropped -- the multi-turn
    requirement from the task spec."""

    def test_second_turn_passes_the_first_turns_interaction_id_as_previous_interaction_id(self, context_text):
        mock_client = MagicMock()
        mock_client.interactions.create.side_effect = [
            types.SimpleNamespace(output_text="First answer.", id="turn-1-id"),
            types.SimpleNamespace(output_text="Second answer.", id="turn-2-id"),
        ]

        history = []
        with patch("src.chat_engine.genai.Client", return_value=mock_client):
            answer_account_question("First question?", context_text, history, gemini_api_key="fake-gemini-key")
            answer_account_question("Second question?", context_text, history, gemini_api_key="fake-gemini-key")

        assert mock_client.interactions.create.call_count == 2
        _, second_call_kwargs = mock_client.interactions.create.call_args_list[1]
        assert second_call_kwargs["previous_interaction_id"] == "turn-1-id"
        # system_instruction is re-sent on every turn (see module docstring),
        # not just the first.
        assert context_text in second_call_kwargs["system_instruction"]
        # History now carries both turns, in order.
        assert [t["content"] for t in history] == ["First question?", "First answer.", "Second question?", "Second answer."]

    def test_last_gemini_interaction_id_helper_scans_backwards_for_the_most_recent(self):
        history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "gemini_interaction_id": "old-id"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2", "gemini_interaction_id": "newer-id"},
        ]
        assert _last_gemini_interaction_id(history) == "newer-id"

    def test_last_gemini_interaction_id_helper_returns_none_for_empty_or_anthropic_only_history(self):
        assert _last_gemini_interaction_id([]) is None
        anthropic_only_history = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
        assert _last_gemini_interaction_id(anthropic_only_history) is None


class TestGeminiFailuresDegradeGracefully:
    def test_missing_key_falls_back_without_any_network_call(self, context_text):
        with patch("src.chat_engine.genai.Client") as mock_client_cls:
            result = answer_account_question("A question?", context_text, [], gemini_api_key=None)
            mock_client_cls.assert_not_called()
        assert "temporarily unavailable" in result

    def test_genai_package_missing_falls_back(self, context_text):
        with patch("src.chat_engine.genai", None):
            result = answer_account_question("A question?", context_text, [], gemini_api_key="fake-gemini-key")
        assert "temporarily unavailable" in result
        assert "not installed" in result

    def test_rate_limit_falls_back_with_specific_reason(self, context_text):
        exc = _make_gemini_api_error(429, "quota exceeded", reason="RESOURCE_EXHAUSTED")
        mock_client = MagicMock()
        mock_client.interactions.create.side_effect = exc

        history = []
        with patch("src.chat_engine.genai.Client", return_value=mock_client):
            result = answer_account_question("A question?", context_text, history, gemini_api_key="fake-gemini-key")

        assert "temporarily unavailable" in result
        assert "Gemini free-tier rate limit reached" in result
        assert history == []

    def test_invalid_key_falls_back_with_specific_reason(self, context_text):
        real_invalid_key_body = {
            "error": {
                "code": 400,
                "message": "API key not valid. Please pass a valid API key.",
                "status": "INVALID_ARGUMENT",
                "details": [{"reason": "API_KEY_INVALID"}],
            }
        }
        exc = _make_gemini_api_error(400, "API key not valid.", body=real_invalid_key_body)
        mock_client = MagicMock()
        mock_client.interactions.create.side_effect = exc

        with patch("src.chat_engine.genai.Client", return_value=mock_client):
            result = answer_account_question("A question?", context_text, [], gemini_api_key="bad-key")

        assert "temporarily unavailable" in result
        assert "invalid Gemini API key" in result

    def test_generic_client_error_falls_back_with_generic_reason(self, context_text):
        exc = _make_gemini_api_error(404, "not found")
        mock_client = MagicMock()
        mock_client.interactions.create.side_effect = exc

        with patch("src.chat_engine.genai.Client", return_value=mock_client):
            result = answer_account_question("A question?", context_text, [], gemini_api_key="fake-gemini-key")

        assert "temporarily unavailable" in result
        assert "Gemini API error" in result

    def test_connection_error_is_caught_by_the_broad_apierror_base_class(self, context_text):
        # APIConnectionError has no HTTP response/status_code (network-level
        # failure) -- NOT an APIStatusError subclass, so this specifically
        # exercises the broader `except genai_errors.APIError` branch, mirroring
        # tests/test_digest_engine.py's equivalent test exactly.
        exc = genai_errors.APIConnectionError(message="connection failed", request=httpx.Request(
            "POST", "https://generativelanguage.googleapis.com/v1beta/interactions"
        ))
        assert isinstance(exc, genai_errors.APIError)
        assert not isinstance(exc, genai_errors.APIStatusError)
        mock_client = MagicMock()
        mock_client.interactions.create.side_effect = exc

        with patch("src.chat_engine.genai.Client", return_value=mock_client):
            result = answer_account_question("A question?", context_text, [], gemini_api_key="fake-gemini-key")

        assert "temporarily unavailable" in result
        assert "Gemini API error" in result
        assert "unexpected error" not in result

    def test_generic_exception_falls_back_instead_of_raising(self, context_text):
        mock_client = MagicMock()
        mock_client.interactions.create.side_effect = RuntimeError("boom")

        history = []
        with patch("src.chat_engine.genai.Client", return_value=mock_client):
            result = answer_account_question("A question?", context_text, history, gemini_api_key="fake-gemini-key")

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

    def test_gemini_bookkeeping_key_is_stripped_from_anthropic_messages(self, context_text):
        # If a conversation started on Gemini and then fell back to Anthropic
        # mid-session, history turns may carry a gemini_interaction_id key --
        # Anthropic's messages list must never see it.
        history_with_gemini_key = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "gemini_interaction_id": "some-id"},
        ]
        messages = _anthropic_messages_from_history(history_with_gemini_key)
        assert messages == [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]

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
    def test_gemini_only_configuration_calls_gemini_not_anthropic(self, context_text):
        mock_interaction = types.SimpleNamespace(output_text="Gemini-generated answer.", id="interaction-x")
        mock_gemini_client = MagicMock()
        mock_gemini_client.interactions.create.return_value = mock_interaction

        with patch("src.chat_engine.genai.Client", return_value=mock_gemini_client) as mock_genai_cls, \
             patch("src.chat_engine.anthropic.Anthropic") as mock_anthropic_cls:
            result = answer_account_question(
                "A question?", context_text, [],
                anthropic_api_key=None, gemini_api_key="fake-gemini-key",
            )

        assert result == "Gemini-generated answer."
        mock_genai_cls.assert_called_once()
        mock_anthropic_cls.assert_not_called()

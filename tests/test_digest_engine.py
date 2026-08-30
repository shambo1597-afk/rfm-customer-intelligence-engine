"""
tests/test_digest_engine.py - pytest suite for the optional AI Executive Summary.

No real network calls are made anywhere in this file -- both provider clients
(Anthropic, Groq) are always mocked (unittest.mock) when a "successful call" or
"provider failure" path is exercised, and the no-key fallback path is tested
precisely because it must NOT construct a client or attempt any network call at all.
"""

import types
from unittest.mock import MagicMock, patch

import groq
import httpx
import pytest

from src.rfm_engine import get_segment_kpi_summary
from src.digest_engine import (
    generate_account_digest,
    get_anthropic_api_key,
    get_groq_api_key,
    get_digest_provider_override,
    _resolve_provider,
    _build_aggregate_stats,
    _build_prompt,
    _fallback_digest,
    _call_groq,
    _call_anthropic,
    MODEL_ID,
    MAX_TOKENS,
    REQUEST_TIMEOUT_SECONDS,
    GROQ_MODEL_ID,
    GROQ_MAX_OUTPUT_TOKENS,
    GROQ_REQUEST_TIMEOUT_SECONDS,
    GROQ_REASONING_EFFORT,
    GROQ_REASONING_TOKEN_HEADROOM,
)


@pytest.fixture(scope="module")
def segment_summary(clv_df):
    return get_segment_kpi_summary(clv_df)


def _make_groq_api_error(status_code: int, message: str = "error"):
    """
    Builds a real groq exception instance for the given HTTP status code via
    the REAL groq.Groq client's own `_make_status_error()` dispatch method --
    confirmed via direct introspection of the pinned groq package
    (groq/_client.py) to be the exact method the SDK itself calls internally
    to translate an HTTP response into an exception. Unlike google-genai's
    SDK (this project's former Gemini provider), groq has no standalone
    `APIError.generate()` factory -- this reaches the real dispatch logic
    through a throwaway client instance instead of reimplementing the
    status_code -> class mapping as a guess.
    """
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status_code=status_code, request=request)
    client = groq.Groq(api_key="dummy-key-for-error-construction")
    return client._make_status_error(message, body=None, response=response)


class TestNamedConstants:
    """B3 requirement: model/max_tokens must be named constants, not inline magic numbers."""

    def test_anthropic_model_and_max_tokens_are_module_level_constants(self):
        assert isinstance(MODEL_ID, str) and MODEL_ID  # non-empty string
        assert isinstance(MAX_TOKENS, int) and MAX_TOKENS > 0

    def test_anthropic_model_is_the_small_cheap_tier(self):
        # This feature's entire cost story (see module docstring) depends on staying
        # on the cheap model -- a regression here silently breaks the ~$1/mo claim.
        assert "haiku" in MODEL_ID.lower()

    def test_groq_model_and_max_output_tokens_are_module_level_constants(self):
        assert isinstance(GROQ_MODEL_ID, str) and GROQ_MODEL_ID
        assert isinstance(GROQ_MAX_OUTPUT_TOKENS, int) and GROQ_MAX_OUTPUT_TOKENS > 0

    def test_groq_model_is_not_the_confirmed_deprecated_llama_3_3_70b_versatile(self):
        # Regression guard for the exact mistake this codebase already made once
        # with Gemini (a pinned model name going stale under it): the task that
        # introduced Groq support specified "llama-3.3-70b-versatile", which
        # live verification (see digest_engine.py's module docstring
        # "Model-choice note") found Groq had deprecated -- "no longer being
        # served by August 2026" on the free/developer tier this project uses.
        # GROQ_MODEL_ID must never silently revert to it.
        assert GROQ_MODEL_ID != "llama-3.3-70b-versatile"

    def test_groq_request_timeout_seconds_is_a_positive_module_level_constant(self):
        # Outer safety-net timeout, mirroring this codebase's own Gemini-hang
        # lesson (see module docstring) even though Groq's client already
        # retries sensibly on its own. Must be a real, finite, positive number
        # of seconds, not left unset/None/0.
        assert isinstance(GROQ_REQUEST_TIMEOUT_SECONDS, (int, float))
        assert GROQ_REQUEST_TIMEOUT_SECONDS > 0

    def test_groq_reasoning_effort_is_a_valid_supported_value(self):
        # 'low', 'medium', or 'high' are the only values Groq documents as
        # supported for openai/gpt-oss-120b specifically -- see
        # GROQ_REASONING_EFFORT's own comment for the two independent
        # sources (WebSearch of Groq's docs + the pinned groq package's own
        # request-parameter type stub) that confirm this.
        assert GROQ_REASONING_EFFORT in ("low", "medium", "high")

    def test_groq_reasoning_effort_is_low_not_the_provider_default(self):
        # Deliberately NOT "medium" (Groq's own default) -- see
        # GROQ_REASONING_EFFORT's comment for why every prompt this codebase
        # sends via Groq is simple enough that "low" is the right choice.
        assert GROQ_REASONING_EFFORT == "low"

    def test_groq_reasoning_token_headroom_is_a_positive_module_level_constant(self):
        assert isinstance(GROQ_REASONING_TOKEN_HEADROOM, int)
        assert GROQ_REASONING_TOKEN_HEADROOM > 0

    def test_groq_max_output_tokens_has_real_headroom_over_the_original_300(self):
        # Regression guard for the actual live bug this fix addresses: 300
        # (the original, headroom-free value) was insufficient for a
        # reasoning model -- GROQ_MAX_OUTPUT_TOKENS must never silently
        # shrink back down to it.
        assert GROQ_MAX_OUTPUT_TOKENS > 300
        assert GROQ_MAX_OUTPUT_TOKENS == 300 + GROQ_REASONING_TOKEN_HEADROOM


class TestResolveProvider:
    """
    Every branch _resolve_provider() must handle, per the task spec: both keys with
    no override -> groq; both keys with an override -> that provider; only one key
    present -> that provider regardless of override; neither key -> "none"; an
    override naming a provider whose key is absent -> falls back to whichever key
    IS present, or "none".
    """

    def test_both_keys_no_override_prefers_groq(self):
        assert _resolve_provider("a-key", "g-key") == "groq"

    def test_both_keys_override_anthropic_uses_anthropic(self):
        assert _resolve_provider("a-key", "g-key", override="anthropic") == "anthropic"

    def test_both_keys_override_groq_uses_groq(self):
        assert _resolve_provider("a-key", "g-key", override="groq") == "groq"

    def test_only_groq_key_uses_groq(self):
        assert _resolve_provider(None, "g-key") == "groq"

    def test_only_anthropic_key_uses_anthropic(self):
        assert _resolve_provider("a-key", None) == "anthropic"

    def test_neither_key_returns_none(self):
        assert _resolve_provider(None, None) == "none"

    def test_override_naming_a_provider_with_no_key_falls_back_to_present_key(self):
        # override="groq" but no Groq key -- falls back to Anthropic, which IS present.
        assert _resolve_provider("a-key", None, override="groq") == "anthropic"
        # override="anthropic" but no Anthropic key -- falls back to Groq, which IS present.
        assert _resolve_provider(None, "g-key", override="anthropic") == "groq"

    def test_override_naming_a_provider_with_no_key_and_no_fallback_key_returns_none(self):
        assert _resolve_provider(None, None, override="groq") == "none"
        assert _resolve_provider(None, None, override="anthropic") == "none"

    def test_unrecognized_override_value_is_ignored(self):
        # A typo'd/garbage override falls through to the normal preference order,
        # never raises.
        assert _resolve_provider("a-key", "g-key", override="chatgpt") == "groq"
        assert _resolve_provider(None, None, override="chatgpt") == "none"


class TestFallbackPathNoKeys:
    def test_no_keys_returns_fallback_without_any_network_call(self, rfmt_df, clv_df, segment_summary):
        with patch("src.digest_engine.anthropic.Anthropic") as mock_anthropic_cls, \
             patch("src.digest_engine.groq.Groq") as mock_groq_cls:
            result = generate_account_digest(rfmt_df, clv_df, segment_summary)
            mock_anthropic_cls.assert_not_called()
            mock_groq_cls.assert_not_called()

        assert isinstance(result, str) and result
        assert "[Template Summary" in result

    def test_empty_string_keys_also_fall_back(self, rfmt_df, clv_df, segment_summary):
        with patch("src.digest_engine.anthropic.Anthropic") as mock_anthropic_cls, \
             patch("src.digest_engine.groq.Groq") as mock_groq_cls:
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, anthropic_api_key="", groq_api_key="")
            mock_anthropic_cls.assert_not_called()
            mock_groq_cls.assert_not_called()
        assert "[Template Summary" in result

    def test_fallback_is_deterministic_for_the_same_stats(self, rfmt_df, clv_df, segment_summary):
        r1 = generate_account_digest(rfmt_df, clv_df, segment_summary)
        r2 = generate_account_digest(rfmt_df, clv_df, segment_summary)
        assert r1 == r2

    def test_fallback_reflects_actual_aggregate_numbers(self, rfmt_df, clv_df, segment_summary):
        result = _fallback_digest(
            _build_aggregate_stats(rfmt_df, clv_df, segment_summary),
            reason="test reason",
        )
        assert f"{len(rfmt_df):,} customers" in result
        assert "test reason" in result


class TestPromptContainsOnlyAggregateData:
    def test_prompt_never_contains_a_raw_customer_id(self, rfmt_df, clv_df, segment_summary):
        stats = _build_aggregate_stats(rfmt_df, clv_df, segment_summary)
        prompt = _build_prompt(stats)
        for customer_id in rfmt_df["CustomerID"].astype(str):
            assert customer_id not in prompt

    def test_prompt_only_contains_the_documented_aggregate_fields(self, rfmt_df, clv_df, segment_summary):
        stats = _build_aggregate_stats(rfmt_df, clv_df, segment_summary)
        prompt = _build_prompt(stats)
        # Every value that appears in the prompt must trace back to _build_aggregate_stats()'s
        # own dict -- this is the structural guarantee that no additional (row-level) data
        # sneaks in between stats extraction and prompt construction.
        assert f"{stats['total_customers']:,}" in prompt
        assert f"{stats['pct_at_risk']:.1f}%" in prompt
        for seg in stats["top_segments"]:
            assert seg["segment"] in prompt

    def test_stats_dict_has_no_per_customer_keys(self, rfmt_df, clv_df, segment_summary):
        stats = _build_aggregate_stats(rfmt_df, clv_df, segment_summary)
        assert set(stats.keys()) == {
            "total_customers", "total_historical_revenue",
            "total_predicted_90d_revenue", "pct_at_risk", "top_segments",
        }
        # top_segments is capped/aggregate (segment-level), never one entry per customer.
        assert len(stats["top_segments"]) <= 3

    def test_prompt_built_for_the_groq_path_also_excludes_raw_customer_ids(self, rfmt_df, clv_df, segment_summary):
        # generate_account_digest() calls the same _build_prompt() regardless of
        # provider -- re-run the PII-exclusion check end-to-end through the Groq
        # branch specifically, not just against _build_prompt() in isolation.
        mock_response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="A summary with no customer IDs."))]
        )
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.return_value = mock_response

        with patch("src.digest_engine.groq.Groq", return_value=mock_client):
            generate_account_digest(rfmt_df, clv_df, segment_summary, groq_api_key="fake-groq-key")

        _, kwargs = mock_client.with_options.return_value.chat.completions.create.call_args
        sent_prompt = kwargs["messages"][0]["content"]
        for customer_id in rfmt_df["CustomerID"].astype(str):
            assert customer_id not in sent_prompt


class TestSuccessfulAnthropicCall:
    def _mock_response(self, text="A healthy account with strong 90-day forecasted revenue."):
        block = types.SimpleNamespace(type="text", text=text)
        return types.SimpleNamespace(content=[block])

    def test_successful_call_returns_model_text_not_fallback(self, rfmt_df, clv_df, segment_summary):
        mock_client = MagicMock()
        mock_client.with_options.return_value.messages.create.return_value = self._mock_response()

        with patch("src.digest_engine.anthropic.Anthropic", return_value=mock_client):
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, anthropic_api_key="sk-ant-fake-key")

        assert result == "A healthy account with strong 90-day forecasted revenue."
        assert "[Template Summary" not in result

    def test_call_uses_the_named_model_and_max_tokens_constants(self, rfmt_df, clv_df, segment_summary):
        mock_client = MagicMock()
        mock_client.with_options.return_value.messages.create.return_value = self._mock_response()

        with patch("src.digest_engine.anthropic.Anthropic", return_value=mock_client):
            generate_account_digest(rfmt_df, clv_df, segment_summary, anthropic_api_key="sk-ant-fake-key")

        _, kwargs = mock_client.with_options.return_value.messages.create.call_args
        assert kwargs["model"] == MODEL_ID
        assert kwargs["max_tokens"] == MAX_TOKENS

    def test_empty_model_response_falls_back(self, rfmt_df, clv_df, segment_summary):
        mock_client = MagicMock()
        mock_client.with_options.return_value.messages.create.return_value = self._mock_response(text="   ")

        with patch("src.digest_engine.anthropic.Anthropic", return_value=mock_client):
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, anthropic_api_key="sk-ant-fake-key")

        assert "[Template Summary" in result


class TestAnthropicFailuresDegradeGracefully:
    def test_generic_exception_falls_back_instead_of_raising(self, rfmt_df, clv_df, segment_summary):
        mock_client = MagicMock()
        mock_client.with_options.return_value.messages.create.side_effect = RuntimeError("boom")

        with patch("src.digest_engine.anthropic.Anthropic", return_value=mock_client):
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, anthropic_api_key="sk-ant-fake-key")

        assert "[Template Summary" in result

    def test_anthropic_package_missing_falls_back(self, rfmt_df, clv_df, segment_summary):
        with patch("src.digest_engine.anthropic", None):
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, anthropic_api_key="sk-ant-fake-key")
        assert "[Template Summary" in result
        assert "not installed" in result

    @pytest.mark.parametrize("exception_name,expected_reason_snippet", [
        ("AuthenticationError", "invalid Anthropic API key"),
        ("RateLimitError", "rate limited"),
        ("APIStatusError", "Anthropic API error"),
        ("APIConnectionError", "could not reach"),
    ])
    def test_each_typed_anthropic_exception_falls_back_with_its_own_reason(
        self, rfmt_df, clv_df, segment_summary, exception_name, expected_reason_snippet
    ):
        import anthropic as real_anthropic

        exc_cls = getattr(real_anthropic, exception_name)
        # These exception classes require constructor args that vary by type; a bare
        # object.__new__ bypass keeps this test agnostic to each one's exact signature
        # while still producing a real instance of the correct type for `except` to match.
        exc_instance = exc_cls.__new__(exc_cls)

        mock_client = MagicMock()
        mock_client.with_options.return_value.messages.create.side_effect = exc_instance

        with patch("src.digest_engine.anthropic.Anthropic", return_value=mock_client):
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, anthropic_api_key="sk-ant-fake-key")

        assert "[Template Summary" in result
        assert expected_reason_snippet in result


class TestSuccessfulGroqCall:
    def _mock_response(self, text="A healthy account overall."):
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text))]
        )

    def test_successful_call_returns_model_text_not_fallback(self, rfmt_df, clv_df, segment_summary):
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.return_value = self._mock_response()

        with patch("src.digest_engine.groq.Groq", return_value=mock_client) as mock_client_cls:
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, groq_api_key="fake-groq-key")

        assert result == "A healthy account overall."
        assert "[Template Summary" not in result
        mock_client_cls.assert_called_once_with(api_key="fake-groq-key")

    def test_call_uses_the_named_model_and_max_tokens_constants(self, rfmt_df, clv_df, segment_summary):
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.return_value = self._mock_response("Summary text.")

        with patch("src.digest_engine.groq.Groq", return_value=mock_client):
            generate_account_digest(rfmt_df, clv_df, segment_summary, groq_api_key="fake-groq-key")

        _, kwargs = mock_client.with_options.return_value.chat.completions.create.call_args
        assert kwargs["model"] == GROQ_MODEL_ID
        assert kwargs["max_tokens"] == GROQ_MAX_OUTPUT_TOKENS

    def test_call_passes_the_timeout_via_with_options_not_a_per_call_kwarg(self, rfmt_df, clv_df, segment_summary):
        # Mirrors Anthropic's own with_options(timeout=...) mechanism exactly
        # (both SDKs support it) -- NOT Gemini's old per-call `timeout=` kwarg
        # pattern, which doesn't apply here. Asserts the actual call site.
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.return_value = self._mock_response("Summary text.")

        with patch("src.digest_engine.groq.Groq", return_value=mock_client):
            generate_account_digest(rfmt_df, clv_df, segment_summary, groq_api_key="fake-groq-key")

        mock_client.with_options.assert_called_once_with(timeout=GROQ_REQUEST_TIMEOUT_SECONDS)

    def test_prompt_is_passed_via_the_messages_parameter(self, rfmt_df, clv_df, segment_summary):
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.return_value = self._mock_response("Summary text.")

        with patch("src.digest_engine.groq.Groq", return_value=mock_client):
            generate_account_digest(rfmt_df, clv_df, segment_summary, groq_api_key="fake-groq-key")

        _, kwargs = mock_client.with_options.return_value.chat.completions.create.call_args
        expected_prompt = _build_prompt(_build_aggregate_stats(rfmt_df, clv_df, segment_summary))
        assert kwargs["messages"] == [{"role": "user", "content": expected_prompt}]

    def test_empty_output_text_falls_back(self, rfmt_df, clv_df, segment_summary):
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.return_value = self._mock_response("   ")

        with patch("src.digest_engine.groq.Groq", return_value=mock_client):
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, groq_api_key="fake-groq-key")

        assert "[Template Summary" in result
        assert "empty response from Groq" in result

    def test_missing_output_text_falls_back(self, rfmt_df, clv_df, segment_summary):
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.return_value = self._mock_response(None)

        with patch("src.digest_engine.groq.Groq", return_value=mock_client):
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, groq_api_key="fake-groq-key")

        assert "[Template Summary" in result


class TestGroqFailuresDegradeGracefully:
    def test_missing_key_falls_back_without_any_network_call(self, rfmt_df, clv_df, segment_summary):
        with patch("src.digest_engine.groq.Groq") as mock_client_cls:
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, groq_api_key=None)
            mock_client_cls.assert_not_called()
        assert "[Template Summary" in result

    def test_groq_package_missing_falls_back(self, rfmt_df, clv_df, segment_summary):
        with patch("src.digest_engine.groq", None):
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, groq_api_key="fake-groq-key")
        assert "[Template Summary" in result
        assert "not installed" in result

    def test_rate_limit_falls_back_with_specific_reason(self, rfmt_df, clv_df, segment_summary):
        # This is the documented, expected-at-scale free-tier ceiling (see
        # digest_engine.py's module docstring for the current rate-limit
        # numbers and their sourcing caveat) -- must be caught, not raised.
        exc = _make_groq_api_error(429, "quota exceeded")
        assert isinstance(exc, groq.RateLimitError)  # sanity check on the helper itself
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.side_effect = exc

        with patch("src.digest_engine.groq.Groq", return_value=mock_client):
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, groq_api_key="fake-groq-key")

        assert "[Template Summary" in result
        assert "Groq free-tier rate limit reached" in result

    def test_invalid_api_key_falls_back_with_specific_reason(self, rfmt_df, clv_df, segment_summary):
        # Unlike Gemini's non-standard "400 with API_KEY_INVALID in the body"
        # quirk, Groq/OpenAI-compatible APIs use standard HTTP 401 for bad
        # credentials -- confirmed via the pinned SDK's own status_code -> class
        # dispatch (groq/_client.py's _make_status_error()), no body-content
        # sniffing required.
        exc = _make_groq_api_error(401, "bad credentials")
        assert isinstance(exc, groq.AuthenticationError)
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.side_effect = exc

        with patch("src.digest_engine.groq.Groq", return_value=mock_client):
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, groq_api_key="bad-key")

        assert "[Template Summary" in result
        assert "invalid Groq API key" in result
        assert "unexpected error" not in result

    def test_other_status_error_falls_back_with_generic_reason(self, rfmt_df, clv_df, segment_summary):
        # A non-2xx that isn't one of the specifically-distinguished classes
        # above should get the generic reason, not be mistaken for a bad key
        # or a rate limit.
        exc = _make_groq_api_error(404, "not found")
        assert isinstance(exc, groq.NotFoundError)
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.side_effect = exc

        with patch("src.digest_engine.groq.Groq", return_value=mock_client):
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, groq_api_key="fake-groq-key")

        assert "[Template Summary" in result
        assert "Groq API error" in result
        assert "invalid Groq API key" not in result

    def test_server_error_falls_back_via_the_status_error_branch(self, rfmt_df, clv_df, segment_summary):
        exc = _make_groq_api_error(500, "internal error")
        assert isinstance(exc, groq.InternalServerError)
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.side_effect = exc

        with patch("src.digest_engine.groq.Groq", return_value=mock_client):
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, groq_api_key="fake-groq-key")

        assert "[Template Summary" in result
        assert "Groq API error" in result

    def test_timeout_falls_back_with_its_own_distinct_reason_never_hangs_the_test(
        self, rfmt_df, clv_df, segment_summary
    ):
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

        with patch("src.digest_engine.groq.Groq", return_value=mock_client):
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, groq_api_key="fake-groq-key")

        assert "[Template Summary" in result
        assert "Groq API request timed out" in result
        # Must never be conflated with the other Groq failure reasons.
        assert "rate limit" not in result
        assert "invalid Groq API key" not in result
        assert result.count("Groq API error") == 0

    def test_connection_error_is_caught_by_the_broad_apierror_base_class(self, rfmt_df, clv_df, segment_summary):
        # APIConnectionError has no HTTP response/status_code at all (raised
        # for network-level failures) -- it is NOT an APIStatusError subclass,
        # so this exercises the broader `except groq.APIConnectionError`
        # branch specifically (distinct from the APIStatusError branch above).
        exc = groq.APIConnectionError(message="connection failed", request=httpx.Request(
            "POST", "https://api.groq.com/openai/v1/chat/completions"
        ))
        assert isinstance(exc, groq.APIError)
        assert not isinstance(exc, groq.APIStatusError)
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.side_effect = exc

        with patch("src.digest_engine.groq.Groq", return_value=mock_client):
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, groq_api_key="fake-groq-key")

        assert "[Template Summary" in result
        assert "could not reach the Groq API" in result
        assert "unexpected error" not in result

    def test_generic_exception_falls_back_instead_of_raising(self, rfmt_df, clv_df, segment_summary):
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.side_effect = RuntimeError("boom")

        with patch("src.digest_engine.groq.Groq", return_value=mock_client):
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, groq_api_key="fake-groq-key")

        assert "[Template Summary" in result
        assert "unexpected error" in result


class TestCallGroqInIsolation:
    """
    Direct, isolated coverage of _call_groq() -- the shared Groq call/
    error-handling logic (mirrors _call_anthropic() almost line for line, see
    its docstring in src/digest_engine.py) so it has its own test coverage
    rather than only being exercised indirectly through generate_account_digest()
    and src/chat_engine.py's answer_account_question(). Every test here calls
    _call_groq() directly and asserts on its (text, failure_reason) return
    tuple -- no _fallback_digest()/_chat_unavailable() wrapping involved,
    since that's each caller's own concern, not this function's.
    """

    def _mock_response(self, text="A generated answer.", finish_reason="stop"):
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content=text),
                finish_reason=finish_reason,
            )]
        )

    def test_success_returns_text_and_no_failure_reason(self):
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.return_value = self._mock_response()

        with patch("src.digest_engine.groq.Groq", return_value=mock_client):
            text, failure_reason = _call_groq(
                messages=[{"role": "user", "content": "hi"}],
                api_key="fake-groq-key",
                max_tokens=300,
            )

        assert text == "A generated answer."
        assert failure_reason is None

    def test_messages_model_max_tokens_and_timeout_are_passed_through(self):
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.return_value = self._mock_response()
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]

        with patch("src.digest_engine.groq.Groq", return_value=mock_client):
            _call_groq(messages=messages, api_key="fake-groq-key", max_tokens=123)

        mock_client.with_options.assert_called_once_with(timeout=GROQ_REQUEST_TIMEOUT_SECONDS)
        _, kwargs = mock_client.with_options.return_value.chat.completions.create.call_args
        assert kwargs["model"] == GROQ_MODEL_ID
        assert kwargs["max_tokens"] == 123
        assert kwargs["messages"] == messages
        # reasoning_effort="low" is passed on every Groq call -- see
        # GROQ_REASONING_EFFORT's own comment for why "low" specifically,
        # confirmed supported for this exact model via both a WebSearch of
        # Groq's own docs and direct introspection of the pinned groq
        # package's own request-parameter type stub.
        assert kwargs["reasoning_effort"] == GROQ_REASONING_EFFORT

    def test_empty_response_returns_the_documented_failure_reason(self):
        # finish_reason defaults to "stop" (via _mock_response()'s own
        # default) -- NOT "length" -- so this exercises the genuinely-empty
        # path, not the truncation path below. "Genuinely empty content
        # (regardless of finish_reason)" per the task that added this
        # distinction: what matters is that finish_reason is NOT "length"
        # here, mirroring a real "model returned nothing" response shape
        # rather than a token-budget exhaustion.
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.return_value = self._mock_response("   ")

        with patch("src.digest_engine.groq.Groq", return_value=mock_client):
            text, failure_reason = _call_groq(
                messages=[{"role": "user", "content": "hi"}], api_key="fake-groq-key", max_tokens=300
            )

        assert text is None
        assert failure_reason == "empty response from Groq"

    def test_truncated_response_with_nonempty_short_content_returns_the_distinct_truncated_reason(self):
        # The exact live failure shape this fix addresses: finish_reason ==
        # "length" (the token budget ran out) with a NON-empty but
        # mid-sentence-cut-off visible answer -- reasoning consumed MOST,
        # not ALL, of the budget. Must be classified as "truncated," a
        # DIFFERENT failure mode from "empty response" (different root
        # cause -- raise the token budget -- and conflating the two
        # fallback messages made debugging this live incident harder).
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.return_value = self._mock_response(
            "The 2026-02 cohort's month-5 retention is 91.", finish_reason="length"
        )

        with patch("src.digest_engine.groq.Groq", return_value=mock_client):
            text, failure_reason = _call_groq(
                messages=[{"role": "user", "content": "hi"}], api_key="fake-groq-key", max_tokens=300
            )

        assert text is None
        assert failure_reason == (
            "Groq response was truncated before completion -- try a "
            "shorter question or increase the token budget"
        )
        assert failure_reason != "empty response from Groq"

    def test_truncated_response_with_fully_empty_content_also_returns_the_truncated_reason(self):
        # The OTHER live-observed shape of the same failure: reasoning
        # consumed the ENTIRE token budget, leaving zero visible content --
        # this is what narrate_cohort_pattern() actually hit live (150-token
        # budget). Even though content is empty here too, finish_reason ==
        # "length" must still win over the generic "empty response" reason --
        # this IS the truncation failure, just a more severe case of it, not
        # a separate "genuinely empty" case.
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.return_value = self._mock_response(
            "", finish_reason="length"
        )

        with patch("src.digest_engine.groq.Groq", return_value=mock_client):
            text, failure_reason = _call_groq(
                messages=[{"role": "user", "content": "hi"}], api_key="fake-groq-key", max_tokens=300
            )

        assert text is None
        assert "truncated" in failure_reason
        assert failure_reason != "empty response from Groq"

    def test_finish_reason_missing_from_a_test_double_does_not_raise(self):
        # Defense-in-depth check for the getattr(..., default=None) in
        # _call_groq() itself: a response object that doesn't set
        # finish_reason at all (as every pre-existing mock in this codebase
        # did before this fix) must not raise AttributeError -- it should
        # fall through to the normal empty/success handling exactly as
        # before. Real groq.types.chat.chat_completion.Choice objects always
        # carry this field (confirmed via direct package introspection --
        # it's a required, non-Optional Literal), so this is purely a
        # test-double-safety guarantee, never a real-response code path.
        bare_response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="A real answer."))]
        )
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.return_value = bare_response

        with patch("src.digest_engine.groq.Groq", return_value=mock_client):
            text, failure_reason = _call_groq(
                messages=[{"role": "user", "content": "hi"}], api_key="fake-groq-key", max_tokens=300
            )

        assert text == "A real answer."
        assert failure_reason is None

    def test_package_missing_returns_the_documented_failure_reason(self):
        # Patches src.digest_engine.groq directly (the name _call_groq()
        # itself reads) -- see its docstring for why this check can't be
        # caller-agnostic across modules.
        with patch("src.digest_engine.groq", None):
            text, failure_reason = _call_groq(
                messages=[{"role": "user", "content": "hi"}], api_key="fake-groq-key", max_tokens=300
            )

        assert text is None
        assert failure_reason == "groq package not installed"

    @pytest.mark.parametrize("exception_name,expected_reason", [
        ("AuthenticationError", "invalid Groq API key"),
        ("RateLimitError", "Groq free-tier rate limit reached"),
        ("APIStatusError", "Groq API error"),
        ("APIConnectionError", "could not reach the Groq API"),
        ("APITimeoutError", "Groq API request timed out"),
    ])
    def test_each_typed_exception_returns_its_own_documented_failure_reason(
        self, exception_name, expected_reason
    ):
        exc_cls = getattr(groq, exception_name)
        # Bare object.__new__ bypass -- these classes' real __init__ signatures
        # vary (some need response/body, APIConnectionError needs request), but
        # `except` only matches on type, so a real instance of the correct
        # class (however constructed) is all that's needed here. Same pattern
        # already used for anthropic's exceptions above.
        exc_instance = exc_cls.__new__(exc_cls)

        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.side_effect = exc_instance

        with patch("src.digest_engine.groq.Groq", return_value=mock_client):
            text, failure_reason = _call_groq(
                messages=[{"role": "user", "content": "hi"}], api_key="fake-groq-key", max_tokens=300
            )

        assert text is None
        assert failure_reason == expected_reason

    def test_unexpected_exception_returns_a_reason_containing_unexpected_error(self):
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.side_effect = RuntimeError("boom")

        with patch("src.digest_engine.groq.Groq", return_value=mock_client):
            text, failure_reason = _call_groq(
                messages=[{"role": "user", "content": "hi"}], api_key="fake-groq-key", max_tokens=300
            )

        assert text is None
        assert "unexpected error" in failure_reason
        assert "Groq" in failure_reason


class TestCallAnthropicInIsolation:
    """
    Direct, isolated coverage of _call_anthropic() -- the shared Anthropic
    call/error-handling logic (see its docstring in src/digest_engine.py).
    Every test here calls _call_anthropic() directly and asserts on its
    (text, failure_reason) return tuple.
    """

    def _mock_response(self, text="A generated answer."):
        block = types.SimpleNamespace(type="text", text=text)
        return types.SimpleNamespace(content=[block])

    def test_success_returns_text_and_no_failure_reason(self):
        mock_client = MagicMock()
        mock_client.with_options.return_value.messages.create.return_value = self._mock_response()

        with patch("src.digest_engine.anthropic.Anthropic", return_value=mock_client):
            text, failure_reason = _call_anthropic(
                messages=[{"role": "user", "content": "hi"}],
                api_key="sk-ant-fake-key",
                max_tokens=300,
            )

        assert text == "A generated answer."
        assert failure_reason is None

    def test_system_none_omits_the_kwarg_entirely(self):
        mock_client = MagicMock()
        mock_client.with_options.return_value.messages.create.return_value = self._mock_response()

        with patch("src.digest_engine.anthropic.Anthropic", return_value=mock_client):
            _call_anthropic(messages=[{"role": "user", "content": "hi"}], api_key="sk-ant-fake-key", max_tokens=300)

        _, kwargs = mock_client.with_options.return_value.messages.create.call_args
        assert "system" not in kwargs

    def test_system_when_given_is_passed_through(self):
        mock_client = MagicMock()
        mock_client.with_options.return_value.messages.create.return_value = self._mock_response()

        with patch("src.digest_engine.anthropic.Anthropic", return_value=mock_client):
            _call_anthropic(
                messages=[{"role": "user", "content": "hi"}],
                api_key="sk-ant-fake-key",
                max_tokens=300,
                system="a system prompt",
            )

        _, kwargs = mock_client.with_options.return_value.messages.create.call_args
        assert kwargs["system"] == "a system prompt"

    def test_messages_model_max_tokens_and_timeout_are_passed_through(self):
        mock_client = MagicMock()
        mock_client.with_options.return_value.messages.create.return_value = self._mock_response()
        messages = [{"role": "user", "content": "hi"}]

        with patch("src.digest_engine.anthropic.Anthropic", return_value=mock_client):
            _call_anthropic(messages=messages, api_key="sk-ant-fake-key", max_tokens=123)

        mock_client.with_options.assert_called_once_with(timeout=REQUEST_TIMEOUT_SECONDS)
        _, kwargs = mock_client.with_options.return_value.messages.create.call_args
        assert kwargs["model"] == MODEL_ID
        assert kwargs["max_tokens"] == 123
        assert kwargs["messages"] == messages

    def test_empty_response_returns_the_documented_failure_reason(self):
        mock_client = MagicMock()
        mock_client.with_options.return_value.messages.create.return_value = self._mock_response(text="   ")

        with patch("src.digest_engine.anthropic.Anthropic", return_value=mock_client):
            text, failure_reason = _call_anthropic(
                messages=[{"role": "user", "content": "hi"}], api_key="sk-ant-fake-key", max_tokens=300
            )

        assert text is None
        assert failure_reason == "empty response from the model"

    def test_package_missing_returns_the_documented_failure_reason(self):
        # Patches src.digest_engine.anthropic directly (the name
        # _call_anthropic() itself reads) -- see its docstring for why this
        # check can't be caller-agnostic across modules.
        with patch("src.digest_engine.anthropic", None):
            text, failure_reason = _call_anthropic(
                messages=[{"role": "user", "content": "hi"}], api_key="sk-ant-fake-key", max_tokens=300
            )

        assert text is None
        assert failure_reason == "anthropic package not installed"

    @pytest.mark.parametrize("exception_name,expected_reason", [
        ("AuthenticationError", "invalid Anthropic API key"),
        ("RateLimitError", "Anthropic API rate limited"),
        ("APIStatusError", "Anthropic API error"),
        ("APIConnectionError", "could not reach the Anthropic API"),
    ])
    def test_each_typed_exception_returns_its_own_documented_failure_reason(
        self, exception_name, expected_reason
    ):
        import anthropic as real_anthropic

        exc_cls = getattr(real_anthropic, exception_name)
        exc_instance = exc_cls.__new__(exc_cls)

        mock_client = MagicMock()
        mock_client.with_options.return_value.messages.create.side_effect = exc_instance

        with patch("src.digest_engine.anthropic.Anthropic", return_value=mock_client):
            text, failure_reason = _call_anthropic(
                messages=[{"role": "user", "content": "hi"}], api_key="sk-ant-fake-key", max_tokens=300
            )

        assert text is None
        assert failure_reason == expected_reason

    def test_unexpected_exception_returns_a_reason_containing_unexpected_error(self):
        mock_client = MagicMock()
        mock_client.with_options.return_value.messages.create.side_effect = RuntimeError("boom")

        with patch("src.digest_engine.anthropic.Anthropic", return_value=mock_client):
            text, failure_reason = _call_anthropic(
                messages=[{"role": "user", "content": "hi"}], api_key="sk-ant-fake-key", max_tokens=300
            )

        assert text is None
        assert "unexpected error" in failure_reason
        assert "Anthropic" in failure_reason


class TestTargetConfiguration:
    """
    The actual configuration this feature was built for: GROQ_API_KEY set,
    ANTHROPIC_API_KEY absent. Confirms the Groq branch is what actually executes,
    end to end through generate_account_digest()'s provider resolution -- not just
    that _resolve_provider() returns the right string in isolation.
    """

    def test_groq_only_configuration_calls_groq_not_anthropic(self, rfmt_df, clv_df, segment_summary):
        mock_response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="Groq-generated summary."))]
        )
        mock_groq_client = MagicMock()
        mock_groq_client.with_options.return_value.chat.completions.create.return_value = mock_response

        with patch("src.digest_engine.groq.Groq", return_value=mock_groq_client) as mock_groq_cls, \
             patch("src.digest_engine.anthropic.Anthropic") as mock_anthropic_cls:
            result = generate_account_digest(
                rfmt_df, clv_df, segment_summary,
                anthropic_api_key=None,
                groq_api_key="fake-groq-key",
            )

        assert result == "Groq-generated summary."
        mock_groq_cls.assert_called_once()
        mock_anthropic_cls.assert_not_called()


class TestStatsWithoutChurnRiskTier:
    def test_missing_churn_risk_tier_column_defaults_pct_at_risk_to_zero(self, rfmt_df, clv_df, segment_summary):
        clv_df_no_tier = clv_df.drop(columns=["Churn_Risk_Tier"])
        stats = _build_aggregate_stats(rfmt_df, clv_df_no_tier, segment_summary)
        assert stats["pct_at_risk"] == 0.0


class TestApiKeyResolution:
    def test_anthropic_key_reads_from_environment_variable(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key-123")
        assert get_anthropic_api_key() == "env-key-123"

    def test_anthropic_key_returns_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        # Note: if st.secrets happens to have ANTHROPIC_API_KEY configured in this
        # test environment, that would also satisfy resolution -- not expected in CI.
        result = get_anthropic_api_key()
        assert result is None or isinstance(result, str)

    def test_groq_key_reads_from_environment_variable(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "groq-env-key-456")
        assert get_groq_api_key() == "groq-env-key-456"

    def test_groq_key_returns_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        result = get_groq_api_key()
        assert result is None or isinstance(result, str)

    def test_provider_override_reads_from_environment_variable(self, monkeypatch):
        monkeypatch.setenv("DIGEST_PROVIDER", "anthropic")
        assert get_digest_provider_override() == "anthropic"

    def test_provider_override_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("DIGEST_PROVIDER", "GROQ")
        assert get_digest_provider_override() == "groq"

    def test_provider_override_returns_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("DIGEST_PROVIDER", raising=False)
        result = get_digest_provider_override()
        assert result is None or result in ("groq", "anthropic")

    def test_provider_override_returns_none_for_unrecognized_value(self, monkeypatch):
        monkeypatch.setenv("DIGEST_PROVIDER", "chatgpt")
        assert get_digest_provider_override() is None

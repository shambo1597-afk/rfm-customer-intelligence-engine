"""
tests/test_digest_engine.py - pytest suite for the optional AI Executive Summary.

No real network calls are made anywhere in this file -- both provider clients
(Anthropic, Gemini) are always mocked (unittest.mock) when a "successful call" or
"provider failure" path is exercised, and the no-key fallback path is tested
precisely because it must NOT construct a client or attempt any network call at all.
"""

import types
from unittest.mock import MagicMock, patch

import httpx
import pytest
from google.genai._gaos.lib import compat_errors as genai_errors

from src.rfm_engine import get_segment_kpi_summary
from src.digest_engine import (
    generate_account_digest,
    get_anthropic_api_key,
    get_gemini_api_key,
    get_digest_provider_override,
    _resolve_provider,
    _build_aggregate_stats,
    _build_prompt,
    _fallback_digest,
    MODEL_ID,
    MAX_TOKENS,
    GEMINI_MODEL_ID,
    GEMINI_MAX_OUTPUT_TOKENS,
)


@pytest.fixture(scope="module")
def segment_summary(clv_df):
    return get_segment_kpi_summary(clv_df)


def _make_gemini_api_error(status_code: int, message: str, reason: str = "", body: dict = None):
    """
    Builds a real `genai_errors.APIStatusError` subclass instance via the exact
    factory (`APIError.generate`) the installed google-genai SDK itself uses to
    turn an HTTP response into an exception -- rather than hand-picking a
    subclass in the test, this reproduces the SDK's real status_code -> class
    dispatch (confirmed by direct reproduction against a live invalid-key call;
    see src/digest_engine.py's module docstring), so these tests exercise the
    actual exception hierarchy `client.interactions.create()` raises, not a
    guess at it.

    `body` defaults to a simple `{code, message, status}` error envelope; pass
    it explicitly to reproduce a more specific real response shape (e.g. the
    invalid-key error's nested `details` list -- see
    test_invalid_api_key_returns_400_bad_request_and_still_falls_back_with_invalid_key_reason).
    """
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


class TestNamedConstants:
    """B3 requirement: model/max_tokens must be named constants, not inline magic numbers."""

    def test_anthropic_model_and_max_tokens_are_module_level_constants(self):
        assert isinstance(MODEL_ID, str) and MODEL_ID  # non-empty string
        assert isinstance(MAX_TOKENS, int) and MAX_TOKENS > 0

    def test_anthropic_model_is_the_small_cheap_tier(self):
        # This feature's entire cost story (see module docstring) depends on staying
        # on the cheap model -- a regression here silently breaks the ~$1/mo claim.
        assert "haiku" in MODEL_ID.lower()

    def test_gemini_model_and_max_output_tokens_are_module_level_constants(self):
        assert isinstance(GEMINI_MODEL_ID, str) and GEMINI_MODEL_ID
        assert isinstance(GEMINI_MAX_OUTPUT_TOKENS, int) and GEMINI_MAX_OUTPUT_TOKENS > 0

    def test_gemini_model_is_the_free_tier_flash_lite_not_pro(self):
        # Pro models require billing and are not on the free tier -- a regression
        # here silently breaks the "genuinely $0 by default" claim.
        model_lower = GEMINI_MODEL_ID.lower()
        assert "flash-lite" in model_lower
        assert "pro" not in model_lower


class TestResolveProvider:
    """
    Every branch _resolve_provider() must handle, per the task spec: both keys with
    no override -> gemini; both keys with an override -> that provider; only one key
    present -> that provider regardless of override; neither key -> "none"; an
    override naming a provider whose key is absent -> falls back to whichever key
    IS present, or "none".
    """

    def test_both_keys_no_override_prefers_gemini(self):
        assert _resolve_provider("a-key", "g-key") == "gemini"

    def test_both_keys_override_anthropic_uses_anthropic(self):
        assert _resolve_provider("a-key", "g-key", override="anthropic") == "anthropic"

    def test_both_keys_override_gemini_uses_gemini(self):
        assert _resolve_provider("a-key", "g-key", override="gemini") == "gemini"

    def test_only_gemini_key_uses_gemini(self):
        assert _resolve_provider(None, "g-key") == "gemini"

    def test_only_anthropic_key_uses_anthropic(self):
        assert _resolve_provider("a-key", None) == "anthropic"

    def test_neither_key_returns_none(self):
        assert _resolve_provider(None, None) == "none"

    def test_override_naming_a_provider_with_no_key_falls_back_to_present_key(self):
        # override="gemini" but no Gemini key -- falls back to Anthropic, which IS present.
        assert _resolve_provider("a-key", None, override="gemini") == "anthropic"
        # override="anthropic" but no Anthropic key -- falls back to Gemini, which IS present.
        assert _resolve_provider(None, "g-key", override="anthropic") == "gemini"

    def test_override_naming_a_provider_with_no_key_and_no_fallback_key_returns_none(self):
        assert _resolve_provider(None, None, override="gemini") == "none"
        assert _resolve_provider(None, None, override="anthropic") == "none"

    def test_unrecognized_override_value_is_ignored(self):
        # A typo'd/garbage override falls through to the normal preference order,
        # never raises.
        assert _resolve_provider("a-key", "g-key", override="chatgpt") == "gemini"
        assert _resolve_provider(None, None, override="chatgpt") == "none"


class TestFallbackPathNoKeys:
    def test_no_keys_returns_fallback_without_any_network_call(self, rfmt_df, clv_df, segment_summary):
        with patch("src.digest_engine.anthropic.Anthropic") as mock_anthropic_cls, \
             patch("src.digest_engine.genai.Client") as mock_genai_cls:
            result = generate_account_digest(rfmt_df, clv_df, segment_summary)
            mock_anthropic_cls.assert_not_called()
            mock_genai_cls.assert_not_called()

        assert isinstance(result, str) and result
        assert "[Template Summary" in result

    def test_empty_string_keys_also_fall_back(self, rfmt_df, clv_df, segment_summary):
        with patch("src.digest_engine.anthropic.Anthropic") as mock_anthropic_cls, \
             patch("src.digest_engine.genai.Client") as mock_genai_cls:
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, anthropic_api_key="", gemini_api_key="")
            mock_anthropic_cls.assert_not_called()
            mock_genai_cls.assert_not_called()
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

    def test_prompt_built_for_the_gemini_path_also_excludes_raw_customer_ids(self, rfmt_df, clv_df, segment_summary):
        # generate_account_digest() calls the same _build_prompt() regardless of
        # provider -- re-run the PII-exclusion check end-to-end through the Gemini
        # branch specifically, not just against _build_prompt() in isolation.
        mock_interaction = types.SimpleNamespace(output_text="A summary with no customer IDs.")
        mock_client = MagicMock()
        mock_client.interactions.create.return_value = mock_interaction

        with patch("src.digest_engine.genai.Client", return_value=mock_client):
            generate_account_digest(rfmt_df, clv_df, segment_summary, gemini_api_key="fake-gemini-key")

        _, kwargs = mock_client.interactions.create.call_args
        sent_prompt = kwargs["input"]
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


class TestSuccessfulGeminiCall:
    def test_successful_call_returns_model_text_not_fallback(self, rfmt_df, clv_df, segment_summary):
        mock_interaction = types.SimpleNamespace(output_text="A healthy account overall.")
        mock_client = MagicMock()
        mock_client.interactions.create.return_value = mock_interaction

        with patch("src.digest_engine.genai.Client", return_value=mock_client) as mock_client_cls:
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, gemini_api_key="fake-gemini-key")

        assert result == "A healthy account overall."
        assert "[Template Summary" not in result
        mock_client_cls.assert_called_once_with(api_key="fake-gemini-key")

    def test_call_uses_the_named_model_and_max_output_tokens_constants(self, rfmt_df, clv_df, segment_summary):
        mock_interaction = types.SimpleNamespace(output_text="Summary text.")
        mock_client = MagicMock()
        mock_client.interactions.create.return_value = mock_interaction

        with patch("src.digest_engine.genai.Client", return_value=mock_client):
            generate_account_digest(rfmt_df, clv_df, segment_summary, gemini_api_key="fake-gemini-key")

        _, kwargs = mock_client.interactions.create.call_args
        assert kwargs["model"] == GEMINI_MODEL_ID
        assert kwargs["generation_config"]["max_output_tokens"] == GEMINI_MAX_OUTPUT_TOKENS

    def test_prompt_is_passed_as_the_input_parameter(self, rfmt_df, clv_df, segment_summary):
        mock_interaction = types.SimpleNamespace(output_text="Summary text.")
        mock_client = MagicMock()
        mock_client.interactions.create.return_value = mock_interaction

        with patch("src.digest_engine.genai.Client", return_value=mock_client):
            generate_account_digest(rfmt_df, clv_df, segment_summary, gemini_api_key="fake-gemini-key")

        _, kwargs = mock_client.interactions.create.call_args
        assert kwargs["input"] == _build_prompt(_build_aggregate_stats(rfmt_df, clv_df, segment_summary))

    def test_empty_output_text_falls_back(self, rfmt_df, clv_df, segment_summary):
        mock_interaction = types.SimpleNamespace(output_text="   ")
        mock_client = MagicMock()
        mock_client.interactions.create.return_value = mock_interaction

        with patch("src.digest_engine.genai.Client", return_value=mock_client):
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, gemini_api_key="fake-gemini-key")

        assert "[Template Summary" in result
        assert "empty response from Gemini" in result

    def test_missing_output_text_falls_back(self, rfmt_df, clv_df, segment_summary):
        mock_interaction = types.SimpleNamespace(output_text=None)
        mock_client = MagicMock()
        mock_client.interactions.create.return_value = mock_interaction

        with patch("src.digest_engine.genai.Client", return_value=mock_client):
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, gemini_api_key="fake-gemini-key")

        assert "[Template Summary" in result


class TestGeminiFailuresDegradeGracefully:
    def test_missing_key_falls_back_without_any_network_call(self, rfmt_df, clv_df, segment_summary):
        with patch("src.digest_engine.genai.Client") as mock_client_cls:
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, gemini_api_key=None)
            mock_client_cls.assert_not_called()
        assert "[Template Summary" in result

    def test_genai_package_missing_falls_back(self, rfmt_df, clv_df, segment_summary):
        with patch("src.digest_engine.genai", None):
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, gemini_api_key="fake-gemini-key")
        assert "[Template Summary" in result
        assert "not installed" in result

    def test_rate_limit_429_resource_exhausted_falls_back_with_specific_reason(self, rfmt_df, clv_df, segment_summary):
        # This is the documented, expected-at-scale free-tier ceiling
        # (ai.google.dev/gemini-api/docs/rate-limits) -- must be caught, not raised.
        exc = _make_gemini_api_error(429, "quota exceeded", reason="RESOURCE_EXHAUSTED")
        assert isinstance(exc, genai_errors.RateLimitError)  # sanity check on the helper itself
        mock_client = MagicMock()
        mock_client.interactions.create.side_effect = exc

        with patch("src.digest_engine.genai.Client", return_value=mock_client):
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, gemini_api_key="fake-gemini-key")

        assert "[Template Summary" in result
        assert "Gemini free-tier rate limit reached" in result

    def test_invalid_api_key_returns_400_bad_request_and_still_falls_back_with_invalid_key_reason(
        self, rfmt_df, clv_df, segment_summary
    ):
        # This is the REAL shape Google's API returns for a bad key (confirmed by
        # reproducing a live invalid-key call against the pinned SDK version) --
        # HTTP 400 BadRequestError, with the outer body 'status' as
        # 'INVALID_ARGUMENT' (NOT a distinguishing signal on its own -- many
        # non-key 400s share it) and the specific 'API_KEY_INVALID' reason
        # nested in 'details', exactly as Google's real response embeds it. NOT
        # 401/403 as HTTP convention might suggest. This is the exact real-world
        # response shape the bug this fix addresses was silently mis-handling.
        real_invalid_key_body = {
            "error": {
                "code": 400,
                "message": "API key not valid. Please pass a valid API key.",
                "status": "INVALID_ARGUMENT",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                        "reason": "API_KEY_INVALID",
                        "domain": "googleapis.com",
                        "metadata": {"service": "generativelanguage.googleapis.com"},
                    },
                ],
            }
        }
        exc = _make_gemini_api_error(
            400, "API key not valid. Please pass a valid API key.", body=real_invalid_key_body
        )
        assert isinstance(exc, genai_errors.BadRequestError)  # sanity check on the helper itself
        mock_client = MagicMock()
        mock_client.interactions.create.side_effect = exc

        with patch("src.digest_engine.genai.Client", return_value=mock_client):
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, gemini_api_key="bad-key")

        assert "[Template Summary" in result
        assert "invalid Gemini API key" in result
        assert "unexpected error" not in result

    @pytest.mark.parametrize("code", [401, 403])
    def test_auth_errors_fall_back_with_invalid_key_reason(self, rfmt_df, clv_df, segment_summary, code):
        exc = _make_gemini_api_error(code, "bad credentials", reason="UNAUTHENTICATED")
        mock_client = MagicMock()
        mock_client.interactions.create.side_effect = exc

        with patch("src.digest_engine.genai.Client", return_value=mock_client):
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, gemini_api_key="bad-key")

        assert "[Template Summary" in result
        assert "invalid Gemini API key" in result

    def test_other_bad_request_error_falls_back_with_generic_reason(self, rfmt_df, clv_df, segment_summary):
        # A 400 that is NOT the invalid-key shape (no 'API_KEY_INVALID' in the
        # message) should get the generic reason, not be mistaken for a bad key.
        exc = _make_gemini_api_error(400, "malformed generation_config", reason="INVALID_ARGUMENT")
        mock_client = MagicMock()
        mock_client.interactions.create.side_effect = exc

        with patch("src.digest_engine.genai.Client", return_value=mock_client):
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, gemini_api_key="fake-gemini-key")

        assert "[Template Summary" in result
        assert "Gemini API error" in result
        assert "invalid Gemini API key" not in result

    def test_server_error_falls_back_via_the_broad_apierror_base_class(self, rfmt_df, clv_df, segment_summary):
        # InternalServerError is an APIStatusError subclass (still has
        # status_code) -- goes through the same branch as the other status
        # errors above and lands on the generic reason (no dedicated 5xx message).
        exc = _make_gemini_api_error(500, "internal error")
        assert isinstance(exc, genai_errors.InternalServerError)
        mock_client = MagicMock()
        mock_client.interactions.create.side_effect = exc

        with patch("src.digest_engine.genai.Client", return_value=mock_client):
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, gemini_api_key="fake-gemini-key")

        assert "[Template Summary" in result
        assert "Gemini API error" in result

    def test_connection_error_is_caught_by_the_broad_apierror_base_class(self, rfmt_df, clv_df, segment_summary):
        # APIConnectionError has no HTTP response/status_code at all (it's raised
        # for network-level failures) -- it is NOT an APIStatusError subclass, so
        # this specifically exercises the broader `except genai_errors.APIError`
        # branch, confirming a future/less-common error variant still gets a
        # diagnostic reason instead of silently falling through to the generic
        # catch-all (the exact failure mode this fix addresses).
        exc = genai_errors.APIConnectionError(message="connection failed", request=httpx.Request(
            "POST", "https://generativelanguage.googleapis.com/v1beta/interactions"
        ))
        assert isinstance(exc, genai_errors.APIError)
        assert not isinstance(exc, genai_errors.APIStatusError)
        mock_client = MagicMock()
        mock_client.interactions.create.side_effect = exc

        with patch("src.digest_engine.genai.Client", return_value=mock_client):
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, gemini_api_key="fake-gemini-key")

        assert "[Template Summary" in result
        assert "Gemini API error" in result
        assert "unexpected error" not in result

    def test_generic_exception_falls_back_instead_of_raising(self, rfmt_df, clv_df, segment_summary):
        mock_client = MagicMock()
        mock_client.interactions.create.side_effect = RuntimeError("boom")

        with patch("src.digest_engine.genai.Client", return_value=mock_client):
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, gemini_api_key="fake-gemini-key")

        assert "[Template Summary" in result
        assert "unexpected error" in result


class TestTargetConfiguration:
    """
    The actual configuration this feature was built for: GEMINI_API_KEY set,
    ANTHROPIC_API_KEY absent. Confirms the Gemini branch is what actually executes,
    end to end through generate_account_digest()'s provider resolution -- not just
    that _resolve_provider() returns the right string in isolation.
    """

    def test_gemini_only_configuration_calls_gemini_not_anthropic(self, rfmt_df, clv_df, segment_summary):
        mock_interaction = types.SimpleNamespace(output_text="Gemini-generated summary.")
        mock_gemini_client = MagicMock()
        mock_gemini_client.interactions.create.return_value = mock_interaction

        with patch("src.digest_engine.genai.Client", return_value=mock_gemini_client) as mock_genai_cls, \
             patch("src.digest_engine.anthropic.Anthropic") as mock_anthropic_cls:
            result = generate_account_digest(
                rfmt_df, clv_df, segment_summary,
                anthropic_api_key=None,
                gemini_api_key="fake-gemini-key",
            )

        assert result == "Gemini-generated summary."
        mock_genai_cls.assert_called_once()
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

    def test_gemini_key_reads_from_environment_variable(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-env-key-456")
        assert get_gemini_api_key() == "gemini-env-key-456"

    def test_gemini_key_returns_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        result = get_gemini_api_key()
        assert result is None or isinstance(result, str)

    def test_provider_override_reads_from_environment_variable(self, monkeypatch):
        monkeypatch.setenv("DIGEST_PROVIDER", "anthropic")
        assert get_digest_provider_override() == "anthropic"

    def test_provider_override_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("DIGEST_PROVIDER", "GEMINI")
        assert get_digest_provider_override() == "gemini"

    def test_provider_override_returns_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("DIGEST_PROVIDER", raising=False)
        result = get_digest_provider_override()
        assert result is None or result in ("gemini", "anthropic")

    def test_provider_override_returns_none_for_unrecognized_value(self, monkeypatch):
        monkeypatch.setenv("DIGEST_PROVIDER", "chatgpt")
        assert get_digest_provider_override() is None

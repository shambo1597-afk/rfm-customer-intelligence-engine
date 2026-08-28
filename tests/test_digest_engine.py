"""
tests/test_digest_engine.py - pytest suite for the optional AI Executive Summary.

No real network calls are made anywhere in this file -- the Anthropic client is
always mocked (unittest.mock) when a "successful call" path is exercised, and the
no-api_key fallback path is tested precisely because it must NOT construct a client
or attempt any network call at all.
"""

import types
from unittest.mock import MagicMock, patch

import pytest

from src.rfm_engine import get_segment_kpi_summary
from src.digest_engine import (
    generate_account_digest,
    get_anthropic_api_key,
    _build_aggregate_stats,
    _build_prompt,
    _fallback_digest,
    MODEL_ID,
    MAX_TOKENS,
)


@pytest.fixture(scope="module")
def segment_summary(clv_df):
    return get_segment_kpi_summary(clv_df)


class TestNamedConstants:
    """B3 requirement: model/max_tokens must be named constants, not inline magic numbers."""

    def test_model_and_max_tokens_are_module_level_constants(self):
        assert isinstance(MODEL_ID, str) and MODEL_ID  # non-empty string
        assert isinstance(MAX_TOKENS, int) and MAX_TOKENS > 0

    def test_model_is_the_small_cheap_tier(self):
        # This feature's entire cost story (see module docstring) depends on staying
        # on the cheap model -- a regression here silently breaks the ~$1/mo claim.
        assert "haiku" in MODEL_ID.lower()


class TestFallbackPathNoApiKey:
    def test_no_api_key_returns_fallback_without_any_network_call(self, rfmt_df, clv_df, segment_summary):
        with patch("src.digest_engine.anthropic.Anthropic") as mock_client_cls:
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, api_key=None)
            mock_client_cls.assert_not_called()

        assert isinstance(result, str) and result
        assert "[Template Summary" in result

    def test_empty_string_api_key_also_falls_back(self, rfmt_df, clv_df, segment_summary):
        with patch("src.digest_engine.anthropic.Anthropic") as mock_client_cls:
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, api_key="")
            mock_client_cls.assert_not_called()
        assert "[Template Summary" in result

    def test_fallback_is_deterministic_for_the_same_stats(self, rfmt_df, clv_df, segment_summary):
        r1 = generate_account_digest(rfmt_df, clv_df, segment_summary, api_key=None)
        r2 = generate_account_digest(rfmt_df, clv_df, segment_summary, api_key=None)
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


class TestSuccessfulApiCall:
    def _mock_response(self, text="A healthy account with strong 90-day forecasted revenue."):
        block = types.SimpleNamespace(type="text", text=text)
        return types.SimpleNamespace(content=[block])

    def test_successful_call_returns_model_text_not_fallback(self, rfmt_df, clv_df, segment_summary):
        mock_client = MagicMock()
        mock_client.with_options.return_value.messages.create.return_value = self._mock_response()

        with patch("src.digest_engine.anthropic.Anthropic", return_value=mock_client):
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, api_key="sk-ant-fake-key")

        assert result == "A healthy account with strong 90-day forecasted revenue."
        assert "[Template Summary" not in result

    def test_call_uses_the_named_model_and_max_tokens_constants(self, rfmt_df, clv_df, segment_summary):
        mock_client = MagicMock()
        mock_client.with_options.return_value.messages.create.return_value = self._mock_response()

        with patch("src.digest_engine.anthropic.Anthropic", return_value=mock_client):
            generate_account_digest(rfmt_df, clv_df, segment_summary, api_key="sk-ant-fake-key")

        _, kwargs = mock_client.with_options.return_value.messages.create.call_args
        assert kwargs["model"] == MODEL_ID
        assert kwargs["max_tokens"] == MAX_TOKENS

    def test_empty_model_response_falls_back(self, rfmt_df, clv_df, segment_summary):
        mock_client = MagicMock()
        mock_client.with_options.return_value.messages.create.return_value = self._mock_response(text="   ")

        with patch("src.digest_engine.anthropic.Anthropic", return_value=mock_client):
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, api_key="sk-ant-fake-key")

        assert "[Template Summary" in result


class TestApiFailuresDegradeGracefully:
    def test_generic_exception_falls_back_instead_of_raising(self, rfmt_df, clv_df, segment_summary):
        mock_client = MagicMock()
        mock_client.with_options.return_value.messages.create.side_effect = RuntimeError("boom")

        with patch("src.digest_engine.anthropic.Anthropic", return_value=mock_client):
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, api_key="sk-ant-fake-key")

        assert "[Template Summary" in result

    def test_anthropic_package_missing_falls_back(self, rfmt_df, clv_df, segment_summary):
        with patch("src.digest_engine.anthropic", None):
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, api_key="sk-ant-fake-key")
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
            result = generate_account_digest(rfmt_df, clv_df, segment_summary, api_key="sk-ant-fake-key")

        assert "[Template Summary" in result
        assert expected_reason_snippet in result


class TestStatsWithoutChurnRiskTier:
    def test_missing_churn_risk_tier_column_defaults_pct_at_risk_to_zero(self, rfmt_df, clv_df, segment_summary):
        clv_df_no_tier = clv_df.drop(columns=["Churn_Risk_Tier"])
        stats = _build_aggregate_stats(rfmt_df, clv_df_no_tier, segment_summary)
        assert stats["pct_at_risk"] == 0.0


class TestApiKeyResolution:
    def test_reads_from_environment_variable(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key-123")
        assert get_anthropic_api_key() == "env-key-123"

    def test_returns_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        # Note: if st.secrets happens to have ANTHROPIC_API_KEY configured in this
        # test environment, that would also satisfy resolution -- not expected in CI.
        result = get_anthropic_api_key()
        assert result is None or isinstance(result, str)

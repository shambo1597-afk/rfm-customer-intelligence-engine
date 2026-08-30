"""
tests/test_cohort_narration.py - pytest suite for the deterministic cohort-
pattern finder (src/cohort_engine.py's find_notable_cohort_pattern()) and the
optional AI narration built on top of it (src/cohort_narration.py).

No real network calls are made anywhere in this file -- both provider clients
(Anthropic, Groq) are always mocked (unittest.mock) when a "successful call" or
"provider failure" path is exercised, and the no-key/no-pattern fallback paths
are tested precisely because they must NOT construct a client or attempt any
network call at all.
"""

import types
from unittest.mock import MagicMock, patch

import groq
import httpx
import numpy as np
import pandas as pd
import pytest

from src.cohort_engine import find_notable_cohort_pattern, NOTABLE_COHORT_MIN_COLUMN_SAMPLES
from src.cohort_narration import (
    narrate_cohort_pattern,
    COHORT_NARRATION_MAX_OUTPUT_TOKENS,
    GROQ_COHORT_NARRATION_MAX_OUTPUT_TOKENS,
)


def _make_groq_api_error(status_code: int, message: str = "error"):
    """Same helper, duplicated per this codebase's established per-test-file
    convention -- see tests/test_digest_engine.py / tests/test_chat_engine.py
    / tests/test_roi_advisor.py's identical copies."""
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status_code=status_code, request=request)
    client = groq.Groq(api_key="dummy-key-for-error-construction")
    return client._make_status_error(message, body=None, response=response)


class TestFindNotableCohortPattern:
    """
    A synthetic retention matrix with a known, deliberately-injected anomaly
    -- asserts the function identifies THAT exact cell, not just that it
    returns something.
    """

    def _matrix_with_injected_anomaly(self):
        # 5 cohorts x Month 0-3. Month 2's values are all clustered around
        # 50%, EXCEPT Cohort D, which is deliberately anomalous at 5%.
        # Month 0 is always 100% (excluded by the function itself). Month 1
        # and Month 3 are flat (0 std -- skipped). Month 2 is the only
        # column with real signal, and Cohort D/Month 2 is the one cell that
        # should win.
        return pd.DataFrame(
            {
                0: [100.0, 100.0, 100.0, 100.0, 100.0],
                1: [80.0, 80.0, 80.0, 80.0, 80.0],
                2: [50.0, 52.0, 49.0, 5.0, 51.0],
                3: [30.0, 30.0, 30.0, 30.0, 30.0],
            },
            index=["Cohort A", "Cohort B", "Cohort C", "Cohort D", "Cohort E"],
        )

    def test_identifies_the_exact_injected_anomalous_cell(self):
        matrix = self._matrix_with_injected_anomaly()
        result = find_notable_cohort_pattern(matrix)

        assert result["cohort"] == "Cohort D"
        assert result["month_index"] == 2
        assert result["retention_pct"] == pytest.approx(5.0)
        assert result["direction"] == "unusually low"

    def test_column_mean_and_deviation_are_computed_from_the_actual_column(self):
        matrix = self._matrix_with_injected_anomaly()
        result = find_notable_cohort_pattern(matrix)

        month2 = matrix[2]
        expected_mean = float(month2.mean())
        expected_std = float(month2.std(ddof=0))
        expected_z = (5.0 - expected_mean) / expected_std

        assert result["column_mean_pct"] == pytest.approx(round(expected_mean, 1))
        assert result["z_score"] == pytest.approx(round(expected_z, 2))
        assert result["deviation_pct_points"] == pytest.approx(round(5.0 - expected_mean, 1))

    def test_month_0_is_always_excluded_even_though_its_values_never_vary(self):
        # Every cohort is 100% at Month 0 by construction -- std is exactly
        # 0 there, so it would be skipped anyway, but this test pins that
        # behavior explicitly rather than relying on it as a side effect.
        matrix = self._matrix_with_injected_anomaly()
        result = find_notable_cohort_pattern(matrix)
        assert result["month_index"] != 0

    def test_unusually_high_direction_is_reported_correctly(self):
        matrix = pd.DataFrame(
            {
                0: [100.0, 100.0, 100.0, 100.0],
                1: [40.0, 41.0, 39.0, 95.0],  # Cohort D unusually HIGH
            },
            index=["Cohort A", "Cohort B", "Cohort C", "Cohort D"],
        )
        result = find_notable_cohort_pattern(matrix)
        assert result["cohort"] == "Cohort D"
        assert result["month_index"] == 1
        assert result["direction"] == "unusually high"
        assert result["z_score"] > 0

    def test_column_with_zero_std_is_skipped_not_a_false_positive(self):
        # Every cohort identical at Month 1 -- std=0, must be skipped
        # entirely (no meaningful "unusual" cell can exist there), leaving
        # Month 2 (which DOES have spread) as the only candidate.
        matrix = pd.DataFrame(
            {
                0: [100.0, 100.0, 100.0],
                1: [50.0, 50.0, 50.0],
                2: [60.0, 61.0, 20.0],
            },
            index=["Cohort A", "Cohort B", "Cohort C"],
        )
        result = find_notable_cohort_pattern(matrix)
        assert result["month_index"] == 2

    def test_column_with_fewer_than_min_samples_is_skipped(self):
        # Only 2 non-NaN cohorts at Month 1 (below
        # NOTABLE_COHORT_MIN_COLUMN_SAMPLES=3) -- must be skipped even
        # though it technically has nonzero std, leaving Month 2 (3+
        # cohorts) as the only usable column.
        assert NOTABLE_COHORT_MIN_COLUMN_SAMPLES == 3
        matrix = pd.DataFrame(
            {
                0: [100.0, 100.0, 100.0],
                1: [10.0, 90.0, np.nan],
                2: [60.0, 61.0, 20.0],
            },
            index=["Cohort A", "Cohort B", "Cohort C"],
        )
        result = find_notable_cohort_pattern(matrix)
        assert result["month_index"] == 2

    def test_nan_cells_future_months_are_excluded_not_treated_as_zero(self):
        # Cohort C hasn't reached Month 2 yet (NaN) -- must be excluded from
        # that column's mean/std, not counted as a 0% retention data point
        # (which would badly distort the statistics).
        matrix = pd.DataFrame(
            {
                0: [100.0, 100.0, 100.0],
                1: [50.0, 52.0, 48.0],
                2: [60.0, 62.0, np.nan],
            },
            index=["Cohort A", "Cohort B", "Cohort C"],
        )
        result = find_notable_cohort_pattern(matrix)
        # Month 2 only has 2 non-NaN values -- below the min-samples
        # threshold -- so Month 1 (3 real values) must win instead.
        assert result["month_index"] == 1

    def test_no_notable_pattern_returns_empty_dict_not_an_error(self):
        # Only Month 0 exists (always 100%, always excluded) -- nothing else
        # to compare.
        matrix = pd.DataFrame({0: [100.0, 100.0, 100.0]}, index=["A", "B", "C"])
        assert find_notable_cohort_pattern(matrix) == {}

    def test_none_or_empty_matrix_returns_empty_dict(self):
        assert find_notable_cohort_pattern(None) == {}
        assert find_notable_cohort_pattern(pd.DataFrame()) == {}

    def test_real_pipeline_fixture_returns_a_well_formed_result_or_empty(self, cohort_retention_matrix):
        # Sanity check against the real bundled dataset's fixture
        # (conftest.py) -- either a well-formed dict or {} (both are valid
        # outcomes; this just confirms it never raises on real data and, if
        # non-empty, has the documented shape).
        result = find_notable_cohort_pattern(cohort_retention_matrix)
        if result:
            assert set(result.keys()) == {
                "cohort", "month_index", "retention_pct", "column_mean_pct",
                "column_std_pct", "deviation_pct_points", "z_score", "direction",
            }
            assert result["month_index"] != 0
            assert result["direction"] in ("unusually high", "unusually low")


@pytest.fixture
def sample_pattern():
    return {
        "cohort": "2026-03",
        "month_index": 2,
        "retention_pct": 5.0,
        "column_mean_pct": 50.5,
        "column_std_pct": 1.2,
        "deviation_pct_points": -45.5,
        "z_score": -3.1,
        "direction": "unusually low",
    }


class TestNarrateCohortPatternNoPatternOrNoKey:
    def test_empty_pattern_falls_back_without_resolving_a_provider_or_any_network_call(self):
        with patch("src.cohort_narration.anthropic.Anthropic") as mock_anthropic_cls, \
             patch("src.cohort_narration.groq.Groq") as mock_groq_cls:
            result = narrate_cohort_pattern({}, groq_api_key="fake-key", anthropic_api_key="fake-key")
            mock_anthropic_cls.assert_not_called()
            mock_groq_cls.assert_not_called()
        assert "temporarily unavailable" in result
        assert "no notable cohort pattern to narrate" in result

    def test_none_pattern_also_falls_back(self):
        result = narrate_cohort_pattern(None)
        assert "temporarily unavailable" in result

    def test_no_keys_falls_back_without_any_network_call(self, sample_pattern):
        with patch("src.cohort_narration.anthropic.Anthropic") as mock_anthropic_cls, \
             patch("src.cohort_narration.groq.Groq") as mock_groq_cls:
            result = narrate_cohort_pattern(sample_pattern)
            mock_anthropic_cls.assert_not_called()
            mock_groq_cls.assert_not_called()
        assert "temporarily unavailable" in result
        assert "no GROQ_API_KEY or ANTHROPIC_API_KEY configured" in result


class TestSuccessfulCalls:
    def test_successful_groq_call_returns_model_text(self, sample_pattern):
        mock_response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(
                content="The 2026-03 cohort's month-2 retention is unusually low."
            ))]
        )
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.return_value = mock_response

        with patch("src.cohort_narration.groq.Groq", return_value=mock_client):
            result = narrate_cohort_pattern(sample_pattern, groq_api_key="fake-groq-key")

        assert result == "The 2026-03 cohort's month-2 retention is unusually low."

    def test_groq_prompt_contains_the_actual_pattern_figures_not_invented_ones(self, sample_pattern):
        mock_response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="An explanation."))]
        )
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.return_value = mock_response

        with patch("src.cohort_narration.groq.Groq", return_value=mock_client):
            narrate_cohort_pattern(sample_pattern, groq_api_key="fake-groq-key")

        _, kwargs = mock_client.with_options.return_value.chat.completions.create.call_args
        prompt = kwargs["messages"][0]["content"]
        assert "2026-03" in prompt
        assert "5.0%" in prompt
        assert "50.5%" in prompt
        assert "Do not invent any numbers" in prompt
        # The Groq call gets COHORT_NARRATION_MAX_OUTPUT_TOKENS PLUS
        # reasoning headroom (GROQ_COHORT_NARRATION_MAX_OUTPUT_TOKENS), not
        # the bare visible-answer target -- see
        # GROQ_REASONING_TOKEN_HEADROOM's comment in digest_engine.py. This
        # is literally the constant whose original, headroom-free value
        # (150) caused the live truncation bug this fix addresses.
        assert kwargs["max_tokens"] == GROQ_COHORT_NARRATION_MAX_OUTPUT_TOKENS

    def test_successful_anthropic_call_returns_model_text(self, sample_pattern):
        block = types.SimpleNamespace(type="text", text="A notable drop was found.")
        mock_response = types.SimpleNamespace(content=[block])
        mock_client = MagicMock()
        mock_client.with_options.return_value.messages.create.return_value = mock_response

        with patch("src.cohort_narration.anthropic.Anthropic", return_value=mock_client):
            result = narrate_cohort_pattern(sample_pattern, anthropic_api_key="sk-ant-fake-key")

        assert result == "A notable drop was found."

    def test_anthropic_call_uses_the_unmodified_visible_answer_target(self, sample_pattern):
        # Regression guard: Claude Haiku is not a reasoning model and was
        # never at risk of this bug -- its call must keep using the
        # original, un-inflated COHORT_NARRATION_MAX_OUTPUT_TOKENS, not the
        # Groq-specific headroom-added constant.
        block = types.SimpleNamespace(type="text", text="An explanation.")
        mock_response = types.SimpleNamespace(content=[block])
        mock_client = MagicMock()
        mock_client.with_options.return_value.messages.create.return_value = mock_response

        with patch("src.cohort_narration.anthropic.Anthropic", return_value=mock_client):
            narrate_cohort_pattern(sample_pattern, anthropic_api_key="sk-ant-fake-key")

        _, kwargs = mock_client.with_options.return_value.messages.create.call_args
        assert kwargs["max_tokens"] == COHORT_NARRATION_MAX_OUTPUT_TOKENS


class TestFailuresDegradeGracefully:
    def test_groq_package_missing_falls_back(self, sample_pattern):
        with patch("src.cohort_narration.groq", None):
            result = narrate_cohort_pattern(sample_pattern, groq_api_key="fake-groq-key")
        assert "temporarily unavailable" in result
        assert "not installed" in result

    def test_anthropic_package_missing_falls_back(self, sample_pattern):
        with patch("src.cohort_narration.anthropic", None):
            result = narrate_cohort_pattern(sample_pattern, anthropic_api_key="sk-ant-fake-key")
        assert "temporarily unavailable" in result
        assert "not installed" in result

    def test_groq_rate_limit_falls_back_with_specific_reason(self, sample_pattern):
        exc = _make_groq_api_error(429, "quota exceeded")
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.side_effect = exc

        with patch("src.cohort_narration.groq.Groq", return_value=mock_client):
            result = narrate_cohort_pattern(sample_pattern, groq_api_key="fake-groq-key")

        assert "temporarily unavailable" in result
        assert "Groq free-tier rate limit reached" in result

    def test_groq_invalid_key_falls_back_with_specific_reason(self, sample_pattern):
        exc = _make_groq_api_error(401, "bad credentials")
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.side_effect = exc

        with patch("src.cohort_narration.groq.Groq", return_value=mock_client):
            result = narrate_cohort_pattern(sample_pattern, groq_api_key="bad-key")

        assert "temporarily unavailable" in result
        assert "invalid Groq API key" in result

    @pytest.mark.parametrize("exception_name,expected_reason_snippet", [
        ("AuthenticationError", "invalid Anthropic API key"),
        ("RateLimitError", "rate limited"),
        ("APIStatusError", "Anthropic API error"),
        ("APIConnectionError", "could not reach"),
    ])
    def test_each_typed_anthropic_exception_falls_back_with_its_own_reason(
        self, sample_pattern, exception_name, expected_reason_snippet
    ):
        import anthropic as real_anthropic

        exc_cls = getattr(real_anthropic, exception_name)
        exc_instance = exc_cls.__new__(exc_cls)

        mock_client = MagicMock()
        mock_client.with_options.return_value.messages.create.side_effect = exc_instance

        with patch("src.cohort_narration.anthropic.Anthropic", return_value=mock_client):
            result = narrate_cohort_pattern(sample_pattern, anthropic_api_key="sk-ant-fake-key")

        assert "temporarily unavailable" in result
        assert expected_reason_snippet in result

    def test_empty_model_response_falls_back(self, sample_pattern):
        mock_response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=""))]
        )
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.return_value = mock_response

        with patch("src.cohort_narration.groq.Groq", return_value=mock_client):
            result = narrate_cohort_pattern(sample_pattern, groq_api_key="fake-groq-key")

        assert "temporarily unavailable" in result

    def test_generic_exception_falls_back_instead_of_raising(self, sample_pattern):
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.side_effect = RuntimeError("boom")

        with patch("src.cohort_narration.groq.Groq", return_value=mock_client):
            result = narrate_cohort_pattern(sample_pattern, groq_api_key="fake-groq-key")

        assert "temporarily unavailable" in result
        assert "unexpected error" in result

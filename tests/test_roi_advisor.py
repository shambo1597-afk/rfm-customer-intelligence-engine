"""
tests/test_roi_advisor.py - pytest suite for the optional AI Budget Advisor
(src/roi_advisor.py).

No real network calls are made anywhere in this file -- both provider clients
(Anthropic, Groq) are always mocked (unittest.mock) when a "successful call" or
"provider failure" path is exercised, and the no-key fallback path is tested
precisely because it must NOT construct a client or attempt any network call at all.
"""

import types
from unittest.mock import MagicMock, patch

import groq
import httpx
import pandas as pd
import pytest

from src.roi_advisor import (
    simulate_campaign_roi,
    simulate_all_segment_allocations,
    build_roi_advisor_context,
    get_roi_recommendation,
    DEFAULT_CONV_RATE_PCT,
    DEFAULT_GROSS_MARGIN_PCT,
    ROI_ADVISOR_MAX_OUTPUT_TOKENS,
)


def _make_groq_api_error(status_code: int, message: str = "error"):
    """
    Builds a real groq exception instance via the REAL groq.Groq client's own
    `_make_status_error()` dispatch method -- same helper as
    tests/test_digest_engine.py and tests/test_chat_engine.py, duplicated
    here per this codebase's established per-test-file convention (see
    those files' identical copies).
    """
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status_code=status_code, request=request)
    client = groq.Groq(api_key="dummy-key-for-error-construction")
    return client._make_status_error(message, body=None, response=response)


class TestSimulateCampaignRoi:
    """
    Hand-calculated expected values -- same rigor already used for
    src/clv_engine.py's tests -- against README's "What-If Campaign ROI
    Simulation Framework" formulas (the exact math app.py's Tab 5 slider
    simulator used to compute inline, before it was extracted here).
    """

    def test_matches_hand_calculated_values(self):
        result = simulate_campaign_roi(
            audience_size=100,
            avg_segment_aov=200.0,
            campaign_budget=1000.0,
            conv_rate_pct=10.0,
            gross_margin_pct=50.0,
        )
        # projected_conversions = 100 * (10/100) = 10.0
        assert result["projected_conversions"] == pytest.approx(10.0)
        # projected_gross_revenue = 10.0 * 200.0 = 2000.0
        assert result["projected_gross_revenue"] == pytest.approx(2000.0)
        # projected_gross_profit = 2000.0 * (50/100) = 1000.0
        assert result["projected_gross_profit"] == pytest.approx(1000.0)
        # net_incremental_profit = 1000.0 - 1000.0 = 0.0
        assert result["net_incremental_profit"] == pytest.approx(0.0)
        # campaign_roi_pct = (0.0 / max(1000.0, 1)) * 100 = 0.0
        assert result["campaign_roi_pct"] == pytest.approx(0.0)
        # cost_per_acquisition = 1000.0 / max(10.0, 1) = 100.0
        assert result["cost_per_acquisition"] == pytest.approx(100.0)

    def test_returns_exactly_the_documented_keys(self):
        result = simulate_campaign_roi(
            audience_size=50, avg_segment_aov=100.0, campaign_budget=500.0,
            conv_rate_pct=5.0, gross_margin_pct=40.0,
        )
        assert set(result.keys()) == {
            "projected_conversions", "projected_gross_revenue", "projected_gross_profit",
            "net_incremental_profit", "campaign_roi_pct", "cost_per_acquisition",
        }

    def test_zero_conversions_uses_the_max_one_guard_for_cpa(self):
        # conv_rate_pct=0 -> projected_conversions=0 -> cost_per_acquisition
        # must use max(projected_conversions, 1), not divide by zero.
        result = simulate_campaign_roi(
            audience_size=100, avg_segment_aov=200.0, campaign_budget=1000.0,
            conv_rate_pct=0.0, gross_margin_pct=50.0,
        )
        assert result["projected_conversions"] == 0.0
        assert result["cost_per_acquisition"] == pytest.approx(1000.0)  # 1000.0 / max(0, 1)

    def test_negative_net_profit_yields_negative_roi_pct(self):
        result = simulate_campaign_roi(
            audience_size=10, avg_segment_aov=50.0, campaign_budget=1000.0,
            conv_rate_pct=10.0, gross_margin_pct=40.0,
        )
        # conversions = 1.0, revenue = 50.0, profit = 20.0, net = 20.0 - 1000.0 = -980.0
        assert result["net_incremental_profit"] == pytest.approx(-980.0)
        assert result["campaign_roi_pct"] == pytest.approx(-98.0)  # (-980/1000)*100


@pytest.fixture
def two_segment_fixture():
    """
    A minimal, hand-verifiable 2-segment fixture (2 customers each) --
    Champions AOV 100.0, Loyalists AOV 150.0 -- small enough that every
    resulting number in TestSimulateAllSegmentAllocations below is derived
    by hand, not just asserted to "be a number."
    """
    df = pd.DataFrame({
        "CustomerID": ["C1", "C2", "C3", "C4"],
        "Monetary": [100.0, 200.0, 300.0, 400.0],
        "Recency": [10, 20, 30, 40],
        "Frequency": [1, 2, 3, 4],
        "Tenure": [50, 60, 70, 80],
        "AvgOrderValue": [100.0, 100.0, 150.0, 150.0],
        "Segment": ["Champions", "Champions", "Loyalists", "Loyalists"],
    })
    return df, df.copy()  # (rfmt_df, clv_df) -- identical here, as in production


class TestSimulateAllSegmentAllocations:
    def test_budget_is_split_proportionally_to_segment_size(self, two_segment_fixture):
        rfmt_df, clv_df = two_segment_fixture
        result = simulate_all_segment_allocations(rfmt_df, clv_df, total_budget=1000.0)

        champions = result[result["segment"] == "Champions"].iloc[0]
        loyalists = result[result["segment"] == "Loyalists"].iloc[0]
        # Both segments have 2 of 4 customers -> 50/50 split.
        assert champions["allocated_budget"] == pytest.approx(500.0)
        assert loyalists["allocated_budget"] == pytest.approx(500.0)
        assert champions["customer_count"] == 2
        assert loyalists["customer_count"] == 2

    def test_avg_segment_aov_matches_the_segments_actual_customer_rows(self, two_segment_fixture):
        rfmt_df, clv_df = two_segment_fixture
        result = simulate_all_segment_allocations(rfmt_df, clv_df, total_budget=1000.0)

        champions = result[result["segment"] == "Champions"].iloc[0]
        loyalists = result[result["segment"] == "Loyalists"].iloc[0]
        assert champions["avg_segment_aov"] == pytest.approx(100.0)
        assert loyalists["avg_segment_aov"] == pytest.approx(150.0)

    def test_default_conversion_rate_matches_app_pys_tab_5_slider_default(self, two_segment_fixture):
        rfmt_df, clv_df = two_segment_fixture
        result = simulate_all_segment_allocations(rfmt_df, clv_df, total_budget=1000.0)

        champions = result[result["segment"] == "Champions"].iloc[0]
        # audience_size=2, conv_rate=DEFAULT_CONV_RATE_PCT (8.5), aov=100.0,
        # budget=500.0, margin=DEFAULT_GROSS_MARGIN_PCT (40) -- matches
        # simulate_campaign_roi() called directly with the same inputs.
        expected = simulate_campaign_roi(
            audience_size=2, avg_segment_aov=100.0, campaign_budget=500.0,
            conv_rate_pct=DEFAULT_CONV_RATE_PCT, gross_margin_pct=DEFAULT_GROSS_MARGIN_PCT,
        )
        assert champions["projected_conversions"] == pytest.approx(expected["projected_conversions"])
        assert champions["net_incremental_profit"] == pytest.approx(expected["net_incremental_profit"])
        assert champions["campaign_roi_pct"] == pytest.approx(expected["campaign_roi_pct"])
        assert champions["cost_per_acquisition"] == pytest.approx(expected["cost_per_acquisition"])

    def test_conv_rate_by_segment_override_applies_only_to_the_named_segment(self, two_segment_fixture):
        rfmt_df, clv_df = two_segment_fixture
        result = simulate_all_segment_allocations(
            rfmt_df, clv_df, total_budget=1000.0, conv_rate_by_segment={"Champions": 20.0}
        )
        champions = result[result["segment"] == "Champions"].iloc[0]
        loyalists = result[result["segment"] == "Loyalists"].iloc[0]

        # Champions: audience=2, conv=20% -> projected_conversions = 2*0.20 = 0.4
        assert champions["projected_conversions"] == pytest.approx(0.4)
        # Loyalists untouched -- still uses DEFAULT_CONV_RATE_PCT (8.5%).
        assert loyalists["projected_conversions"] == pytest.approx(2 * (DEFAULT_CONV_RATE_PCT / 100.0))

    def test_returns_one_row_per_segment_present_in_rfmt_df(self, two_segment_fixture):
        rfmt_df, clv_df = two_segment_fixture
        result = simulate_all_segment_allocations(rfmt_df, clv_df, total_budget=1000.0)
        assert set(result["segment"]) == {"Champions", "Loyalists"}
        assert len(result) == 2

    def test_returns_the_documented_columns(self, two_segment_fixture):
        rfmt_df, clv_df = two_segment_fixture
        result = simulate_all_segment_allocations(rfmt_df, clv_df, total_budget=1000.0)
        assert set(result.columns) == {
            "segment", "customer_count", "allocated_budget", "avg_segment_aov",
            "projected_conversions", "projected_gross_profit", "net_incremental_profit",
            "campaign_roi_pct", "cost_per_acquisition",
        }

    def test_empty_segment_in_clv_df_falls_back_to_the_150_default_aov(self, two_segment_fixture):
        # Mirrors app.py Tab 5's own fallback exactly: a segment with zero
        # rows in clv_df (e.g. present in rfmt_df's Segment categories but
        # with no matching customers in this particular clv_df slice) uses
        # 150.0, not NaN/an error.
        rfmt_df, clv_df = two_segment_fixture
        clv_df_missing_loyalists = clv_df[clv_df["Segment"] != "Loyalists"]
        result = simulate_all_segment_allocations(rfmt_df, clv_df_missing_loyalists, total_budget=1000.0)
        loyalists = result[result["segment"] == "Loyalists"].iloc[0]
        assert loyalists["avg_segment_aov"] == pytest.approx(150.0)

    def test_real_pipeline_fixture_covers_every_segment_present(self, rfmt_df, clv_df):
        # Full end-to-end sanity check against the real bundled dataset's
        # rfmt_df/clv_df fixtures (conftest.py) -- every segment actually
        # present gets exactly one row, and the budget shares sum to 1.0.
        result = simulate_all_segment_allocations(rfmt_df, clv_df, total_budget=5000.0)
        assert set(result["segment"]) == set(rfmt_df["Segment"].astype(str).unique())
        assert result["allocated_budget"].sum() == pytest.approx(5000.0)


class TestBuildRoiAdvisorContext:
    def test_context_contains_total_budget_and_every_segments_figures(self, two_segment_fixture):
        rfmt_df, clv_df = two_segment_fixture
        allocations_df = simulate_all_segment_allocations(rfmt_df, clv_df, total_budget=1000.0)
        text = build_roi_advisor_context(allocations_df, total_budget=1000.0)

        assert "$1,000.00" in text
        for _, row in allocations_df.iterrows():
            assert row["segment"] in text
            assert f"${row['allocated_budget']:,.2f}" in text
            assert f"${row['avg_segment_aov']:,.2f}" in text
            assert f"${row['net_incremental_profit']:,.2f}" in text

    def test_empty_allocations_degrades_to_a_no_data_message_not_an_error(self):
        text = build_roi_advisor_context(pd.DataFrame(), total_budget=500.0)
        assert "$500.00" in text
        assert "no segment allocation data available" in text


class TestSystemPromptInstructsUsingOnlyProvidedTable:
    """
    The task's explicit requirement: confirm the system prompt instructs the
    model to use ONLY the provided allocations table -- mirrors
    tests/test_chat_engine.py's "answer only from context" tests.
    """

    def test_groq_system_prompt_forbids_inventing_or_recomputing_figures(self, two_segment_fixture):
        rfmt_df, clv_df = two_segment_fixture
        allocations_df = simulate_all_segment_allocations(rfmt_df, clv_df, total_budget=1000.0)

        mock_response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="A recommendation."))]
        )
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.return_value = mock_response

        with patch("src.roi_advisor.groq.Groq", return_value=mock_client):
            get_roi_recommendation(
                "Recommend an allocation.", allocations_df, 1000.0, groq_api_key="fake-groq-key"
            )

        _, kwargs = mock_client.with_options.return_value.chat.completions.create.call_args
        system_msg = kwargs["messages"][0]["content"]
        assert "ONLY the numbers in this table" in system_msg
        assert "do not calculate a new ROI" in system_msg
        # The actual table figures (not invented ones) must be present.
        for _, row in allocations_df.iterrows():
            assert row["segment"] in system_msg

    def test_anthropic_system_prompt_carries_the_same_instruction(self, two_segment_fixture):
        rfmt_df, clv_df = two_segment_fixture
        allocations_df = simulate_all_segment_allocations(rfmt_df, clv_df, total_budget=1000.0)

        block = types.SimpleNamespace(type="text", text="A recommendation.")
        mock_response = types.SimpleNamespace(content=[block])
        mock_client = MagicMock()
        mock_client.with_options.return_value.messages.create.return_value = mock_response

        with patch("src.roi_advisor.anthropic.Anthropic", return_value=mock_client):
            get_roi_recommendation(
                "Recommend an allocation.", allocations_df, 1000.0, anthropic_api_key="sk-ant-fake-key"
            )

        _, kwargs = mock_client.with_options.return_value.messages.create.call_args
        assert "ONLY the numbers in this table" in kwargs["system"]
        assert kwargs["messages"] == [{"role": "user", "content": "Recommend an allocation."}]


class TestSuccessfulCalls:
    def test_successful_groq_call_returns_model_text(self, two_segment_fixture):
        rfmt_df, clv_df = two_segment_fixture
        allocations_df = simulate_all_segment_allocations(rfmt_df, clv_df, total_budget=1000.0)

        mock_response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="Split evenly."))]
        )
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.return_value = mock_response

        with patch("src.roi_advisor.groq.Groq", return_value=mock_client):
            result = get_roi_recommendation(
                "Recommend an allocation.", allocations_df, 1000.0, groq_api_key="fake-groq-key"
            )

        assert result == "Split evenly."
        assert "unavailable" not in result

    def test_groq_call_uses_the_named_max_tokens_constant(self, two_segment_fixture):
        rfmt_df, clv_df = two_segment_fixture
        allocations_df = simulate_all_segment_allocations(rfmt_df, clv_df, total_budget=1000.0)

        mock_response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="A recommendation."))]
        )
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.return_value = mock_response

        with patch("src.roi_advisor.groq.Groq", return_value=mock_client):
            get_roi_recommendation("A question.", allocations_df, 1000.0, groq_api_key="fake-groq-key")

        _, kwargs = mock_client.with_options.return_value.chat.completions.create.call_args
        assert kwargs["max_tokens"] == ROI_ADVISOR_MAX_OUTPUT_TOKENS

    def test_successful_anthropic_call_returns_model_text(self, two_segment_fixture):
        rfmt_df, clv_df = two_segment_fixture
        allocations_df = simulate_all_segment_allocations(rfmt_df, clv_df, total_budget=1000.0)

        block = types.SimpleNamespace(type="text", text="Prioritize Champions.")
        mock_response = types.SimpleNamespace(content=[block])
        mock_client = MagicMock()
        mock_client.with_options.return_value.messages.create.return_value = mock_response

        with patch("src.roi_advisor.anthropic.Anthropic", return_value=mock_client):
            result = get_roi_recommendation(
                "Recommend an allocation.", allocations_df, 1000.0, anthropic_api_key="sk-ant-fake-key"
            )

        assert result == "Prioritize Champions."


class TestFailuresDegradeGracefully:
    def test_no_keys_falls_back_without_any_network_call(self, two_segment_fixture):
        rfmt_df, clv_df = two_segment_fixture
        allocations_df = simulate_all_segment_allocations(rfmt_df, clv_df, total_budget=1000.0)

        with patch("src.roi_advisor.anthropic.Anthropic") as mock_anthropic_cls, \
             patch("src.roi_advisor.groq.Groq") as mock_groq_cls:
            result = get_roi_recommendation("A question.", allocations_df, 1000.0)
            mock_anthropic_cls.assert_not_called()
            mock_groq_cls.assert_not_called()

        assert "temporarily unavailable" in result
        assert "no GROQ_API_KEY or ANTHROPIC_API_KEY configured" in result

    def test_groq_package_missing_falls_back(self, two_segment_fixture):
        rfmt_df, clv_df = two_segment_fixture
        allocations_df = simulate_all_segment_allocations(rfmt_df, clv_df, total_budget=1000.0)
        with patch("src.roi_advisor.groq", None):
            result = get_roi_recommendation("A question.", allocations_df, 1000.0, groq_api_key="fake-groq-key")
        assert "temporarily unavailable" in result
        assert "not installed" in result

    def test_anthropic_package_missing_falls_back(self, two_segment_fixture):
        rfmt_df, clv_df = two_segment_fixture
        allocations_df = simulate_all_segment_allocations(rfmt_df, clv_df, total_budget=1000.0)
        with patch("src.roi_advisor.anthropic", None):
            result = get_roi_recommendation("A question.", allocations_df, 1000.0, anthropic_api_key="sk-ant-fake-key")
        assert "temporarily unavailable" in result
        assert "not installed" in result

    def test_groq_rate_limit_falls_back_with_specific_reason(self, two_segment_fixture):
        rfmt_df, clv_df = two_segment_fixture
        allocations_df = simulate_all_segment_allocations(rfmt_df, clv_df, total_budget=1000.0)

        exc = _make_groq_api_error(429, "quota exceeded")
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.side_effect = exc

        with patch("src.roi_advisor.groq.Groq", return_value=mock_client):
            result = get_roi_recommendation("A question.", allocations_df, 1000.0, groq_api_key="fake-groq-key")

        assert "temporarily unavailable" in result
        assert "Groq free-tier rate limit reached" in result

    def test_groq_invalid_key_falls_back_with_specific_reason(self, two_segment_fixture):
        rfmt_df, clv_df = two_segment_fixture
        allocations_df = simulate_all_segment_allocations(rfmt_df, clv_df, total_budget=1000.0)

        exc = _make_groq_api_error(401, "bad credentials")
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.side_effect = exc

        with patch("src.roi_advisor.groq.Groq", return_value=mock_client):
            result = get_roi_recommendation("A question.", allocations_df, 1000.0, groq_api_key="bad-key")

        assert "temporarily unavailable" in result
        assert "invalid Groq API key" in result

    def test_groq_timeout_falls_back_with_its_own_distinct_reason(self, two_segment_fixture):
        rfmt_df, clv_df = two_segment_fixture
        allocations_df = simulate_all_segment_allocations(rfmt_df, clv_df, total_budget=1000.0)

        exc = groq.APITimeoutError(request=httpx.Request(
            "POST", "https://api.groq.com/openai/v1/chat/completions"
        ))
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.side_effect = exc

        with patch("src.roi_advisor.groq.Groq", return_value=mock_client):
            result = get_roi_recommendation("A question.", allocations_df, 1000.0, groq_api_key="fake-groq-key")

        assert "temporarily unavailable" in result
        assert "Groq API request timed out" in result

    @pytest.mark.parametrize("exception_name,expected_reason_snippet", [
        ("AuthenticationError", "invalid Anthropic API key"),
        ("RateLimitError", "rate limited"),
        ("APIStatusError", "Anthropic API error"),
        ("APIConnectionError", "could not reach"),
    ])
    def test_each_typed_anthropic_exception_falls_back_with_its_own_reason(
        self, two_segment_fixture, exception_name, expected_reason_snippet
    ):
        import anthropic as real_anthropic

        rfmt_df, clv_df = two_segment_fixture
        allocations_df = simulate_all_segment_allocations(rfmt_df, clv_df, total_budget=1000.0)

        exc_cls = getattr(real_anthropic, exception_name)
        exc_instance = exc_cls.__new__(exc_cls)

        mock_client = MagicMock()
        mock_client.with_options.return_value.messages.create.side_effect = exc_instance

        with patch("src.roi_advisor.anthropic.Anthropic", return_value=mock_client):
            result = get_roi_recommendation("A question.", allocations_df, 1000.0, anthropic_api_key="sk-ant-fake-key")

        assert "temporarily unavailable" in result
        assert expected_reason_snippet in result

    def test_empty_model_response_falls_back(self, two_segment_fixture):
        rfmt_df, clv_df = two_segment_fixture
        allocations_df = simulate_all_segment_allocations(rfmt_df, clv_df, total_budget=1000.0)

        mock_response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="   "))]
        )
        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.return_value = mock_response

        with patch("src.roi_advisor.groq.Groq", return_value=mock_client):
            result = get_roi_recommendation("A question.", allocations_df, 1000.0, groq_api_key="fake-groq-key")

        assert "temporarily unavailable" in result

    def test_generic_exception_falls_back_instead_of_raising(self, two_segment_fixture):
        rfmt_df, clv_df = two_segment_fixture
        allocations_df = simulate_all_segment_allocations(rfmt_df, clv_df, total_budget=1000.0)

        mock_client = MagicMock()
        mock_client.with_options.return_value.chat.completions.create.side_effect = RuntimeError("boom")

        with patch("src.roi_advisor.groq.Groq", return_value=mock_client):
            result = get_roi_recommendation("A question.", allocations_df, 1000.0, groq_api_key="fake-groq-key")

        assert "temporarily unavailable" in result
        assert "unexpected error" in result

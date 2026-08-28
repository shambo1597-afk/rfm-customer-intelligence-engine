"""
src/digest_engine.py - Optional per-account AI Executive Summary.

Generates ONE natural-language executive-summary paragraph per account, after the
batch RFM-T/CLV scoring run completes, built entirely from already-computed
AGGREGATE output (segment counts, % at-risk, total forecasted revenue, top
segments) -- never raw per-customer rows. This keeps the prompt small and keeps
customer PII (IDs, names, individual transaction history) out of the API call
entirely.

This is a narrative WRAPPER on existing output, not a new inference step: no
per-record LLM calls are made anywhere in this pipeline. RFM-T, K-Means, PCA, and
the CLV/churn model all remain 100% baked-in/zero-API, exactly as before this
feature existed -- see README "AI Executive Summary (Optional)".

--- Cost model this was designed against ---

This design's whole point is captured here so the reasoning survives in the code,
not just in a slide, and doesn't get silently violated by a well-intentioned
"let's personalize it per customer" change later.

  Per-end-customer digest (REJECTED): one LLM call per end-customer, per batch run.
    At an 8-account pilot averaging ~450 end-customers/account, one batch run/day:
        8 accounts * 450 customers = 3,600 calls/day = ~108,000 calls/month.
    Even at Haiku's cheap per-1M-token rate with a tiny ~150-in/~60-out-token
    prompt, that's real money at this call volume, and per-request overhead (rate
    limiting, latency, retries) at 100K+ calls/month pushes realistic pilot cost
    toward ~$208/month -- for a summary granularity (one paragraph per end-customer)
    nobody on the account actually reads end to end.

  Per-account digest (THIS DESIGN): one LLM call per ACCOUNT, per batch run.
        8 accounts * 1 call/day = 8 calls/day = ~240 calls/month.
    At the same per-call token cost, that's under $1/month at pilot volume --
    roughly two orders of magnitude cheaper, for materially the same executive-
    facing value: a marketer wants one summary of their whole book of business per
    batch run, not one summary per customer.

Both figures assume Claude Haiku 4.5 (MODEL_ID below, $1.00 / $5.00 per 1M
input/output tokens) -- the "small/cheap model" this feature is deliberately built
around. Do not swap MODEL_ID for a larger model without re-deriving this comparison.
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

# Small/cheap model by design -- see the module docstring's cost model above.
MODEL_ID = "claude-haiku-4-5"
# Short paragraph only; also bounds worst-case per-call cost.
MAX_TOKENS = 300
REQUEST_TIMEOUT_SECONDS = 20.0

# How many of the largest segments to name in the prompt (keeps the prompt compact).
SEGMENTS_SHOWN_IN_PROMPT = 3

# Must match src/clv_engine.py's Churn_Risk_Tier label for the high-risk bucket.
HIGH_CHURN_RISK_TIER = "🔴 High Churn Risk"


def get_anthropic_api_key():
    """
    Resolves an Anthropic API key from st.secrets first, then the ANTHROPIC_API_KEY
    environment variable. Never hardcoded, never logged. Returns None if neither is
    configured -- callers should treat that as "AI Digest unavailable, use the
    fallback template", not as an error.
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
        f"Configure ANTHROPIC_API_KEY to enable the AI-generated version of this summary."
    )


def generate_account_digest(rfmt_df, clv_df, segment_summary, api_key: str = None) -> str:
    """
    Generates ONE natural-language executive summary paragraph for an account, from
    already-computed aggregate output only -- see the module docstring for the full
    cost-model rationale (one call per account per batch run, never per end-customer).

    Parameters:
    - rfmt_df, clv_df, segment_summary: the already-computed pipeline outputs for
      this account (see src/rfm_engine.py, src/clv_engine.py). Only aggregate
      statistics are ever read from them -- never raw per-customer rows.
    - api_key: an Anthropic API key. If falsy (None/empty), or if the API call fails
      for ANY reason (network error, rate limit, invalid key, empty response, ...),
      this function ALWAYS returns a clearly-labeled fallback string built from the
      same aggregate stats via an f-string template -- it never raises, so the
      feature degrades gracefully and the app never breaks without a key.
    """
    stats = _build_aggregate_stats(rfmt_df, clv_df, segment_summary)

    if not api_key:
        return _fallback_digest(stats, reason="no ANTHROPIC_API_KEY configured")
    if anthropic is None:
        return _fallback_digest(stats, reason="anthropic package not installed")

    prompt = _build_prompt(stats)
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.with_options(timeout=REQUEST_TIMEOUT_SECONDS).messages.create(
            model=MODEL_ID,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        return text if text else _fallback_digest(stats, reason="empty response from the model")
    except anthropic.AuthenticationError:
        return _fallback_digest(stats, reason="invalid Anthropic API key")
    except anthropic.RateLimitError:
        return _fallback_digest(stats, reason="Anthropic API rate limited")
    except anthropic.APIStatusError:
        return _fallback_digest(stats, reason="Anthropic API error")
    except anthropic.APIConnectionError:
        return _fallback_digest(stats, reason="could not reach the Anthropic API")
    except Exception:
        # Last-resort safety net -- this optional feature must never crash the app.
        return _fallback_digest(stats, reason="unexpected error generating the AI summary")

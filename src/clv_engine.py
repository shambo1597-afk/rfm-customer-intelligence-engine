"""
src/clv_engine.py - Probabilistic Customer Lifetime Value (CLV) & Real-Time Churn Radar
Features:
1. Probabilistic Buy-Till-You-Die (BTYD / BG-NBD style) P(Alive) estimation.
2. Expected 90-Day Transaction Volume Forecasting.
3. 90-Day Predictive Customer Lifetime Value (CLV $) calculation.
4. Urgent Intervention Churn Watchlist for high-value decaying customers.
"""

import numpy as np
import pandas as pd

# Empirical Bayesian smoothing priors for the per-day transaction rate (lambda).
# alpha_prior/beta_prior act as a pseudo-count / pseudo-duration that pulls thin
# purchase histories toward a plausible baseline cadence rather than an extreme
# early estimate.
DEFAULT_ALPHA_PRIOR = 1.2
DEFAULT_BETA_PRIOR_DAYS = 60.0

# Logistic churn-hazard coefficients. `churn_hazard` grows (pushing P(Alive) down)
# as a customer misses more of their own expected purchase cycles, and as their
# inactive time becomes a larger share of their total tenure. The *_offset terms
# set the "on schedule" break-even point; the *_weight terms set how sharply the
# hazard escalates once that point is crossed.
HAZARD_MISSED_CYCLES_WEIGHT = 1.4
HAZARD_MISSED_CYCLES_OFFSET = 1.8
HAZARD_INACTIVITY_RATIO_WEIGHT = 1.2
HAZARD_INACTIVITY_RATIO_OFFSET = 0.4

# Churn Risk Tier P(Alive) cut points.
CHURN_TIER_LOW_RISK_THRESHOLD = 0.75
CHURN_TIER_MODERATE_THRESHOLD = 0.45


def estimate_btyd_clv(
    rfmt_df: pd.DataFrame,
    prediction_horizon_days: int = 90,
    gross_margin: float = 0.35,
    alpha_prior: float = DEFAULT_ALPHA_PRIOR,
    beta_prior_days: float = DEFAULT_BETA_PRIOR_DAYS,
    hazard_missed_cycles_weight: float = HAZARD_MISSED_CYCLES_WEIGHT,
    hazard_missed_cycles_offset: float = HAZARD_MISSED_CYCLES_OFFSET,
    hazard_inactivity_ratio_weight: float = HAZARD_INACTIVITY_RATIO_WEIGHT,
    hazard_inactivity_ratio_offset: float = HAZARD_INACTIVITY_RATIO_OFFSET,
) -> pd.DataFrame:
    """
    Computes a heuristic, BTYD-inspired P(Alive), 90-day expected purchase frequency,
    and 90-day predicted CLV for each customer.

    IMPORTANT: this is not a fitted probabilistic model (no likelihood maximization, no
    posterior estimation) — it's a hand-tuned logistic hazard function shaped after BTYD
    models. The alpha_prior/beta_prior_days/hazard_* defaults below were calibrated to
    produce plausible-looking P(Alive) distributions on this project's synthetic dataset,
    not fit against real churn outcomes. See backtest_clv.py and the README's "Model
    Validation" section for out-of-sample accuracy numbers (they are mediocre-to-poor on
    real transaction data) and treat every value as a *deployment-specific calibration
    target*, not a universal constant. All six are exposed as parameters (rather than
    only module-level constants) specifically so a real deployment can recalibrate them
    against its own known-outcome data instead of trusting these synthetic-data defaults.

    Parameters:
    - rfmt_df: DataFrame containing CustomerID, Recency, Frequency, Monetary, Tenure, AvgOrderValue
    - prediction_horizon_days: Forecast period in days (default 90 days)
    - gross_margin: Assumed product gross profit margin (default 35%)
    - alpha_prior, beta_prior_days: Laplace-style smoothing constants for the per-day
      transaction rate lambda = (Frequency + alpha_prior) / (Tenure + beta_prior_days).
      Larger beta_prior_days pulls thin purchase histories harder toward a low baseline
      rate; larger alpha_prior raises that baseline.
    - hazard_missed_cycles_weight, hazard_missed_cycles_offset: scale and break-even
      point (in multiples of a customer's own expected purchase cadence) for the
      "missed cycles" term of the churn hazard.
    - hazard_inactivity_ratio_weight, hazard_inactivity_ratio_offset: scale and
      break-even point (as a fraction of tenure) for the "inactivity ratio" term.
    """
    df_clv = rfmt_df.copy()

    # Recency (days since last purchase), Tenure (days since first purchase)
    r = df_clv["Recency"].astype(float).values
    t = df_clv["Tenure"].astype(float).values
    f = df_clv["Frequency"].astype(float).values
    m = df_clv["Monetary"].astype(float).values
    aov = df_clv["AvgOrderValue"].astype(float).values

    # 1. Laplace-smoothed transaction rate (lambda) — fixed additive smoothing constants,
    #    not a fitted Bayesian posterior. Baseline frequency per day.
    lambda_rate = (f + alpha_prior) / (t + beta_prior_days)  # expected purchases per day

    # 2. Expected purchase interval (in days)
    cadence = np.maximum(t / np.maximum(f, 1.0), 7.0)

    # 3. Ratio of inactive days to expected cadence (missed purchasing cycles)
    missed_cycles = r / cadence

    # 4. Heuristic P(Alive) using logistic sigmoid decay based on missed cycles and tenure ratio.
    # When customer is on schedule, P(Alive) is ~95-99%. As missed cycles grow beyond 2.5x, P(Alive) drops rapidly.
    inactivity_ratio = r / np.maximum(t, 1.0)
    churn_hazard = (
        hazard_missed_cycles_weight * (missed_cycles - hazard_missed_cycles_offset)
        + hazard_inactivity_ratio_weight * (inactivity_ratio - hazard_inactivity_ratio_offset)
    )
    p_alive = 1.0 / (1.0 + np.exp(np.clip(churn_hazard, -10.0, 10.0)))
    
    # Bound P(Alive) gracefully in [0.02, 0.99]
    p_alive = np.clip(p_alive, 0.02, 0.99)
    df_clv["P_Alive"] = np.round(p_alive, 4)
    df_clv["P_Alive_Pct"] = np.round(p_alive * 100, 1)
    
    # 5. Churn Risk Index (1 - P(Alive))
    df_clv["Churn_Risk_Pct"] = np.round((1.0 - p_alive) * 100, 1)
    
    # Churn Risk Tier
    conditions = [
        df_clv["P_Alive"] >= CHURN_TIER_LOW_RISK_THRESHOLD,
        (df_clv["P_Alive"] >= CHURN_TIER_MODERATE_THRESHOLD) & (df_clv["P_Alive"] < CHURN_TIER_LOW_RISK_THRESHOLD),
        df_clv["P_Alive"] < CHURN_TIER_MODERATE_THRESHOLD
    ]
    tiers = ["🟢 Low Churn Risk", "🟡 Moderate Watch", "🔴 High Churn Risk"]
    df_clv["Churn_Risk_Tier"] = np.select(conditions, tiers, default="🟡 Moderate Watch")
    
    # 6. Expected Purchases in the Next Horizon (e.g. 90 days)
    expected_purchases_90d = p_alive * lambda_rate * prediction_horizon_days
    df_clv["Expected_Orders_90d"] = np.round(expected_purchases_90d, 2)
    
    # 7. 90-Day Forecast Gross Revenue ($)
    predicted_revenue_90d = expected_purchases_90d * aov
    df_clv["Predicted_Spend_90d"] = np.round(predicted_revenue_90d, 2)
    
    # 8. 90-Day Predictive Net CLV ($)
    df_clv["Predictive_CLV_90d"] = np.round(predicted_revenue_90d * gross_margin, 2)
    
    return df_clv


def get_urgent_churn_watchlist(clv_df: pd.DataFrame, p_alive_threshold: float = CHURN_TIER_MODERATE_THRESHOLD) -> pd.DataFrame:
    """
    Identifies high historical spenders whose P(Alive) has fallen below the critical safety threshold.
    """
    median_spend = clv_df["Monetary"].median()
    
    watchlist = clv_df[
        (clv_df["P_Alive"] < p_alive_threshold) &
        (clv_df["Monetary"] >= median_spend)
    ].sort_values(by="Monetary", ascending=False).copy()
    
    return watchlist


def get_top_future_growth_targets(clv_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Returns the highest-value forecasted revenue targets for the next 90 days.
    """
    return clv_df.sort_values(by="Predicted_Spend_90d", ascending=False).head(top_n).copy()

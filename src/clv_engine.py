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

def estimate_btyd_clv(
    rfmt_df: pd.DataFrame,
    prediction_horizon_days: int = 90,
    gross_margin: float = 0.35
) -> pd.DataFrame:
    """
    Computes probabilistic BTYD P(Alive), 90-day expected purchase frequency,
    and 90-day predicted CLV for each customer.
    
    Parameters:
    - rfmt_df: DataFrame containing CustomerID, Recency, Frequency, Monetary, Tenure, AvgOrderValue
    - prediction_horizon_days: Forecast period in days (default 90 days)
    - gross_margin: Assumed product gross profit margin (default 35%)
    """
    df_clv = rfmt_df.copy()
    
    # Recency (days since last purchase), Tenure (days since first purchase)
    r = df_clv["Recency"].astype(float).values
    t = df_clv["Tenure"].astype(float).values
    f = df_clv["Frequency"].astype(float).values
    m = df_clv["Monetary"].astype(float).values
    aov = df_clv["AvgOrderValue"].astype(float).values
    
    # 1. Empirical Bayesian Transaction Rate (lambda) with smoothing
    # Baseline frequency per day
    alpha_prior = 1.2
    beta_prior = 60.0  # prior scale of 60 days
    lambda_rate = (f + alpha_prior) / (t + beta_prior)  # expected purchases per day
    
    # 2. Expected purchase interval (in days)
    cadence = np.maximum(t / np.maximum(f, 1.0), 7.0)
    
    # 3. Ratio of inactive days to expected cadence (missed purchasing cycles)
    missed_cycles = r / cadence
    
    # 4. Probabilistic P(Alive) using logistic sigmoid decay based on missed cycles and tenure ratio
    # When customer is on schedule, P(Alive) is ~95-99%. As missed cycles grow beyond 2.5x, P(Alive) drops rapidly.
    inactivity_ratio = r / np.maximum(t, 1.0)
    churn_hazard = 1.4 * (missed_cycles - 1.8) + 1.2 * (inactivity_ratio - 0.4)
    p_alive = 1.0 / (1.0 + np.exp(np.clip(churn_hazard, -10.0, 10.0)))
    
    # Bound P(Alive) gracefully in [0.02, 0.99]
    p_alive = np.clip(p_alive, 0.02, 0.99)
    df_clv["P_Alive"] = np.round(p_alive, 4)
    df_clv["P_Alive_Pct"] = np.round(p_alive * 100, 1)
    
    # 5. Churn Risk Index (1 - P(Alive))
    df_clv["Churn_Risk_Pct"] = np.round((1.0 - p_alive) * 100, 1)
    
    # Churn Risk Tier
    conditions = [
        df_clv["P_Alive"] >= 0.75,
        (df_clv["P_Alive"] >= 0.45) & (df_clv["P_Alive"] < 0.75),
        df_clv["P_Alive"] < 0.45
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


def get_urgent_churn_watchlist(clv_df: pd.DataFrame, p_alive_threshold: float = 0.45) -> pd.DataFrame:
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

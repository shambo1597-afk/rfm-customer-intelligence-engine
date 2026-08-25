"""
backtest_clv.py - Out-of-sample validation of the 90-day CLV forecast and churn flag.

`estimate_btyd_clv()` (src/clv_engine.py) produces `Predicted_Spend_90d` and a binary
"High Churn Risk" flag from a heuristic, hand-tuned hazard formula (see the module's
HAZARD_* constants). Those numbers had never been checked against what customers
actually did next — this script does that check honestly, on real held-out data.

Method (a standard temporal train/test split, not a synthetic-data circularity check):
  1. Cutoff date C = (max transaction date T) - `--horizon-days` (default 90).
  2. TRAIN = every transaction on or before C. The RFM-T + CLV pipeline runs on
     TRAIN only, with snapshot_date=C, exactly as if C were "today".
  3. TEST = every transaction after C (i.e. the actual next `--horizon-days` days).
  4. For each customer present in TRAIN, compare:
       - the model's Predicted_Spend_90d (computed from TRAIN only)
     against
       - their ACTUAL realized spend in TEST (0.0 if they bought nothing).
  5. Two naive baselines are reported alongside the model, computed the same
     out-of-sample way (no peeking at TEST):
       - trailing: assume each customer repeats their own trailing `horizon-days`
         spend (the `horizon-days` immediately before C) — "next period looks
         like last period."
       - population_mean: assume every customer spends the population-average
         trailing `horizon-days` amount — the simplest possible baseline.
  6. The "High Churn Risk" flag at C is checked against the actual outcome
     (zero purchases in TEST) via precision/recall, not just asserted to be
     "probabilistic" and left unverified.

Usage:
    python backtest_clv.py [--dataset data/ecommerce_transactions.csv] [--horizon-days 90]

Numbers from this script (not just "it works") belong in the README's
"Model Validation" section — including if they're mediocre. An unvalidated
confident claim is worse than a validated modest one.
"""

import argparse
import sys

import numpy as np
import pandas as pd

from src.rfm_engine import standardize_transactions, compute_rfmt, calculate_rfmt_scores
from src.clv_engine import estimate_btyd_clv, CHURN_TIER_MODERATE_THRESHOLD


def _mae(pred: pd.Series, actual: pd.Series) -> float:
    return float(np.abs(pred - actual).mean())


def _rmse(pred: pd.Series, actual: pd.Series) -> float:
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def run_backtest(dataset_path: str, horizon_days: int = 90, gross_margin: float = 0.35,
                  churn_threshold: float = CHURN_TIER_MODERATE_THRESHOLD) -> dict:
    df_raw = pd.read_csv(dataset_path)
    clean_tx = standardize_transactions(df_raw)

    snapshot_t = clean_tx["PurchaseDate"].max()
    cutoff = snapshot_t - pd.Timedelta(days=horizon_days)
    trailing_start = cutoff - pd.Timedelta(days=horizon_days)

    train_tx = clean_tx[clean_tx["PurchaseDate"] <= cutoff].copy()
    test_tx = clean_tx[clean_tx["PurchaseDate"] > cutoff].copy()

    if train_tx["CustomerID"].nunique() < 5:
        print(
            f"⚠️  Only {train_tx['CustomerID'].nunique()} customers have purchase history before the "
            f"{cutoff.date()} cutoff — too few for a meaningful backtest on this dataset/horizon.",
            file=sys.stderr,
        )

    # --- Model: RFM-T + CLV forecast computed on TRAIN only, "as of" the cutoff ---
    rfmt_train = compute_rfmt(train_tx, snapshot_date=cutoff)
    rfmt_train = calculate_rfmt_scores(rfmt_train)
    clv_train = estimate_btyd_clv(rfmt_train, prediction_horizon_days=horizon_days, gross_margin=gross_margin)
    clv_train = clv_train.set_index("CustomerID")

    backtest_population = clv_train.index

    # --- Ground truth: actual realized spend in the held-out TEST window ---
    actual_spend = test_tx.groupby("CustomerID")["TotalSpend"].sum()
    actual_spend = actual_spend.reindex(backtest_population, fill_value=0.0)

    # --- Baseline 1: each customer repeats their own trailing-window spend ---
    trailing_tx = train_tx[(train_tx["PurchaseDate"] > trailing_start) & (train_tx["PurchaseDate"] <= cutoff)]
    trailing_spend = trailing_tx.groupby("CustomerID")["TotalSpend"].sum()
    trailing_spend = trailing_spend.reindex(backtest_population, fill_value=0.0)

    # --- Baseline 2: every customer gets the population-average trailing spend ---
    population_mean_pred = pd.Series(trailing_spend.mean(), index=backtest_population)

    model_pred = clv_train["Predicted_Spend_90d"].reindex(backtest_population, fill_value=0.0)

    results = {
        "cutoff_date": str(cutoff.date()),
        "snapshot_date": str(snapshot_t.date()),
        "horizon_days": horizon_days,
        "n_customers_backtested": int(len(backtest_population)),
        "spend_forecast": {
            "model": {"mae": _mae(model_pred, actual_spend), "rmse": _rmse(model_pred, actual_spend)},
            "baseline_trailing": {"mae": _mae(trailing_spend, actual_spend), "rmse": _rmse(trailing_spend, actual_spend)},
            "baseline_population_mean": {"mae": _mae(population_mean_pred, actual_spend), "rmse": _rmse(population_mean_pred, actual_spend)},
        },
    }

    # --- Churn flag: "High Churn Risk" at cutoff vs. actually zero purchases in TEST ---
    flagged_high_risk = clv_train["P_Alive"] < churn_threshold
    actually_churned = actual_spend.reindex(clv_train.index) == 0.0

    tp = int((flagged_high_risk & actually_churned).sum())
    fp = int((flagged_high_risk & ~actually_churned).sum())
    fn = int((~flagged_high_risk & actually_churned).sum())
    tn = int((~flagged_high_risk & ~actually_churned).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    results["churn_flag"] = {
        "threshold_p_alive_below": churn_threshold,
        "base_rate_actually_zero_purchase": float(actually_churned.mean()),
        "flagged_rate": float(flagged_high_risk.mean()),
        "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
        "precision": precision, "recall": recall,
    }

    return results


def print_report(results: dict) -> None:
    print("=" * 78)
    print("CLV FORECAST & CHURN FLAG BACKTEST (out-of-sample, temporal split)")
    print("=" * 78)
    print(f"Snapshot (max transaction date): {results['snapshot_date']}")
    print(f"Cutoff (train/test split, T - {results['horizon_days']}d): {results['cutoff_date']}")
    print(f"Customers backtested (had history before cutoff): {results['n_customers_backtested']:,}")
    print()
    print(f"--- {results['horizon_days']}-Day Spend Forecast vs. Actual Realized Spend ---")
    sf = results["spend_forecast"]
    print(f"{'Method':<28}{'MAE ($)':>14}{'RMSE ($)':>14}")
    print(f"{'Model (Predicted_Spend_90d)':<28}{sf['model']['mae']:>14,.2f}{sf['model']['rmse']:>14,.2f}")
    print(f"{'Baseline: trailing spend':<28}{sf['baseline_trailing']['mae']:>14,.2f}{sf['baseline_trailing']['rmse']:>14,.2f}")
    print(f"{'Baseline: population mean':<28}{sf['baseline_population_mean']['mae']:>14,.2f}{sf['baseline_population_mean']['rmse']:>14,.2f}")
    model_mae = sf["model"]["mae"]
    best_baseline_mae = min(sf["baseline_trailing"]["mae"], sf["baseline_population_mean"]["mae"])
    if model_mae < best_baseline_mae:
        print(f"\n✅ Model beats the best naive baseline on MAE ({model_mae:,.2f} < {best_baseline_mae:,.2f}).")
    else:
        print(f"\n⚠️  Model does NOT beat the best naive baseline on MAE ({model_mae:,.2f} >= {best_baseline_mae:,.2f}).")
    print()
    print(f"--- Churn Flag (P(Alive) < {results['churn_flag']['threshold_p_alive_below']}) vs. Actual Zero Purchases in Window ---")
    cf = results["churn_flag"]
    print(f"Base rate (actually made zero purchases in the {results['horizon_days']}d window): {cf['base_rate_actually_zero_purchase']*100:.1f}%")
    print(f"Flagged 'High Churn Risk' at cutoff: {cf['flagged_rate']*100:.1f}%")
    print(f"Confusion matrix: TP={cf['true_positive']}  FP={cf['false_positive']}  FN={cf['false_negative']}  TN={cf['true_negative']}")
    print(f"Precision: {cf['precision']*100:.1f}%  (of flagged customers, % who actually churned)")
    print(f"Recall:    {cf['recall']*100:.1f}%  (of customers who actually churned, % that were flagged)")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="Backtest the 90-day CLV forecast and churn flag against actual held-out outcomes.")
    parser.add_argument("--dataset", default="data/ecommerce_transactions.csv", help="Path to a raw transactions CSV.")
    parser.add_argument("--horizon-days", type=int, default=90, help="Forecast horizon / train-test split window, in days.")
    parser.add_argument("--gross-margin", type=float, default=0.35, help="Gross margin assumption passed to estimate_btyd_clv.")
    parser.add_argument("--churn-threshold", type=float, default=CHURN_TIER_MODERATE_THRESHOLD, help="P(Alive) below this = 'High Churn Risk' flag.")
    args = parser.parse_args()

    results = run_backtest(
        dataset_path=args.dataset,
        horizon_days=args.horizon_days,
        gross_margin=args.gross_margin,
        churn_threshold=args.churn_threshold,
    )
    print_report(results)


if __name__ == "__main__":
    main()

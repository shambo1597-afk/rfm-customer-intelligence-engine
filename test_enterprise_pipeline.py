"""
test_enterprise_pipeline.py - Automated Testing Suite for Enterprise RFM-T & AI Platform
"""

import sys
import pandas as pd
import numpy as np

from src.rfm_engine import process_rfmt_pipeline, get_segment_kpi_summary, SEGMENT_PLAYBOOKS
from src.ml_engine import evaluate_kmeans_candidates, perform_kmeans_clustering, compute_pca_3d
from src.clv_engine import estimate_btyd_clv, get_urgent_churn_watchlist, get_top_future_growth_targets
from src.cohort_engine import compute_monthly_cohort_matrix

def run_tests():
    print("[1/5] Testing Data Generation...")
    df_raw = pd.read_csv("data/ecommerce_transactions.csv")
    assert len(df_raw) >= 3500, f"Expected >=3500 rows, got {len(df_raw)}"
    assert df_raw["CustomerID"].nunique() == 450, f"Expected 450 customers, got {df_raw['CustomerID'].nunique()}"
    print(f"      PASSED: {len(df_raw):,} records across {df_raw['CustomerID'].nunique()} unique customers.")

    print("[2/5] Testing RFM-T Engine...")
    clean_tx, rfmt = process_rfmt_pipeline(df_raw)
    assert len(rfmt) == 450, f"Expected 450 customer rows, got {len(rfmt)}"
    expected_segs = {"Champions", "Loyalists", "Potential Growth", "At-Risk VIPs", "Can't Lose Them", "Hibernating", "New Customers"}
    actual_segs = set(rfmt["Segment"].unique())
    assert actual_segs.issubset(expected_segs), f"Unexpected segments: {actual_segs}"
    assert not rfmt["RFM_Score"].isnull().any(), "Found null RFM scores"
    summary = get_segment_kpi_summary(rfmt)
    assert len(summary) == 7, f"Expected 7 segment summaries, got {len(summary)}"
    print("      PASSED: RFM-T calculations, 1-5 scoring, and 7-segment taxonomy.")

    print("[3/5] Testing Machine Learning Engine...")
    eval_df = evaluate_kmeans_candidates(rfmt[["Recency", "Frequency", "Monetary", "Tenure"]].values, min_k=2, max_k=7)
    assert len(eval_df) == 6, f"Expected 6 k-evaluations, got {len(eval_df)}"
    assert "Silhouette_Score" in eval_df.columns and "Inertia" in eval_df.columns
    
    df_ml, km, cluster_summary = perform_kmeans_clustering(rfmt, n_clusters=4)
    assert "ML_Cluster" in df_ml.columns
    assert len(cluster_summary) == 4
    
    df_pca, pca_model, exp_var = compute_pca_3d(df_ml)
    assert "PCA_1" in df_pca.columns and "PCA_2" in df_pca.columns and "PCA_3" in df_pca.columns
    assert len(exp_var) == 3
    print("      PASSED: K-Means, Silhouette evaluation, and PCA 3D projection.")

    print("[4/5] Testing CLV & Churn Radar Engine...")
    clv_df = estimate_btyd_clv(rfmt, prediction_horizon_days=90, gross_margin=0.35)
    assert "P_Alive" in clv_df.columns
    assert "Predicted_Spend_90d" in clv_df.columns
    assert "Predictive_CLV_90d" in clv_df.columns
    assert clv_df["P_Alive"].between(0.0, 1.0).all()
    
    watchlist = get_urgent_churn_watchlist(clv_df, p_alive_threshold=0.45)
    assert isinstance(watchlist, pd.DataFrame)
    print(f"      PASSED: P(Alive) estimation & {len(watchlist)} urgent churn accounts identified.")

    print("[5/5] Testing Cohort Retention Triangle Engine...")
    count_mat, ret_mat, cohort_sizes = compute_monthly_cohort_matrix(clean_tx)
    assert not ret_mat.empty
    assert (ret_mat.iloc[:, 0] == 100.0).all(), "Month 0 retention must be 100%"
    print(f"      PASSED: {len(cohort_sizes)} monthly acquisition cohorts evaluated.")

    print("\n>>> ALL ENTERPRISE ENGINE MODULES PASSED WITH 100% SUCCESS! <<<")

if __name__ == "__main__":
    run_tests()

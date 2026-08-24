"""
generate_action_plan.py - Autonomous Execution Script for Customer RFM-T & AI Segmentation
Generates the complete enterprise deliverable 'customer_segmentation_action_plan.csv'
incorporating RFM-T Quintiles, Unsupervised K-Means, 3D PCA, Probabilistic BTYD CLV,
and Actionable Marketing Playbooks.
"""

import os
import sys
import pandas as pd
import numpy as np

from src.rfm_engine import process_rfmt_pipeline, SEGMENT_PLAYBOOKS, get_segment_kpi_summary
from src.ml_engine import preprocess_rfmt_features, evaluate_kmeans_candidates, perform_kmeans_clustering, compute_pca_3d
from src.clv_engine import estimate_btyd_clv, get_urgent_churn_watchlist, get_top_future_growth_targets
from src.cohort_engine import compute_monthly_cohort_matrix

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("=" * 70)
    print("[*] ENTERPRISE CUSTOMER RFM-T & AI SEGMENTATION ENGINE EXECUTION")
    print("=" * 70)

    # 1. Load Dataset
    data_path = "data/ecommerce_transactions.csv"
    if not os.path.exists(data_path):
        data_path = "sample_transactions.csv"
    
    print(f"[*] Step 1: Ingesting Transaction Dataset from '{data_path}'...")
    df_raw = pd.read_csv(data_path)
    print(f"    -> Ingested {len(df_raw):,} transaction records across {df_raw['CustomerID'].nunique():,} unique customers.")

    # 2. RFM-T Processing Pipeline
    print("[*] Step 2: Computing Recency, Frequency, Monetary, and Customer Tenure (RFM-T)...")
    clean_tx, rfmt_df = process_rfmt_pipeline(df_raw)
    print(f"    -> Standardized transactions: {len(clean_tx):,} records.")
    print(f"    -> Scored customers across 7 distinct behavioral segments.")

    # 3. Probabilistic CLV & Churn Radar Engine
    print("[*] Step 3: Estimating Probabilistic BTYD P(Alive) & 90-Day Predictive CLV...")
    clv_df = estimate_btyd_clv(rfmt_df, prediction_horizon_days=90, gross_margin=0.35)
    
    active_count = (clv_df["P_Alive"] >= 0.50).sum()
    print(f"    -> Active accounts: {active_count:,} ({active_count / len(clv_df) * 100:.1f}%)")
    print(f"    -> 90-Day Projected Gross Revenue: ${clv_df['Predicted_Spend_90d'].sum():,.2f}")
    print(f"    -> 90-Day Projected Net CLV (35% Margin): ${clv_df['Predictive_CLV_90d'].sum():,.2f}")

    # 4. Unsupervised ML K-Means & PCA 3D
    print("[*] Step 4: Optimizing K-Means Clustering & Computing 3D PCA Projections...")
    # Evaluate candidate k values on the same log1p + StandardScaler feature space used by
    # the final K-Means fit (perform_kmeans_clustering), so the recommended k is consistent
    # with the model actually fit below (PCA is computed separately, purely for 3D display).
    X_scaled, _, _ = preprocess_rfmt_features(clv_df)
    eval_df = evaluate_kmeans_candidates(X_scaled, min_k=2, max_k=7)
    optimal_k = int(eval_df.loc[eval_df["Silhouette_Score"].idxmax()]["k"])
    print(f"    -> Optimal K evaluated by Silhouette Max: k = {optimal_k} (Silhouette = {eval_df.loc[eval_df['k']==optimal_k, 'Silhouette_Score'].values[0]:.4f})")

    df_ml, km_model, cluster_summary = perform_kmeans_clustering(clv_df, n_clusters=optimal_k)
    final_df, pca_model, exp_var = compute_pca_3d(df_ml)
    print(f"    -> 3D PCA Variance Explained: PC1={exp_var[0]}%, PC2={exp_var[1]}%, PC3={exp_var[2]}% (Total={exp_var[:3].sum():.1f}%)")

    # 5. Enrich with Marketing Playbook Guidance
    print("[*] Step 5: Attaching Tactical Marketing Playbooks & Campaign Blueprints...")
    final_df["Strategic_Objective"] = final_df["Segment"].apply(lambda s: SEGMENT_PLAYBOOKS.get(str(s), {}).get("objective", ""))
    final_df["Recommended_Channel"] = final_df["Segment"].apply(lambda s: SEGMENT_PLAYBOOKS.get(str(s), {}).get("best_channel", ""))
    final_df["Recommended_Promotion"] = final_df["Segment"].apply(lambda s: SEGMENT_PLAYBOOKS.get(str(s), {}).get("promo_type", ""))
    final_df["Primary_Action_Item"] = final_df["Segment"].apply(lambda s: SEGMENT_PLAYBOOKS.get(str(s), {}).get("actions", [""])[0])
    final_df["Campaign_Email_Subject"] = final_df["Segment"].apply(lambda s: SEGMENT_PLAYBOOKS.get(str(s), {}).get("campaign_template", {}).get("subject", ""))
    final_df["Campaign_CTA"] = final_df["Segment"].apply(lambda s: SEGMENT_PLAYBOOKS.get(str(s), {}).get("campaign_template", {}).get("cta", ""))

    # 6. Reorganize & Format Columns for Executive Submission
    output_columns = [
        "CustomerID",
        "Segment",
        "ML_Cluster",
        "Churn_Risk_Tier",
        "P_Alive_Pct",
        "Churn_Risk_Pct",
        "Recency",
        "Frequency",
        "Monetary",
        "Tenure",
        "AvgOrderValue",
        "TopCategory",
        "R_Score",
        "F_Score",
        "M_Score",
        "T_Score",
        "RFM_Score",
        "RFMT_Mean",
        "Expected_Orders_90d",
        "Predicted_Spend_90d",
        "Predictive_CLV_90d",
        "PCA_1",
        "PCA_2",
        "PCA_3",
        "Strategic_Objective",
        "Recommended_Channel",
        "Recommended_Promotion",
        "Primary_Action_Item",
        "Campaign_Email_Subject",
        "Campaign_CTA",
        "FirstPurchase",
        "LastPurchase",
        "TotalItems",
        "TotalTransactions"
    ]

    export_df = final_df[output_columns].copy()
    
    # Sort for optimal readability: High Churn Risk VIPs & Top Spenders first
    export_df = export_df.sort_values(
        by=["Monetary", "Predictive_CLV_90d"],
        ascending=[False, False]
    ).reset_index(drop=True)

    # 7. Save Final Deliverable
    output_filename = "customer_segmentation_action_plan.csv"
    export_df.to_csv(output_filename, index=False)
    print("=" * 70)
    print(f"✅ SUCCESS: Deliverable saved to '{output_filename}' ({len(export_df):,} customers, {len(output_columns)} attributes).")
    print("=" * 70)

    # Print Quick Executive Breakdown
    print("\n📊 Segment Distribution & Revenue Contribution:")
    kpi_summary = get_segment_kpi_summary(final_df)
    for _, row in kpi_summary.iterrows():
        print(f"  • {row['Segment']:<18} | Custs: {row['CustomerCount']:>3} ({row['CustomerSharePct']:>4.1f}%) | Rev: ${row['TotalRevenue']:>10,.2f} ({row['RevenueSharePct']:>4.1f}%) | Avg Spend: ${row['AvgRevenue']:>8,.2f}")

    print("\n🔍 Urgent Churn Watchlist Sample (Top 3 High-Value Accounts at Risk):")
    watchlist = get_urgent_churn_watchlist(final_df, p_alive_threshold=0.45)
    for _, w in watchlist.head(3).iterrows():
        print(f"  ⚠️ Customer {w['CustomerID']}: Historical Spend=${w['Monetary']:,.2f}, Recency={w['Recency']}d, P(Alive)={w['P_Alive_Pct']}%, Action={w['Primary_Action_Item']}")

if __name__ == "__main__":
    main()

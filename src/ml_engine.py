"""
src/ml_engine.py - Unsupervised Machine Learning & Dimensionality Reduction Engine
Features:
1. Log-Transformation & StandardScaler feature normalization.
2. Multi-k K-Means Clustering evaluation (k=2 to k=7) with Silhouette & Elbow analysis.
3. 3-Component Principal Component Analysis (PCA) for 3D spatial cluster visualization.
4. Cluster profiling and centroid characterization.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def preprocess_rfmt_features(rfmt_df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler, list[str]]:
    """
    Applies log1p transformation to mitigate skewness in RFM-T metrics,
    followed by standard z-score scaling.
    """
    features = ["Recency", "Frequency", "Monetary", "Tenure"]
    X_raw = rfmt_df[features].values
    
    # Log1p transform (handles non-negatives gracefully)
    X_log = np.log1p(X_raw)
    
    # Standard Scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)
    
    return X_scaled, scaler, features


def evaluate_kmeans_candidates(X_scaled: np.ndarray, min_k: int = 2, max_k: int = 7, random_state: int = 42) -> pd.DataFrame:
    """
    Evaluates K-Means clustering across a range of k values, computing Inertia (Elbow)
    and Silhouette Scores with sub-sampling to prevent CPU freezing on large datasets.
    """
    eval_records = []
    n_samples = len(X_scaled)
    samp_size = min(1000, n_samples) if n_samples > 1000 else None
    
    for k in range(min_k, max_k + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        cluster_labels = km.fit_predict(X_scaled)
        
        inertia = km.inertia_
        if samp_size is not None:
            sil_score = silhouette_score(X_scaled, cluster_labels, sample_size=samp_size, random_state=random_state)
        else:
            sil_score = silhouette_score(X_scaled, cluster_labels)
        
        eval_records.append({
            "k": k,
            "Inertia": round(inertia, 2),
            "Silhouette_Score": round(sil_score, 4)
        })
        
    eval_df = pd.DataFrame(eval_records)
    return eval_df


def perform_kmeans_clustering(
    rfmt_df: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42
) -> tuple[pd.DataFrame, KMeans, pd.DataFrame]:
    """
    Executes K-Means clustering with specified or optimal k, and generates cluster summary metrics.
    """
    df_ml = rfmt_df.copy()
    X_scaled, _, features = preprocess_rfmt_features(df_ml)
    
    kmeans_model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=15,
        random_state=random_state
    )
    cluster_labels = kmeans_model.fit_predict(X_scaled)
    
    # Format cluster names as Cluster 0, Cluster 1, etc.
    df_ml["ML_Cluster"] = [f"Cluster {c+1}" for c in cluster_labels]
    
    # Generate Cluster Profiles
    cluster_summary = df_ml.groupby("ML_Cluster").agg(
        CustomerCount=("CustomerID", "count"),
        TotalRevenue=("Monetary", "sum"),
        MeanRecency=("Recency", "mean"),
        MeanFrequency=("Frequency", "mean"),
        MeanMonetary=("Monetary", "mean"),
        MeanTenure=("Tenure", "mean"),
        MeanAOV=("AvgOrderValue", "mean")
    ).reset_index()
    
    total_rev = df_ml["Monetary"].sum()
    total_cust = len(df_ml)
    
    cluster_summary["CustomerPct"] = (cluster_summary["CustomerCount"] / max(total_cust, 1) * 100).round(1)
    cluster_summary["RevenuePct"] = (cluster_summary["TotalRevenue"] / max(total_rev, 1) * 100).round(1)
    cluster_summary["MeanRecency"] = cluster_summary["MeanRecency"].round(1)
    cluster_summary["MeanFrequency"] = cluster_summary["MeanFrequency"].round(1)
    cluster_summary["MeanMonetary"] = cluster_summary["MeanMonetary"].round(2)
    cluster_summary["MeanTenure"] = cluster_summary["MeanTenure"].round(1)
    cluster_summary["MeanAOV"] = cluster_summary["MeanAOV"].round(2)
    
    return df_ml, kmeans_model, cluster_summary


def compute_segment_cluster_crosstab(df_clustered: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cross-tabulates the rule-based `Segment` taxonomy (src/rfm_engine.py, hand-authored
    RFM-T quintile thresholds) against the unsupervised `ML_Cluster` assignment (K-Means
    on log1p + StandardScaler RFM-T features). `df_clustered` must have both columns —
    i.e. it's the output of perform_kmeans_clustering() run on a Segment-scored frame.

    The two labels are computed completely independently: the rule system has no
    knowledge of the clustering, and K-Means has no knowledge of the segment rules.
    This crosstab is a validation/sanity-check tool, not a merge of the two into a new
    label. Where a Segment's customers land overwhelmingly in one ML_Cluster, that's
    corroborating evidence the rule-based boundary is capturing a real behavioral
    grouping. Where a Segment splits roughly evenly across multiple clusters, that
    segment's quintile thresholds may be cutting across a natural cluster boundary and
    are worth revisiting.

    Returns (counts, row_pct):
    - counts: raw crosstab of customer counts, Segment (rows) x ML_Cluster (columns).
    - row_pct: the same table with each Segment's row normalized to sum to 100%, for
      comparing segments of very different sizes on the same visual scale.
    """
    counts = pd.crosstab(df_clustered["Segment"], df_clustered["ML_Cluster"])
    row_totals = counts.sum(axis=1).replace(0, 1)
    row_pct = (counts.div(row_totals, axis=0) * 100.0).round(1)
    return counts, row_pct


def compute_pca_3d(rfmt_df: pd.DataFrame) -> tuple[pd.DataFrame, PCA, np.ndarray]:
    """
    Applies Principal Component Analysis (3 components) on RFM-T features
    for 3D spatial clustering visualization.
    """
    df_pca = rfmt_df.copy()
    X_scaled, _, _ = preprocess_rfmt_features(df_pca)
    
    pca = PCA(n_components=3, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    
    df_pca["PCA_1"] = pca_coords[:, 0].round(4)
    df_pca["PCA_2"] = pca_coords[:, 1].round(4)
    df_pca["PCA_3"] = pca_coords[:, 2].round(4)
    
    explained_variance = (pca.explained_variance_ratio_ * 100).round(2)
    
    return df_pca, pca, explained_variance

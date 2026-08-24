"""
src/cohort_engine.py - Monthly Acquisition Cohort Retention Triangle Engine
Calculates monthly acquisition cohorts and builds the triangle retention rate matrix
with clean Plotly heatmap visualization and accurate unique customer counts.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go

def compute_monthly_cohort_matrix(
    df_transactions: pd.DataFrame,
    max_months: int = 13
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Groups transactions by monthly acquisition cohort and computes:
    1. Active Customer Count Matrix (Triangle): unique customer count per (CohortPeriod, CohortIndex)
    2. Retention Percentage Matrix (Triangle): retention rate relative to Month 0 cohort size (0% - 100%)
    3. Cohort Sizes (Series): initial unique customer count at acquisition Month 0

    Parameters:
    - max_months: Maximum number of cohort-index columns to retain (Month 0 through
      Month max_months-1). Defaults to 13 (a full year of post-acquisition history
      plus Month 0). Pass a larger value for longer-running datasets, or None for
      no cap.
    """
    df = df_transactions.copy()
    df["PurchaseDate"] = pd.to_datetime(df["PurchaseDate"])
    df = df.dropna(subset=["CustomerID", "PurchaseDate"]).copy()
    
    # 1. Transaction Month (OrderPeriod)
    df["OrderPeriod"] = df["PurchaseDate"].dt.to_period("M")
    
    # 2. Customer Cohort Month (Acquisition Month - First Transaction)
    df["CohortPeriod"] = df.groupby("CustomerID")["PurchaseDate"].transform("min").dt.to_period("M")
    
    # 3. Calculate Cohort Index (Months elapsed since acquisition: 0, 1, 2, ...)
    cohort_year = df["CohortPeriod"].dt.year
    cohort_month = df["CohortPeriod"].dt.month
    order_year = df["OrderPeriod"].dt.year
    order_month = df["OrderPeriod"].dt.month
    
    years_diff = order_year - cohort_year
    months_diff = order_month - cohort_month
    df["CohortIndex"] = years_diff * 12 + months_diff
    
    # Filter out any negative cohort index edge cases
    df = df[df["CohortIndex"] >= 0]
    
    # Group strictly by CohortPeriod and CohortIndex to count unique active customers
    cohort_data = df.groupby(["CohortPeriod", "CohortIndex"])["CustomerID"].nunique().reset_index()
    
    # Pivot into customer count matrix
    count_matrix = cohort_data.pivot(index="CohortPeriod", columns="CohortIndex", values="CustomerID")
    
    # Ensure index is string formatted for clean charting (e.g., "2024-09")
    count_matrix.index = count_matrix.index.astype(str)
    
    # Initial cohort size is strictly Month 0 unique customer count
    cohort_sizes = count_matrix.iloc[:, 0].copy()
    
    # Compute retention percentage: (Active Customers in Month k / Cohort Size in Month 0) * 100
    retention_matrix = count_matrix.divide(cohort_sizes, axis=0) * 100.0
    
    # Ensure Month 0 is strictly 100.0% and subsequent values are bounded between 0% and 100%
    retention_matrix.iloc[:, 0] = 100.0
    retention_matrix = retention_matrix.clip(lower=0.0, upper=100.0).round(1)
    
    # Limit max columns shown if history is long (e.g. up to 13 columns: Month 0 to Month 12)
    if max_months is not None and retention_matrix.shape[1] > max_months:
        retention_matrix = retention_matrix.iloc[:, :max_months]
        count_matrix = count_matrix.iloc[:, :max_months]
        
    return count_matrix, retention_matrix, cohort_sizes


def create_cohort_retention_heatmap(retention_matrix: pd.DataFrame, count_matrix: pd.DataFrame) -> go.Figure:
    """
    Renders an enterprise-grade Plotly Triangle Heatmap of Cohort Retention.
    - Sized at 650px height
    - Color scale strictly from 0% to 100% (zmin=0, zmax=100)
    - Clean single-line 'XX%' text per cell (empty for unelapsed future months) without overlap
    - Clean hover tooltips displaying Cohort, Month, Active Customers / Cohort Size, and Retention Rate %
    - No duplicate add_annotation loops
    """
    z_values = retention_matrix.values
    y_labels = [f"Cohort {idx}" for idx in retention_matrix.index]
    x_labels = [f"Month {col}" for col in retention_matrix.columns]
    
    # Build clean 2D text matrix with only 'XX%' (or '' for NaN) to prevent text overlap
    text_matrix = []
    # Build 3D customdata array for rich hover tooltips: [active_count, cohort_size]
    customdata = np.empty((len(retention_matrix), len(retention_matrix.columns), 2), dtype=object)
    
    for r_idx in range(len(retention_matrix)):
        row_text = []
        cohort_size = count_matrix.iloc[r_idx, 0]
        for c_idx in range(len(retention_matrix.columns)):
            val = z_values[r_idx, c_idx]
            count_val = count_matrix.values[r_idx, c_idx]
            
            if pd.isna(val):
                row_text.append("")
                customdata[r_idx, c_idx] = ["-", int(cohort_size) if not pd.isna(cohort_size) else 0]
            else:
                row_text.append(f"{val:.0f}%")
                customdata[r_idx, c_idx] = [int(count_val), int(cohort_size)]
                
        text_matrix.append(row_text)
        
    # Custom hover tooltip
    hovertemplate = (
        "<b>Cohort:</b> %{y}<br>"
        "<b>Timeline:</b> %{x}<br>"
        "<b>Active Customers:</b> %{customdata[0]} of %{customdata[1]} acquired<br>"
        "<b>Retention Rate:</b> %{z:.1f}%"
        "<extra></extra>"
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=x_labels,
        y=y_labels,
        customdata=customdata,
        hovertemplate=hovertemplate,
        colorscale="Blues",
        zmin=0,
        zmax=100,
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(size=10, color="white", family="Inter, sans-serif"),
        hoverongaps=False,
        colorbar=dict(
            title=dict(text="Retention %", font=dict(size=12, color="#E2E8F0")),
            ticksuffix="%",
            tickfont=dict(color="#CBD5E1")
        )
    ))
    
    fig.update_layout(
        title="<b>Monthly Acquisition Cohort Retention Triangle (%)</b>",
        template="plotly_dark",
        margin=dict(l=60, r=20, t=50, b=40),
        height=650,
        xaxis=dict(
            title="Months Since Acquisition",
            showgrid=False,
            tickfont=dict(size=11, color="#CBD5E1")
        ),
        yaxis=dict(
            title="Acquisition Cohort (Year-Month)",
            autorange="reversed",
            showgrid=False,
            tickfont=dict(size=11, color="#CBD5E1")
        )
    )
    
    return fig


def create_average_retention_curve(retention_matrix: pd.DataFrame) -> go.Figure:
    """
    Renders the overall average retention decay curve across all cohorts.
    """
    mean_retention = retention_matrix.mean(axis=0).round(1)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[f"Month {c}" for c in mean_retention.index],
        y=mean_retention.values,
        mode="lines+markers+text",
        text=[f"{v:.1f}%" for v in mean_retention.values],
        textposition="top right",
        line=dict(color="#38BDF8", width=3),
        marker=dict(size=8, color="#0284C7"),
        name="Average Retention",
        hovertemplate="<b>%{x}</b><br>Average Retention: %{y:.1f}%<extra></extra>"
    ))
    
    fig.update_layout(
        title="<b>Average Customer Retention Decay Curve</b>",
        template="plotly_dark",
        margin=dict(l=40, r=20, t=50, b=40),
        height=320,
        xaxis_title="Month",
        yaxis_title="Average Retention (%)",
        yaxis=dict(range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    )
    
    return fig

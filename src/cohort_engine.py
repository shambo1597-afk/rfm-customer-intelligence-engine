"""
src/cohort_engine.py - Monthly Acquisition Cohort Retention Triangle Engine
Calculates monthly acquisition cohorts and builds the triangle retention rate matrix.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go

def compute_monthly_cohort_matrix(df_transactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Groups transactions by monthly acquisition cohort and computes:
    1. Active Customer Count Matrix (Triangle)
    2. Retention Percentage Matrix (Triangle)
    3. Cohort Sizes
    """
    df = df_transactions.copy()
    df["PurchaseDate"] = pd.to_datetime(df["PurchaseDate"])
    
    # 1. Transaction Month
    df["OrderPeriod"] = df["PurchaseDate"].dt.to_period("M")
    
    # 2. Customer Cohort Month (Month of first order)
    df["CohortPeriod"] = df.groupby("CustomerID")["PurchaseDate"].transform("min").dt.to_period("M")
    
    # 3. Calculate Cohort Index (Months elapsed since acquisition: 0, 1, 2, ...)
    cohort_year = df["CohortPeriod"].dt.year
    cohort_month = df["CohortPeriod"].dt.month
    order_year = df["OrderPeriod"].dt.year
    order_month = df["OrderPeriod"].dt.month
    
    years_diff = order_year - cohort_year
    months_diff = order_month - cohort_month
    df["CohortIndex"] = years_diff * 12 + months_diff
    
    # Group by CohortPeriod and CohortIndex to count unique active customers
    cohort_data = df.groupby(["CohortPeriod", "CohortIndex"])["CustomerID"].nunique().reset_index()
    
    # Pivot into customer count matrix
    count_matrix = cohort_data.pivot(index="CohortPeriod", columns="CohortIndex", values="CustomerID")
    
    # Format index strings (e.g., "2025-01")
    count_matrix.index = count_matrix.index.astype(str)
    
    # Initial cohort size is CohortIndex 0
    cohort_sizes = count_matrix.iloc[:, 0]
    
    # Compute retention percentages (Month 0 = 100%)
    retention_matrix = count_matrix.divide(cohort_sizes, axis=0) * 100.0
    retention_matrix = retention_matrix.round(1)
    
    # Limit max columns shown if history is long (e.g. up to 12 months for clean visual presentation)
    if retention_matrix.shape[1] > 13:
        retention_matrix = retention_matrix.iloc[:, :13]
        count_matrix = count_matrix.iloc[:, :13]
        
    return count_matrix, retention_matrix, cohort_sizes


def create_cohort_retention_heatmap(retention_matrix: pd.DataFrame, count_matrix: pd.DataFrame) -> go.Figure:
    """
    Renders an enterprise-grade Plotly Triangle Heatmap of Cohort Retention.
    """
    z_values = retention_matrix.values
    y_labels = [f"Cohort {idx}" for idx in retention_matrix.index]
    x_labels = [f"Month {col}" for col in retention_matrix.columns]
    
    # Prepare text annotations
    text_matrix = []
    for r_idx in range(len(retention_matrix)):
        row_text = []
        for c_idx in range(len(retention_matrix.columns)):
            val = z_values[r_idx, c_idx]
            count_val = count_matrix.values[r_idx, c_idx]
            if pd.isna(val):
                row_text.append("")
            else:
                row_text.append(f"<b>{val:.0f}%</b><br><span style='font-size:10px;'>({int(count_val)} custs)</span>")
        text_matrix.append(row_text)
        
    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=x_labels,
        y=y_labels,
        colorscale="Blues",
        zmin=0,
        zmax=100,
        text=text_matrix,
        texttemplate="%{text}",
        textfont={"size": 11, "color": "white"},
        hoverongaps=False,
        colorbar=dict(title="Retention %", ticksuffix="%")
    ))
    
    fig.update_layout(
        title="<b>Monthly Acquisition Cohort Retention Triangle (%)</b>",
        template="plotly_dark",
        margin=dict(l=50, r=20, t=50, b=40),
        height=520,
        xaxis=dict(title="Months Since Acquisition", showgrid=False),
        yaxis=dict(title="Acquisition Cohort (Year-Month)", autorange="reversed", showgrid=False)
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
        name="Average Retention"
    ))
    
    fig.update_layout(
        title="<b>Average Customer Retention Decay Curve</b>",
        template="plotly_dark",
        margin=dict(l=40, r=20, t=50, b=40),
        height=350,
        xaxis_title="Month",
        yaxis_title="Average Retention (%)",
        yaxis=dict(range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    )
    
    return fig

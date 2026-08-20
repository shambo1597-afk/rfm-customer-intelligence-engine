"""
rfm_engine.py - Core RFM Segmentation & Marketing Recommendation Engine
Computes Recency, Frequency, and Monetary scores (1-4), customer segmentation,
and actionable marketing plays.
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Standard color palette for RFM segments
SEGMENT_COLORS = {
    "Champions": "#10B981",        # Emerald Green
    "Loyal Customers": "#3B82F6",  # Royal Blue
    "New/Promising": "#8B5CF6",    # Purple
    "At Risk": "#F59E0B",          # Amber / Orange
    "Lost/Inactive": "#EF4444"     # Rose / Red
}

SEGMENT_ICONS = {
    "Champions": "👑",
    "Loyal Customers": "💎",
    "New/Promising": "🌱",
    "At Risk": "⚠️",
    "Lost/Inactive": "💤"
}

# Rich actionable marketing playbooks for each segment
MARKETING_RECOMMENDATIONS = {
    "Champions": {
        "title": "Champions",
        "icon": "👑",
        "color": "#10B981",
        "badge": "Top 1% High-Value Spenders",
        "summary": "Your most loyal, frequent, and highest spending customers. They buy often and recently.",
        "goal": "Reward loyalty, maintain high engagement, and turn them into brand evangelists.",
        "actions": [
            "Grant exclusive VIP early access to product launches, seasonal drops, and limited editions.",
            "Provide dedicated VIP concierge / priority support channels.",
            "Invite to brand advisory panel or beta testing programs.",
            "Implement a premium tier loyalty program with double points and milestone gifts."
        ],
        "channels": ["Direct VIP Concierge", "Personalized Executive Email", "Exclusive In-App Perk"],
        "sample_campaign": {
            "subject": "👑 VIP Early Access: You're invited to explore our newest collection first",
            "body": "Hi {Customer_Name}, as one of our most valued members, you get exclusive 48-hour early access to our newest drop before public release, plus complimentary priority shipping on your next order."
        }
    },
    "Loyal Customers": {
        "title": "Loyal Customers",
        "icon": "💎",
        "color": "#3B82F6",
        "badge": "Steady Repeat Buyers",
        "summary": "Regular buyers with healthy order values. They have high trust in your store.",
        "goal": "Increase average order value (AOV), cross-sell complementary categories, and elevate to Champion status.",
        "actions": [
            "Recommend high-margin accessories and personalized product bundles based on purchase history.",
            "Offer tiered volume discounts (e.g. 'Spend $150, get $30 off') to increase basket size.",
            "Provide loyalty rewards when they refer a colleague or friend.",
            "Send personalized quarterly check-ins and product care guides."
        ],
        "channels": ["Targeted Email Newsletters", "SMS Milestones", "Personalized On-Site Recommendations"],
        "sample_campaign": {
            "subject": "💎 Unlock your exclusive bundle: Handpicked items for you",
            "body": "Hi {Customer_Name}, thank you for choosing us! Based on your recent orders, here is an exclusive 15% bundle discount on our top-rated accessories."
        }
    },
    "New/Promising": {
        "title": "New/Promising",
        "icon": "🌱",
        "color": "#8B5CF6",
        "badge": "Recent First-Time Buyers",
        "summary": "Customers who purchased recently with low-to-moderate frequency. High potential for second purchase.",
        "goal": "Build purchasing habit, foster brand trust, and convert into repeat buyers.",
        "actions": [
            "Deliver an automated 3-part onboarding email sequence with tutorials, tips, and FAQs.",
            "Provide a time-sensitive second-order incentive ('Enjoy $15 off your next order within 14 days').",
            "Highlight best-sellers and customer reviews to spark curiosity.",
            "Prompt for immediate post-purchase feedback to resolve any friction."
        ],
        "channels": ["Welcome Email Flow", "In-App Onboarding Guides", "SMS Welcome Voucher"],
        "sample_campaign": {
            "subject": "🌱 How is your order? Here is a special gift for your next visit",
            "body": "Hi {Customer_Name}, we hope you are loving your purchase! To help you get the best experience, here is a quick setup guide plus $15 off your next order with code WELCOME15."
        }
    },
    "At Risk": {
        "title": "At Risk",
        "icon": "⚠️",
        "color": "#F59E0B",
        "badge": "Dormant High-Value Customers",
        "summary": "Formerly active and high-spending customers who have not made a purchase recently.",
        "goal": "Re-engage before churn becomes permanent and diagnose reasons for disengagement.",
        "actions": [
            "Deploy an aggressive win-back campaign with a high-value discount (e.g. 20-25% off).",
            "Send a personalized survey ('How can we do better?') with an incentive for completion.",
            "Highlight major improvements, new product lines, or solved pain points since their last order.",
            "Test multi-channel outreach (Email + SMS) with urgency triggers."
        ],
        "channels": ["Urgent Win-back Email", "Re-engagement SMS", "Paid Retargeting Ads"],
        "sample_campaign": {
            "subject": "⚠️ We miss you! Take 20% off your next order — this week only",
            "body": "Hi {Customer_Name}, it has been a while since your last visit. We've introduced several exciting updates you might like. Enjoy 20% off your next purchase with code WE_MISS_YOU."
        }
    },
    "Lost/Inactive": {
        "title": "Lost/Inactive",
        "icon": "💤",
        "color": "#EF4444",
        "badge": "Churned Low-Frequency Buyers",
        "summary": "Low spend and long periods of inactivity. Unlikely to return without drastic incentive.",
        "goal": "Low-cost reactivation or email list hygiene to protect sender reputation.",
        "actions": [
            "Launch a clearance / massive liquidation reactivation campaign (up to 35% off).",
            "Send a gentle sunset notification asking if they wish to remain subscribed.",
            "Cleanse unengaged contacts from high-cost marketing workflows.",
            "Run lookalike exclusion on advertising platforms to reduce acquisition waste."
        ],
        "channels": ["Automated Sunset Email", "Annual Clearance Blast", "List Hygiene Routine"],
        "sample_campaign": {
            "subject": "💤 Is this goodbye? Last chance to keep your exclusive member perks",
            "body": "Hi {Customer_Name}, we haven't seen you in a long time. If you'd like to stay connected and enjoy 30% off our seasonal clearance, click below — otherwise we'll gracefully reduce our emails."
        }
    }
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names to handle flexible CSV uploads.
    """
    col_mapping = {}
    for col in df.columns:
        clean = col.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
        if clean in ["invoiceno", "invoice", "invoicenumber", "orderno", "orderid", "transactionid"]:
            col_mapping[col] = "InvoiceNo"
        elif clean in ["customerid", "customer", "customerno", "clientid", "userid"]:
            col_mapping[col] = "CustomerID"
        elif clean in ["purchasedate", "date", "invoicedate", "orderdate", "timestamp", "datetime"]:
            col_mapping[col] = "PurchaseDate"
        elif clean in ["product", "productname", "item", "description", "sku"]:
            col_mapping[col] = "Product"
        elif clean in ["quantity", "qty", "count"]:
            col_mapping[col] = "Quantity"
        elif clean in ["unitprice", "price", "itemprice", "rate"]:
            col_mapping[col] = "UnitPrice"
        elif clean in ["totalspend", "total", "spend", "amount", "sales", "revenue"]:
            col_mapping[col] = "TotalSpend"

    df_renamed = df.rename(columns=col_mapping).copy()

    # Validate essential columns
    required_cols = ["CustomerID", "PurchaseDate"]
    for req in required_cols:
        if req not in df_renamed.columns:
            raise ValueError(f"Missing mandatory column: '{req}'. Please map your dataset columns.")

    if "InvoiceNo" not in df_renamed.columns:
        df_renamed["InvoiceNo"] = [f"INV-{i+1:05d}" for i in range(len(df_renamed))]

    if "Product" not in df_renamed.columns:
        df_renamed["Product"] = "Standard Product"

    if "Quantity" not in df_renamed.columns:
        df_renamed["Quantity"] = 1
    else:
        df_renamed["Quantity"] = pd.to_numeric(df_renamed["Quantity"], errors="coerce").fillna(1)

    if "UnitPrice" not in df_renamed.columns:
        if "TotalSpend" in df_renamed.columns:
            df_renamed["UnitPrice"] = pd.to_numeric(df_renamed["TotalSpend"], errors="coerce").fillna(0.0) / df_renamed["Quantity"].replace(0, 1)
        else:
            df_renamed["UnitPrice"] = 10.0
    else:
        df_renamed["UnitPrice"] = pd.to_numeric(df_renamed["UnitPrice"], errors="coerce").fillna(0.0)

    if "TotalSpend" not in df_renamed.columns:
        df_renamed["TotalSpend"] = round(df_renamed["Quantity"] * df_renamed["UnitPrice"], 2)
    else:
        df_renamed["TotalSpend"] = pd.to_numeric(df_renamed["TotalSpend"], errors="coerce").fillna(
            round(df_renamed["Quantity"] * df_renamed["UnitPrice"], 2)
        )

    # Parse dates
    df_renamed["PurchaseDate"] = pd.to_datetime(df_renamed["PurchaseDate"], errors="coerce")
    df_renamed = df_renamed.dropna(subset=["CustomerID", "PurchaseDate"]).copy()
    df_renamed["CustomerID"] = df_renamed["CustomerID"].astype(str)

    return df_renamed


def compute_rfm_table(df: pd.DataFrame, snapshot_date=None) -> pd.DataFrame:
    """
    Computes Recency, Frequency, and Monetary metrics for each customer.
    """
    if snapshot_date is None:
        snapshot_date = df["PurchaseDate"].max() + timedelta(days=1)
    else:
        snapshot_date = pd.to_datetime(snapshot_date)

    # Customer level aggregations
    rfm_grouped = df.groupby("CustomerID").agg(
        Recency=("PurchaseDate", lambda x: int((snapshot_date - x.max()).total_seconds() // 86400)),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalSpend", "sum"),
        FirstPurchase=("PurchaseDate", "min"),
        LastPurchase=("PurchaseDate", "max"),
        TotalItems=("Quantity", "sum"),
        TotalTransactions=("InvoiceNo", "count")
    ).reset_index()

    # Recency cannot be negative
    rfm_grouped["Recency"] = rfm_grouped["Recency"].clip(lower=0)
    rfm_grouped["Monetary"] = rfm_grouped["Monetary"].round(2)
    rfm_grouped["AvgOrderValue"] = (rfm_grouped["Monetary"] / rfm_grouped["Frequency"].replace(0, 1)).round(2)

    # Identify top purchased product per customer
    top_prods = df.groupby(["CustomerID", "Product"])["Quantity"].sum().reset_index()
    top_prods = top_prods.sort_values(["CustomerID", "Quantity"], ascending=[True, False]).drop_duplicates("CustomerID")
    top_prod_map = dict(zip(top_prods["CustomerID"], top_prods["Product"]))
    rfm_grouped["FavoriteProduct"] = rfm_grouped["CustomerID"].map(top_prod_map).fillna("N/A")

    return rfm_grouped


def calculate_rfm_scores(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes 1-4 scale scores for Recency, Frequency, and Monetary metrics using quartile bins.
    Uses rank-based qcut to cleanly avoid duplicate bin errors with non-unique thresholds.
    """
    df_score = rfm_df.copy()
    n_customers = len(df_score)

    if n_customers < 4:
        # Fallback for very small datasets
        df_score["R_Score"] = 4
        df_score["F_Score"] = 4
        df_score["M_Score"] = 4
    else:
        # Recency: Lower days = better = higher score (4)
        df_score["R_Score"] = pd.qcut(
            df_score["Recency"].rank(method="first", ascending=False),
            q=4,
            labels=[1, 2, 3, 4]
        ).astype(int)

        # Frequency: Higher count = better = higher score (4)
        df_score["F_Score"] = pd.qcut(
            df_score["Frequency"].rank(method="first", ascending=True),
            q=4,
            labels=[1, 2, 3, 4]
        ).astype(int)

        # Monetary: Higher spend = better = higher score (4)
        df_score["M_Score"] = pd.qcut(
            df_score["Monetary"].rank(method="first", ascending=True),
            q=4,
            labels=[1, 2, 3, 4]
        ).astype(int)

    # Composite RFM Score String and Average Score
    df_score["RFM_Score"] = df_score["R_Score"].astype(str) + df_score["F_Score"].astype(str) + df_score["M_Score"].astype(str)
    df_score["RFM_Mean"] = ((df_score["R_Score"] + df_score["F_Score"] + df_score["M_Score"]) / 3.0).round(2)

    return df_score


def assign_rfm_segment(row: pd.Series) -> str:
    """
    Classifies customer into one of 5 distinct behavioral segments based on RFM scores (1-4).
    """
    r = int(row["R_Score"])
    f = int(row["F_Score"])
    m = int(row["M_Score"])

    # 1. Champions: Highest recency, high frequency, high spend
    if (r == 4 and f >= 3 and m >= 3) or (r >= 3 and f == 4 and m == 4) or (r == 4 and f == 4):
        return "Champions"

    # 2. Loyal Customers: Active repeat buyers with strong frequency and solid spend
    if (r >= 3 and f >= 3) or (r >= 2 and f >= 3 and m >= 3) or (r >= 3 and m >= 3):
        return "Loyal Customers"

    # 3. New / Promising: Recent buyers with low frequency (newly acquired)
    if (r >= 3 and f <= 2):
        return "New/Promising"

    # 4. At Risk: High past value/frequency, but have not bought recently (dormant)
    if (r <= 2 and (f >= 2 or m >= 3)):
        return "At Risk"

    # 5. Lost / Inactive: Low recency, low frequency, low monetary
    return "Lost/Inactive"


def process_rfm_pipeline(df: pd.DataFrame, snapshot_date=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: cleans raw transactions, calculates RFM metrics, scores 1-4,
    and assigns customer segments.
    Returns: (cleaned_transactions_df, rfm_customer_df)
    """
    clean_tx = standardize_columns(df)
    rfm_table = compute_rfm_table(clean_tx, snapshot_date=snapshot_date)
    rfm_scored = calculate_rfm_scores(rfm_table)
    rfm_scored["Segment"] = rfm_scored.apply(assign_rfm_segment, axis=1)

    # Segment metadata
    rfm_scored["Segment_Icon"] = rfm_scored["Segment"].map(SEGMENT_ICONS)
    rfm_scored["Segment_Color"] = rfm_scored["Segment"].map(SEGMENT_COLORS)

    # Order segments logically
    segment_order = ["Champions", "Loyal Customers", "New/Promising", "At Risk", "Lost/Inactive"]
    rfm_scored["Segment"] = pd.Categorical(rfm_scored["Segment"], categories=segment_order, ordered=True)

    return clean_tx, rfm_scored


def get_segment_summary(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes high-level aggregated statistics per segment.
    """
    summary = rfm_df.groupby("Segment", observed=False).agg(
        CustomerCount=("CustomerID", "count"),
        TotalRevenue=("Monetary", "sum"),
        AvgRevenue=("Monetary", "mean"),
        AvgRecency=("Recency", "mean"),
        AvgFrequency=("Frequency", "mean"),
        AvgAOV=("AvgOrderValue", "mean")
    ).reset_index()

    total_customers = len(rfm_df)
    total_rev = rfm_df["Monetary"].sum()

    summary["CustomerSharePct"] = (summary["CustomerCount"] / max(total_customers, 1) * 100).round(1)
    summary["RevenueSharePct"] = (summary["TotalRevenue"] / max(total_rev, 1) * 100).round(1)
    summary["AvgRevenue"] = summary["AvgRevenue"].round(2)
    summary["AvgRecency"] = summary["AvgRecency"].round(1)
    summary["AvgFrequency"] = summary["AvgFrequency"].round(1)
    summary["AvgAOV"] = summary["AvgAOV"].round(2)

    return summary


if __name__ == "__main__":
    df_raw = pd.read_csv("sample_transactions.csv")
    tx, rfm = process_rfm_pipeline(df_raw)
    print("RFM Pipeline Test Success!")
    print(f"Total customers: {len(rfm)}")
    print("\nSegment Distribution:")
    print(rfm["Segment"].value_counts())
    print("\nSegment Summary:")
    print(get_segment_summary(rfm))

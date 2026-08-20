"""
src/rfm_engine.py - Enterprise RFM-T Customer Segmentation & Action Engine
Calculates Recency, Frequency, Monetary, and Customer Tenure (RFM-T),
scores them on a 1-5 quintile scale, classifies customers into 7 distinct segments,
and generates actionable marketing playbooks.
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 7-Segment Color Palette
SEGMENT_COLORS = {
    "Champions": "#10B981",        # Vibrant Emerald
    "Loyalists": "#3B82F6",        # Royal Blue
    "Potential Growth": "#8B5CF6", # Violet / Indigo
    "At-Risk VIPs": "#F59E0B",     # Amber
    "Can't Lose Them": "#EF4444",  # Crimson / Red Alert
    "Hibernating": "#64748B",      # Slate Muted Grey
    "New Customers": "#06B6D4"     # Cyan / Turquoise
}

SEGMENT_ICONS = {
    "Champions": "👑",
    "Loyalists": "💎",
    "Potential Growth": "🚀",
    "At-Risk VIPs": "⚠️",
    "Can't Lose Them": "🚨",
    "Hibernating": "💤",
    "New Customers": "🌱"
}

# Segment Strategic Playbooks & Campaign Blueprints
SEGMENT_PLAYBOOKS = {
    "Champions": {
        "title": "Champions",
        "icon": "👑",
        "color": "#10B981",
        "badge": "Top Tier Advocates",
        "profile": "Bought recently, buy often, and generate top spend. Long tenured, loyal advocates.",
        "objective": "Reward loyalty, maximize retention, elevate brand advocacy, and introduce exclusive high-ticket drops.",
        "best_channel": "VIP Concierge / Direct Email / Exclusive Portal",
        "promo_type": "VIP Early Access, Concierge Perks, Loyalty Multipliers",
        "actions": [
            "Grant 48-hour early access to upcoming product lines.",
            "Assign dedicated VIP customer care specialist.",
            "Provide invite-only roundtables & beta product feedback rewards.",
            "Deliver anniversary appreciation gifts & tier status upgrades."
        ],
        "campaign_template": {
            "subject": "👑 VIP Exclusive: You have first access to our flagship release",
            "headline": "A Private Invitation for our Most Valued Member",
            "body": "Hi {Customer_Name},\n\nAs one of our top 1% community leaders, we want you to experience our newest drop before public release. Enjoy complimentary expedited delivery + double loyalty tokens on your order.",
            "cta": "Explore Private Drop"
        }
    },
    "Loyalists": {
        "title": "Loyalists",
        "icon": "💎",
        "color": "#3B82F6",
        "badge": "Steady Core Revenue",
        "profile": "Regular repeat shoppers with strong average order values and healthy tenure.",
        "objective": "Increase basket size, cross-sell complementary categories, and nurture into Champions.",
        "best_channel": "Automated Email Sequences / SMS Alerts",
        "promo_type": "Category Bundles, Volume Discounts, Referral Multipliers",
        "actions": [
            "Trigger automated cross-sell bundles based on preferred product categories.",
            "Provide tiered spending incentives (e.g. 'Spend $200, receive $40 bonus').",
            "Introduce gamified referral bonuses for bringing colleagues/peers.",
            "Send personalized quarterly product usage & care insights."
        ],
        "campaign_template": {
            "subject": "💎 Handpicked upgrades for your setup + Member Bonus",
            "headline": "Curated Exclusively for Your Taste",
            "body": "Hi {Customer_Name},\n\nThank you for choosing us over the past year! We've assembled custom accessory bundles that seamlessly pair with your past favorites. Enjoy 15% off when bundling today.",
            "cta": "Unlock Your Bundle"
        }
    },
    "Potential Growth": {
        "title": "Potential Growth",
        "icon": "🚀",
        "color": "#8B5CF6",
        "badge": "High Upside Accelerators",
        "profile": "Recent buyers with medium frequency or high basket sizes. Strong expansion potential.",
        "objective": "Build habitual repurchase cadence, educate on full product ecosystem, and increase frequency.",
        "best_channel": "Educational Email Drips / On-Site Personalization",
        "promo_type": "Next-Purchase Vouchers, Category Discovery Incentives",
        "actions": [
            "Enroll into multi-part ecosystem education workflows.",
            "Provide limited-time second-order incentive voucher.",
            "Highlight verified reviews of best-selling adjacent categories.",
            "Offer free trial of premium subscription or support warranty."
        ],
        "campaign_template": {
            "subject": "🚀 Take your experience further: $25 gift for your next order",
            "headline": "Elevate Your Routine",
            "body": "Hi {Customer_Name},\n\nWe noticed you recently explored our top-rated line. Here is a curated guide to maximizing performance, plus $25 off your next order with code GROW25.",
            "cta": "Claim $25 Credit"
        }
    },
    "At-Risk VIPs": {
        "title": "At-Risk VIPs",
        "icon": "⚠️",
        "color": "#F59E0B",
        "badge": "High Value Churn Danger",
        "profile": "Substantial lifetime spenders who have become inactive over the last 90–180 days.",
        "objective": "Immediate intervention to reignite brand affinity before permanent churn occurs.",
        "best_channel": "High-Priority Win-Back Email / SMS / Direct Outreach",
        "promo_type": "High-Value Reactivation Voucher (20-25% off), Free Renewal",
        "actions": [
            "Send personalized executive outreach asking for direct feedback.",
            "Offer significant time-sensitive reactivation coupon with clear expiration.",
            "Showcase all new features and products launched since their last visit.",
            "Trigger dedicated retargeting campaign on digital channels."
        ],
        "campaign_template": {
            "subject": "⚠️ We miss you, {Customer_Name} — Here is 25% off your return",
            "headline": "Let's Reconnect: Special Executive Offer Inside",
            "body": "Hi {Customer_Name},\n\nIt's been a while since your last order. We've rolled out major upgrades we know you'll love. Enjoy an exclusive 25% discount on anything in store for the next 7 days.",
            "cta": "Reclaim Your 25% Discount"
        }
    },
    "Can't Lose Them": {
        "title": "Can't Lose Them",
        "icon": "🚨",
        "color": "#EF4444",
        "badge": "Critical High-Spender Defection",
        "profile": "Made significant purchases and frequent orders in the past, but completely dormant for >180 days.",
        "objective": "High-touch, high-urgency win-back campaigns and qualitative feedback discovery.",
        "best_channel": "Executive Win-back Email / Phone Concierge / High-Incentive Push",
        "promo_type": "Aggressive 30% Off Voucher, Complimentary Premium Gift",
        "actions": [
            "Deploy 'Voice of Customer' survey with a substantial gift card incentive.",
            "Assign customer success lead for manual VIP relationship re-establishment.",
            "Offer complimentary product replacements or premium warranties.",
            "Provide steep liquidation or major flagship upgrade pricing."
        ],
        "campaign_template": {
            "subject": "🚨 An urgent message from our leadership team",
            "headline": "We Want to Make Things Right",
            "body": "Hi {Customer_Name},\n\nYou were one of our most valued foundational customers, and your absence hasn't gone unnoticed. Take 30% off your next purchase, or reply directly to this email with any feedback.",
            "cta": "Restore My VIP Status"
        }
    },
    "Hibernating": {
        "title": "Hibernating",
        "icon": "💤",
        "color": "#64748B",
        "badge": "Dormant Low-Tier Accounts",
        "profile": "Low frequency, low spend, and long inactivity (>200 days).",
        "objective": "Low-cost reactivation or systematic email hygiene to protect sender reputation.",
        "best_channel": "Automated Re-permission Email / Social Lookalike Exclusion",
        "promo_type": "Clearance Discounts, Re-Opt-in Incentives",
        "actions": [
            "Include in seasonal flash liquidation and clearance blasts.",
            "Execute automated opt-in verification / sunset workflow.",
            "Exclude from high-cost paid ads to prevent acquisition waste.",
            "Cleanse unengaged inboxes to maintain 99%+ deliverability."
        ],
        "campaign_template": {
            "subject": "💤 Should we say goodbye? Seasonal clearance inside",
            "headline": "Last Call for Your Member Profile",
            "body": "Hi {Customer_Name},\n\nWe are tidying our mailing lists. If you still want to hear about massive clearance events (up to 40% off), click below. Otherwise, no action is needed and we'll pause our emails.",
            "cta": "Stay Subscribed & Save"
        }
    },
    "New Customers": {
        "title": "New Customers",
        "icon": "🌱",
        "color": "#06B6D4",
        "badge": "Newly Acquired (0-60d)",
        "profile": "First purchase occurred recently with 1–2 orders. Clean slate for relationship building.",
        "objective": "Ensure exceptional first impression, guide product onboarding, and secure 2nd purchase.",
        "best_channel": "Onboarding Welcome Series / SMS Tips",
        "promo_type": "Welcome Voucher, Setup Guides, Community Access",
        "actions": [
            "Send immediate order confirmation with step-by-step setup guides.",
            "Deliver 3-part educational onboarding series over 14 days.",
            "Provide $15 second-purchase gift voucher expiring in 21 days.",
            "Prompt for quick satisfaction rating after delivery."
        ],
        "campaign_template": {
            "subject": "🌱 Welcome to the Family! Here is a gift for your journey",
            "headline": "Getting Started with Your New Purchase",
            "body": "Hi {Customer_Name},\n\nWelcome! We are thrilled to have you with us. Check out our quick setup walkthrough, and enjoy $15 off your next order when you return within 3 weeks with code WELCOME15.",
            "cta": "View Setup Guide & Claim $15"
        }
    }
}


def standardize_transactions(df: pd.DataFrame, custom_mapping: dict = None) -> pd.DataFrame:
    """
    Standardizes transaction columns from various e-commerce schemas.
    Accepts optional custom_mapping dict (e.g. {'CustomerID': 'client_id', 'PurchaseDate': 'tx_date', 'TotalSpend': 'amount'})
    """
    df_clean = df.copy()

    # Apply explicit user-defined custom mappings first
    if custom_mapping:
        rename_dict = {}
        for std_col, src_col in custom_mapping.items():
            if src_col and src_col in df_clean.columns:
                rename_dict[src_col] = std_col
        if rename_dict:
            df_clean = df_clean.rename(columns=rename_dict)

    # Standard auto-detection for any unmapped standard columns
    col_mapping = {}
    for col in df_clean.columns:
        if col in ["InvoiceNo", "CustomerID", "PurchaseDate", "ProductCategory", "Product", "Quantity", "UnitPrice", "TotalSpend"]:
            continue
        clean = col.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
        if clean in ["invoiceno", "invoice", "invoicenumber", "orderno", "orderid", "transactionid"]:
            col_mapping[col] = "InvoiceNo"
        elif clean in ["customerid", "customer", "customerno", "clientid", "userid"]:
            col_mapping[col] = "CustomerID"
        elif clean in ["purchasedate", "date", "invoicedate", "orderdate", "timestamp", "datetime"]:
            col_mapping[col] = "PurchaseDate"
        elif clean in ["productcategory", "category", "itemcategory", "dept"]:
            col_mapping[col] = "ProductCategory"
        elif clean in ["product", "productname", "item", "description", "sku"]:
            col_mapping[col] = "Product"
        elif clean in ["quantity", "qty", "count"]:
            col_mapping[col] = "Quantity"
        elif clean in ["unitprice", "price", "itemprice", "rate"]:
            col_mapping[col] = "UnitPrice"
        elif clean in ["totalspend", "total", "spend", "amount", "sales", "revenue"]:
            col_mapping[col] = "TotalSpend"

    df_clean = df_clean.rename(columns=col_mapping).copy()

    # Verify essential columns
    for req in ["CustomerID", "PurchaseDate"]:
        if req not in df_clean.columns:
            raise ValueError(f"Required column '{req}' is missing from dataset.")

    if "InvoiceNo" not in df_clean.columns:
        df_clean["InvoiceNo"] = [f"INV-{i+1:06d}" for i in range(len(df_clean))]
    if "ProductCategory" not in df_clean.columns:
        df_clean["ProductCategory"] = "General Merchandise"
    if "Product" not in df_clean.columns:
        df_clean["Product"] = "Catalog Product"
    if "Quantity" not in df_clean.columns:
        df_clean["Quantity"] = 1
    else:
        df_clean["Quantity"] = pd.to_numeric(df_clean["Quantity"], errors="coerce").fillna(1)

    if "UnitPrice" not in df_clean.columns:
        if "TotalSpend" in df_clean.columns:
            df_clean["UnitPrice"] = pd.to_numeric(df_clean["TotalSpend"], errors="coerce").fillna(0.0) / df_clean["Quantity"].replace(0, 1)
        else:
            df_clean["UnitPrice"] = 25.0
    else:
        df_clean["UnitPrice"] = pd.to_numeric(df_clean["UnitPrice"], errors="coerce").fillna(0.0)

    if "TotalSpend" not in df_clean.columns:
        df_clean["TotalSpend"] = round(df_clean["Quantity"] * df_clean["UnitPrice"], 2)
    else:
        df_clean["TotalSpend"] = pd.to_numeric(df_clean["TotalSpend"], errors="coerce").fillna(
            round(df_clean["Quantity"] * df_clean["UnitPrice"], 2)
        )

    df_clean["PurchaseDate"] = pd.to_datetime(df_clean["PurchaseDate"], errors="coerce")
    df_clean = df_clean.dropna(subset=["CustomerID", "PurchaseDate"]).copy()
    df_clean["CustomerID"] = df_clean["CustomerID"].astype(str)

    return df_clean


def compute_rfmt(df: pd.DataFrame, snapshot_date=None) -> pd.DataFrame:
    """
    Computes Recency, Frequency, Monetary, and Tenure (RFM-T) metrics for every customer.
    """
    if snapshot_date is None:
        snapshot_date = df["PurchaseDate"].max() + timedelta(days=1)
    else:
        snapshot_date = pd.to_datetime(snapshot_date)

    # Customer level RFM-T aggregations
    rfmt = df.groupby("CustomerID").agg(
        FirstPurchase=("PurchaseDate", "min"),
        LastPurchase=("PurchaseDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalSpend", "sum"),
        TotalItems=("Quantity", "sum"),
        TotalTransactions=("InvoiceNo", "count")
    ).reset_index()

    # Recency: Days since last purchase
    rfmt["Recency"] = (snapshot_date - rfmt["LastPurchase"]).dt.total_seconds() // 86400
    rfmt["Recency"] = rfmt["Recency"].clip(lower=0).astype(int)

    # Tenure: Days since first purchase
    rfmt["Tenure"] = (snapshot_date - rfmt["FirstPurchase"]).dt.total_seconds() // 86400
    rfmt["Tenure"] = rfmt["Tenure"].clip(lower=1).astype(int)

    rfmt["Monetary"] = rfmt["Monetary"].round(2)
    rfmt["AvgOrderValue"] = (rfmt["Monetary"] / rfmt["Frequency"].replace(0, 1)).round(2)

    # Favorite product category
    cat_counts = df.groupby(["CustomerID", "ProductCategory"])["TotalSpend"].sum().reset_index()
    cat_counts = cat_counts.sort_values(["CustomerID", "TotalSpend"], ascending=[True, False]).drop_duplicates("CustomerID")
    cat_map = dict(zip(cat_counts["CustomerID"], cat_counts["ProductCategory"]))
    rfmt["TopCategory"] = rfmt["CustomerID"].map(cat_map).fillna("General Merchandise")

    return rfmt


def calculate_rfmt_scores(rfmt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns 1-5 quintile scores for Recency, Frequency, Monetary, and Tenure.
    Uses rank-based qcut to prevent non-unique bin edge errors.
    """
    df_scored = rfmt_df.copy()
    n = len(df_scored)

    if n < 5:
        df_scored["R_Score"] = 5
        df_scored["F_Score"] = 5
        df_scored["M_Score"] = 5
        df_scored["T_Score"] = 5
    else:
        # Recency: Lower days = higher score (5 is most recent)
        df_scored["R_Score"] = pd.qcut(
            df_scored["Recency"].rank(method="first", ascending=False),
            q=5,
            labels=[1, 2, 3, 4, 5]
        ).astype(int)

        # Frequency: Higher count = higher score (5 is most frequent)
        df_scored["F_Score"] = pd.qcut(
            df_scored["Frequency"].rank(method="first", ascending=True),
            q=5,
            labels=[1, 2, 3, 4, 5]
        ).astype(int)

        # Monetary: Higher spend = higher score (5 is highest spend)
        df_scored["M_Score"] = pd.qcut(
            df_scored["Monetary"].rank(method="first", ascending=True),
            q=5,
            labels=[1, 2, 3, 4, 5]
        ).astype(int)

        # Tenure: Higher tenure = higher score (5 is oldest customer)
        df_scored["T_Score"] = pd.qcut(
            df_scored["Tenure"].rank(method="first", ascending=True),
            q=5,
            labels=[1, 2, 3, 4, 5]
        ).astype(int)

    df_scored["RFM_Score"] = (
        df_scored["R_Score"].astype(str) +
        df_scored["F_Score"].astype(str) +
        df_scored["M_Score"].astype(str)
    )
    df_scored["RFMT_Mean"] = ((df_scored["R_Score"] + df_scored["F_Score"] + df_scored["M_Score"] + df_scored["T_Score"]) / 4.0).round(2)

    return df_scored


def assign_7_segment_taxonomy(row: pd.Series) -> str:
    """
    Categorizes each customer into one of 7 distinct enterprise segments.
    """
    r = int(row["R_Score"])
    f = int(row["F_Score"])
    m = int(row["M_Score"])
    tenure_days = int(row["Tenure"])

    # 1. New Customers: Acquired recently (tenure <= 65 days) with low frequency
    if tenure_days <= 65 and r >= 4 and f <= 2:
        return "New Customers"

    # 2. Champions: High recency, high frequency, top spenders
    if (r >= 4 and f >= 4 and m >= 4) or (r == 5 and f >= 3 and m >= 4):
        return "Champions"

    # 3. Can't Lose Them: Formerly top spenders & frequent, but completely dormant (R=1)
    if r == 1 and (f >= 4 or m >= 4):
        return "Can't Lose Them"

    # 4. At-Risk VIPs: High/moderate spenders dormant for moderate period (R=2 or R=1 with F>=3)
    if (r <= 2 and (m >= 3 or f >= 3)):
        return "At-Risk VIPs"

    # 5. Potential Growth: Recent buyers with high monetary upside or emerging frequency
    if (r >= 4 and f <= 3) or (r >= 3 and m >= 4):
        return "Potential Growth"

    # 6. Loyalists: Consistent repeat buyers with solid frequency and spend
    if (r >= 3 and f >= 3) or (r >= 2 and f >= 4 and m >= 3):
        return "Loyalists"

    # 7. Hibernating: Lowest recency, frequency, and spend
    return "Hibernating"


def process_rfmt_pipeline(df: pd.DataFrame, snapshot_date=None, custom_mapping: dict = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full RFM-T processing pipeline:
    Returns (cleaned_transactions_df, rfmt_customer_df)
    """
    clean_tx = standardize_transactions(df, custom_mapping=custom_mapping)
    rfmt = compute_rfmt(clean_tx, snapshot_date=snapshot_date)
    rfmt_scored = calculate_rfmt_scores(rfmt)
    rfmt_scored["Segment"] = rfmt_scored.apply(assign_7_segment_taxonomy, axis=1)

    # Segment metadata
    rfmt_scored["Segment_Icon"] = rfmt_scored["Segment"].map(SEGMENT_ICONS)
    rfmt_scored["Segment_Color"] = rfmt_scored["Segment"].map(SEGMENT_COLORS)

    segment_order = [
        "Champions", "Loyalists", "Potential Growth",
        "At-Risk VIPs", "Can't Lose Them", "Hibernating", "New Customers"
    ]
    rfmt_scored["Segment"] = pd.Categorical(rfmt_scored["Segment"], categories=segment_order, ordered=True)

    return clean_tx, rfmt_scored


def get_segment_kpi_summary(rfmt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates summary KPIs for the 7 segments.
    """
    summary = rfmt_df.groupby("Segment", observed=False).agg(
        CustomerCount=("CustomerID", "count"),
        TotalRevenue=("Monetary", "sum"),
        AvgRevenue=("Monetary", "mean"),
        AvgRecency=("Recency", "mean"),
        AvgFrequency=("Frequency", "mean"),
        AvgTenure=("Tenure", "mean"),
        AvgAOV=("AvgOrderValue", "mean")
    ).reset_index()

    total_customers = len(rfmt_df)
    total_rev = rfmt_df["Monetary"].sum()

    summary["CustomerSharePct"] = (summary["CustomerCount"] / max(total_customers, 1) * 100).round(1)
    summary["RevenueSharePct"] = (summary["TotalRevenue"] / max(total_rev, 1) * 100).round(1)
    summary["AvgRevenue"] = summary["AvgRevenue"].round(2)
    summary["AvgRecency"] = summary["AvgRecency"].round(1)
    summary["AvgFrequency"] = summary["AvgFrequency"].round(1)
    summary["AvgTenure"] = summary["AvgTenure"].round(1)
    summary["AvgAOV"] = summary["AvgAOV"].round(2)

    return summary

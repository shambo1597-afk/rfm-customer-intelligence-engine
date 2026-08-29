"""
app.py - Enterprise Customer RFM-T & Customer Intelligence AI Platform
Multi-tab Streamlit dashboard combining RFM-T Segmentation, Unsupervised K-Means & 3D PCA,
Probabilistic BTYD CLV & Churn Radar, Cohort Retention Triangle, and What-If Campaign ROI Simulator.
"""

import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.rfm_engine import (
    process_rfmt_pipeline,
    get_segment_kpi_summary,
    ensure_series,
    RFMPipelineError,
    SEGMENT_COLORS,
    SEGMENT_ICONS,
    SEGMENT_PLAYBOOKS
)
from src.ml_engine import (
    preprocess_rfmt_features,
    evaluate_kmeans_candidates,
    perform_kmeans_clustering,
    compute_pca_3d,
    compute_segment_cluster_crosstab
)
from src.clv_engine import (
    estimate_btyd_clv,
    get_urgent_churn_watchlist,
    get_top_future_growth_targets
)
from src.cohort_engine import (
    compute_monthly_cohort_matrix,
    create_cohort_retention_heatmap,
    create_average_retention_curve
)
from src.shopify_ingest import (
    fetch_shopify_orders,
    shopify_dataframe_column_mapping,
    get_shopify_credentials,
    ShopifyIngestError
)
from src.digest_engine import (
    generate_account_digest,
    get_anthropic_api_key,
    get_groq_api_key,
    get_digest_provider_override
)
from src.chat_context import build_account_context_blob, build_context_text
from src.chat_engine import answer_account_question, escape_markdown_dollar_signs

# Configure Streamlit Page
st.set_page_config(
    page_title="Enterprise RFM-T & Customer Intelligence Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design System (Glassmorphic Dark Styling)
ENTERPRISE_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.01em;
    }

    /* Modern Glassmorphic KPI Cards */
    .kpi-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-container:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.5);
    }
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-family: 'Outfit', sans-serif;
        font-size: 1.85rem;
        font-weight: 800;
        color: #F8FAFC;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 0.80rem;
        font-weight: 500;
        margin-top: 4px;
    }
    .sub-green { color: #10B981; }
    .sub-blue { color: #38BDF8; }
    .sub-amber { color: #F59E0B; }
    .sub-red { color: #EF4444; }

    /* Enhanced Prominent Loaders & Status Elements */
    .stSpinner {
        text-align: center;
        padding: 18px 0;
    }
    .stSpinner > div {
        border-top-color: #38BDF8 !important;
        border-right-color: #818CF8 !important;
        border-bottom-color: #A855F7 !important;
        animation: spinGlow 0.75s cubic-bezier(0.4, 0, 0.2, 1) infinite;
    }
    .stSpinner span {
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.0rem !important;
        font-weight: 600 !important;
        color: #38BDF8 !important;
        letter-spacing: 0.02em !important;
        margin-left: 10px;
    }
    @keyframes spinGlow {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Status Pill Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 9999px;
        padding: 5px 14px;
        font-size: 0.80rem;
        font-weight: 600;
        color: #38BDF8;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 8px #10B981;
        animation: pulseAnimation 1.5s infinite;
    }
    @keyframes pulseAnimation {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.85); }
    }

    /* Action Playbook Cards */
    .playbook-box {
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 18px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: rgba(15, 23, 42, 0.75);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
    }

    /* Simulation Results Banner */
    .sim-card {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.4) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(59, 130, 246, 0.4);
        border-radius: 14px;
        padding: 20px;
        margin-top: 15px;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }

    /* Streamlit Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        border-radius: 8px 8px 0 0;
        padding: 10px 18px;
        font-weight: 600;
        font-family: 'Outfit', sans-serif;
    }
</style>
"""

st.markdown(ENTERPRISE_CSS, unsafe_allow_html=True)


# -------------------------------------------------------------------------------------------------
# Streamlit High-Performance Caching Layer
# -------------------------------------------------------------------------------------------------
@st.cache_data
def load_default_transactions(dataset_key: str = "enterprise"):
    if dataset_key == "uci":
        for uci_path in ["data/real_online_retail.csv.gz", "data/real_online_retail.csv"]:
            if os.path.exists(uci_path):
                return pd.read_csv(uci_path)
    for path in ["data/ecommerce_transactions.csv", "sample_transactions.csv"]:
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception:
                pass
    st.error("Default transaction dataset not found. Please upload a CSV.")
    return pd.DataFrame()


@st.cache_data(show_spinner="Computing RFM-T & Scoring Segments...")
def cached_process_rfmt_pipeline(df: pd.DataFrame, snapshot_date=None, custom_mapping: dict = None):
    return process_rfmt_pipeline(df, snapshot_date=snapshot_date, custom_mapping=custom_mapping)


@st.cache_data(show_spinner="Estimating BTYD P(Alive) & Predictive CLV...")
def cached_estimate_btyd_clv(rfmt_df: pd.DataFrame, prediction_horizon_days: int = 90, gross_margin: float = 0.35):
    return estimate_btyd_clv(rfmt_df, prediction_horizon_days=prediction_horizon_days, gross_margin=gross_margin)


@st.cache_data(show_spinner="Computing PCA 3D Projections...")
def cached_compute_pca_3d(clv_df: pd.DataFrame):
    return compute_pca_3d(clv_df)


@st.cache_data(show_spinner="Normalizing RFM-T Feature Space...")
def cached_preprocess_rfmt_features(clv_df: pd.DataFrame):
    return preprocess_rfmt_features(clv_df)


@st.cache_data(show_spinner="Evaluating Optimal Clusters...")
def cached_evaluate_kmeans_candidates(X_values: np.ndarray, min_k: int = 2, max_k: int = 7):
    return evaluate_kmeans_candidates(X_values, min_k=min_k, max_k=max_k)


@st.cache_data(show_spinner="Clustering Customer Profiles...")
def cached_perform_kmeans_clustering(clv_df: pd.DataFrame, n_clusters: int = 4):
    return perform_kmeans_clustering(clv_df, n_clusters=n_clusters)


@st.cache_data(show_spinner="Building Monthly Cohort Retention Matrix...")
def cached_compute_monthly_cohort_matrix(clean_tx: pd.DataFrame):
    return compute_monthly_cohort_matrix(clean_tx)


@st.cache_data(show_spinner="Generating AI executive summary...")
def cached_generate_account_digest(rfmt_df: pd.DataFrame, clv_df: pd.DataFrame,
                                    segment_summary: pd.DataFrame,
                                    anthropic_api_key: str = None,
                                    groq_api_key: str = None,
                                    provider_override: str = None):
    # Cached (keyed on the data + keys/override) so this only calls the LLM provider
    # once per account per batch run -- Streamlit reruns the whole script on every
    # widget interaction elsewhere in the app, and without this cache each of those
    # reruns would fire a fresh API call, silently multiplying the "one call per
    # account per batch run" cost model this feature was built around (see
    # src/digest_engine.py). Groq is preferred by default when both keys are set.
    return generate_account_digest(
        rfmt_df, clv_df, segment_summary,
        anthropic_api_key=anthropic_api_key,
        groq_api_key=groq_api_key,
        provider_override=provider_override
    )


@st.cache_data(show_spinner="Building account context for Chat Q&A...")
def cached_build_account_context_blob(rfmt_df: pd.DataFrame, clv_df: pd.DataFrame,
                                       segment_summary: pd.DataFrame,
                                       cohort_matrix: pd.DataFrame,
                                       crosstab_counts: pd.DataFrame):
    # Built ONCE per batch run (cached, same as every other pipeline output),
    # never rebuilt per question -- see src/chat_context.py's module docstring
    # for why this is a static blob, not a live query, and
    # src/chat_engine.py's cost-model section for what "once per batch run"
    # buys here specifically (it bounds the CONTEXT side of the cost, not the
    # per-question LLM call itself -- chat's cost model is usage-based, unlike
    # the digest's fixed-per-batch-run one).
    return build_account_context_blob(rfmt_df, clv_df, segment_summary, cohort_matrix, crosstab_counts)


def render_kpi(label: str, value: str, subtext: str = "", style: str = "blue"):
    sub_class = f"sub-{style}"
    st.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub {sub_class}">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def detect_default_column(columns: list[str], keywords: list[str], fallback_idx: int = 0) -> int:
    """
    Finds the index of the best matching column for a list of priority keywords.
    First checks exact matches across keywords in priority order,
    then checks substring matches across keywords in priority order.
    """
    if not columns:
        return 0
    cleaned_cols = [col.strip().lower().replace(" ", "").replace("_", "").replace("-", "") for col in columns]

    # Priority 1: Exact matches in priority order of keywords
    for kw in keywords:
        for i, c in enumerate(cleaned_cols):
            if c == kw:
                return i

    # Priority 2: Substring matches in priority order of keywords
    for kw in keywords:
        for i, c in enumerate(cleaned_cols):
            if kw in c:
                return i

    return min(fallback_idx, max(0, len(columns) - 1))


# -------------------------------------------------------------------------------------------------
# Sidebar: File Upload, Snapshot Date, Column Mapping & Analysis Settings
# -------------------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎯 Intelligence Hub")
    st.caption("Enterprise RFM-T & Customer Analytics")

    st.markdown("---")
    st.subheader("📁 Data Source")
    
    dataset_source_mode = st.radio(
        "Select Data Mode",
        options=["Preloaded Datasets", "Upload Custom File", "Connect Shopify Store"],
        horizontal=True
    )

    # Set by the Shopify branch below when synced data already has canonical column
    # names — skips the manual mapping dropdowns rather than asking the user to
    # re-map columns we already know.
    skip_column_mapping_ui = False

    if dataset_source_mode == "Connect Shopify Store":
        st.caption(
            "Pulls orders on demand from a connected Shopify store via the Admin API "
            "and feeds them into the same batch pipeline as an uploaded CSV — an "
            "ingest-path change only. RFM-T/K-Means/PCA/CLV scoring still runs "
            "in-process, zero API calls. See README § Connecting a Shopify Store."
        )
        default_shop, default_token = get_shopify_credentials()
        shopify_shop_domain = st.text_input(
            "Shop Domain",
            value=default_shop or "",
            placeholder="your-store.myshopify.com",
            help="Pre-filled from SHOPIFY_SHOP_DOMAIN in st.secrets/env if configured."
        )
        shopify_access_token = st.text_input(
            "Admin API Access Token",
            value=default_token or "",
            type="password",
            placeholder="shpat_...",
            help="Pre-filled from SHOPIFY_ACCESS_TOKEN in st.secrets/env if configured. Never logged."
        )
        sync_clicked = st.button("🔄 Sync Now", use_container_width=True)

        if sync_clicked:
            if not shopify_shop_domain or not shopify_access_token:
                st.error("Enter both a shop domain and an access token before syncing.")
            else:
                with st.spinner(f"Pulling orders from {shopify_shop_domain} via the Shopify Admin API..."):
                    try:
                        fetched_df = fetch_shopify_orders(shopify_shop_domain, shopify_access_token)
                        if fetched_df.empty:
                            st.warning("Sync succeeded but returned zero orders (check the date range / order status on the store).")
                        else:
                            st.session_state["shopify_orders_df"] = fetched_df
                            st.session_state["shopify_shop_label"] = shopify_shop_domain
                            st.success(f"Synced {len(fetched_df):,} line items from {shopify_shop_domain}.")
                    except ShopifyIngestError as e:
                        st.error(f"⚠️ Shopify sync failed: {e}")

        if "shopify_orders_df" in st.session_state:
            df_raw = st.session_state["shopify_orders_df"]
            data_source_label = f"Shopify: {st.session_state['shopify_shop_label']}"
            skip_column_mapping_ui = True
        else:
            st.info("Enter your store credentials above and click Sync Now to pull live orders.")
            df_raw = load_default_transactions("enterprise")
            data_source_label = "data/ecommerce_transactions.csv (Shopify not yet synced)"
    elif dataset_source_mode == "Upload Custom File":
        uploaded_file = st.file_uploader(
            "Upload Transactions (CSV / XLSX)",
            type=["csv", "xlsx", "xls"],
            help="Upload transactional records with CustomerID, PurchaseDate, and Spend/Product details."
        )
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df_raw = pd.read_csv(uploaded_file)
                else:
                    df_raw = pd.read_excel(uploaded_file)
                data_source_label = uploaded_file.name
                st.success(f"Loaded `{uploaded_file.name}` ({len(df_raw):,} records)")
            except Exception as e:
                st.error(f"Error loading file: {e}. Falling back to default data.")
                df_raw = load_default_transactions("enterprise")
                data_source_label = "ecommerce_transactions.csv (Fallback)"
        else:
            df_raw = load_default_transactions("enterprise")
            data_source_label = "data/ecommerce_transactions.csv"
            st.info("Upload your custom CSV or XLSX file above.")
    else:
        dataset_options = {
            "Enterprise Synthetic (450 Custs / 24 Months)": ("enterprise", "data/ecommerce_transactions.csv"),
            "Authentic UCI Online Retail (4,338 Custs / 397K Tx)": ("uci", "data/real_online_retail.csv")
        }
        selected_dataset_label = st.selectbox(
            "Choose Preloaded Dataset",
            options=list(dataset_options.keys()),
            index=0
        )
        ds_key, ds_path = dataset_options[selected_dataset_label]
        df_raw = load_default_transactions(ds_key)
        data_source_label = ds_path
        st.info(f"Using `{selected_dataset_label}`")

    if df_raw.empty:
        st.stop()

    raw_columns = list(df_raw.columns)

    # ---------------------------------------------------------------------------------------------
    # Dynamic Column Mapping Dropdowns
    # ---------------------------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🧩 Column Mapping")

    if skip_column_mapping_ui:
        # fetch_shopify_orders() already emits the pipeline's canonical column names --
        # reuse the same custom_mapping dict shape the manual dropdowns below build,
        # rather than asking the user to re-map columns we already know.
        column_mapping = shopify_dataframe_column_mapping()
        col_cust, col_date, col_spend = "CustomerID", "PurchaseDate", "TotalSpend"
        st.caption(
            "Auto-mapped from the Shopify sync (CustomerID, PurchaseDate, TotalSpend, "
            "InvoiceNo, ProductCategory) — no manual mapping needed."
        )
    else:
        cust_default_idx = detect_default_column(raw_columns, ["customerid", "customer", "customerno", "clientid", "userid", "user", "client", "id"], 0)
        date_default_idx = detect_default_column(raw_columns, ["purchasedate", "invoicedate", "orderdate", "date", "timestamp", "time", "datetime"], min(1, len(raw_columns)-1))
        spend_default_idx = detect_default_column(raw_columns, ["totalspend", "total", "spend", "amount", "sales", "revenue", "price", "unitprice"], min(2, len(raw_columns)-1))

        with st.expander("⚙️ Map Dataset Columns", expanded=True):
            st.caption("Select the columns matching each required engine field:")
            col_cust = st.selectbox(
                "Customer ID Column *",
                options=raw_columns,
                index=cust_default_idx,
                help="Unique customer identifier (e.g. CustomerID, User ID, Client ID)."
            )
            col_date = st.selectbox(
                "Transaction Date Column *",
                options=raw_columns,
                index=date_default_idx,
                help="Transaction timestamp or date (e.g. PurchaseDate, OrderDate, InvoiceDate)."
            )
            col_spend = st.selectbox(
                "Total Spend / Amount Column *",
                options=raw_columns,
                index=spend_default_idx,
                help="Monetary transaction value or total sales (e.g. TotalSpend, Amount, Sales)."
            )

            optional_options = ["(Auto-Detect / None)"] + raw_columns

            inv_default = detect_default_column(raw_columns, ["invoiceno", "invoice", "orderno", "orderid", "transactionid"], -1)
            inv_idx = (inv_default + 1) if inv_default >= 0 else 0
            col_invoice = st.selectbox(
                "Invoice / Order ID (Optional)",
                options=optional_options,
                index=inv_idx,
                help="Order or invoice number for frequency deduplication."
            )

            cat_default = detect_default_column(raw_columns, ["productcategory", "category", "itemcategory", "dept", "product"], -1)
            cat_idx = (cat_default + 1) if cat_default >= 0 else 0
            col_cat = st.selectbox(
                "Product / Category (Optional)",
                options=optional_options,
                index=cat_idx,
                help="Product category or item description for category affinity."
            )

        # Build active custom column mapping dictionary
        column_mapping = {
            "CustomerID": col_cust,
            "PurchaseDate": col_date,
            "TotalSpend": col_spend
        }
        if col_invoice != "(Auto-Detect / None)":
            column_mapping["InvoiceNo"] = col_invoice
        if col_cat != "(Auto-Detect / None)":
            column_mapping["ProductCategory"] = col_cat

    st.markdown("---")
    st.subheader("⚙️ Reference Timeline")

    # Detect max transaction date from the mapped date column
    try:
        parsed_dates = pd.to_datetime(ensure_series(df_raw[col_date]), errors="coerce")
        max_dt = parsed_dates.max()
        default_snapshot = (max_dt + timedelta(days=1)).date() if pd.notnull(max_dt) else datetime.now().date()
    except Exception:
        default_snapshot = datetime.now().date()

    use_custom_date = st.checkbox("Custom Snapshot Date", value=False)
    if use_custom_date:
        snapshot_date = st.date_input("Analysis Reference Date", value=default_snapshot)
    else:
        snapshot_date = default_snapshot

    st.markdown("---")
    st.subheader("🤖 AI Executive Summary")
    enable_ai_digest = st.checkbox(
        "Enable AI Digest — Groq (default) or Anthropic",
        value=False,
        help=(
            "Off by default. Generates ONE narrative paragraph per account per view "
            "from already-computed aggregate stats (never raw customer rows) — see "
            "README § AI Executive Summary (Optional) for the cost model "
            "(Groq's free tier — no credit card, genuinely $0 at this project's call "
            "volume — by default; ~$1/month on Anthropic at pilot volume; vs. "
            "~$208/month for a per-customer design). Configure GROQ_API_KEY "
            "and/or ANTHROPIC_API_KEY in st.secrets/env — either enables this feature, "
            "and Groq is used by default when both are present. See README for the "
            "current free-tier rate limits and data-usage disclosure before enabling "
            "in a production deployment."
        )
    )
    enable_chat_qa = st.checkbox(
        "Enable Chat Q&A — uses the same API key as AI Digest",
        value=False,
        help=(
            "Off by default. A separate feature from the AI Digest above — lets you "
            "ask natural-language questions about this account, answered ONLY from "
            "the same precomputed aggregate stats shown across this dashboard "
            "(segments, churn watchlist, growth targets, cohort retention, ML "
            "clustering) — never a live query, never raw per-customer rows, no "
            "tool-calling back into the pipeline mid-conversation. See README § Chat "
            "Q&A (Optional) — its cost model is USAGE-based (one call per question "
            "asked), unlike the AI Digest's fixed one-call-per-batch-run cost."
        )
    )
    resolve_llm_keys = enable_ai_digest or enable_chat_qa
    anthropic_api_key = get_anthropic_api_key() if resolve_llm_keys else None
    groq_api_key = get_groq_api_key() if resolve_llm_keys else None
    digest_provider_override = get_digest_provider_override() if resolve_llm_keys else None
    if resolve_llm_keys and not anthropic_api_key and not groq_api_key:
        st.caption("⚠️ No GROQ_API_KEY or ANTHROPIC_API_KEY found in st.secrets/env — AI Digest will show a template summary, and Chat Q&A will be unavailable.")

    st.markdown("---")
    st.caption("RFM-T / K-Means / PCA / CLV Scoring: 100% Baked-In • Zero External API Cost")


# -------------------------------------------------------------------------------------------------
# Core Processing Pipeline (RFM-T, CLV, ML, Cohorts)
# -------------------------------------------------------------------------------------------------
with st.spinner(f"⚡ Processing '{data_source_label}' & calculating RFM-T Customer Intelligence..."):
    try:
        clean_tx, rfmt_df = cached_process_rfmt_pipeline(df_raw, snapshot_date=snapshot_date, custom_mapping=column_mapping)
        clv_df = cached_estimate_btyd_clv(rfmt_df, prediction_horizon_days=90, gross_margin=0.35)
        rfmt_ml, pca_model, exp_var = cached_compute_pca_3d(clv_df)
    except RFMPipelineError as e:
        st.error(f"⚠️ Could not process this dataset: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error processing transaction dataset: {e}")
        st.stop()


# Detect parameter changes and trigger real-time feedback toast
active_state_sig = f"{data_source_label}#{snapshot_date}#{col_cust}#{col_date}#{col_spend}"
if "prev_state_sig" in st.session_state and st.session_state["prev_state_sig"] != active_state_sig:
    st.toast(f"Intelligence refreshed: {len(rfmt_df):,} customers loaded from {data_source_label}", icon="🎯")
st.session_state["prev_state_sig"] = active_state_sig


# -------------------------------------------------------------------------------------------------
# Batch-Level Context Outputs (segment KPIs, cohort retention, Segment x ML_Cluster crosstab)
# -------------------------------------------------------------------------------------------------
# Computed once per batch run, independent of any tab's own interactive widgets
# (e.g. tab 3's live k-selection slider) -- both the AI Digest and the Chat Q&A
# context blob need a STABLE snapshot of these, not one that silently changes
# depending on which tab the user last visited or what they happen to be
# exploring there. The clustering/crosstab step here picks k via the same
# Silhouette-argmax procedure tab 3 uses for its own default, wrapped
# defensively since clustering can legitimately fail on a very small or
# degenerate dataset -- build_account_context_blob() (src/chat_context.py)
# already treats a missing crosstab as "no ML clustering data available"
# rather than raising, so a failure here only narrows the context blob, never
# crashes the app.
segment_summary = get_segment_kpi_summary(clv_df)

try:
    _, batch_retention_matrix, _ = cached_compute_monthly_cohort_matrix(clean_tx)
except Exception:
    batch_retention_matrix = None

try:
    X_scaled_batch, _, _ = cached_preprocess_rfmt_features(clv_df)
    eval_df_batch = cached_evaluate_kmeans_candidates(X_scaled_batch, min_k=2, max_k=7)
    optimal_k_batch = int(eval_df_batch.loc[eval_df_batch["Silhouette_Score"].idxmax()]["k"])
    df_clustered_batch, _, _ = cached_perform_kmeans_clustering(clv_df, n_clusters=optimal_k_batch)
    batch_crosstab_counts, _ = compute_segment_cluster_crosstab(df_clustered_batch)
except Exception:
    batch_crosstab_counts = None


# -------------------------------------------------------------------------------------------------
# Top Executive Header & Global KPI Banner
# -------------------------------------------------------------------------------------------------
hdr_left, hdr_right = st.columns([3, 1])

with hdr_left:
    st.title("🎯 Customer RFM-T & AI Intelligence Platform")
    st.markdown(
        f"""
        <div style='display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-top: -6px;'>
            <span class='status-badge'><span class='pulse-dot'></span> Live AI Engine Active</span>
            <span style='color:#94A3B8; font-size:0.88rem;'><strong>Source:</strong> <code>{data_source_label}</code></span>
            <span style='color:#94A3B8; font-size:0.88rem;'><strong>Snapshot:</strong> <code>{snapshot_date}</code></span>
            <span style='color:#94A3B8; font-size:0.88rem;'><strong>Customers:</strong> <strong style='color:#F8FAFC;'>{len(rfmt_df):,}</strong></span>
            <span style='color:#94A3B8; font-size:0.88rem;'><strong>Transactions:</strong> <strong style='color:#F8FAFC;'>{len(clean_tx):,}</strong></span>
        </div>
        """,
        unsafe_allow_html=True
    )

with hdr_right:
    full_export_csv = clv_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Full Intelligence (CSV)",
        data=full_export_csv,
        file_name=f"customer_intelligence_export_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

# Compute Top Global KPIs
total_custs = len(clv_df)
active_custs = (clv_df["P_Alive"] >= 0.50).sum()
active_pct = (active_custs / max(total_custs, 1)) * 100
total_historical_rev = clv_df["Monetary"].sum()
total_pred_90d_rev = clv_df["Predicted_Spend_90d"].sum()
avg_tenure_days = clv_df["Tenure"].mean()

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    render_kpi("Total Customers", f"{total_custs:,}", "Total unique accounts", "blue")
with kpi2:
    render_kpi("Active Rate", f"{active_pct:.1f}%", f"{active_custs:,} with P(Alive) ≥ 50%", "green")
with kpi3:
    render_kpi("Realized Revenue", f"${total_historical_rev:,.0f}", "Historical lifetime spend", "blue")
with kpi4:
    render_kpi("90-Day Forecast", f"${total_pred_90d_rev:,.0f}", "Predicted future revenue", "green")
with kpi5:
    render_kpi("Avg Customer Tenure", f"{avg_tenure_days:.0f} days", "Since initial acquisition", "amber")

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)


# -------------------------------------------------------------------------------------------------
# 5 Interactive Enterprise Tabs
# -------------------------------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive KPI & Cohort Matrix",
    "🎯 RFM-T Rule Segmentation",
    "🤖 Unsupervised ML Clustering",
    "🔮 Predictive CLV & Churn Radar",
    "💰 What-If ROI Simulator & Playbook"
])


# -------------------------------------------------------------------------------------------------
# TAB 1: Executive KPI & Triangle Cohort Retention Matrix
# -------------------------------------------------------------------------------------------------
with tab1:
    st.subheader("📈 Month-over-Month Acquisition Cohort Retention")
    st.caption("Track retention decay dynamics across monthly customer acquisition cohorts over a 24-month horizon.")

    with st.spinner("📈 Analyzing monthly acquisition cohorts & computing retention triangle matrix..."):
        try:
            count_matrix, retention_matrix, cohort_sizes = cached_compute_monthly_cohort_matrix(clean_tx)
            
            c_left, c_right = st.columns([2, 1])
            with c_left:
                fig_cohort = create_cohort_retention_heatmap(retention_matrix, count_matrix)
                st.plotly_chart(fig_cohort, use_container_width=True)
                
            with c_right:
                fig_curve = create_average_retention_curve(retention_matrix)
                st.plotly_chart(fig_curve, use_container_width=True)
                
                # Retention Health Check
                m1_ret = retention_matrix.mean().get(1, 0)
                m3_ret = retention_matrix.mean().get(3, 0)
                m6_ret = retention_matrix.mean().get(6, 0)
                
                st.markdown(
                    f"""
                    <div class="kpi-container" style="margin-top: 10px;">
                        <div class="kpi-label">Retention Benchmarks</div>
                        <div style="font-size: 0.90rem; line-height: 1.8; color: #E2E8F0;">
                            <div>⚡ <strong>Month 1 Retention:</strong> <span class="sub-green">{m1_ret:.1f}%</span></div>
                            <div>⚡ <strong>Month 3 Retention:</strong> <span class="sub-blue">{m3_ret:.1f}%</span></div>
                            <div>⚡ <strong>Month 6 Retention:</strong> <span class="sub-amber">{m6_ret:.1f}%</span></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
        except Exception as err:
            st.warning(f"Unable to render cohort matrix: {err}")


# -------------------------------------------------------------------------------------------------
# TAB 2: RFM-T Rule Segmentation
# -------------------------------------------------------------------------------------------------
with tab2:
    st.subheader("🎯 Enterprise RFM-T 7-Segment Value Hierarchy")
    st.caption("Customer classification using Recency, Frequency, Monetary, and Tenure quintile scoring (1-5 scale).")

    # segment_summary is computed once at the batch level (above, alongside the
    # other Chat Q&A / AI Digest context inputs) rather than re-derived here.

    col_treemap, col_charts = st.columns([1, 1])

    with col_treemap:
        st.markdown("**Segment Revenue & Volume Hierarchy Treemap**")
        fig_tree = px.treemap(
            clv_df,
            path=["Segment", "CustomerID"],
            values="Monetary",
            color="Segment",
            color_discrete_map=SEGMENT_COLORS,
            template="plotly_dark"
        )
        fig_tree.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=400)
        st.plotly_chart(fig_tree, use_container_width=True)

    with col_charts:
        st.markdown("**Segment Share: Customer Volume vs. Revenue Contribution**")
        fig_donut = px.pie(
            segment_summary,
            names="Segment",
            values="TotalRevenue",
            color="Segment",
            color_discrete_map=SEGMENT_COLORS,
            hole=0.45,
            template="plotly_dark"
        )
        fig_donut.update_traces(textposition='inside', textinfo='percent+label')
        fig_donut.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=400, showlegend=False)
        st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Segment Performance Benchmark Summary")

    display_seg = segment_summary.rename(columns={
        "CustomerCount": "Customers",
        "CustomerSharePct": "Cust %",
        "TotalRevenue": "Total Revenue ($)",
        "RevenueSharePct": "Rev %",
        "AvgRevenue": "Avg Spend / Cust ($)",
        "AvgRecency": "Avg Recency (Days)",
        "AvgFrequency": "Avg Orders",
        "AvgTenure": "Avg Tenure (Days)",
        "AvgAOV": "Avg AOV ($)"
    })
    
    st.dataframe(
        display_seg,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Total Revenue ($)": st.column_config.NumberColumn(format="$%,.2f"),
            "Avg Spend / Cust ($)": st.column_config.NumberColumn(format="$%,.2f"),
            "Avg AOV ($)": st.column_config.NumberColumn(format="$%,.2f"),
            "Cust %": st.column_config.NumberColumn(format="%.1f%%"),
            "Rev %": st.column_config.NumberColumn(format="%.1f%%")
        }
    )

    st.markdown("---")
    with st.expander("🤖 AI Executive Summary", expanded=enable_ai_digest):
        st.caption(
            "One narrative paragraph built from the aggregate stats above (segment "
            "sizes, % at-risk, 90-day forecast) — never raw per-customer rows. "
            "Off by default; enable via the sidebar. See README § AI Executive "
            "Summary (Optional) for the cost model."
        )
        if enable_ai_digest:
            digest_text = cached_generate_account_digest(
                rfmt_df, clv_df, segment_summary,
                anthropic_api_key=anthropic_api_key,
                groq_api_key=groq_api_key,
                provider_override=digest_provider_override
            )
            st.markdown(digest_text)
            if not anthropic_api_key and not groq_api_key:
                st.caption("ℹ️ Showing the template summary — configure GROQ_API_KEY (free) or ANTHROPIC_API_KEY for the AI-generated version.")
        else:
            st.info("Enable \"AI Digest\" in the sidebar to generate this summary.")

    st.markdown("---")
    with st.expander("💬 Ask About This Account", expanded=False):
        st.caption(
            "Ask natural-language questions about this account — answered ONLY "
            "from the same precomputed aggregate stats shown across this "
            "dashboard (segments, churn watchlist, growth targets, cohort "
            "retention, ML clustering) — never a live query against raw "
            "customer rows, never tool-calling back into the pipeline "
            "mid-conversation. A separate, optional feature from the AI Digest "
            "above; off by default. See README § Chat Q&A (Optional) for the "
            "usage-based cost model."
        )
        if enable_chat_qa:
            # Reset the conversation whenever the active account/dataset changes
            # (active_state_sig, computed once above) -- stale chat history from
            # a previous dataset must never bleed into a new one.
            if st.session_state.get("chat_context_sig") != active_state_sig:
                st.session_state["chat_conversation_history"] = []
                st.session_state["chat_context_sig"] = active_state_sig

            context_blob = cached_build_account_context_blob(
                rfmt_df, clv_df, segment_summary, batch_retention_matrix, batch_crosstab_counts
            )
            context_text = build_context_text(context_blob)

            # escape_markdown_dollar_signs() at every render site below --
            # Streamlit's markdown renderer treats bare '$' pairs as LaTeX
            # math delimiters by default (confirmed via the installed
            # package's own JS bundle, not assumed -- see that function's
            # docstring in src/chat_engine.py for the full root-cause and why
            # this escaping, not a Streamlit config flag, is the actual fix).
            # Applied to every message in this panel, not just the model's
            # answer, since a user typing a literal '$' in their own question
            # would hit the exact same bug when echoed back.
            for turn in st.session_state["chat_conversation_history"]:
                with st.chat_message(turn["role"]):
                    st.markdown(escape_markdown_dollar_signs(turn["content"]))

            question = st.chat_input("Ask a question about this account...")
            if question:
                with st.chat_message("user"):
                    st.markdown(escape_markdown_dollar_signs(question))
                with st.spinner("Thinking..."):
                    answer = answer_account_question(
                        question, context_text,
                        st.session_state["chat_conversation_history"],
                        anthropic_api_key=anthropic_api_key,
                        groq_api_key=groq_api_key,
                        provider_override=digest_provider_override,
                    )
                with st.chat_message("assistant"):
                    st.markdown(escape_markdown_dollar_signs(answer))
                # answer_account_question() already appended the successful turn
                # to chat_conversation_history in place -- see its docstring. A
                # failed turn is intentionally NOT recorded, so nothing more to
                # do here either way.

            if not anthropic_api_key and not groq_api_key:
                st.caption("ℹ️ Configure GROQ_API_KEY (free) or ANTHROPIC_API_KEY to enable Chat Q&A answers.")
        else:
            st.info("Enable \"Chat Q&A\" in the sidebar to ask questions about this account.")


# -------------------------------------------------------------------------------------------------
# TAB 3: Unsupervised Machine Learning Clustering & PCA 3D
# -------------------------------------------------------------------------------------------------
with tab3:
    st.subheader("🤖 Unsupervised Machine Learning (K-Means & PCA 3D)")
    st.caption("Normalized feature space ($\log(1+x)$ + StandardScaler) with dynamic Silhouette Optimization and 3D PCA projection.")

    # Evaluate k=2 to k=7 on the SAME log1p + StandardScaler feature space used by the
    # final K-Means fit below (perform_kmeans_clustering), so the Silhouette/Elbow curves
    # and the recommended "optimal k" accurately reflect the model that actually gets fit —
    # rather than a 3D PCA-reduced projection that discards one dimension of information.
    with st.spinner("🤖 Optimizing cluster partitions across k=2...7 via Silhouette scoring..."):
        X_scaled, _, _ = cached_preprocess_rfmt_features(clv_df)
        eval_df = cached_evaluate_kmeans_candidates(X_scaled, min_k=2, max_k=7)
        optimal_k = int(eval_df.loc[eval_df["Silhouette_Score"].idxmax()]["k"])

    c_sel, c_opt = st.columns([2, 1])
    with c_sel:
        selected_k = st.slider("Select Number of Clusters (k)", min_value=2, max_value=7, value=optimal_k)
    with c_opt:
        st.markdown(
            f"""
            <div class="kpi-container" style="padding: 10px 16px; margin-top: 4px;">
                <div class="kpi-label">Algorithm Recommendation</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #10B981;">Optimal k = {optimal_k}</div>
                <div style="font-size: 0.75rem; color: #94A3B8;">Max Silhouette Score: {eval_df.loc[eval_df['k']==optimal_k, 'Silhouette_Score'].values[0]:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Perform clustering with spinner
    with st.spinner(f"🤖 Fitting K-Means (k={selected_k}) & calculating 3D PCA coordinates..."):
        df_clustered, km_model, cluster_summary = cached_perform_kmeans_clustering(clv_df, n_clusters=selected_k)
        df_clustered, _, exp_variance = cached_compute_pca_3d(df_clustered)

    col_sil, col_elbow = st.columns([1, 1])
    with col_sil:
        fig_sil = px.line(
            eval_df,
            x="k",
            y="Silhouette_Score",
            markers=True,
            title="<b>Silhouette Score Analysis (Higher is Better)</b>",
            template="plotly_dark",
            labels={"Silhouette_Score": "Silhouette Score", "k": "Clusters (k)"}
        )
        fig_sil.add_vline(x=selected_k, line_dash="dash", line_color="#38BDF8", annotation_text=f"Selected k={selected_k}")
        fig_sil.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_sil, use_container_width=True)

    with col_elbow:
        fig_elbow = px.line(
            eval_df,
            x="k",
            y="Inertia",
            markers=True,
            title="<b>Elbow Method (Inertia Decay)</b>",
            template="plotly_dark",
            labels={"Inertia": "Within-Cluster Sum of Squares", "k": "Clusters (k)"}
        )
        fig_elbow.add_vline(x=selected_k, line_dash="dash", line_color="#F59E0B")
        fig_elbow.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_elbow, use_container_width=True)

    st.markdown("---")
    st.subheader("🌐 3D Spatial PCA Cluster Visualization")
    st.caption(f"Principal Components Variance Explained: PC1: {exp_variance[0]}% | PC2: {exp_variance[1]}% | PC3: {exp_variance[2]}% (Total: {exp_variance[:3].sum():.1f}%)")

    fig_3d_pca = px.scatter_3d(
        df_clustered,
        x="PCA_1",
        y="PCA_2",
        z="PCA_3",
        color="ML_Cluster",
        hover_name="CustomerID",
        hover_data={
            "Segment": True,
            "Monetary": ":$,.2f",
            "Recency": True,
            "Frequency": True,
            "Tenure": True,
            "PCA_1": False,
            "PCA_2": False,
            "PCA_3": False
        },
        size="Monetary",
        size_max=24,
        opacity=0.85,
        template="plotly_dark"
    )
    fig_3d_pca.update_layout(
        height=580,
        margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(
            xaxis_title="Principal Component 1 (Volume/Spend)",
            yaxis_title="Principal Component 2 (Tenure/Recency)",
            zaxis_title="Principal Component 3 (AOV)"
        ),
        legend=dict(orientation="h", yanchor="bottom", y=0.02, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_3d_pca, use_container_width=True)

    st.markdown("### 📊 ML Cluster Profiling Matrix")
    st.dataframe(cluster_summary, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🔎 Segment × ML Cluster Agreement")
    st.caption(
        "The 7-segment rule taxonomy and this K-Means clustering are computed completely "
        "independently — neither one knows about the other. This crosstab is a sanity check "
        "on the rule thresholds, not a merge into a new label: where a segment's customers land "
        "overwhelmingly in one cluster, that corroborates the rule boundaries; where a segment "
        "splits roughly evenly across clusters, its quintile thresholds may be cutting across a "
        "real behavioral grouping and could be worth revisiting."
    )
    seg_cluster_counts, seg_cluster_pct = compute_segment_cluster_crosstab(df_clustered)
    fig_crosstab = px.imshow(
        seg_cluster_pct,
        text_auto=".0f",
        color_continuous_scale="Blues",
        labels=dict(x="ML Cluster", y="Segment", color="% of Segment"),
        template="plotly_dark",
        aspect="auto"
    )
    fig_crosstab.update_layout(height=360, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_crosstab, use_container_width=True)
    with st.expander("View raw Segment × ML Cluster customer counts"):
        st.dataframe(seg_cluster_counts, use_container_width=True)


# -------------------------------------------------------------------------------------------------
# TAB 4: Predictive CLV & Real-Time Churn Radar
# -------------------------------------------------------------------------------------------------
with tab4:
    st.subheader("🔮 Probabilistic BTYD CLV & Real-Time Churn Radar")
    st.caption("Continuous-time Buy-Till-You-Die (BTYD) P(Alive) estimation and 90-day forward value projection.")

    with st.spinner("🔮 Calculating active probabilities P(Alive) & churn risks..."):
        # Churn Tier Summary
        churn_counts = clv_df["Churn_Risk_Tier"].value_counts()
        c_low = churn_counts.get("🟢 Low Churn Risk", 0)
        c_mod = churn_counts.get("🟡 Moderate Watch", 0)
        c_high = churn_counts.get("🔴 High Churn Risk", 0)

        ct1, ct2, ct3 = st.columns(3)
        with ct1:
            render_kpi("Low Churn Risk", f"{c_low:,}", f"{c_low/total_custs*100:.1f}% healthy customers (P(Alive) ≥ 75%)", "green")
        with ct2:
            render_kpi("Moderate Watch", f"{c_mod:,}", f"{c_mod/total_custs*100:.1f}% showing decay (45% ≤ P(Alive) < 75%)", "amber")
        with ct3:
            render_kpi("High Churn Risk", f"{c_high:,}", f"{c_high/total_custs*100:.1f}% critical danger (P(Alive) < 45%)", "red")

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        col_pdist, col_scatter_clv = st.columns([1, 1])

        with col_pdist:
            fig_p_alive = px.histogram(
                clv_df,
                x="P_Alive_Pct",
                nbins=30,
                color="Churn_Risk_Tier",
                color_discrete_map={
                    "🟢 Low Churn Risk": "#10B981",
                    "🟡 Moderate Watch": "#F59E0B",
                    "🔴 High Churn Risk": "#EF4444"
                },
                title="<b>P(Alive) Customer Active Probability Distribution</b>",
                template="plotly_dark",
                labels={"P_Alive_Pct": "P(Alive) %"}
            )
            fig_p_alive.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_p_alive, use_container_width=True)

        with col_scatter_clv:
            fig_scatter_clv = px.scatter(
                clv_df,
                x="Monetary",
                y="Predicted_Spend_90d",
                color="Segment",
                color_discrete_map=SEGMENT_COLORS,
                size="P_Alive_Pct",
                hover_name="CustomerID",
                hover_data={
                    "P_Alive_Pct": ":.1f%",
                    "Expected_Orders_90d": True,
                    "Predictive_CLV_90d": ":$,.2f"
                },
                title="<b>Historical Spend vs. 90-Day Forecasted Revenue</b>",
                template="plotly_dark",
                labels={"Monetary": "Historical Spend ($)", "Predicted_Spend_90d": "Predicted 90-Day Spend ($)"}
            )
            fig_scatter_clv.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_scatter_clv, use_container_width=True)

    st.markdown("---")

    # Urgent Churn Watchlist Table
    st.subheader("🚨 Urgent Churn Watchlist (High-Spend Accounts in Critical Decay)")
    st.caption("Customers with above-median historical spend whose active probability $P(\\text{Alive}) < 45\\%$. Immediate intervention required.")

    urgent_watchlist = get_urgent_churn_watchlist(clv_df, p_alive_threshold=0.45)

    col_tbl, col_dl = st.columns([3, 1])
    with col_tbl:
        st.markdown(f"**Found {len(urgent_watchlist)} high-value at-risk accounts.**")
    with col_dl:
        watchlist_csv = urgent_watchlist.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Churn Watchlist (CSV)",
            data=watchlist_csv,
            file_name="urgent_churn_intervention_watchlist.csv",
            mime="text/csv",
            key="dl_watchlist"
        )

    watch_display = urgent_watchlist[[
        "CustomerID", "Segment", "P_Alive_Pct", "Churn_Risk_Tier",
        "Monetary", "Frequency", "Recency", "Tenure", "AvgOrderValue", "TopCategory"
    ]].copy()

    watch_display["Monetary"] = watch_display["Monetary"].map("${:,.2f}".format)
    watch_display["AvgOrderValue"] = watch_display["AvgOrderValue"].map("${:,.2f}".format)
    watch_display["P_Alive_Pct"] = watch_display["P_Alive_Pct"].map("{:.1f}%".format)
    watch_display["Recency"] = watch_display["Recency"].map("{} days".format)

    st.dataframe(watch_display, hide_index=True, use_container_width=True)


# -------------------------------------------------------------------------------------------------
# TAB 5: What-If Marketing ROI Simulator & Action Playbooks
# -------------------------------------------------------------------------------------------------
with tab5:
    st.subheader("💰 'What-If' Marketing Campaign ROI & Action Playbook Simulator")
    st.caption("Simulate expected revenue, incremental margin, net ROI, and payback period across targeted segments.")

    st.markdown(
        """
        <div style='margin-bottom: 14px;'>
            <span class='status-badge'>
                <span class='pulse-dot'></span> Real-Time ROI Engine Active
            </span>
            &nbsp; <span style='font-size:0.84rem; color:#94A3B8;'>Adjust inputs below to simulate instant financial returns.</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    all_segs = list(SEGMENT_PLAYBOOKS.keys())

    col_inputs, col_sim_results = st.columns([1, 1])

    with col_inputs:
        st.markdown("### 🎛️ Campaign Configuration")
        target_segment = st.selectbox("1. Target Customer Segment", all_segs, index=3)  # Default: At-Risk VIPs
        
        seg_cust_df = clv_df[clv_df["Segment"] == target_segment]
        max_reach = len(seg_cust_df)
        
        target_audience_pct = st.slider("2. Segment Penetration Reach (%)", min_value=10, max_value=100, value=80, step=5)
        audience_size = int(max_reach * (target_audience_pct / 100.0))
        
        campaign_budget = st.number_input("3. Total Campaign Budget ($)", min_value=500, max_value=50000, value=3500, step=500)
        expected_conv_rate = st.slider("4. Expected Campaign Conversion Rate (%)", min_value=1.0, max_value=30.0, value=8.5, step=0.5)
        gross_margin_pct = st.slider("5. Product Gross Margin (%)", min_value=15, max_value=85, value=40, step=5)

    # Calculate Simulation Financials
    avg_segment_aov = seg_cust_df["AvgOrderValue"].mean() if len(seg_cust_df) > 0 else 150.0
    projected_conversions = audience_size * (expected_conv_rate / 100.0)
    projected_gross_revenue = projected_conversions * avg_segment_aov
    projected_gross_profit = projected_gross_revenue * (gross_margin_pct / 100.0)
    net_incremental_profit = projected_gross_profit - campaign_budget
    campaign_roi_pct = (net_incremental_profit / max(campaign_budget, 1)) * 100.0
    cost_per_acquisition = campaign_budget / max(projected_conversions, 1)

    with col_sim_results:
        st.markdown("### 📊 Projected Financial ROI")
        
        roi_color = "sub-green" if net_incremental_profit > 0 else "sub-red"
        st.markdown(
            f"""
            <div class="sim-card">
                <div style="font-size: 0.85rem; color: #94A3B8; text-transform: uppercase;">Campaign Performance Estimate</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #F8FAFC; margin-top: 4px;">
                    ${projected_gross_revenue:,.2f}
                </div>
                <div style="font-size: 0.95rem; margin-top: 6px;" class="{roi_color}">
                    <strong>Net Incremental Profit:</strong> ${net_incremental_profit:,.2f} ({campaign_roi_pct:+.1f}% Net ROI)
                </div>
                <hr style="border-color: rgba(255,255,255,0.15); margin: 14px 0;" />
                <div style="font-size: 0.90rem; line-height: 1.8; color: #E2E8F0;">
                    <div>👥 <strong>Targeted Audience:</strong> {audience_size:,} of {max_reach:,} customers</div>
                    <div>🎯 <strong>Projected Orders:</strong> {projected_conversions:.1f} orders</div>
                    <div>💵 <strong>Estimated Segment AOV:</strong> ${avg_segment_aov:,.2f}</div>
                    <div>💳 <strong>Cost Per Converted Order:</strong> ${cost_per_acquisition:,.2f}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Segment Playbook & Actionable Campaign
    st.subheader(f"🚀 Targeted Action Playbook: {SEGMENT_ICONS[target_segment]} {target_segment}")
    playbook = SEGMENT_PLAYBOOKS[target_segment]

    st.markdown(
        f"""
        <div class="playbook-box" style="border-left: 6px solid {playbook['color']};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h3 style="margin: 0; color: {playbook['color']};">{playbook['icon']} {playbook['title']}</h3>
                <span class="badge" style="background: {playbook['color']}22; color: {playbook['color']}; border: 1px solid {playbook['color']};">
                    {playbook['badge']}
                </span>
            </div>
            <p style="font-size: 1.0rem; color: #E2E8F0; margin-top: 8px;">{playbook['profile']}</p>
            <div style="background: rgba(255,255,255,0.04); border-radius: 8px; padding: 10px 16px; margin-top: 8px;">
                <strong style="color: #38BDF8;">🎯 Strategic Objective:</strong> {playbook['objective']}
            </div>
            <div style="margin-top: 10px;">
                <strong>📡 Best Channels:</strong> <span class="badge" style="background:#334155; color:#F8FAFC;">{playbook['best_channel']}</span> &nbsp;
                <strong>🎁 Recommended Promo:</strong> <span class="badge" style="background:#334155; color:#F8FAFC;">{playbook['promo_type']}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c_acts, c_email = st.columns([1, 1])

    with c_acts:
        st.markdown("#### 📋 Recommended Playbook Action Items")
        for act in playbook["actions"]:
            st.markdown(f"- ⚡ **{act}**")

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        # Segment download button
        target_export_csv = seg_cust_df.head(audience_size).to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📥 Export Targeted Audience List ({audience_size} Customers CSV)",
            data=target_export_csv,
            file_name=f"campaign_target_{target_segment.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            key="dl_target_campaign"
        )

    with c_email:
        st.markdown("#### ✉️ Campaign Copy Blueprint")
        camp = playbook["campaign_template"]
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px dashed rgba(148, 163, 184, 0.3); border-radius: 12px; padding: 16px;">
                <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 4px;">
                    <strong>Subject:</strong> {camp['subject']}
                </div>
                <div style="font-size: 0.95rem; font-weight: 700; color: #38BDF8; margin-bottom: 8px;">
                    {camp['headline']}
                </div>
                <div style="font-size: 0.92rem; color: #F8FAFC; line-height: 1.6; white-space: pre-wrap;">
{camp['body']}
                </div>
                <div style="margin-top: 12px;">
                    <span style="background: #3B82F6; color: white; padding: 6px 14px; border-radius: 6px; font-size: 0.85rem; font-weight: 600;">
                        🔘 {camp['cta']}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

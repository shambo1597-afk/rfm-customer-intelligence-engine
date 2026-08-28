"""
src/shopify_ingest.py - Shopify Admin API transaction ingest.

Pulls orders from a connected Shopify store via the Admin REST API and maps them
into the exact transaction schema the existing batch pipeline already consumes --
CustomerID, InvoiceNo, PurchaseDate, TotalSpend, Quantity, UnitPrice, Product,
ProductCategory (matching data/real_online_retail.csv.gz's schema exactly). Once
mapped, the DataFrame is handed to the same standardize_transactions() /
process_rfmt_pipeline() entry point a CSV/Excel upload already goes through.

This is an INGEST-PATH change only. RFM-T, K-Means, PCA, and the CLV/churn model
all continue to run in-process on the returned DataFrame, unchanged -- zero API
calls for the scoring step itself, exactly as with a manual CSV upload.

Cost note: Shopify's Admin REST API has no per-call charge at this data volume for
a standard/custom app -- calls are governed by the leaky-bucket rate limit handled
below (HTTP 429 + Retry-After), not billed per request. That is why this
integration only touches the "Ingest & Pipeline" cost category and never "Model /
API Inference" -- there is no metered inference anywhere in this module.
"""

import os
import re
import time

import pandas as pd
import requests

try:
    import streamlit as st
except ImportError:  # pragma: no cover - streamlit is a hard dependency of the app,
    st = None          # but this module should still be importable/testable without it.

# Shopify Admin REST API version this module was written against. Bump when Shopify
# deprecates it; the orders.json line-item shape has been stable across versions.
API_VERSION = "2024-10"

# Shopify's documented maximum orders per page for REST Admin API list endpoints.
DEFAULT_PAGE_LIMIT = 250

# Retry/backoff for 429 responses. Shopify always sends a Retry-After header on a
# 429; DEFAULT_RETRY_AFTER_SECONDS is only a fallback for the (unexpected) case
# where it's missing.
MAX_RETRY_ATTEMPTS = 5
DEFAULT_RETRY_AFTER_SECONDS = 2.0

# Hard cap on pages fetched per sync click, so a manual "Sync Now" has a bounded
# worst-case run time/cost regardless of store size.
DEFAULT_MAX_PAGES = 200

# Column order/names the rest of the pipeline (src/rfm_engine.py) expects, matching
# data/real_online_retail.csv.gz exactly.
PIPELINE_COLUMNS = [
    "CustomerID",
    "InvoiceNo",
    "PurchaseDate",
    "TotalSpend",
    "Quantity",
    "UnitPrice",
    "Product",
    "ProductCategory",
]


class ShopifyIngestError(RuntimeError):
    """
    Raised when a Shopify Admin API request fails after exhausting retries, or the
    response indicates bad credentials / an invalid shop domain. Callers (app.py)
    should catch this specifically and surface an actionable message, the same way
    RFMPipelineError is handled for the standardize_transactions() path.
    """
    pass


def get_shopify_credentials() -> tuple:
    """
    Resolves (shop_domain, access_token) from st.secrets first, then environment
    variables (SHOPIFY_SHOP_DOMAIN, SHOPIFY_ACCESS_TOKEN). Never hardcoded, never
    logged. Returns (None, None) for whichever value isn't configured anywhere --
    callers should treat that as "not pre-configured" and fall back to letting the
    user type it into the UI, not as an error.
    """
    shop_domain = None
    access_token = None

    if st is not None:
        try:
            shop_domain = st.secrets.get("SHOPIFY_SHOP_DOMAIN")
            access_token = st.secrets.get("SHOPIFY_ACCESS_TOKEN")
        except Exception:
            # st.secrets raises when no secrets.toml exists at all in some Streamlit
            # versions -- that's a normal "not configured" state here, not an error.
            pass

    shop_domain = shop_domain or os.environ.get("SHOPIFY_SHOP_DOMAIN")
    access_token = access_token or os.environ.get("SHOPIFY_ACCESS_TOKEN")
    return shop_domain, access_token


def _parse_next_page_info(link_header: str):
    """
    Extracts the `page_info` cursor for the 'next' relation from Shopify's Link
    response header, e.g.:
        <https://shop.myshopify.com/admin/api/2024-10/orders.json?page_info=abc&limit=250>; rel="next"
    Returns None when there is no next page (Link header absent, or only a "previous"
    relation present -- i.e. the last page has been reached).
    """
    if not link_header:
        return None
    for part in link_header.split(","):
        if 'rel="next"' not in part:
            continue
        match = re.search(r"page_info=([^&>]+)", part)
        if match:
            return match.group(1)
    return None


def _get_with_rate_limit_backoff(url: str, headers: dict, params: dict = None,
                                  max_attempts: int = MAX_RETRY_ATTEMPTS) -> requests.Response:
    """
    Issues a single GET request, respecting Shopify's 429 `Retry-After` header with
    a sleep/backoff -- never hammering immediate retries. Raises ShopifyIngestError
    on repeated rate-limiting, bad credentials, or any other non-recoverable error.
    """
    last_status = None
    for attempt in range(1, max_attempts + 1):
        response = requests.get(url, headers=headers, params=params, timeout=30)
        last_status = response.status_code

        if response.status_code == 429:
            if attempt == max_attempts:
                break
            retry_after = DEFAULT_RETRY_AFTER_SECONDS
            try:
                retry_after = float(response.headers.get("Retry-After", DEFAULT_RETRY_AFTER_SECONDS))
            except (TypeError, ValueError):
                pass
            time.sleep(retry_after)
            continue

        if response.status_code == 401:
            raise ShopifyIngestError("Shopify authentication failed (401) -- check the access token.")
        if response.status_code == 404:
            raise ShopifyIngestError("Shopify shop not found (404) -- check the shop domain.")
        if response.status_code >= 400:
            raise ShopifyIngestError(
                f"Shopify API request failed with HTTP {response.status_code}: {response.text[:300]}"
            )

        return response

    raise ShopifyIngestError(
        f"Shopify API rate limit exceeded after {max_attempts} attempts (last status: {last_status})."
    )


def _orders_to_rows(orders: list) -> list:
    """
    Explodes a list of Shopify order objects into one row per line item, mapped to
    PIPELINE_COLUMNS -- each row mirrors one product line within one order/invoice,
    the same granularity as the bundled UCI Online Retail dataset.

    Cancelled orders (cancelled_at set) are excluded, mirroring how
    download_uci_retail.py already excludes cancelled invoices from the UCI dataset:
    a cancelled order isn't a completed transaction. Zero/negative quantity or price
    line items are skipped for the same reason (standardize_transactions() would
    filter them anyway; skipping here keeps this module's output self-consistent
    without depending on that downstream behavior).
    """
    rows = []
    for order in orders:
        if order.get("cancelled_at"):
            continue

        customer = order.get("customer") or {}
        customer_id = customer.get("id")
        if customer_id is None:
            # Guest checkout with no linked customer record: use a stable per-order
            # pseudo-ID rather than dropping the row -- it's still a real completed
            # transaction, it just can't be linked to a repeat-customer history.
            customer_id = f"GUEST-{order.get('id')}"

        invoice_no = order.get("name") or str(order.get("order_number") or order.get("id"))
        purchase_date = order.get("created_at")

        for line_item in order.get("line_items", []):
            quantity = line_item.get("quantity") or 0
            if quantity <= 0:
                continue
            try:
                unit_price = float(line_item.get("price") or 0.0)
            except (TypeError, ValueError):
                unit_price = 0.0
            if unit_price <= 0:
                continue

            rows.append({
                "CustomerID": str(customer_id),
                "InvoiceNo": str(invoice_no),
                "PurchaseDate": purchase_date,
                "TotalSpend": round(quantity * unit_price, 2),
                "Quantity": quantity,
                "UnitPrice": unit_price,
                "Product": line_item.get("title") or line_item.get("name") or "Catalog Product",
                "ProductCategory": line_item.get("product_type") or "General Merchandise",
            })

    return rows


def fetch_shopify_orders(
    shop_domain: str,
    access_token: str,
    since_date=None,
    api_version: str = API_VERSION,
    page_limit: int = DEFAULT_PAGE_LIMIT,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> pd.DataFrame:
    """
    Fetches orders (and their line items) from a Shopify store's Admin REST API and
    returns a DataFrame already in the pipeline's target schema (PIPELINE_COLUMNS).

    Parameters:
    - shop_domain: e.g. "your-store.myshopify.com" (scheme/trailing slash optional,
      stripped automatically).
    - access_token: a Shopify Admin API access token (custom/private app token, or
      OAuth token for a public app) with `read_orders` scope.
    - since_date: optional datetime/date/str. Only orders created on/after this date
      are fetched (maps to Shopify's `created_at_min` filter). None fetches all
      orders, bounded by max_pages.
    - api_version: Shopify Admin API version, e.g. "2024-10".
    - page_limit: orders per page (Shopify's REST Admin API max is 250).
    - max_pages: hard cap on pagination so one "Sync Now" click has a bounded
      worst-case run time/cost; stops silently (not an error) if reached.

    Pagination uses the cursor-based `page_info` value in the response's `Link`
    header, per Shopify's Admin REST API -- not an offset/page-number parameter
    (Shopify deprecated offset paging; a `page` query param is silently ignored past
    the first page on current API versions).
    """
    if not shop_domain or not access_token:
        raise ShopifyIngestError(
            "Shopify credentials are missing. Provide a shop domain and access token, "
            "or configure SHOPIFY_SHOP_DOMAIN / SHOPIFY_ACCESS_TOKEN via st.secrets "
            "or environment variables."
        )

    shop_domain = shop_domain.strip().replace("https://", "").replace("http://", "").rstrip("/")
    headers = {
        "X-Shopify-Access-Token": access_token,
        "Accept": "application/json",
    }
    base_url = f"https://{shop_domain}/admin/api/{api_version}/orders.json"
    params = {"status": "any", "limit": page_limit, "order": "created_at asc"}
    if since_date is not None:
        params["created_at_min"] = pd.to_datetime(since_date).isoformat()

    all_rows = []
    next_url, next_params = base_url, params
    page_count = 0

    while next_url and page_count < max_pages:
        response = _get_with_rate_limit_backoff(next_url, headers=headers, params=next_params)
        payload = response.json()
        all_rows.extend(_orders_to_rows(payload.get("orders", [])))

        page_info = _parse_next_page_info(response.headers.get("Link", ""))
        if page_info:
            next_url = base_url
            next_params = {"limit": page_limit, "page_info": page_info}
        else:
            next_url = None
            next_params = None
        page_count += 1

    if not all_rows:
        return pd.DataFrame(columns=PIPELINE_COLUMNS)
    return pd.DataFrame(all_rows, columns=PIPELINE_COLUMNS)


def shopify_dataframe_column_mapping() -> dict:
    """
    The custom_mapping dict for fetch_shopify_orders()'s output, in the same shape
    app.py already builds for the CSV/Excel upload flow's dynamic column-mapping
    sidebar (see `column_mapping` there). Since fetch_shopify_orders() already emits
    the pipeline's canonical column names, this is an identity mapping -- built
    explicitly, rather than passing custom_mapping=None, so every data-source path in
    app.py funnels through process_rfmt_pipeline() the exact same way.
    """
    return {
        "CustomerID": "CustomerID",
        "PurchaseDate": "PurchaseDate",
        "TotalSpend": "TotalSpend",
        "InvoiceNo": "InvoiceNo",
        "ProductCategory": "ProductCategory",
    }

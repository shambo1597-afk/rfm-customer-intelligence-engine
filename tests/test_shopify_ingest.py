"""
tests/test_shopify_ingest.py - pytest suite for the Shopify Admin API ingest module.

All Shopify API calls are mocked via the `responses` library -- no real network
calls are made anywhere in this file.
"""

import pandas as pd
import pytest
import responses

from src.shopify_ingest import (
    fetch_shopify_orders,
    shopify_dataframe_column_mapping,
    get_shopify_credentials,
    ShopifyIngestError,
    PIPELINE_COLUMNS,
)

SHOP = "test-store.myshopify.com"
TOKEN = "shpat_testtoken123"
ORDERS_URL = f"https://{SHOP}/admin/api/2024-10/orders.json"


def _order(order_id, customer_id, created_at, line_items, cancelled_at=None, order_number=None):
    return {
        "id": order_id,
        "order_number": order_number or order_id,
        "name": f"#{order_number or order_id}",
        "created_at": created_at,
        "cancelled_at": cancelled_at,
        "customer": {"id": customer_id} if customer_id is not None else None,
        "line_items": line_items,
    }


def _line_item(quantity=1, price="10.00", title="Test Product", product_type="Widgets"):
    return {"quantity": quantity, "price": price, "title": title, "product_type": product_type}


class TestFetchShopifyOrdersMapping:
    @responses.activate
    def test_maps_to_exact_pipeline_schema(self):
        orders = [_order(1001, 555, "2026-01-01T10:00:00-05:00", [_line_item(quantity=2, price="15.50")])]
        responses.add(responses.GET, ORDERS_URL, json={"orders": orders}, status=200)

        df = fetch_shopify_orders(SHOP, TOKEN)

        assert list(df.columns) == PIPELINE_COLUMNS
        assert len(df) == 1
        row = df.iloc[0]
        assert row["CustomerID"] == "555"
        assert row["InvoiceNo"] == "#1001"
        assert row["Quantity"] == 2
        assert row["UnitPrice"] == 15.50
        assert row["TotalSpend"] == 31.00
        assert row["ProductCategory"] == "Widgets"

    @responses.activate
    def test_multiple_line_items_produce_multiple_rows_same_invoice(self):
        orders = [_order(2001, 777, "2026-02-01T00:00:00Z", [
            _line_item(quantity=1, price="5.00", title="A"),
            _line_item(quantity=3, price="2.00", title="B"),
        ])]
        responses.add(responses.GET, ORDERS_URL, json={"orders": orders}, status=200)

        df = fetch_shopify_orders(SHOP, TOKEN)
        assert len(df) == 2
        assert set(df["Product"]) == {"A", "B"}
        assert df["InvoiceNo"].nunique() == 1  # One order == one invoice.

    @responses.activate
    def test_cancelled_orders_are_excluded(self):
        orders = [
            _order(3001, 111, "2026-01-01T00:00:00Z", [_line_item()], cancelled_at="2026-01-02T00:00:00Z"),
            _order(3002, 111, "2026-01-03T00:00:00Z", [_line_item()]),
        ]
        responses.add(responses.GET, ORDERS_URL, json={"orders": orders}, status=200)

        df = fetch_shopify_orders(SHOP, TOKEN)
        assert len(df) == 1
        assert df.iloc[0]["InvoiceNo"] == "#3002"

    @responses.activate
    def test_zero_price_or_quantity_line_items_are_skipped(self):
        orders = [_order(3500, 42, "2026-01-01T00:00:00Z", [
            _line_item(quantity=0, price="10.00"),
            _line_item(quantity=1, price="0.00"),
            _line_item(quantity=1, price="10.00"),
        ])]
        responses.add(responses.GET, ORDERS_URL, json={"orders": orders}, status=200)

        df = fetch_shopify_orders(SHOP, TOKEN)
        assert len(df) == 1

    @responses.activate
    def test_guest_checkout_gets_a_stable_pseudo_customer_id(self):
        orders = [_order(4001, None, "2026-01-01T00:00:00Z", [_line_item()])]
        responses.add(responses.GET, ORDERS_URL, json={"orders": orders}, status=200)

        df = fetch_shopify_orders(SHOP, TOKEN)
        assert df.iloc[0]["CustomerID"] == "GUEST-4001"

    @responses.activate
    def test_no_orders_returns_empty_dataframe_with_correct_columns(self):
        responses.add(responses.GET, ORDERS_URL, json={"orders": []}, status=200)
        df = fetch_shopify_orders(SHOP, TOKEN)
        assert df.empty
        assert list(df.columns) == PIPELINE_COLUMNS


class TestPagination:
    @responses.activate
    def test_follows_link_header_page_info_cursor_across_pages(self):
        page1_orders = [_order(5001, 1, "2026-01-01T00:00:00Z", [_line_item()])]
        page2_orders = [_order(5002, 2, "2026-01-02T00:00:00Z", [_line_item()])]

        next_link = f'<{ORDERS_URL}?limit=250&page_info=abc123>; rel="next"'
        responses.add(responses.GET, ORDERS_URL, json={"orders": page1_orders}, status=200,
                       headers={"Link": next_link})
        responses.add(responses.GET, ORDERS_URL, json={"orders": page2_orders}, status=200)  # no next -> last page

        df = fetch_shopify_orders(SHOP, TOKEN)

        assert len(df) == 2
        assert set(df["CustomerID"]) == {"1", "2"}
        assert len(responses.calls) == 2
        assert "page_info=abc123" in responses.calls[1].request.url

    @responses.activate
    def test_max_pages_bounds_pagination(self):
        next_link = f'<{ORDERS_URL}?limit=250&page_info=infinite>; rel="next"'
        order = _order(1, 1, "2026-01-01T00:00:00Z", [_line_item()])
        # This store would paginate forever without a cap -- every page links to a next page.
        for _ in range(3):
            responses.add(responses.GET, ORDERS_URL, json={"orders": [order]}, status=200,
                           headers={"Link": next_link})

        df = fetch_shopify_orders(SHOP, TOKEN, max_pages=2)
        assert len(responses.calls) == 2
        assert len(df) == 2


class TestRateLimitBackoff:
    @responses.activate
    def test_429_respects_retry_after_and_then_succeeds(self, monkeypatch):
        sleep_calls = []
        monkeypatch.setattr("src.shopify_ingest.time.sleep", lambda s: sleep_calls.append(s))

        order = _order(6001, 9, "2026-01-01T00:00:00Z", [_line_item()])
        responses.add(responses.GET, ORDERS_URL, status=429, headers={"Retry-After": "1.5"})
        responses.add(responses.GET, ORDERS_URL, json={"orders": [order]}, status=200)

        df = fetch_shopify_orders(SHOP, TOKEN)

        assert len(df) == 1
        assert sleep_calls == [1.5]  # Respected Retry-After; did not hammer immediate retries.

    @responses.activate
    def test_exhausted_retries_raise_ingest_error_not_a_crash(self, monkeypatch):
        monkeypatch.setattr("src.shopify_ingest.time.sleep", lambda s: None)
        for _ in range(5):
            responses.add(responses.GET, ORDERS_URL, status=429, headers={"Retry-After": "0"})

        with pytest.raises(ShopifyIngestError):
            fetch_shopify_orders(SHOP, TOKEN, max_pages=1)


class TestErrorHandling:
    @responses.activate
    def test_bad_credentials_401_raises_ingest_error(self):
        responses.add(responses.GET, ORDERS_URL, status=401)
        with pytest.raises(ShopifyIngestError):
            fetch_shopify_orders(SHOP, TOKEN)

    @responses.activate
    def test_invalid_shop_domain_404_raises_ingest_error(self):
        responses.add(responses.GET, ORDERS_URL, status=404)
        with pytest.raises(ShopifyIngestError):
            fetch_shopify_orders(SHOP, TOKEN)

    def test_missing_credentials_raise_before_any_network_call(self):
        with pytest.raises(ShopifyIngestError):
            fetch_shopify_orders("", "")


class TestColumnMappingAndCredentials:
    def test_column_mapping_is_identity_and_matches_pipeline_columns(self):
        mapping = shopify_dataframe_column_mapping()
        assert set(mapping.keys()) == {"CustomerID", "PurchaseDate", "TotalSpend", "InvoiceNo", "ProductCategory"}
        for target, source in mapping.items():
            assert target == source
            assert source in PIPELINE_COLUMNS

    def test_get_shopify_credentials_reads_environment_variables(self, monkeypatch):
        monkeypatch.setenv("SHOPIFY_SHOP_DOMAIN", "env-store.myshopify.com")
        monkeypatch.setenv("SHOPIFY_ACCESS_TOKEN", "env-token")
        shop, token = get_shopify_credentials()
        assert shop == "env-store.myshopify.com"
        assert token == "env-token"

    def test_get_shopify_credentials_returns_none_when_fully_unset(self, monkeypatch):
        monkeypatch.delenv("SHOPIFY_SHOP_DOMAIN", raising=False)
        monkeypatch.delenv("SHOPIFY_ACCESS_TOKEN", raising=False)
        shop, token = get_shopify_credentials()
        assert shop is None
        assert token is None

    def test_get_shopify_credentials_reads_st_secrets_when_present(self, monkeypatch):
        monkeypatch.delenv("SHOPIFY_SHOP_DOMAIN", raising=False)
        monkeypatch.delenv("SHOPIFY_ACCESS_TOKEN", raising=False)
        monkeypatch.setattr(
            "src.shopify_ingest.st.secrets",
            {"SHOPIFY_SHOP_DOMAIN": "secrets-store.myshopify.com", "SHOPIFY_ACCESS_TOKEN": "secrets-token"},
        )
        shop, token = get_shopify_credentials()
        assert shop == "secrets-store.myshopify.com"
        assert token == "secrets-token"


class TestLinkHeaderParsing:
    @responses.activate
    def test_link_header_with_only_previous_relation_returns_none(self):
        order = _order(1, 1, "2026-01-01T00:00:00Z", [_line_item()])
        prev_only_link = f'<{ORDERS_URL}?limit=250&page_info=prevcursor>; rel="previous"'
        responses.add(responses.GET, ORDERS_URL, json={"orders": [order]}, status=200,
                       headers={"Link": prev_only_link})

        df = fetch_shopify_orders(SHOP, TOKEN)
        assert len(df) == 1
        assert len(responses.calls) == 1  # No "next" relation -> stopped after page 1.

    @responses.activate
    def test_link_header_with_both_previous_and_next_relations(self):
        page1_order = _order(1, 1, "2026-01-01T00:00:00Z", [_line_item()])
        page2_order = _order(2, 2, "2026-01-02T00:00:00Z", [_line_item()])
        both_link = (
            f'<{ORDERS_URL}?limit=250&page_info=prevcursor>; rel="previous", '
            f'<{ORDERS_URL}?limit=250&page_info=nextcursor>; rel="next"'
        )
        responses.add(responses.GET, ORDERS_URL, json={"orders": [page1_order]}, status=200,
                       headers={"Link": both_link})
        responses.add(responses.GET, ORDERS_URL, json={"orders": [page2_order]}, status=200)

        df = fetch_shopify_orders(SHOP, TOKEN)
        assert len(df) == 2
        assert "page_info=nextcursor" in responses.calls[1].request.url


class TestAdditionalErrorPaths:
    @responses.activate
    def test_non_numeric_retry_after_falls_back_to_default_delay(self, monkeypatch):
        sleep_calls = []
        monkeypatch.setattr("src.shopify_ingest.time.sleep", lambda s: sleep_calls.append(s))
        order = _order(1, 1, "2026-01-01T00:00:00Z", [_line_item()])
        responses.add(responses.GET, ORDERS_URL, status=429, headers={"Retry-After": "not-a-number"})
        responses.add(responses.GET, ORDERS_URL, json={"orders": [order]}, status=200)

        df = fetch_shopify_orders(SHOP, TOKEN)
        assert len(df) == 1
        assert sleep_calls == [2.0]  # DEFAULT_RETRY_AFTER_SECONDS fallback.

    @responses.activate
    def test_server_error_500_raises_ingest_error(self):
        responses.add(responses.GET, ORDERS_URL, status=500)
        with pytest.raises(ShopifyIngestError):
            fetch_shopify_orders(SHOP, TOKEN)

    @responses.activate
    def test_non_numeric_line_item_price_is_treated_as_zero_and_skipped(self):
        orders = [_order(1, 1, "2026-01-01T00:00:00Z", [
            _line_item(quantity=1, price="not-a-price"),
            _line_item(quantity=1, price="10.00"),
        ])]
        responses.add(responses.GET, ORDERS_URL, json={"orders": orders}, status=200)

        df = fetch_shopify_orders(SHOP, TOKEN)
        assert len(df) == 1  # Only the valid-price line item survives.

    @responses.activate
    def test_since_date_is_passed_as_created_at_min_filter(self):
        order = _order(1, 1, "2026-06-01T00:00:00Z", [_line_item()])
        responses.add(responses.GET, ORDERS_URL, json={"orders": [order]}, status=200)

        fetch_shopify_orders(SHOP, TOKEN, since_date="2026-01-01")
        assert "created_at_min" in responses.calls[0].request.url

"""
generate_data.py - Enterprise Synthetic E-Commerce Transaction Generator
Generates >=3,500 transactions across 450 unique customers spanning a 24-month timeline.
Outputs dataset to 'data/ecommerce_transactions.csv'.
"""

import os
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def generate_enterprise_transactions(
    output_path="data/ecommerce_transactions.csv",
    num_customers=450,
    min_transactions=3500,
    random_seed=42
):
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Reference snapshot date
    reference_date = datetime(2026, 8, 20)
    start_history_date = reference_date - timedelta(days=730)  # 24 months ago

    # Create target output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # E-Commerce Product Catalog across diverse categories
    catalog = [
        {"category": "Electronics & Audio", "product": "Noise-Cancelling Studio Pro Headphones", "base_price": 299.99, "weight": 8},
        {"category": "Electronics & Audio", "product": "4K Curved Ultra-Wide Gaming Monitor", "base_price": 499.00, "weight": 5},
        {"category": "Electronics & Audio", "product": "Thunderbolt 4 Docking Station", "base_price": 189.50, "weight": 7},
        {"category": "Computer Accessories", "product": "Custom Mechanical RGB Keyboard", "base_price": 149.99, "weight": 11},
        {"category": "Computer Accessories", "product": "Ergonomic Multi-Device Wireless Mouse", "base_price": 79.90, "weight": 13},
        {"category": "Computer Accessories", "product": "Premium Aluminum Monitor Arm", "base_price": 64.50, "weight": 9},
        {"category": "Smart Office & Furniture", "product": "Motorized Standing Desk Frame", "base_price": 380.00, "weight": 4},
        {"category": "Smart Office & Furniture", "product": "Ergonomic Lumbar Executive Chair", "base_price": 275.00, "weight": 6},
        {"category": "Smart Office & Furniture", "product": "Smart LED Desk Bar Lamp", "base_price": 49.99, "weight": 10},
        {"category": "Wearables & Fitness", "product": "Titanium GPS Smart Fitness Watch", "base_price": 249.00, "weight": 8},
        {"category": "Wearables & Fitness", "product": "Heart Rate Performance Sensor Band", "base_price": 89.00, "weight": 9},
        {"category": "Apparel & Lifestyle", "product": "Full-Grain Leather Briefcase Sleeve", "base_price": 85.00, "weight": 10},
        {"category": "Apparel & Lifestyle", "product": "Thermal Double-Wall Stainless Flask", "base_price": 36.00, "weight": 14},
        {"category": "Specialty Gourmet", "product": "Single-Origin Espresso Bean Subscription (1kg)", "base_price": 32.50, "weight": 16},
    ]

    categories = list(set(item["category"] for item in catalog))
    product_names = [item["product"] for item in catalog]
    product_weights = [item["weight"] for item in catalog]
    product_category_map = {item["product"]: item["category"] for item in catalog}
    product_price_map = {item["product"]: item["base_price"] for item in catalog}

    # Generate customer IDs
    customer_ids = [f"CUST-{i:04d}" for i in range(1, num_customers + 1)]
    random.shuffle(customer_ids)

    # Distribute 450 customers across 7 realistic behavioral archetypes
    # 1. Champions (14% / ~63 custs): 12-24 orders across 1-2 years, very recent (1-30d)
    # 2. Loyalists (20% / ~90 custs): 8-16 orders, steady cadence, recent (15-60d)
    # 3. Potential Growth (14% / ~63 custs): 3-7 orders, recent (5-45d), high basket value
    # 4. At-Risk VIPs (14% / ~63 custs): 7-15 orders historically, but dormant for 90-180d
    # 5. Can't Lose Them (8% / ~36 custs): 10-20 orders historically, high spend, dormant for 180-365d
    # 6. Hibernating (18% / ~81 custs): 1-4 orders, dormant for 200-500d, low spend
    # 7. New Customers (12% / ~54 custs): 1-3 orders, joined in last 10-50d

    n_champions = int(num_customers * 0.14)
    n_loyalists = int(num_customers * 0.20)
    n_growth = int(num_customers * 0.14)
    n_at_risk = int(num_customers * 0.14)
    n_cant_lose = int(num_customers * 0.08)
    n_hibernating = int(num_customers * 0.18)
    n_new = num_customers - (n_champions + n_loyalists + n_growth + n_at_risk + n_cant_lose + n_hibernating)

    personas = {}
    idx = 0

    # Champions
    for cid in customer_ids[idx:idx + n_champions]:
        first_join_days_ago = random.randint(300, 700)
        personas[cid] = {
            "archetype": "Champions",
            "order_count": random.randint(12, 22),
            "first_join_days_ago": first_join_days_ago,
            "last_recency_days": random.randint(1, 28),
            "qty_range": (1, 6),
            "price_multiplier": 1.10
        }
    idx += n_champions

    # Loyalists
    for cid in customer_ids[idx:idx + n_loyalists]:
        first_join_days_ago = random.randint(240, 680)
        personas[cid] = {
            "archetype": "Loyalists",
            "order_count": random.randint(8, 15),
            "first_join_days_ago": first_join_days_ago,
            "last_recency_days": random.randint(15, 60),
            "qty_range": (1, 4),
            "price_multiplier": 1.0
        }
    idx += n_loyalists

    # Potential Growth
    for cid in customer_ids[idx:idx + n_growth]:
        first_join_days_ago = random.randint(60, 240)
        personas[cid] = {
            "archetype": "Potential Growth",
            "order_count": random.randint(3, 7),
            "first_join_days_ago": first_join_days_ago,
            "last_recency_days": random.randint(5, 45),
            "qty_range": (1, 4),
            "price_multiplier": 1.15
        }
    idx += n_growth

    # At-Risk VIPs
    for cid in customer_ids[idx:idx + n_at_risk]:
        first_join_days_ago = random.randint(300, 710)
        personas[cid] = {
            "archetype": "At-Risk VIPs",
            "order_count": random.randint(7, 14),
            "first_join_days_ago": first_join_days_ago,
            "last_recency_days": random.randint(90, 180),
            "qty_range": (1, 5),
            "price_multiplier": 1.05
        }
    idx += n_at_risk

    # Can't Lose Them
    for cid in customer_ids[idx:idx + n_cant_lose]:
        first_join_days_ago = random.randint(400, 720)
        personas[cid] = {
            "archetype": "Can't Lose Them",
            "order_count": random.randint(10, 18),
            "first_join_days_ago": first_join_days_ago,
            "last_recency_days": random.randint(185, 380),
            "qty_range": (2, 6),
            "price_multiplier": 1.20
        }
    idx += n_cant_lose

    # Hibernating
    for cid in customer_ids[idx:idx + n_hibernating]:
        first_join_days_ago = random.randint(250, 700)
        personas[cid] = {
            "archetype": "Hibernating",
            "order_count": random.randint(1, 4),
            "first_join_days_ago": first_join_days_ago,
            "last_recency_days": random.randint(210, 520),
            "qty_range": (1, 2),
            "price_multiplier": 0.95
        }
    idx += n_hibernating

    # New Customers
    for cid in customer_ids[idx:]:
        first_join_days_ago = random.randint(10, 55)
        personas[cid] = {
            "archetype": "New Customers",
            "order_count": random.randint(1, 3),
            "first_join_days_ago": first_join_days_ago,
            "last_recency_days": random.randint(2, first_join_days_ago),
            "qty_range": (1, 3),
            "price_multiplier": 1.0
        }

    records = []
    invoice_counter = 100001

    for cid, profile in personas.items():
        n_orders = profile["order_count"]
        first_date = reference_date - timedelta(days=profile["first_join_days_ago"])
        last_date = reference_date - timedelta(days=profile["last_recency_days"])

        if first_date > last_date:
            first_date, last_date = last_date - timedelta(days=30), last_date

        if n_orders == 1:
            order_dates = [last_date]
        else:
            # Interpolate order dates between first and last purchase
            span_days = max((last_date - first_date).days, 1)
            random_offsets = sorted([random.randint(0, span_days) for _ in range(n_orders - 2)])
            order_dates = [first_date] + [first_date + timedelta(days=off) for off in random_offsets] + [last_date]

        for o_date in order_dates:
            # An order may have 1 to 3 distinct line items
            num_items = random.choices([1, 2, 3], weights=[0.68, 0.24, 0.08])[0]
            invoice_no = f"INV-{invoice_counter}"
            invoice_counter += 1

            chosen_prods = random.choices(product_names, weights=product_weights, k=num_items)
            for prod in chosen_prods:
                qty = random.randint(profile["qty_range"][0], profile["qty_range"][1])
                base_price = product_price_map[prod]
                category = product_category_map[prod]
                
                # Pricing variations
                price_jitter = random.uniform(0.96, 1.04) * profile["price_multiplier"]
                unit_price = round(base_price * price_jitter, 2)
                total_spend = round(qty * unit_price, 2)

                purchase_datetime = o_date.replace(
                    hour=random.randint(8, 22),
                    minute=random.randint(0, 59),
                    second=random.randint(0, 59)
                )

                records.append({
                    "InvoiceNo": invoice_no,
                    "CustomerID": cid,
                    "PurchaseDate": purchase_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                    "ProductCategory": category,
                    "Product": prod,
                    "Quantity": qty,
                    "UnitPrice": unit_price,
                    "TotalSpend": total_spend
                })

    df = pd.DataFrame(records)
    df["PurchaseDate"] = pd.to_datetime(df["PurchaseDate"])
    df = df.sort_values(by="PurchaseDate").reset_index(drop=True)

    print(f"Generated {len(df):,} transactions across {df['CustomerID'].nunique()} unique customers.")
    print(f"Timeline: {df['PurchaseDate'].min().strftime('%Y-%m-%d')} to {df['PurchaseDate'].max().strftime('%Y-%m-%d')}")
    print(f"Gross Realized Revenue: ${df['TotalSpend'].sum():,.2f}")

    df.to_csv(output_path, index=False)
    # Also save a copy as sample_transactions.csv for seamless backwards compatibility
    df.to_csv("sample_transactions.csv", index=False)
    print(f"Saved dataset successfully to '{output_path}' and 'sample_transactions.csv'")
    return df

if __name__ == "__main__":
    generate_enterprise_transactions()

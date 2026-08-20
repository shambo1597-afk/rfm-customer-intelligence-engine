"""
download_uci_retail.py - Downloads and cleans the authentic UCI Online Retail Dataset
"""

import os
import sys
import requests
import pandas as pd
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def main():
    print("=" * 70)
    print("[*] DOWNLOADING & CLEANING AUTHENTIC UCI ONLINE RETAIL DATASET")
    print("=" * 70)
    
    os.makedirs("data", exist_ok=True)
    raw_path = "data/raw_retail.csv"
    output_path = "data/real_online_retail.csv"
    
    url = "https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/07_Visualization/Online_Retail/Online_Retail.csv"
    
    if not os.path.exists(raw_path):
        print(f"[*] Fetching raw dataset from: {url}")
        r = requests.get(url, stream=True, timeout=60)
        total = 0
        with open(raw_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)
                    print(f"    Downloaded {total / (1024*1024):.1f} MB...", flush=True)
        print("[+] Finished raw file download.")
    else:
        print(f"[+] Using cached raw file at '{raw_path}'.")

    print("[*] Parsing and cleaning raw dataset...")
    df_raw = pd.read_csv(raw_path, encoding="latin1")
    print(f"    Raw records: {len(df_raw):,}")

    # 1. Clean null Customer IDs
    df_clean = df_raw.dropna(subset=["CustomerID"]).copy()
    print(f"    After dropping null CustomerID: {len(df_clean):,} records")

    # 2. Filter cancelled orders and positive quantities / unit prices
    df_clean["InvoiceNo"] = df_clean["InvoiceNo"].astype(str)
    df_clean = df_clean[~df_clean["InvoiceNo"].str.startswith("C")]
    df_clean["Quantity"] = pd.to_numeric(df_clean["Quantity"], errors="coerce").fillna(0).astype(int)
    df_clean["UnitPrice"] = pd.to_numeric(df_clean["UnitPrice"], errors="coerce").fillna(0.0)
    df_clean = df_clean[(df_clean["Quantity"] > 0) & (df_clean["UnitPrice"] > 0)].copy()
    print(f"    After filtering Quantity > 0 & UnitPrice > 0: {len(df_clean):,} records")

    # 3. Format CustomerID as clean string integer
    df_clean["CustomerID"] = df_clean["CustomerID"].astype(int).astype(str)

    # 4. Standardize PurchaseDate
    df_clean["PurchaseDate"] = pd.to_datetime(df_clean["InvoiceDate"], errors="coerce")
    df_clean = df_clean.dropna(subset=["PurchaseDate"]).copy()
    df_clean["PurchaseDate"] = df_clean["PurchaseDate"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # 5. Compute TotalSpend
    df_clean["TotalSpend"] = (df_clean["Quantity"] * df_clean["UnitPrice"]).round(2)

    # 6. Format Product & Category
    df_clean["Product"] = df_clean["Description"].fillna("Catalog Item").str.strip()
    df_clean["ProductCategory"] = df_clean["Country"].fillna("United Kingdom")

    # 7. Select & Reorder Target Columns
    target_cols = [
        "CustomerID",
        "InvoiceNo",
        "PurchaseDate",
        "TotalSpend",
        "Quantity",
        "UnitPrice",
        "Product",
        "ProductCategory"
    ]
    df_final = df_clean[target_cols].sort_values(by="PurchaseDate").reset_index(drop=True)

    # 8. Save Clean CSV
    df_final.to_csv(output_path, index=False)

    # Clean up raw temp file
    if os.path.exists(raw_path):
        os.remove(raw_path)

    print("=" * 70)
    print(f"✅ SUCCESS: Authentic cleaned dataset saved to '{output_path}'")
    print(f"   • Total Clean Transactions: {len(df_final):,}")
    print(f"   • Unique Customers: {df_final['CustomerID'].nunique():,}")
    print(f"   • Date Range: {df_final['PurchaseDate'].min()} to {df_final['PurchaseDate'].max()}")
    print(f"   • Gross Realized Revenue: ${df_final['TotalSpend'].sum():,.2f}")
    print(f"   • Formatted Columns: {list(df_final.columns)}")
    print("=" * 70)

if __name__ == "__main__":
    main()

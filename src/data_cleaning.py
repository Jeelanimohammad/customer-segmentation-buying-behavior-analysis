"""
Data Cleaning and Preprocessing Module.

Handles:
- Missing CustomerID logging and filtering
- Duplicate rows removal
- Cancellation filtering (InvoiceNo starting with 'C' and negative quantities)
- Non-positive price and quantity cleaning
- Datetime parsing and validation
- Line revenue computation (Quantity * UnitPrice)
- Summary statistics tracking for governance and transparency
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


def clean_retail_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw retail transaction data and returns cleaned DataFrame along with
    a comprehensive audit report of removed vs retained rows.

    Args:
        df: Raw pandas DataFrame

    Returns:
        (cleaned_df, cleaning_summary_dict)
    """
    if df is None or df.empty:
        return pd.DataFrame(), {
            "initial_rows": 0,
            "cleaned_rows": 0,
            "missing_customer_ids": 0,
            "duplicate_rows": 0,
            "cancelled_invoices": 0,
            "invalid_quantities": 0,
            "invalid_prices": 0,
            "retained_percentage": 0.0,
            "audit_log": ["Input dataset was empty."]
        }

    # Work on a fresh copy so original dataset remains immutable
    data = df.copy()
    initial_rows = len(data)
    audit_log = []

    # 1. Standardize column names
    expected_cols = {
        "invoiceno": "InvoiceNo",
        "stockcode": "StockCode",
        "description": "Description",
        "quantity": "Quantity",
        "invoicedate": "InvoiceDate",
        "unitprice": "UnitPrice",
        "customerid": "CustomerID",
        "country": "Country"
    }
    col_mapping = {}
    for col in data.columns:
        clean_key = str(col).strip().lower()
        if clean_key in expected_cols:
            col_mapping[col] = expected_cols[clean_key]
    data.rename(columns=col_mapping, inplace=True)

    # Validate essential columns
    required_fields = ["InvoiceNo", "Quantity", "InvoiceDate", "UnitPrice"]
    for field in required_fields:
        if field not in data.columns:
            raise ValueError(f"Required column '{field}' is missing from the dataset.")

    # 2. Track & Handle Missing CustomerID
    missing_cust_count = 0
    if "CustomerID" in data.columns:
        missing_cust_count = int(data["CustomerID"].isna().sum())
        data = data.dropna(subset=["CustomerID"])
        # Format CustomerID as integer string
        data["CustomerID"] = data["CustomerID"].astype(int).astype(str)
        audit_log.append(f"Excluded {missing_cust_count:,} transactions without CustomerID (guest purchases).")
    else:
        # If CustomerID column doesn't exist, create an empty one
        data["CustomerID"] = "Unknown"

    # 3. Deduplication
    dups_count = int(data.duplicated().sum())
    if dups_count > 0:
        data = data.drop_duplicates()
        audit_log.append(f"Removed {dups_count:,} duplicate rows.")

    # 4. Handle Cancellations (InvoiceNo starts with 'C' or negative Quantity)
    cancelled_mask = data["InvoiceNo"].astype(str).str.startswith("C", na=False) | (pd.to_numeric(data["Quantity"], errors="coerce") < 0)
    cancelled_count = int(cancelled_mask.sum())
    if cancelled_count > 0:
        data = data[~cancelled_mask]
        audit_log.append(f"Filtered out {cancelled_count:,} cancelled orders/returns.")

    # 5. Filter invalid / zero quantities
    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")
    invalid_qty_mask = (data["Quantity"] <= 0) | data["Quantity"].isna()
    invalid_qty_count = int(invalid_qty_mask.sum())
    if invalid_qty_count > 0:
        data = data[~invalid_qty_mask]
        audit_log.append(f"Removed {invalid_qty_count:,} items with zero or negative quantity.")

    # 6. Filter invalid unit prices (UnitPrice <= 0)
    data["UnitPrice"] = pd.to_numeric(data["UnitPrice"], errors="coerce")
    invalid_price_mask = (data["UnitPrice"] <= 0) | data["UnitPrice"].isna()
    invalid_price_count = int(invalid_price_mask.sum())
    if invalid_price_count > 0:
        data = data[~invalid_price_mask]
        audit_log.append(f"Removed {invalid_price_count:,} rows with zero or negative unit price.")

    # 7. Parse InvoiceDate
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    invalid_dates = int(data["InvoiceDate"].isna().sum())
    if invalid_dates > 0:
        data = data.dropna(subset=["InvoiceDate"])
        audit_log.append(f"Removed {invalid_dates:,} rows with unparseable transaction dates.")

    # 8. Calculate Line Revenue = Quantity * UnitPrice
    data["Revenue"] = np.round(data["Quantity"] * data["UnitPrice"], 2)

    # 9. Clean Country string
    if "Country" in data.columns:
        data["Country"] = data["Country"].fillna("Unspecified").astype(str).str.strip()
    else:
        data["Country"] = "Unspecified"

    cleaned_rows = len(data)
    retained_pct = round((cleaned_rows / initial_rows) * 100, 2) if initial_rows > 0 else 0.0

    summary = {
        "initial_rows": initial_rows,
        "cleaned_rows": cleaned_rows,
        "missing_customer_ids": missing_cust_count,
        "duplicate_rows": dups_count,
        "cancelled_invoices": cancelled_count,
        "invalid_quantities": invalid_qty_count,
        "invalid_prices": invalid_price_count,
        "retained_percentage": retained_pct,
        "audit_log": audit_log
    }

    return data.reset_index(drop=True), summary

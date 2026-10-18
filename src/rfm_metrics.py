"""
RFM (Recency, Frequency, Monetary) Calculation Module.

Computes:
- Recency: Days between the customer's last purchase and reference date (max date + 1 day)
- Frequency: Count of distinct valid purchases (distinct InvoiceNo)
- Monetary: Total spending across all completed transactions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any


def calculate_rfm(df: pd.DataFrame, reference_date: datetime = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Computes RFM metrics per customer.
    
    Args:
        df: Cleaned transaction DataFrame with CustomerID, InvoiceNo, InvoiceDate, Revenue
        reference_date: Optional custom reference date. If None, max(InvoiceDate) + 1 day is used.
    
    Returns:
        (rfm_df, metadata_dict)
    """
    if df is None or df.empty or "CustomerID" not in df.columns:
        return pd.DataFrame(), {
            "reference_date": None,
            "total_customers": 0,
            "repeat_customers": 0,
            "repeat_customer_rate": 0.0,
            "avg_monetary": 0.0,
            "median_monetary": 0.0,
            "avg_frequency": 0.0,
            "avg_recency": 0.0
        }

    # Reference date determination: latest transaction date + 1 day
    # This prevents recency = 0 for the most recent day's shoppers and simulates running analysis 'the next morning'
    max_invoice_date = df["InvoiceDate"].max()
    if reference_date is None:
        ref_date = max_invoice_date + timedelta(days=1)
    else:
        ref_date = pd.to_datetime(reference_date)

    # Group by CustomerID
    # Frequency: Number of unique invoices
    # Monetary: Sum of Revenue
    # Recency: (ref_date - max(InvoiceDate)).days
    rfm = df.groupby("CustomerID").agg(
        LastPurchaseDate=("InvoiceDate", "max"),
        FirstPurchaseDate=("InvoiceDate", "min"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
        TotalUnits=("Quantity", "sum"),
        Country=("Country", lambda s: s.mode()[0] if not s.empty else "Unknown")
    ).reset_index()

    # Calculate Recency in integer days
    rfm["Recency"] = (ref_date - rfm["LastPurchaseDate"]).dt.days

    # Round monetary value to 2 decimals
    rfm["Monetary"] = np.round(rfm["Monetary"], 2)

    # Calculate Repeat vs One-Time metrics
    total_customers = len(rfm)
    repeat_customers = int((rfm["Frequency"] > 1).sum())
    repeat_rate = round((repeat_customers / total_customers * 100), 2) if total_customers > 0 else 0.0

    metadata = {
        "reference_date": ref_date.strftime("%Y-%m-%d %H:%M:%S"),
        "max_invoice_date": max_invoice_date.strftime("%Y-%m-%d %H:%M:%S"),
        "total_customers": total_customers,
        "repeat_customers": repeat_customers,
        "repeat_customer_rate": repeat_rate,
        "avg_monetary": round(float(rfm["Monetary"].mean()), 2) if total_customers > 0 else 0.0,
        "median_monetary": round(float(rfm["Monetary"].median()), 2) if total_customers > 0 else 0.0,
        "avg_frequency": round(float(rfm["Frequency"].mean()), 2) if total_customers > 0 else 0.0,
        "avg_recency": round(float(rfm["Recency"].mean()), 1) if total_customers > 0 else 0.0
    }

    return rfm, metadata

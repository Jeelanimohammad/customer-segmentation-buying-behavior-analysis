"""
Comprehensive Test Suite for Customer Segmentation & Buying Behavior Analysis.

Tests:
1. Data cleaning (missing CustomerID, duplicate rows, cancellations, invalid quantities & prices)
2. Revenue calculation
3. Empty dataframe handling
4. RFM metric computation & reference date determination
5. Segment assignment rules and edge cases
6. Database initialization and parameterized SQL queries
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.data_cleaning import clean_retail_data
from src.rfm_metrics import calculate_rfm
from src.segmentation import segment_customers_rfm, run_exploratory_kmeans
from src.db_manager import init_database, execute_parameterized_query


@pytest.fixture
def sample_raw_data():
    return pd.DataFrame([
        # Normal row 1
        {"InvoiceNo": "536365", "StockCode": "85123A", "Description": "WHITE HEART", "Quantity": 6, "InvoiceDate": "2023-01-01 10:00:00", "UnitPrice": 2.55, "CustomerID": 12345, "Country": "United Kingdom"},
        # Normal row 2 (same customer, different item)
        {"InvoiceNo": "536365", "StockCode": "71053", "Description": "LANTERN", "Quantity": 2, "InvoiceDate": "2023-01-01 10:00:00", "UnitPrice": 3.00, "CustomerID": 12345, "Country": "United Kingdom"},
        # Cancelled row (starts with C, negative qty)
        {"InvoiceNo": "C536366", "StockCode": "85123A", "Description": "CANCELLED ITEM", "Quantity": -6, "InvoiceDate": "2023-01-02 11:00:00", "UnitPrice": 2.55, "CustomerID": 12345, "Country": "United Kingdom"},
        # Missing CustomerID (guest purchase)
        {"InvoiceNo": "536367", "StockCode": "22752", "Description": "BOXES", "Quantity": 4, "InvoiceDate": "2023-01-03 12:00:00", "UnitPrice": 5.00, "CustomerID": np.nan, "Country": "United Kingdom"},
        # Zero quantity item
        {"InvoiceNo": "536368", "StockCode": "22633", "Description": "ZERO QTY", "Quantity": 0, "InvoiceDate": "2023-01-04 14:00:00", "UnitPrice": 2.10, "CustomerID": 12346, "Country": "France"},
        # Negative / zero price
        {"InvoiceNo": "536369", "StockCode": "22632", "Description": "ZERO PRICE", "Quantity": 5, "InvoiceDate": "2023-01-05 15:00:00", "UnitPrice": 0.00, "CustomerID": 12347, "Country": "Germany"},
        # Duplicate row of row 1
        {"InvoiceNo": "536365", "StockCode": "85123A", "Description": "WHITE HEART", "Quantity": 6, "InvoiceDate": "2023-01-01 10:00:00", "UnitPrice": 2.55, "CustomerID": 12345, "Country": "United Kingdom"},
        # Valid transaction for customer 12348
        {"InvoiceNo": "536370", "StockCode": "22423", "Description": "CAKESTAND", "Quantity": 10, "InvoiceDate": "2023-02-10 16:00:00", "UnitPrice": 10.00, "CustomerID": 12348, "Country": "United Kingdom"},
    ])


def test_data_cleaning_filters_and_revenue(sample_raw_data):
    clean_df, summary = clean_retail_data(sample_raw_data)

    # Initial was 8 rows
    assert summary["initial_rows"] == 8
    # Retained valid rows: row 1, row 2, and row 8 (3 unique valid records)
    assert summary["cleaned_rows"] == 3
    assert summary["missing_customer_ids"] == 1
    assert summary["cancelled_invoices"] == 1
    assert summary["duplicate_rows"] == 1
    assert summary["invalid_quantities"] == 1
    assert summary["invalid_prices"] == 1

    # Verify Revenue calculation: Quantity * UnitPrice
    row1 = clean_df[(clean_df["CustomerID"] == "12345") & (clean_df["StockCode"] == "85123A")].iloc[0]
    assert row1["Revenue"] == 15.30  # 6 * 2.55


def test_clean_retail_data_empty():
    empty_df = pd.DataFrame()
    clean_df, summary = clean_retail_data(empty_df)
    assert clean_df.empty
    assert summary["initial_rows"] == 0
    assert summary["cleaned_rows"] == 0


def test_rfm_calculation(sample_raw_data):
    clean_df, _ = clean_retail_data(sample_raw_data)
    rfm_df, meta = calculate_rfm(clean_df)

    assert meta["total_customers"] == 2  # Customer 12345 and 12348
    assert "CustomerID" in rfm_df.columns
    assert "Recency" in rfm_df.columns
    assert "Frequency" in rfm_df.columns
    assert "Monetary" in rfm_df.columns

    # Check Customer 12345: 1 invoice (536365), spend = 15.30 + 6.00 = 21.30
    c12345 = rfm_df[rfm_df["CustomerID"] == "12345"].iloc[0]
    assert c12345["Frequency"] == 1
    assert c12345["Monetary"] == 21.30

    # Check Customer 12348: 1 invoice (536370), spend = 100.00
    c12348 = rfm_df[rfm_df["CustomerID"] == "12348"].iloc[0]
    assert c12348["Frequency"] == 1
    assert c12348["Monetary"] == 100.00


def test_segmentation_assignment():
    test_rfm = pd.DataFrame([
        {"CustomerID": "1", "Recency": 2, "Frequency": 10, "Monetary": 1500.0, "Country": "UK"},
        {"CustomerID": "2", "Recency": 4, "Frequency": 6, "Monetary": 600.0, "Country": "UK"},
        {"CustomerID": "3", "Recency": 10, "Frequency": 2, "Monetary": 150.0, "Country": "UK"},
        {"CustomerID": "4", "Recency": 150, "Frequency": 8, "Monetary": 900.0, "Country": "UK"},
        {"CustomerID": "5", "Recency": 200, "Frequency": 1, "Monetary": 20.0, "Country": "UK"},
    ])

    segmented = segment_customers_rfm(test_rfm)
    assert "Segment" in segmented.columns
    assert "RFM_Score" in segmented.columns
    assert len(segmented["Segment"].dropna()) == 5

    # Check that each customer belongs to one of valid segments
    valid_segments = {
        "Champions", "Loyal Customers", "Potential Loyalists",
        "Recent Customers", "At Risk", "Others / Hibernating"
    }
    for seg in segmented["Segment"]:
        assert seg in valid_segments


def test_sqlite_database_and_parameterized_query(sample_raw_data):
    clean_df, _ = clean_retail_data(sample_raw_data)
    rfm_df, _ = calculate_rfm(clean_df)
    segmented = segment_customers_rfm(rfm_df)

    conn = init_database(clean_df, segmented, db_path=":memory:")
    
    # Test transactions table
    df_tx, status_tx = execute_parameterized_query(conn, "SELECT COUNT(*) AS cnt FROM transactions;")
    assert status_tx == "Success"
    assert df_tx.iloc[0]["cnt"] == 3

    # Test customers_rfm table with parameterization
    df_rfm, status_rfm = execute_parameterized_query(
        conn,
        "SELECT CustomerID, Monetary FROM customers_rfm WHERE CustomerID = ?;",
        params=("12348",)
    )
    assert status_rfm == "Success"
    assert len(df_rfm) == 1
    assert df_rfm.iloc[0]["Monetary"] == 100.00

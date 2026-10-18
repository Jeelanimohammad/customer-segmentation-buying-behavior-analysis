"""
Database Operations and SQL Analytics Module.

Creates and populates a local SQLite database with cleaned retail transactions
and customer-level RFM metrics. Provides parameterized SQL queries for interactive exploration.
"""

import sqlite3
import pandas as pd
from typing import Tuple, List, Dict, Any


def init_database(
    transactions_df: pd.DataFrame,
    rfm_df: pd.DataFrame,
    db_path: str = "retail_analytics.db"
) -> sqlite3.Connection:
    """
    Initializes a local SQLite database and loads transactions and customer RFM tables.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)

    # Write cleaned transactions
    if transactions_df is not None and not transactions_df.empty:
        # Convert datetime to string for clean SQLite storage
        tx_copy = transactions_df.copy()
        if "InvoiceDate" in tx_copy.columns:
            tx_copy["InvoiceDate"] = tx_copy["InvoiceDate"].astype(str)
        tx_copy.to_sql("transactions", conn, if_exists="replace", index=False)

    # Write customer RFM
    if rfm_df is not None and not rfm_df.empty:
        rfm_copy = rfm_df.copy()
        if "LastPurchaseDate" in rfm_copy.columns:
            rfm_copy["LastPurchaseDate"] = rfm_copy["LastPurchaseDate"].astype(str)
        if "FirstPurchaseDate" in rfm_copy.columns:
            rfm_copy["FirstPurchaseDate"] = rfm_copy["FirstPurchaseDate"].astype(str)
        rfm_copy.to_sql("customers_rfm", conn, if_exists="replace", index=False)

    # Create helpful indexes for performance
    cursor = conn.cursor()
    try:
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tx_customer ON transactions(CustomerID);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tx_invoice ON transactions(InvoiceNo);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rfm_segment ON customers_rfm(Segment);")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    return conn


def execute_parameterized_query(
    conn: sqlite3.Connection,
    query: str,
    params: Tuple = ()
) -> Tuple[pd.DataFrame, str]:
    """
    Executes a SQL query against the connection using parameter substitution for security.
    Returns (result_df, status_or_error_message).
    """
    try:
        df = pd.read_sql_query(query, conn, params=params)
        return df, "Success"
    except Exception as e:
        return pd.DataFrame(), f"SQL Error: {str(e)}"


def get_predefined_sql_queries() -> List[Dict[str, Any]]:
    """
    Returns curated SQL business queries demonstrating retail analytics skills.
    """
    return [
        {
            "title": "Top 10 High-Value Customers by Total Spend",
            "description": "Identifies the highest revenue generating accounts and their current segment status.",
            "sql": """
                SELECT 
                    CustomerID, 
                    Country, 
                    Frequency AS TotalOrders, 
                    ROUND(Monetary, 2) AS TotalSpent, 
                    Recency AS DaysSinceLastOrder,
                    Segment 
                FROM customers_rfm 
                ORDER BY Monetary DESC 
                LIMIT 10;
            """,
            "params": ()
        },
        {
            "title": "Segment Performance & Average Spend Benchmark",
            "description": "Aggregates revenue, customer counts, and average ticket size across segments.",
            "sql": """
                SELECT 
                    Segment,
                    COUNT(CustomerID) AS TotalCustomers,
                    ROUND(SUM(Monetary), 2) AS TotalRevenue,
                    ROUND(AVG(Monetary), 2) AS AvgSpendPerCustomer,
                    ROUND(AVG(Frequency), 1) AS AvgPurchaseFrequency,
                    ROUND(AVG(Recency), 0) AS AvgDaysInactive
                FROM customers_rfm
                GROUP BY Segment
                ORDER BY TotalRevenue DESC;
            """,
            "params": ()
        },
        {
            "title": "At-Risk Customers with High Prior Spending",
            "description": "Finds customers who spent above a threshold but have not purchased in over 60 days.",
            "sql": """
                SELECT 
                    CustomerID,
                    Country,
                    Recency AS DaysInactive,
                    Frequency AS LifetimeOrders,
                    ROUND(Monetary, 2) AS TotalSpent,
                    Segment
                FROM customers_rfm
                WHERE Recency >= ? AND Monetary >= ?
                ORDER BY Monetary DESC
                LIMIT 15;
            """,
            "params": (60, 200.0)
        },
        {
            "title": "Top 10 Best-Selling Products by Revenue",
            "description": "Ranks individual catalog products by total sales and units sold.",
            "sql": """
                SELECT 
                    StockCode,
                    Description,
                    SUM(Quantity) AS TotalUnitsSold,
                    ROUND(SUM(Revenue), 2) AS TotalRevenue,
                    COUNT(DISTINCT InvoiceNo) AS UniqueOrdersAppearingIn
                FROM transactions
                GROUP BY StockCode, Description
                ORDER BY TotalRevenue DESC
                LIMIT 10;
            """,
            "params": ()
        },
        {
            "title": "Revenue and Order Count by Country",
            "description": "Geographical distribution of orders and revenue.",
            "sql": """
                SELECT 
                    Country,
                    COUNT(DISTINCT InvoiceNo) AS TotalInvoices,
                    COUNT(DISTINCT CustomerID) AS TotalCustomers,
                    ROUND(SUM(Revenue), 2) AS TotalRevenue,
                    ROUND(SUM(Revenue) / COUNT(DISTINCT InvoiceNo), 2) AS AvgOrderValue
                FROM transactions
                GROUP BY Country
                ORDER BY TotalRevenue DESC
                LIMIT 10;
            """,
            "params": ()
        }
    ]

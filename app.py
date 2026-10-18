"""
Customer Segmentation & Buying Behavior Analysis
Interactive Business Development & Analytics Application.

Built for Retail Growth, Retention Strategy, and Portfolio Demonstration.
Tech Stack: Python, Pandas, SQLite, Streamlit, Plotly, Scikit-learn.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

from src.data_cleaning import clean_retail_data
from src.rfm_metrics import calculate_rfm
from src.segmentation import (
    segment_customers_rfm,
    run_exploratory_kmeans,
    get_segment_business_recommendations,
)
from src.db_manager import (
    init_database,
    execute_parameterized_query,
    get_predefined_sql_queries,
)
from src.charts import (
    create_segment_distribution_chart,
    create_segment_revenue_chart,
    create_rfm_scatter_2d,
    create_rfm_scatter_3d,
    create_monthly_trend_chart,
    create_order_value_distribution,
    create_repeat_buyer_chart,
    SEGMENT_COLORS,
)

# Page configuration
st.set_page_config(
    page_title="Customer Analytics & RFM Segmentation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished dashboard feel
st.markdown("""
<style>
    .metric-card {
        background-color: #1E293B;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #334155;
    }
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Cleaning and transforming data...")
def load_and_process_data(file_source, is_uploaded=False):
    """Cached pipeline for data ingestion, cleaning, and RFM scoring."""
    if is_uploaded:
        raw_df = pd.read_csv(file_source)
    else:
        if not os.path.exists(file_source):
            return None, None, None, {"error": f"File not found: {file_source}"}
        raw_df = pd.read_csv(file_source)

    cleaned_df, cleaning_summary = clean_retail_data(raw_df)
    rfm_df, rfm_meta = calculate_rfm(cleaned_df)
    segmented_rfm = segment_customers_rfm(rfm_df)

    return raw_df, cleaned_df, segmented_rfm, {
        "cleaning": cleaning_summary,
        "rfm": rfm_meta,
    }


def main():
    # ------------------ SIDEBAR CONTROLS ------------------
    st.sidebar.title("Retail Intelligence")
    st.sidebar.caption("Customer Segmentation & Buying Behavior")

    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Data Source Selection")

    data_option = st.sidebar.radio(
        "Choose Data Source:",
        ["Synthetic Demo Data (Fast)", "Upload Custom CSV", "UCI Dataset File (data/online_retail.csv)"]
    )

    data_path = "data/sample_retail_data.csv"
    is_uploaded = False
    uploaded_file = None

    if data_option == "Synthetic Demo Data (Fast)":
        data_path = "data/sample_retail_data.csv"
        st.sidebar.info("Using reproducible synthetic dataset (3,500 transactions). Demonstrates real edge cases without API calls.")
    elif data_option == "Upload Custom CSV":
        uploaded_file = st.sidebar.file_uploader("Upload CSV (Online Retail Format)", type=["csv"])
        if uploaded_file is not None:
            is_uploaded = True
        else:
            st.sidebar.warning("Please upload a CSV or switch back to Synthetic Demo Data.")
    else:
        data_path = "data/online_retail.csv"
        if not os.path.exists(data_path):
            st.sidebar.warning("UCI file not found at `data/online_retail.csv`. Falling back to synthetic demo data.")
            data_path = "data/sample_retail_data.csv"

    # Ingestion & Processing
    with st.spinner("Processing transaction records..."):
        if is_uploaded and uploaded_file is not None:
            raw_df, clean_df, rfm_df, meta = load_and_process_data(uploaded_file, is_uploaded=True)
        else:
            raw_df, clean_df, rfm_df, meta = load_and_process_data(data_path, is_uploaded=False)

    if raw_df is None or clean_df is None or clean_df.empty:
        st.error("Could not process the dataset. Please ensure the CSV contains valid retail transaction columns.")
        return

    # Global Filters in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Global Filters")

    # Country filter
    available_countries = ["All Countries"] + sorted(list(clean_df["Country"].unique()))
    selected_country = st.sidebar.selectbox("Country Filter", available_countries)

    # Date filter
    min_date = clean_df["InvoiceDate"].min().date()
    max_date = clean_df["InvoiceDate"].max().date()
    date_range = st.sidebar.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    # Apply filters
    filtered_tx = clean_df.copy()
    if selected_country != "All Countries":
        filtered_tx = filtered_tx[filtered_tx["Country"] == selected_country]

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_d, end_d = date_range
        filtered_tx = filtered_tx[
            (filtered_tx["InvoiceDate"].dt.date >= start_d) &
            (filtered_tx["InvoiceDate"].dt.date <= end_d)
        ]

    # Re-calculate RFM on filtered transactions if filters applied
    if len(filtered_tx) != len(clean_df):
        rfm_display, _ = calculate_rfm(filtered_tx)
        rfm_display = segment_customers_rfm(rfm_display)
    else:
        rfm_display = rfm_df.copy()

    # Initialize SQLite Database connection
    conn = init_database(filtered_tx, rfm_display)

    # Top Navigation Tabs
    tab_overview, tab_segmentation, tab_behavior, tab_insights, tab_sql = st.tabs([
        "📈 Executive Overview",
        "👥 Customer Segmentation",
        "🛒 Buying Behavior",
        "💡 Customer Insights & Actions",
        "🗄️ Data & SQL Explorer"
    ])

    # ------------------ TAB 1: EXECUTIVE OVERVIEW ------------------
    with tab_overview:
        st.header("Executive Retail Overview")
        st.caption("High-level key performance metrics and customer lifecycle benchmarks.")

        # KPI Metrics Row
        total_customers = len(rfm_display)
        total_revenue = filtered_tx["Revenue"].sum()
        total_orders = filtered_tx["InvoiceNo"].nunique()
        repeat_cust_count = (rfm_display["Frequency"] > 1).sum() if total_customers > 0 else 0
        repeat_rate = round((repeat_cust_count / total_customers * 100), 1) if total_customers > 0 else 0.0
        avg_spend_per_cust = round(total_revenue / total_customers, 2) if total_customers > 0 else 0.0
        avg_order_value = round(total_revenue / total_orders, 2) if total_orders > 0 else 0.0

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Customers", f"{total_customers:,}")
        m2.metric("Total Net Revenue", f"${total_revenue:,.2f}")
        m3.metric("Repeat Customer Rate", f"{repeat_rate}%")
        m4.metric("Avg Spend / Customer", f"${avg_spend_per_cust:,.2f}")
        m5.metric("Avg Order Value (AOV)", f"${avg_order_value:,.2f}")

        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(create_segment_distribution_chart(rfm_display), use_container_width=True)
        with c2:
            st.plotly_chart(create_segment_revenue_chart(rfm_display), use_container_width=True)

        st.markdown("### Executive Performance Summary")
        top_segment = rfm_display.groupby("Segment")["Monetary"].sum().idxmax() if not rfm_display.empty else "N/A"
        top_segment_rev = rfm_display.groupby("Segment")["Monetary"].sum().max() if not rfm_display.empty else 0
        top_rev_pct = round((top_segment_rev / total_revenue * 100), 1) if total_revenue > 0 else 0

        st.info(
            f"**Key Business Takeaway**: **{top_segment}** is currently the top revenue driver, contributing "
            f"**${top_segment_rev:,.2f} ({top_rev_pct}%)** of all sales. "
            f"**{repeat_rate}%** of buyers are repeat purchasers, indicating consistent retention opportunities."
        )

    # ------------------ TAB 2: CUSTOMER SEGMENTATION ------------------
    with tab_segmentation:
        st.header("RFM Customer Segmentation")
        st.caption("Categorization based on Recency (inactivity), Frequency (orders), and Monetary value (spend).")

        # Segment breakdown table
        seg_summary = rfm_display.groupby("Segment").agg(
            CustomerCount=("CustomerID", "count"),
            TotalRevenue=("Monetary", "sum"),
            AvgRecencyDays=("Recency", "mean"),
            AvgOrders=("Frequency", "mean"),
            AvgCustomerSpend=("Monetary", "mean")
        ).reset_index()

        seg_summary["RevenueSharePct"] = (seg_summary["TotalRevenue"] / seg_summary["TotalRevenue"].sum() * 100).round(1)
        seg_summary["AvgRecencyDays"] = seg_summary["AvgRecencyDays"].round(1)
        seg_summary["AvgOrders"] = seg_summary["AvgOrders"].round(1)
        seg_summary["AvgCustomerSpend"] = seg_summary["AvgCustomerSpend"].round(2)

        st.dataframe(
            seg_summary.sort_values(by="TotalRevenue", ascending=False),
            use_container_width=True,
            hide_index=True
        )

        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.plotly_chart(create_rfm_scatter_2d(rfm_display), use_container_width=True)
        with col_right:
            st.plotly_chart(create_rfm_scatter_3d(rfm_display), use_container_width=True)

        # Optional Exploratory K-Means Clustering
        st.markdown("---")
        with st.expander("🔬 Exploratory K-Means Clustering (Machine Learning Diagnostic)"):
            st.markdown("""
            **Analytical Context**: Rule-based RFM gives deterministic business categories. 
            K-Means clustering is provided here as an *exploratory* statistical method to verify natural mathematical groupings.
            Features are log-transformed (`log(1 + x)`) to mitigate severe monetary skew, then standardized with `StandardScaler`.
            """)
            k_val = st.slider("Select Number of Clusters (k):", min_value=2, max_value=6, value=4)
            clustered_df, kmeans_meta = run_exploratory_kmeans(rfm_display, k=k_val)

            kcol1, kcol2 = st.columns(2)
            kcol1.metric("Silhouette Score", kmeans_meta.get("silhouette_score", "N/A"), help="Measures cluster separation (-1 to 1; >0.3 indicates reasonable cohesion).")
            kcol2.metric("Inertia (Sum of Squared Distances)", kmeans_meta.get("inertia", "N/A"))

            if "cluster_profile" in kmeans_meta:
                st.write("**Cluster Statistical Profiles:**")
                st.dataframe(kmeans_meta["cluster_profile"].round(2), use_container_width=True, hide_index=True)

        # Filterable Customer Table & CSV Export
        st.markdown("---")
        st.subheader("Filterable Customer Roster & Export")
        selected_seg = st.multiselect("Filter by Segment:", options=list(rfm_display["Segment"].unique()), default=list(rfm_display["Segment"].unique()))
        
        filtered_table = rfm_display[rfm_display["Segment"].isin(selected_seg)]
        st.dataframe(filtered_table[["CustomerID", "Segment", "Recency", "Frequency", "Monetary", "Country", "LastPurchaseDate"]], use_container_width=True)

        csv_data = filtered_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Customer RFM Table as CSV",
            data=csv_data,
            file_name="customer_rfm_segments.csv",
            mime="text/csv",
        )

    # ------------------ TAB 3: BUYING BEHAVIOR ------------------
    with tab_behavior:
        st.header("Buying Behavior & Transaction Analysis")
        st.caption("Purchase rhythms, basket size distribution, and repeat buyer trends.")

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(create_monthly_trend_chart(filtered_tx), use_container_width=True)
        with c2:
            st.plotly_chart(create_order_value_distribution(filtered_tx), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(create_repeat_buyer_chart(rfm_display), use_container_width=True)
        with c4:
            st.markdown("### Repeat Purchase Insights")
            single_buyers = (rfm_display["Frequency"] == 1).sum()
            rep_buyers = (rfm_display["Frequency"] > 1).sum()
            rep_rev = rfm_display[rfm_display["Frequency"] > 1]["Monetary"].sum()
            rep_share = round((rep_rev / total_revenue * 100), 1) if total_revenue > 0 else 0.0

            st.write(f"- **One-Time Shoppers**: {single_buyers:,} customers")
            st.write(f"- **Repeat Buyers**: {rep_buyers:,} customers")
            st.write(f"- **Repeat Revenue Share**: **{rep_share}%** of all net sales")
            st.write(
                "A high concentration of revenue among repeat buyers confirms the importance of retention campaigns. "
                "Converting even 5% of one-time shoppers to second orders can significantly elevate total top-line revenue."
            )

    # ------------------ TAB 4: CUSTOMER INSIGHTS & ACTIONS ------------------
    with tab_insights:
        st.header("Customer Insights & Actionable Strategies")
        st.caption("Identifies priority customer groups and pairs them with rule-based business development actions.")

        rec_catalog = get_segment_business_recommendations()

        col_hi, col_risk = st.columns(2)
        with col_hi:
            st.subheader("⭐ Top 5 High-Value Champions")
            top_5 = rfm_display[rfm_display["Segment"] == "Champions"].sort_values(by="Monetary", ascending=False).head(5)
            if not top_5.empty:
                st.dataframe(top_5[["CustomerID", "Monetary", "Frequency", "Recency", "Country"]], hide_index=True)
            else:
                st.info("No Champions found under current filters.")

        with col_risk:
            st.subheader("⚠️ Priority At-Risk Customers")
            at_risk = rfm_display[rfm_display["Segment"] == "At Risk"].sort_values(by="Monetary", ascending=False).head(5)
            if not at_risk.empty:
                st.dataframe(at_risk[["CustomerID", "Monetary", "Frequency", "Recency", "Country"]], hide_index=True)
            else:
                st.info("No At-Risk customers found under current filters.")

        st.markdown("---")
        st.subheader("Actionable Business Development Recommendations")
        st.caption("Note: These are hypothesis-driven suggested interventions. Actual revenue uplift requires A/B testing verification.")

        for segment_name, details in rec_catalog.items():
            count_in_seg = (rfm_display["Segment"] == segment_name).sum()
            with st.expander(f"📌 {segment_name} ({count_in_seg:,} customers) — Focus: {details['priority']}"):
                st.markdown(f"**Description**: {details['description']}")
                st.markdown(f"**Key Metric to Track**: `{details['kpi_focus']}`")
                st.markdown("**Suggested Actions:**")
                for action in details["suggested_actions"]:
                    st.markdown(f"- {action}")

    # ------------------ TAB 5: DATA & SQL EXPLORER ------------------
    with tab_sql:
        st.header("Data Governance & SQL Explorer")
        st.caption("Cleaned transaction inspection, cleaning audit log, and parameterized SQL queries against SQLite.")

        with st.expander("🧹 Data Cleaning Audit Log & Metrics", expanded=True):
            clean_meta = meta.get("cleaning", {})
            ca, cb, cc, cd = st.columns(4)
            ca.metric("Initial Raw Rows", f"{clean_meta.get('initial_rows', 0):,}")
            cb.metric("Cleaned Rows Retained", f"{clean_meta.get('cleaned_rows', 0):,}")
            cc.metric("Excluded Guest Checkouts", f"{clean_meta.get('missing_customer_ids', 0):,}")
            cd.metric("Cancellations Filtered", f"{clean_meta.get('cancelled_invoices', 0):,}")

            st.write("**Cleaning Decisions & Rules Applied:**")
            for log_item in clean_meta.get("audit_log", []):
                st.markdown(f"- {log_item}")

        st.markdown("---")
        st.subheader("Cleaned Transactions Preview (First 50 Rows)")
        st.dataframe(filtered_tx.head(50), use_container_width=True)

        st.markdown("---")
        st.subheader("Interactive SQL Explorer (SQLite)")

        queries = get_predefined_sql_queries()
        query_titles = [q["title"] for q in queries] + ["Custom SQL Query"]
        chosen_query_title = st.selectbox("Select SQL Template or Write Custom:", query_titles)

        if chosen_query_title != "Custom SQL Query":
            selected_item = next(q for q in queries if q["title"] == chosen_query_title)
            st.caption(selected_item["description"])
            sql_text = st.text_area("SQL Statement (Read-Only)", value=selected_item["sql"].strip(), height=140)
            params = selected_item["params"]
        else:
            sql_text = st.text_area("Write SQL Query (Tables: `transactions`, `customers_rfm`):", value="SELECT * FROM customers_rfm LIMIT 10;", height=140)
            params = ()

        if st.button("▶ Run SQL Query"):
            res_df, status = execute_parameterized_query(conn, sql_text, params=params)
            if status == "Success":
                st.success(f"Query returned {len(res_df)} rows.")
                st.dataframe(res_df, use_container_width=True)
            else:
                st.error(status)


if __name__ == "__main__":
    main()

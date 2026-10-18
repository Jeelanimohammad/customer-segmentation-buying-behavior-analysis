"""
Interactive Visualizations Module using Plotly.

Features:
- Segment Distribution Donut & Bar Charts
- 3D & 2D RFM Scatter Plots
- Monthly Sales Trends & Order Frequency Distributions
- Repeat vs One-Time Buyer Comparisons
- Clean, professional styling with consistent color palette
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


# Curated visual palette
SEGMENT_COLORS = {
    "Champions": "#10B981",          # Emerald Green
    "Loyal Customers": "#3B82F6",    # Vibrant Blue
    "Potential Loyalists": "#8B5CF6",# Violet
    "Recent Customers": "#06B6D4",   # Cyan
    "At Risk": "#F59E0B",            # Amber Orange
    "Others / Hibernating": "#6B7280"# Cool Slate Gray
}


def create_segment_distribution_chart(rfm_df: pd.DataFrame) -> go.Figure:
    """Creates a donut chart showing customer share by segment."""
    seg_counts = rfm_df["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]

    colors = [SEGMENT_COLORS.get(s, "#94A3B8") for s in seg_counts["Segment"]]

    fig = go.Figure(data=[go.Pie(
        labels=seg_counts["Segment"],
        values=seg_counts["Count"],
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#1E293B", width=1.5)),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Customers: %{value:,}<br>Share: %{percent}<extra></extra>"
    )])

    fig.update_layout(
        title="Customer Distribution by Segment",
        showlegend=False,
        margin=dict(t=50, b=20, l=20, r=20),
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif", size=13)
    )
    return fig


def create_segment_revenue_chart(rfm_df: pd.DataFrame) -> go.Figure:
    """Creates a horizontal bar chart comparing monetary contribution by segment."""
    seg_rev = rfm_df.groupby("Segment")["Monetary"].sum().reset_index()
    seg_rev = seg_rev.sort_values(by="Monetary", ascending=True)

    colors = [SEGMENT_COLORS.get(s, "#94A3B8") for s in seg_rev["Segment"]]

    fig = go.Figure(go.Bar(
        x=seg_rev["Monetary"],
        y=seg_rev["Segment"],
        orientation="h",
        marker=dict(color=colors, line=dict(color="#1E293B", width=1)),
        hovertemplate="<b>%{y}</b><br>Total Revenue: $%{x:,.2f}<extra></extra>"
    ))

    fig.update_layout(
        title="Total Revenue Contribution by Segment ($)",
        xaxis_title="Monetary Contribution ($)",
        yaxis_title="",
        margin=dict(t=50, b=30, l=40, r=20),
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif", size=13)
    )
    fig.update_xaxes(showgrid=True, gridcolor="#334155")
    return fig


def create_rfm_scatter_3d(rfm_df: pd.DataFrame) -> go.Figure:
    """Creates a 3D scatter plot of Recency vs Frequency vs Monetary."""
    fig = px.scatter_3d(
        rfm_df,
        x="Recency",
        y="Frequency",
        z="Monetary",
        color="Segment",
        color_discrete_map=SEGMENT_COLORS,
        hover_data=["CustomerID", "Country"],
        title="3D RFM Customer Landscape",
        opacity=0.85
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(
        scene=dict(
            xaxis_title="Recency (Days)",
            yaxis_title="Frequency (Orders)",
            zaxis_title="Monetary ($)"
        ),
        margin=dict(t=40, b=20, l=20, r=20),
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif")
    )
    return fig


def create_rfm_scatter_2d(rfm_df: pd.DataFrame) -> go.Figure:
    """Creates an interactive 2D bubble scatter plot of Recency vs Frequency."""
    fig = px.scatter(
        rfm_df,
        x="Recency",
        y="Frequency",
        size="Monetary",
        color="Segment",
        color_discrete_map=SEGMENT_COLORS,
        hover_name="CustomerID",
        hover_data={"Monetary": ":$.2f", "Country": True, "Recency": True, "Frequency": True},
        title="Customer Engagement: Recency vs. Frequency (Bubble Size = Total Spend)",
        size_max=35
    )
    fig.update_layout(
        xaxis_title="Recency (Days Inactive - Lower is More Recent)",
        yaxis_title="Order Frequency (Lifetime Orders)",
        margin=dict(t=50, b=40, l=40, r=20),
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    fig.update_xaxes(showgrid=True, gridcolor="#334155")
    fig.update_yaxes(showgrid=True, gridcolor="#334155")
    return fig


def create_monthly_trend_chart(clean_tx_df: pd.DataFrame) -> go.Figure:
    """Monthly revenue and order volume trends."""
    df = clean_tx_df.copy()
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    
    monthly = df.groupby("YearMonth").agg(
        Revenue=("Revenue", "sum"),
        Orders=("InvoiceNo", "nunique")
    ).reset_index()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=monthly["YearMonth"],
        y=monthly["Revenue"],
        name="Revenue ($)",
        marker_color="#3B82F6",
        yaxis="y",
        hovertemplate="<b>%{x}</b><br>Revenue: $%{y:,.2f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=monthly["YearMonth"],
        y=monthly["Orders"],
        name="Order Count",
        mode="lines+markers",
        marker=dict(size=8, color="#F59E0B"),
        line=dict(width=3, color="#F59E0B"),
        yaxis="y2",
        hovertemplate="<b>%{x}</b><br>Orders: %{y:,}<extra></extra>"
    ))

    fig.update_layout(
        title="Monthly Revenue & Order Volume Trends",
        xaxis=dict(title="Month"),
        yaxis=dict(title="Revenue ($)", showgrid=True, gridcolor="#334155"),
        yaxis2=dict(
            title="Order Count",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(15,23,42,0.6)"),
        margin=dict(t=50, b=40, l=50, r=50),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_order_value_distribution(clean_tx_df: pd.DataFrame) -> go.Figure:
    """Histogram of individual invoice order values."""
    order_totals = clean_tx_df.groupby("InvoiceNo")["Revenue"].sum().reset_index()
    
    # Cap 99th percentile for clean visualization
    p99 = order_totals["Revenue"].quantile(0.99)
    filtered = order_totals[order_totals["Revenue"] <= p99]

    fig = px.histogram(
        filtered,
        x="Revenue",
        nbins=40,
        title="Order Value Distribution (Per Invoice, up to 99th percentile)",
        color_discrete_sequence=["#10B981"]
    )
    fig.update_layout(
        xaxis_title="Invoice Total ($)",
        yaxis_title="Count of Orders",
        margin=dict(t=50, b=40, l=40, r=20),
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    fig.update_xaxes(showgrid=True, gridcolor="#334155")
    fig.update_yaxes(showgrid=True, gridcolor="#334155")
    return fig


def create_repeat_buyer_chart(rfm_df: pd.DataFrame) -> go.Figure:
    """Pie/Donut chart comparing Single vs Repeat purchasers."""
    rfm_df_copy = rfm_df.copy()
    rfm_df_copy["BuyerType"] = np.where(rfm_df_copy["Frequency"] > 1, "Repeat Customers (>1 order)", "One-Time Customers (1 order)")
    
    summary = rfm_df_copy.groupby("BuyerType").agg(
        Customers=("CustomerID", "count"),
        Revenue=("Monetary", "sum")
    ).reset_index()

    fig = go.Figure(data=[go.Pie(
        labels=summary["BuyerType"],
        values=summary["Revenue"],
        hole=0.6,
        marker=dict(colors=["#3B82F6", "#94A3B8"], line=dict(color="#1E293B", width=1.5)),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Revenue: $%{value:,.2f}<extra></extra>"
    )])

    fig.update_layout(
        title="Revenue Split: Repeat Buyers vs One-Time Buyers",
        showlegend=False,
        margin=dict(t=50, b=20, l=20, r=20),
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

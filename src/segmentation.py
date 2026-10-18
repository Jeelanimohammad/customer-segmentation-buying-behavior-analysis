"""
Customer Segmentation and Exploratory Clustering Module.

Features:
1. Rule-Based RFM Segmentation:
   - Recency, Frequency, Monetary quantile scoring (1-4 or 1-5)
   - Explainable segment definitions: Champions, Loyal Customers, Potential Loyalists,
     Recent Customers, At Risk, and Others/Hibernating.
2. Exploratory K-Means Clustering:
   - Log-transformation for right-skewed revenue/frequency
   - StandardScaler normalization
   - Silhouette analysis & inertia tracking
   - Transparently framed as an exploratory diagnostic tool, not ground truth labels.
3. Rule-based actionable business recommendations per segment.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def segment_customers_rfm(df_rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns RFM scores (1 to 4) and maps customers to transparent business segments:
    - Champions: High recency, high frequency, high spend
    - Loyal Customers: Responsive repeat buyers with solid spend
    - Potential Loyalists: Recent buyers with above-average activity
    - Recent Customers: New or recent shoppers with lower frequency
    - At Risk: Formerly loyal or frequent customers who have not bought recently
    - Others / Hibernating: Inactive buyers with low historical frequency and spend
    """
    if df_rfm is None or df_rfm.empty:
        return df_rfm

    rfm = df_rfm.copy()

    # Recency score: Lower days = higher score (4 is best, 1 is worst)
    # Using rank(method='first') to ensure robust binning even when ties exist
    try:
        rfm["R_Score"] = pd.qcut(rfm["Recency"], q=4, labels=[4, 3, 2, 1])
    except ValueError:
        rfm["R_Score"] = pd.qcut(rfm["Recency"].rank(method="first"), q=4, labels=[4, 3, 2, 1])

    # Frequency score: Higher orders = higher score (4 is best, 1 is worst)
    try:
        rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=4, labels=[1, 2, 3, 4])
    except ValueError:
        rfm["F_Score"] = pd.cut(rfm["Frequency"], bins=4, labels=[1, 2, 3, 4])

    # Monetary score: Higher spend = higher score (4 is best, 1 is worst)
    try:
        rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"), q=4, labels=[1, 2, 3, 4])
    except ValueError:
        rfm["M_Score"] = pd.cut(rfm["Monetary"], bins=4, labels=[1, 2, 3, 4])

    # Convert to numeric for segment evaluation
    rfm["R_Score"] = rfm["R_Score"].astype(int)
    rfm["F_Score"] = rfm["F_Score"].astype(int)
    rfm["M_Score"] = rfm["M_Score"].astype(int)

    rfm["RFM_Score"] = rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)

    # Segment Mapping Logic
    def assign_segment(row):
        r = row["R_Score"]
        f = row["F_Score"]
        m = row["M_Score"]

        # Champions: Top recency and high engagement/spend
        if r >= 3 and f >= 3 and m >= 3:
            return "Champions"
        # Loyal Customers: Consistent repeat shoppers
        elif f >= 3:
            return "Loyal Customers"
        # Potential Loyalists: Recent shoppers with decent spend
        elif r >= 3 and (f >= 2 or m >= 2):
            return "Potential Loyalists"
        # Recent Customers: High recency but low frequency (newcomers)
        elif r >= 3 and f == 1:
            return "Recent Customers"
        # At Risk: Low recency but previously exhibited moderate-to-high frequency or spend
        elif r <= 2 and (f >= 3 or m >= 3):
            return "At Risk"
        # Others / Hibernating: Inactive, infrequent, low spend
        else:
            return "Others / Hibernating"

    rfm["Segment"] = rfm.apply(assign_segment, axis=1)
    return rfm


def run_exploratory_kmeans(
    df_rfm: pd.DataFrame,
    k: int = 4,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Runs exploratory K-Means clustering on log-transformed and standardized RFM features.
    Transparently frames output as exploratory patterns rather than ground truth labels.
    """
    if df_rfm is None or len(df_rfm) < k:
        return df_rfm, {"error": "Not enough data points for K-Means clustering."}

    rfm_copy = df_rfm.copy()

    # Log-transform to handle severe right-skew in monetary and frequency distributions
    features = pd.DataFrame()
    features["Recency_log"] = np.log1p(rfm_copy["Recency"])
    features["Frequency_log"] = np.log1p(rfm_copy["Frequency"])
    features["Monetary_log"] = np.log1p(np.maximum(0, rfm_copy["Monetary"]))

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)

    rfm_copy["Cluster"] = [f"Cluster {c}" for c in cluster_labels]

    sil_score = float(silhouette_score(scaled_features, cluster_labels))
    inertia = float(kmeans.inertia_)

    # Cluster summary
    cluster_profile = rfm_copy.groupby("Cluster").agg(
        CustomerCount=("CustomerID", "count"),
        AvgRecency=("Recency", "mean"),
        AvgFrequency=("Frequency", "mean"),
        AvgMonetary=("Monetary", "mean")
    ).reset_index()

    metadata = {
        "k": k,
        "silhouette_score": round(sil_score, 3),
        "inertia": round(inertia, 2),
        "cluster_profile": cluster_profile
    }

    return rfm_copy, metadata


def get_segment_business_recommendations() -> Dict[str, Dict[str, Any]]:
    """
    Rule-based, explainable business recommendations per segment.
    Carefully framed as suggested actions (hypotheses to test) rather than guaranteed outcomes.
    """
    return {
        "Champions": {
            "description": "Your most valuable customers: bought recently, buy frequently, and spend the most.",
            "suggested_actions": [
                "VIP reward program invitation and exclusive early access to new seasonal collections.",
                "Solicit qualitative feedback or product testimonials to turn them into brand advocates.",
                "Avoid aggressive discount coupons; instead, offer value-add perks (e.g., free premium gift wrapping)."
            ],
            "priority": "High Retention & Advocacy",
            "kpi_focus": "Customer Lifetime Value (LTV) & Referral Rate"
        },
        "Loyal Customers": {
            "description": "Steady, reliable repeat customers with consistent ordering patterns over time.",
            "suggested_actions": [
                "Cross-sell complementary categories based on historical basket analysis.",
                "Tiered loyalty milestones (e.g., 'Spend $50 more this month to unlock Gold tier').",
                "Personalized product recommendation emails highlighting trending items in their favorite category."
            ],
            "priority": "Upsell & Category Expansion",
            "kpi_focus": "Average Order Value (AOV) & Purchase Frequency"
        },
        "Potential Loyalists": {
            "description": "Recent buyers with above-average spend or multiple initial visits showing strong intent.",
            "suggested_actions": [
                "Onboarding follow-up series showcasing top customer favorites and product care guides.",
                "Time-sensitive second/third purchase incentive to foster habit formation.",
                "Introduce brand community or subscription replenishment options if applicable."
            ],
            "priority": "Habit Building & Nurturing",
            "kpi_focus": "Repeat Purchase Conversion Rate"
        },
        "Recent Customers": {
            "description": "New or recently acquired buyers with single purchases and moderate spend.",
            "suggested_actions": [
                "Warm welcome sequence thanking them for their first purchase and reassuring satisfaction.",
                "Clear return policy reminders and customer support touchpoint to build trust.",
                "Targeted intro discount on their complementary second purchase within 14-21 days."
            ],
            "priority": "First-to-Second Order Conversion",
            "kpi_focus": "30-Day Re-engagement Rate"
        },
        "At Risk": {
            "description": "Valuable customers with past high spend/frequency who haven't purchased in an extended period.",
            "suggested_actions": [
                "Win-back automated email series with subject line: 'We miss you - Here is 15% off your next visit'.",
                "Short 2-question satisfaction survey to identify service breakdowns, delivery delays, or churn causes.",
                "Direct personalized outreach from an account associate for high-ticket accounts."
            ],
            "priority": "Immediate Churn Prevention",
            "kpi_focus": "Reactivation Rate & Churn Containment"
        },
        "Others / Hibernating": {
            "description": "Infrequent buyers with low spend who have been inactive for an extended period.",
            "suggested_actions": [
                "Low-cost, seasonal clearance broadcast campaigns (Black Friday, End-of-Season sales).",
                "Avoid costly ad retargeting spend; maintain email deliverability hygiene by sunsetting non-responders.",
                "Focus outbound BD efforts on higher-probability segments (Potential Loyalists and At Risk)."
            ],
            "priority": "Cost Control & Low-touch Maintenance",
            "kpi_focus": "Acquisition Cost Optimization"
        }
    }

# Customer Segmentation & Buying Behavior Analysis

An intermediate-level customer analytics and business intelligence platform built with **Python, Pandas, SQLite, Streamlit, Plotly, and Scikit-learn**. Designed specifically to help retail businesses understand purchasing rhythms, identify high-value customer groups, measure repeat retention, and inform data-driven business development strategies.

---

## 📌 Executive Summary & Business Objective

In e-commerce and retail, customer acquisition costs (CAC) continue to rise. Maximizing Customer Lifetime Value (LTV) and reducing churn are paramount for sustainable growth. 

This project delivers:
1. **Automated Data Quality & Governance**: Robust cleaning pipeline handling guest checkouts, invoice cancellations, duplicate transactions, and anomalies.
2. **Transparent RFM Modeling**: Customer-level Recency, Frequency, and Monetary scoring with explainable, deterministic segment classification.
3. **Exploratory Machine Learning**: Optional K-Means clustering with log-transformation and standardized features as an exploratory diagnostic.
4. **Actionable Business Interventions**: Rule-based hypotheses and go-to-market recommendations for Business Development (BD) and marketing teams.
5. **Interactive SQL Analytics Engine**: Integrated local SQLite database with parameterized business queries and custom query execution.

---

## 🛠️ Tech Stack & Architecture

- **Core Analytics & Data Processing**: Python 3.10+, Pandas, NumPy
- **Relational Database & SQL**: SQLite3 (zero-configuration, embedded relational engine with parameterized queries)
- **Interactive Web Dashboard**: Streamlit
- **Data Visualizations**: Plotly (Interactive 2D/3D scatter plots, donut charts, distribution histograms, dual-axis trend lines)
- **Machine Learning (Exploratory)**: Scikit-learn (K-Means, StandardScaler, Silhouette Analysis)
- **Testing & Quality Assurance**: Pytest (100% test pass on data cleaning, RFM, segmentation, and SQL queries)
- **Version Control**: Git & GitHub

---

## 📂 Project Structure

```text
├── .gitignore                     # Git ignore rules (prevents tracking raw dumps, cache, venv, scrap)
├── README.md                      # Comprehensive project documentation & interview guide
├── requirements.txt               # Locked production dependencies
├── app.py                         # Streamlit multi-tab analytics application
├── data/
│   ├── README.md                  # Dataset origin, legal compliance, and limitations
│   └── sample_retail_data.csv     # Reproducible synthetic transaction dataset (3,500 rows)
├── src/
│   ├── __init__.py                # Package initialization
│   ├── generate_sample_data.py    # Synthetic retail transaction generator (UCI schema)
│   ├── data_cleaning.py           # Preprocessing, governance & audit tracking
│   ├── rfm_metrics.py             # Recency, Frequency, Monetary calculation engine
│   ├── segmentation.py            # Rule-based RFM, K-Means clustering & recommendations
│   ├── db_manager.py              # SQLite initialization & parameterized SQL queries
│   └── charts.py                  # Interactive Plotly chart builders
└── tests/
    └── test_analytics.py          # Pytest suite validating pipeline integrity
```

---

## 📊 Analytical Methodology

### 1. Data Cleaning & Integrity Rules
The pipeline ingests raw transactional data and applies strict data governance:
- **Guest Purchases / Missing Customer IDs**: Excluded from customer-level RFM metrics while logging total excluded transactions (~11.3% of raw events).
- **Cancellations & Returns**: Invoices starting with `C` or containing negative quantities are filtered out.
- **Data Hygiene**: Removes zero/negative prices, zero/negative quantities, and duplicate rows.
- **Line Revenue Calculation**: $\text{Revenue} = \text{Quantity} \times \text{UnitPrice}$.

### 2. RFM Calculation Logic
- **Reference Date**: Defined as $\text{max}(\text{InvoiceDate}) + 1\text{ day}$. This simulates running the analytics system the morning following the final recorded transaction date and ensures valid non-negative recency.
- **Recency ($R$)**: Days between the customer's most recent purchase and the reference date.
- **Frequency ($F$)**: Count of distinct, unique `InvoiceNo` orders per customer.
- **Monetary ($M$)**: Total net spend across all completed purchases.

### 3. Customer Segments & Business Rules
Customers are scored into quartiles ($1-4$) for $R$, $F$, and $M$ and mapped to business segments:

| Segment | RFM Profile | Strategic Business Focus | Recommended Action |
| :--- | :--- | :--- | :--- |
| **Champions** | High Recency ($R \ge 3$), High Frequency ($F \ge 3$), High Spend ($M \ge 3$) | Retention & Advocacy | VIP rewards, early access to new lines, testimonial requests. Avoid deep discounting. |
| **Loyal Customers** | Consistent Repeat Orders ($F \ge 3$) | Basket Size Expansion | Category cross-selling, loyalty milestones, personalized product suggestions. |
| **Potential Loyalists** | Recent buyers with above-average activity ($R \ge 3$, $F \ge 2$ or $M \ge 2$) | Habit Building | 2nd-purchase incentives, onboarding tips, subscription replenishment. |
| **Recent Customers** | First-time buyers ($R \ge 3, F = 1$) | First-to-Second Order Conversion | Post-purchase follow-up, satisfaction check, intro discount on complementary items. |
| **At Risk** | Inactive ($R \le 2$) with past high spend or frequency | Win-back & Churn Prevention | Automated re-engagement campaign, feedback survey, high-touch outreach. |
| **Others / Hibernating**| Low recency, low frequency, low spend | Cost Control & Maintenance | Low-cost seasonal clearance announcements; sunset non-responsive emails. |

*Disclaimer: All recommendations are hypothesis-driven interventions. Real-world revenue uplift requires controlled A/B testing verification.*

---

## 📈 Sample Findings (Generated from Included Dataset)

Audited output from the bundled 3,500-transaction benchmark dataset:
- **Data Ingestion Audit**:
  - Raw Ingested: 3,500 rows
  - Valid Retained: 2,971 rows (84.89% retention)
  - Excluded Guest Checkouts: 396 rows
  - Cancelled Invoices: 115 rows
- **Customer Lifecycle Benchmarks**:
  - Total Unique Customers: **186**
  - Repeat Customers: **127 (68.28% repeat rate)**
  - Average Customer Lifetime Spend: **$735.76** (Median: $399.32)
  - Average Purchase Frequency: **4.55 orders**
- **Segment Revenue Contribution**:
  - **Champions (60 customers)**: Generated **$94,402.13 (69.0% of total revenue)**.
  - **Loyal Customers (33 customers)**: Generated **$21,482.46 (15.7%)**.
  - **At Risk (9 customers)**: Represent **$4,708.42** in prior revenue requiring active churn containment.

---

## 🚀 Step-by-Step Setup Instructions (Windows & PowerShell)

### Prerequisites
- Python 3.10 to 3.14 installed ([python.org](https://www.python.org/downloads/))
- Git installed ([git-scm.com](https://git-scm.com/))

### 1. Clone the Repository
```powershell
git clone https://github.com/Jeelanimohammad/customer-segmentation-buying-behavior-analysis.git
cd customer-segmentation-buying-behavior-analysis
```

### 2. Create and Activate a Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(If PowerShell displays an ExecutionPolicy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and re-run activate).*

### 3. Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. (Optional) Re-generate Sample Synthetic Data
```powershell
python src/generate_sample_data.py
```

### 5. Run the Automated Test Suite
```powershell
python -m pytest tests/ -v
```

### 6. Launch the Interactive Streamlit Dashboard
```powershell
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 📸 Dashboard Preview & Screenshot Guide

The dashboard is structured into 5 tabs:

1. **Executive Overview**: High-level KPIs (Total Customers, Net Revenue, Repeat Rate, AOV) + Donut chart of customer distribution + Segment revenue contribution.
2. **Customer Segmentation**: Detailed tabular breakdown of all segments, interactive 2D bubble scatter plot, 3D RFM landscape, exploratory K-Means clustering, and downloadable CSV table.
3. **Buying Behavior**: Monthly revenue trends, order volume bars, invoice basket size histogram, and single vs. repeat customer spend split.
4. **Customer Insights & Actions**: Ranked Champions, priority At-Risk accounts, and rule-based business development action playbooks.
5. **Data & SQL Explorer**: Data cleaning audit trail, live data inspection, and parameterized SQL query runner against local SQLite.

> **Tip for Portfolio / Interview Screen Capture**:
> Use `Win + Shift + S` on Windows to capture high-resolution screenshots of each tab when running `streamlit run app.py` locally.

---

## 🎤 3–5 Minute Interview Walkthrough Script

When presenting this project to a hiring manager or interview panel:

1. **Hook & Problem Statement (0:00 - 0:45)**:
   > *"In modern retail and business development, treating all customers identically wastes marketing budget and increases churn. I built this application to ingest transactional sales data, perform rigorous data cleaning, and compute customer-level RFM metrics to prioritize high-value accounts."*

2. **Data Pipeline & Governance (0:45 - 1:30)**:
   > *"Under the Data & SQL Explorer tab, you can see how the pipeline handles data hygiene: it logs guest transactions without IDs, strips out cancelled invoices (prefixed with 'C'), eliminates zero-priced items, and enforces immutable transformations so original data remains intact."*

3. **RFM Segmentation & Repeat Analytics (1:30 - 2:30)**:
   > *"Under Executive Overview and Customer Segmentation, customers are categorized based on Recency, Frequency, and Monetary metrics. In our dataset, 60 Champions drive over 68% of total revenue. We also track repeat purchase rates: nearly 68% of customers have purchased more than once, confirming high retention potential."*

4. **Actionable Recommendations & SQL Integration (2:30 - 3:30)**:
   > *"Rather than presenting abstract charts, the Insights tab pairs segments with actionable BD strategies—such as VIP recognition for Champions and win-back offers for At-Risk accounts. Finally, the app populates an embedded SQLite database where business development associates can execute parameterized SQL queries to extract targeted lists."*

5. **Analytical Humility & Future Enhancements (3:30 - 4:00)**:
   > *"I intentionally framed recommendations as hypotheses to validate via A/B testing rather than making unverified claims. In future iterations, I would add survival analysis for churn prediction and automated webhook triggers for CRM integration."*

---

## 💡 Key Conceptual Explanations for Candidates

- **Why RFM over simple revenue ranking?**
  Revenue alone rewards a customer who made a single massive purchase two years ago and never returned. RFM balances spend with *recency* and *frequency*, highlighting active loyal customers versus lapsed accounts.
- **Why log-transform features before K-Means?**
  Monetary and Frequency variables in e-commerce are heavily right-skewed (Pareto distribution). Extreme outliers pull centroid calculations. Taking $\log(1 + x)$ normalizes variance, allowing distance-based algorithms like K-Means to cluster meaningfully.
- **Why use Parameterized SQL Queries?**
  Concatenating raw strings into SQL queries introduces SQL Injection vulnerabilities. Parameter substitution (`?` placeholders) ensures values are safely treated as data literals rather than executable SQL syntax.

---

## 📄 License & Attribution
- Open-source and distributed under the MIT License.
- Dataset schemas reference the UCI Online Retail Dataset (Dr. Daqing Chen, London South Bank University).
# Dataset Guidelines & Source Information

## Primary Real-World Dataset: UCI Online Retail Dataset
This project supports the real-world **Online Retail Dataset** provided by the UCI Machine Learning Repository.

### How to Obtain the Dataset Legally:
1. Visit the official UCI Machine Learning Repository page:
   [UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail)
2. Download the `Online Retail.xlsx` file or the archive CSV format.
3. Save or convert the file to `data/online_retail.csv` (or upload it directly via the Streamlit dashboard).
4. License & Attribution:
   - Donor: Dr. Daqing Chen, Director of Public Analytics, School of Engineering, London South Bank University.
   - Publicly available for research and educational purposes.

### Data Limitations:
- Missing Customer IDs: Approximately 25% of transactions in the original UCI dataset lack a registered CustomerID (guest checkouts). In customer-level RFM modeling, these must be excluded while keeping line transactions logged.
- Cancellations: Invoices prefixed with 'C' indicate cancellations or returns and contain negative quantities.
- Geographic Imbalance: Over 90% of transactions originate from the United Kingdom.

---

## Synthetic Sample Dataset
For immediate testing, demonstrations, and automated test suites without external downloads:
- A realistic synthetic generator is provided in `src/generate_sample_data.py`.
- Generates `data/sample_retail_data.csv` replicating the exact schema, cancellation patterns, country distributions, and guest checkout scenarios.
- **Important Note**: Synthetic data is created strictly for reproducible demonstrations and testing; it is never claimed to be real commercial data.

"""
Synthetic Retail Data Generator.

Generates realistic retail transactional data mirroring the UCI Online Retail dataset structure:
InvoiceNo, StockCode, Description, Quantity, UnitPrice, InvoiceDate, CustomerID, Country.
Includes realistic edge cases: cancellations, guest checkouts (missing CustomerID), price anomalies.
"""

import os
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


def generate_synthetic_retail_data(
    num_transactions: int = 3500,
    num_customers: int = 300,
    output_filepath: str = "data/sample_retail_data.csv",
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate realistic synthetic retail transaction data.
    
    Args:
        num_transactions: Target row count
        num_customers: Unique customer ID pool
        output_filepath: Destination CSV path
        random_seed: Seed for reproducible output
    
    Returns:
        pd.DataFrame containing synthetic transactions
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    products = [
        ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 2.95),
        ("71053", "WHITE METAL LANTERN", 3.39),
        ("84406B", "CREAM CUPID HEARTS COAT HANGER", 4.25),
        ("84029G", "KNITTED UNION FLAG HOT WATER BOTTLE", 3.75),
        ("84029E", "RED WOOLLY HOTTIE WHITE HEART.", 3.39),
        ("22752", "SET 7 BABUSHKA NESTING BOXES", 8.50),
        ("21730", "GLASS STAR FROSTED T-LIGHT HOLDER", 4.25),
        ("22633", "HAND WARMER UNION JACK", 2.10),
        ("22632", "HAND WARMER RED POLKADOT", 2.10),
        ("84879", "ASSORTED COLOUR BIRD ORNAMENT", 1.69),
        ("22745", "POPPY'S PLAYHOUSE BEDROOM", 2.10),
        ("22748", "POPPY'S PLAYHOUSE KITCHEN", 2.10),
        ("22749", "FELTCRAFT PRINCESS CHARLOTTE DOLL", 3.75),
        ("22310", "IVORY KNITTED CHUNKY BEANIE", 4.15),
        ("84945", "MULTI COLOUR SILVER T-LIGHT HOLDER", 0.85),
        ("22960", "JAM MAKING SET WITH JARS", 4.25),
        ("22961", "JAM MAKING SET PRINTED", 1.45),
        ("22138", "BAKING SET 9 PIECE RETROSPOT", 4.95),
        ("21034", "REX CASH & CARRY JUMBO SHOPPER", 0.95),
        ("22423", "REGENCY CAKESTAND 3 TIER", 12.75),
    ]

    countries = ["United Kingdom"] * 82 + ["Germany"] * 6 + ["France"] * 5 + ["EIRE"] * 3 + ["Spain"] * 2 + ["Australia"] * 2

    start_date = datetime(2023, 1, 1, 8, 30)
    end_date = datetime(2023, 12, 10, 18, 00)
    time_delta = (end_date - start_date).total_seconds()

    customer_ids = [10000 + i for i in range(num_customers)]

    # Assign customer weights to simulate repeat buyers vs one-off buyers
    customer_weights = np.random.pareto(a=1.5, size=len(customer_ids))
    customer_weights /= customer_weights.sum()

    records = []
    invoice_seq = 536365

    # Group into multi-item orders
    total_generated = 0
    while total_generated < num_transactions:
        invoice_no = str(invoice_seq)
        invoice_seq += 1

        # Determine if guest checkout (missing CustomerID) ~ 12%
        is_guest = random.random() < 0.12
        customer_id = None if is_guest else int(np.random.choice(customer_ids, p=customer_weights))
        country = random.choice(countries)
        
        # Determine invoice timestamp
        rand_sec = random.uniform(0, time_delta)
        order_time = start_date + timedelta(seconds=rand_sec)
        formatted_time = order_time.strftime("%Y-%m-%d %H:%M:%S")

        items_in_order = random.randint(1, 6)

        # ~4% chance this is a cancelled invoice
        is_cancelled = random.random() < 0.04
        if is_cancelled:
            invoice_no = "C" + invoice_no

        for _ in range(items_in_order):
            prod = random.choice(products)
            stock_code, desc, unit_price = prod

            # Price variance / discounts
            actual_price = round(unit_price * random.choice([0.9, 1.0, 1.0, 1.0, 1.1]), 2)
            qty = random.randint(1, 24)
            if is_cancelled:
                qty = -qty

            # Edge case anomalies: rare bad unit price or zero qty
            anomaly_dice = random.random()
            if anomaly_dice < 0.003:
                actual_price = 0.0
            elif anomaly_dice < 0.006:
                qty = 0

            records.append({
                "InvoiceNo": invoice_no,
                "StockCode": stock_code,
                "Description": desc,
                "Quantity": qty,
                "InvoiceDate": formatted_time,
                "UnitPrice": actual_price,
                "CustomerID": customer_id if customer_id is not None else np.nan,
                "Country": country
            })
            total_generated += 1
            if total_generated >= num_transactions:
                break

    df = pd.DataFrame(records)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    df.to_csv(output_filepath, index=False)
    print(f"Synthetic sample dataset created at: {output_filepath} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    generate_synthetic_retail_data()

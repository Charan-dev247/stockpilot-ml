
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DAILY_FILE = ROOT / "data" / "processed" / "daily_demand.csv"
TRANSACTIONS_FILE = ROOT / "data" / "processed" / "cleaned_transactions.csv"


def main():
    daily = pd.read_csv(
    DAILY_FILE,
    parse_dates=["Date"],
    dtype={"StockCode": "string"},
    )

    transactions = pd.read_csv(
        TRANSACTIONS_FILE,
        parse_dates=["InvoiceDate"],
        dtype={"StockCode": "string", "Invoice": "string"},
        low_memory=False,
    )

    print("=" * 60)
    print("DAILY DEMAND VALIDATION")
    print("=" * 60)

    print(f"\nCleaned positive-quantity transactions: {len(transactions):,}")
    print(f"Daily product-date records: {len(daily):,}")
    print(f"Unique products: {daily['StockCode'].nunique():,}")
    print(f"Unique dates: {daily['Date'].nunique():,}")
    print(f"First date: {daily['Date'].min()}")
    print(f"Last date: {daily['Date'].max()}")

    print("\nMissing values:")
    print(daily.isna().sum().to_string())

    print("\nDuplicate product-date keys:")
    print(daily.duplicated(["Date", "StockCode"]).sum())

    print("\nDemand checks:")
    print(f"Zero or negative daily demand: {(daily['demand'] <= 0).sum():,}")
    print(f"Non-integer demand: {(daily['demand'] % 1 != 0).sum():,}")

    print("\nDemand summary:")
    print(daily["demand"].describe().to_string())

    print("\nMost frequently appearing products:")
    print(daily["StockCode"].value_counts().head(10).to_string())

    print("\nRecords per product (summary):")
    print(daily.groupby("StockCode")["Date"].nunique().describe().to_string())

    print("\n" + "=" * 60)
    print("Validation completed. No files were modified.")
    print("=" * 60)


if __name__ == "__main__":
    main()

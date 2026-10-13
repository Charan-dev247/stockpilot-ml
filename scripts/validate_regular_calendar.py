
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "data" / "processed" / "regular_daily_demand.csv"


def main():
    data = pd.read_csv(
        FILE,
        parse_dates=["Date"],
        dtype={"StockCode": "string"},
    )

    print("=" * 60)
    print("REGULAR DAILY CALENDAR VALIDATION")
    print("=" * 60)

    print(f"\nTotal rows: {len(data):,}")
    print(f"Unique products: {data['StockCode'].nunique():,}")
    print(f"Unique dates: {data['Date'].nunique():,}")
    print(f"Missing values:\n{data.isna().sum().to_string()}")

    duplicate_count = data.duplicated(["StockCode", "Date"]).sum()
    print(f"\nDuplicate product-date keys: {duplicate_count:,}")

    print("\nZero-demand rows:", f"{(data['demand'] == 0).sum():,}")
    print("Positive-demand rows:", f"{(data['demand'] > 0).sum():,}")
    print("Negative-demand rows:", f"{(data['demand'] < 0).sum():,}")

    # Check whether each product has consecutive calendar dates
    gaps = []
    for stock_code, group in data.groupby("StockCode", sort=False):
        dates = group["Date"].sort_values()
        expected_days = (dates.max() - dates.min()).days + 1

        if len(dates) != expected_days:
            gaps.append(stock_code)

    print(f"\nProducts with date gaps: {len(gaps):,}")

    print("\nExample product histories:")
    for stock_code in data["StockCode"].drop_duplicates().head(3):
        sample = (
            data.loc[data["StockCode"] == stock_code]
            .sort_values("Date")
            .head(7)
        )
        print(f"\nStockCode: {stock_code}")
        print(sample[["Date", "demand"]].to_string(index=False))

    print("\n" + "=" * 60)
    print("Validation complete. No files were modified.")
    print("=" * 60)


if __name__ == "__main__":
    main()

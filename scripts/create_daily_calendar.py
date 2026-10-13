
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "processed" / "daily_demand.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "regular_daily_demand.csv"


def main():
    print("Loading daily demand dataset...")

    daily = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Date"],
        dtype={"StockCode": "string"},
    )

    daily["Date"] = daily["Date"].dt.normalize()

    if daily.duplicated(["StockCode", "Date"]).any():
        raise ValueError("Duplicate product-date records found.")

    original_rows = len(daily)
    regular_parts = []

    print(f"Input records: {original_rows:,}")
    print(f"Products: {daily['StockCode'].nunique():,}")
    print("Building daily calendars for each product...")

    # Build a calendar only between each product's
    # first and last recorded positive-sales dates.
    for stock_code, group in daily.groupby("StockCode", sort=False):
        group = group.sort_values("Date").set_index("Date")

        calendar = pd.date_range(
            start=group.index.min(),
            end=group.index.max(),
            freq="D",
        )

        regular = group.reindex(calendar)
        regular.index.name = "Date"

        regular["StockCode"] = stock_code
        regular["demand"] = regular["demand"].fillna(0)
        regular["transaction_lines"] = (
            regular["transaction_lines"].fillna(0)
        )

        regular_parts.append(
            regular.reset_index()[
                ["Date", "StockCode", "demand", "transaction_lines"]
            ]
        )

    result = pd.concat(regular_parts, ignore_index=True)

    result["demand"] = result["demand"].astype("int64")
    result["transaction_lines"] = (
        result["transaction_lines"].astype("int64")
    )

    result = result.sort_values(
        ["StockCode", "Date"]
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 60)
    print("REGULAR DAILY CALENDAR REPORT")
    print("=" * 60)

    print(f"Original daily records: {original_rows:,}")
    print(f"Regular daily records: {len(result):,}")
    print(f"Added no-sales dates: {len(result) - original_rows:,}")
    print(f"Products: {result['StockCode'].nunique():,}")
    print(f"First date: {result['Date'].min()}")
    print(f"Last date: {result['Date'].max()}")
    print(f"Zero-demand rows: {(result['demand'] == 0).sum():,}")
    print(
        "Duplicate product-date keys:",
        result.duplicated(["StockCode", "Date"]).sum(),
    )

    print("\nSample rows:")
    print(result.head(10).to_string(index=False))

    print(f"\nSaved to: {OUTPUT_FILE}")
    print("Original dataset and daily_demand.csv remain unchanged.")
    print(
        "Zero means no positive sales were recorded within "
        "the product's observed date range."
    )


if __name__ == "__main__":
    main()

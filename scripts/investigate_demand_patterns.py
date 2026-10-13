
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

RAW_FILE = ROOT / "data" / "raw" / "online_retail_II.xlsx"
CALENDAR_FILE = (
    ROOT / "data" / "processed" / "regular_daily_demand.csv"
)


def main():
    print("=" * 65)
    print("STOCKPILOT ML - DEMAND PATTERN INVESTIGATION")
    print("=" * 65)

    # 1. Inspect the extreme product in the original workbook.
    print("\n--- Product 23843: Original Transactions ---")

    sheets = pd.read_excel(
        RAW_FILE,
        sheet_name=None,
        dtype={"StockCode": "string", "Invoice": "string"},
    )

    raw = pd.concat(sheets.values(), ignore_index=True)

    raw["InvoiceDate"] = pd.to_datetime(
        raw["InvoiceDate"], errors="coerce"
    )

    extreme = raw.loc[raw["StockCode"] == "23843"].copy()

    if extreme.empty:
        print("Product 23843 was not found.")
    else:
        print(f"Transaction lines: {len(extreme):,}")
        print(f"Total quantity, including returns: "
              f"{extreme['Quantity'].sum():,.0f}")
        print(f"Positive quantity: "
              f"{extreme.loc[extreme['Quantity'] > 0, 'Quantity'].sum():,.0f}")
        print(f"Negative quantity: "
              f"{extreme.loc[extreme['Quantity'] < 0, 'Quantity'].sum():,.0f}")

        print("\nTransaction details:")
        print(
            extreme[
                [
                    "Invoice",
                    "StockCode",
                    "Description",
                    "Quantity",
                    "InvoiceDate",
                    "Price",
                ]
            ]
            .sort_values("InvoiceDate")
            .to_string(index=False)
        )

    # 2. Investigate weekday activity using positive transaction lines.
    print("\n--- Weekday Activity in Original Data ---")

    positive = raw.loc[raw["Quantity"] > 0].copy()
    positive["Date"] = positive["InvoiceDate"].dt.normalize()
    positive["Weekday"] = positive["Date"].dt.day_name()

    weekday_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    ]

    weekday_activity = (
        positive.groupby("Weekday")
        .agg(
            transaction_lines=("Quantity", "size"),
            distinct_dates=("Date", "nunique"),
            total_quantity=("Quantity", "sum"),
        )
        .reindex(weekday_order)
    )

    print(weekday_activity.fillna(0).round(2).to_string())

    # 3. Inspect product history lengths.
    print("\n--- Product History Lengths ---")

    data = pd.read_csv(
        CALENDAR_FILE,
        parse_dates=["Date"],
        dtype={"StockCode": "string"},
    )

    histories = (
        data.groupby("StockCode")
        .agg(
            first_date=("Date", "min"),
            last_date=("Date", "max"),
            calendar_days=("Date", "size"),
            active_days=("demand", lambda x: (x > 0).sum()),
            total_demand=("demand", "sum"),
        )
    )

    print("\nCalendar days per product:")
    print(histories["calendar_days"].describe().round(2))

    print("\nProducts by history length:")
    for threshold in [30, 90, 180, 365]:
        count = (histories["calendar_days"] >= threshold).sum()
        print(
            f"At least {threshold:>3} calendar days: "
            f"{count:,} products"
        )

    print("\nProducts with fewer than 30 calendar days:")
    print((histories["calendar_days"] < 30).sum())

    print("\nInvestigation complete. No files were modified.")


if __name__ == "__main__":
    main()


from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data" / "processed" / "regular_daily_demand.csv"


def main():
    data = pd.read_csv(
        DATA_FILE,
        parse_dates=["Date"],
        dtype={"StockCode": "string"},
    )

    first_date = data["Date"].min()
    last_date = data["Date"].max()

    # Use chronological cutoffs based on the complete calendar.
    total_days = (last_date - first_date).days + 1

    train_end = first_date + pd.Timedelta(
        days=int(total_days * 0.70) - 1
    )
    validation_end = first_date + pd.Timedelta(
        days=int(total_days * 0.85) - 1
    )

    train = data.loc[data["Date"] <= train_end]
    validation = data.loc[
        (data["Date"] > train_end)
        & (data["Date"] <= validation_end)
    ]
    test = data.loc[data["Date"] > validation_end]

    print("=" * 65)
    print("STOCKPILOT ML - CHRONOLOGICAL SPLIT")
    print("=" * 65)

    print(f"\nComplete date range: {first_date.date()} to {last_date.date()}")
    print(f"Calendar days: {total_days}")

    for name, frame in [
        ("TRAIN", train),
        ("VALIDATION", validation),
        ("TEST", test),
    ]:
        print(f"\n--- {name} ---")
        print(f"Date range: {frame['Date'].min().date()} to "
              f"{frame['Date'].max().date()}")
        print(f"Rows: {len(frame):,}")
        print(f"Products: {frame['StockCode'].nunique():,}")
        print(f"Total recorded demand: {frame['demand'].sum():,.0f}")
        print(f"Zero-demand percentage: "
              f"{(frame['demand'] == 0).mean() * 100:.2f}%")

    print("\nNo files were modified.")
    print("Split inspection completed.")


if __name__ == "__main__":
    main()

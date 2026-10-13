
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
    total_days = (data["Date"].max() - first_date).days + 1

    train_end = first_date + pd.Timedelta(
        days=int(total_days * 0.70) - 1
    )
    validation_end = first_date + pd.Timedelta(
        days=int(total_days * 0.85) - 1
    )

    train = data[data["Date"] <= train_end]
    validation = data[
        (data["Date"] > train_end)
        & (data["Date"] <= validation_end)
    ]
    test = data[data["Date"] > validation_end]

    train_products = set(train["StockCode"].unique())
    validation_products = set(validation["StockCode"].unique())
    test_products = set(test["StockCode"].unique())

    all_products = set(data["StockCode"].unique())

    print("=" * 65)
    print("STOCKPILOT ML - PRODUCT COVERAGE ANALYSIS")
    print("=" * 65)

    print(f"\nProducts overall: {len(all_products):,}")
    print(f"Products in training: {len(train_products):,}")
    print(f"Products in validation: {len(validation_products):,}")
    print(f"Products in test: {len(test_products):,}")

    print("\n--- Product Coverage ---")
    print(
        "Present in all three periods:",
        f"{len(train_products & validation_products & test_products):,}",
    )
    print(
        "In validation but not training:",
        f"{len(validation_products - train_products):,}",
    )
    print(
        "In test but not training:",
        f"{len(test_products - train_products):,}",
    )
    print(
        "In training but not validation:",
        f"{len(train_products - validation_products):,}",
    )
    print(
        "In training but not test:",
        f"{len(train_products - test_products):,}",
    )

    # Inspect when products first appear in the recorded dataset.
    first_sales = data.loc[data["demand"] > 0].groupby(
        "StockCode"
    )["Date"].min()

    print("\n--- First Recorded Positive Sale ---")
    print(f"First sales during training: "
          f"{(first_sales <= train_end).sum():,}")
    print(
        "First sales during validation:",
        f"{((first_sales > train_end) & (first_sales <= validation_end)).sum():,}",
    )
    print(
        "First sales during test:",
        f"{(first_sales > validation_end).sum():,}",
    )

    print("\nCoverage analysis complete. No files were modified.")


if __name__ == "__main__":
    main()

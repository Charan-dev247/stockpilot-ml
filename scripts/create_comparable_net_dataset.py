
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

original_file = PROCESSED / "regular_daily_demand.csv"
net_file = PROCESSED / "regular_net_daily_demand.csv"
output_file = PROCESSED / "comparable_net_daily_demand.csv"

original = pd.read_csv(
    original_file,
    usecols=["StockCode", "Date"],
    dtype={"StockCode": "string"},
    parse_dates=["Date"],
)

net = pd.read_csv(
    net_file,
    dtype={"StockCode": "string"},
    parse_dates=["Date"],
)

# Use only products that belong to the original product cohort.
original_products = set(original["StockCode"].dropna().unique())

comparable = net[net["StockCode"].isin(original_products)].copy()

# Ensure we have exactly the original product-date keys.
original_keys = original[["StockCode", "Date"]].drop_duplicates()
comparable = original_keys.merge(
    comparable,
    on=["StockCode", "Date"],
    how="left",
    validate="one_to_one",
)

if comparable["demand"].isna().any():
    raise ValueError(
        "Some original product-date rows are missing from the net dataset."
    )

comparable.to_csv(output_file, index=False)

print("=" * 64)
print("COMPARABLE NET-DEMAND DATASET")
print("=" * 64)
print(f"Rows: {len(comparable):,}")
print(f"Products: {comparable['StockCode'].nunique():,}")
print(f"Dates: {comparable['Date'].nunique():,}")
print(f"Negative raw net-quantity days: "
      f"{(comparable['net_quantity'] < 0).sum():,}")
print(f"Zero target days: {(comparable['demand'] == 0).sum():,}")
print(f"Positive target days: {(comparable['demand'] > 0).sum():,}")
print(f"Saved to: {output_file}")

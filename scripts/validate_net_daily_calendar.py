
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "data" / "processed" / "regular_net_daily_demand.csv"

df = pd.read_csv(
    FILE,
    parse_dates=["Date"],
    dtype={"StockCode": "string"},
)

products = df["StockCode"].nunique()
dates = df["Date"].nunique()

checks = {
    "No missing values": not df.isna().any().any(),
    "No duplicate product-date pairs": not df.duplicated(
        ["StockCode", "Date"]
    ).any(),
    "Complete product-date calendar": len(df) == products * dates,
    "Exactly 739 dates": dates == 739,
    "Non-negative demand target": (df["demand"] >= 0).all(),
    "Net quantity is preserved": (
        df["net_quantity"]
        == df["positive_quantity"] + df["negative_quantity"]
    ).all(),
}

print("=" * 64)
print("ALTERNATIVE CALENDAR VALIDATION")
print("=" * 64)

for name, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL'}: {name}")

print(f"\nRows: {len(df):,}")
print(f"Products: {products:,}")
print(f"Dates: {dates:,}")
print(f"Negative raw net-quantity days: "
      f"{(df['net_quantity'] < 0).sum():,}")
print(f"Zero demand days: {(df['demand'] == 0).sum():,}")
print(f"Positive demand days: {(df['demand'] > 0).sum():,}")

if not all(checks.values()):
    raise ValueError("Validation failed. Inspect the results.")

print("\nAll checks passed.")

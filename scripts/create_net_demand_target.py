
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "reports" / "cancellation_audit_daily.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["Date"],
    dtype={"StockCode": "string"},
)

# Preserve the raw net quantity for analysis.
# For a non-negative demand target, negative net quantities are
# clipped to zero. This is an experimental definition, not ground truth.
df["demand"] = df["net_quantity"].clip(lower=0)

net_demand = df[
    [
        "Date",
        "StockCode",
        "demand",
        "positive_quantity",
        "negative_quantity",
        "net_quantity",
        "transaction_lines",
        "cancellation_invoice_lines",
    ]
].copy()

output_file = OUTPUT_DIR / "net_daily_demand.csv"
net_demand.to_csv(output_file, index=False)

print("=" * 64)
print("STOCKPILOT ML - ALTERNATIVE DEMAND TARGET")
print("=" * 64)

print(f"Product-day rows: {len(net_demand):,}")
print(f"Unique products: {net_demand['StockCode'].nunique():,}")
print(f"Date range: {net_demand['Date'].min().date()} to "
      f"{net_demand['Date'].max().date()}")
print(f"Rows with negative net quantity: "
      f"{(net_demand['net_quantity'] < 0).sum():,}")
print(f"Rows with zero net quantity: "
      f"{(net_demand['net_quantity'] == 0).sum():,}")
print(f"Rows with positive net quantity: "
      f"{(net_demand['net_quantity'] > 0).sum():,}")

print("\n--- Largest positive quantities after netting ---")
print(
    net_demand.nlargest(10, "demand")[
        ["Date", "StockCode", "positive_quantity",
         "negative_quantity", "net_quantity", "demand"]
    ].to_string(index=False)
)

print(f"\nSaved alternative target to: {output_file}")
print("Existing datasets and model outputs were not modified.")

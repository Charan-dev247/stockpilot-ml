
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "net_daily_demand.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "regular_net_daily_demand.csv"

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["Date"],
    dtype={"StockCode": "string"},
)

# Aggregate to one row per product and date.
daily = (
    df.groupby(["StockCode", "Date"], as_index=False)
    .agg(
        demand=("demand", "sum"),
        positive_quantity=("positive_quantity", "sum"),
        negative_quantity=("negative_quantity", "sum"),
        net_quantity=("net_quantity", "sum"),
    )
)

start_date = daily["Date"].min()
end_date = daily["Date"].max()
all_dates = pd.date_range(start_date, end_date, freq="D")

products = daily["StockCode"].dropna().unique()

calendar = pd.MultiIndex.from_product(
    [products, all_dates],
    names=["StockCode", "Date"],
).to_frame(index=False)

regular = calendar.merge(
    daily,
    on=["StockCode", "Date"],
    how="left",
    validate="one_to_one",
)

# Missing transaction dates represent no recorded transactions.
# These are not guaranteed to be true zero customer demand.
for col in [
    "demand",
    "positive_quantity",
    "negative_quantity",
    "net_quantity",
]:
    regular[col] = regular[col].fillna(0)

regular["demand"] = regular["demand"].astype("int64")

regular.to_csv(OUTPUT_FILE, index=False)

print("=" * 64)
print("STOCKPILOT ML - NET DEMAND CALENDAR")
print("=" * 64)
print(f"Product-day rows: {len(regular):,}")
print(f"Products: {regular['StockCode'].nunique():,}")
print(f"Dates: {regular['Date'].nunique():,}")
print(f"Date range: {start_date.date()} to {end_date.date()}")
print(f"Zero-demand days: {(regular['demand'] == 0).sum():,}")
print(f"Positive-demand days: {(regular['demand'] > 0).sum():,}")
print(f"Negative net-quantity days: {(regular['net_quantity'] < 0).sum():,}")
print(f"Saved to: {OUTPUT_FILE}")

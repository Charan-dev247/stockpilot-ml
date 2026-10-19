
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "comparable_net_daily_demand.csv"
)

REPORT_DIR = ROOT / "reports"

TRAIN_END = "2011-05-01"

LEAD_TIME_DAYS = 7
REVIEW_PERIOD_DAYS = 7
SERVICE_LEVEL = 0.95

# Demonstration assumptions, not actual inventory records.
DEFAULT_ON_HAND = 0
DEFAULT_ON_ORDER = 0

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load net-adjusted demand
# --------------------------------------------------

df = pd.read_csv(
    DATA_PATH,
    dtype={"StockCode": str},
)

df.columns = df.columns.str.strip()

required_columns = {"StockCode", "Date"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Expected columns {required_columns}; "
        f"found {df.columns.tolist()}"
    )

date_column = next(
    (
        c for c in df.columns
        if c.lower().replace(" ", "_") in
        {"date", "invoicedate", "invoice_date"}
    ),
    None,
)

demand_column = next(
    (
        c for c in df.columns
        if c.lower().replace(" ", "_") in
        {"demand_net", "net_demand", "demand", "net_quantity"}
    ),
    None,
)

if date_column is None or demand_column is None:
    raise ValueError(
        "Could not identify date or net-demand column. "
        f"Available columns: {df.columns.tolist()}"
    )

df = df.rename(
    columns={
        date_column: "Date",
        demand_column: "Demand",
    }
)

df["Date"] = pd.to_datetime(df["Date"])
df["Demand"] = pd.to_numeric(
    df["Demand"],
    errors="raise",
)

if df.duplicated(["StockCode", "Date"]).any():
    raise ValueError("Duplicate product-date records found.")

if (df["Demand"] < 0).any():
    raise ValueError(
        "Negative net demand found. Check the input dataset."
    )

# Do not use validation or test observations for planning estimates.
df = df[df["Date"] <= pd.Timestamp(TRAIN_END)].copy()

if df.empty:
    raise ValueError("No training-period demand data found.")


# --------------------------------------------------
# 3. Complete the daily calendar per product
# --------------------------------------------------

products = sorted(df["StockCode"].unique())

calendar = pd.date_range(
    start=df["Date"].min(),
    end=pd.Timestamp(TRAIN_END),
    freq="D",
)

full_index = pd.MultiIndex.from_product(
    [products, calendar],
    names=["StockCode", "Date"],
)

daily = (
    df.set_index(["StockCode", "Date"])["Demand"]
    .reindex(full_index, fill_value=0)
    .rename("Demand")
    .reset_index()
)

# --------------------------------------------------
# 4. Estimate demand statistics
# --------------------------------------------------

stats = daily.groupby("StockCode")["Demand"].agg(
    average_daily_demand="mean",
    daily_demand_std="std",
    active_days=lambda x: int((x > 0).sum()),
    total_demand="sum",
    days_observed="count",
).reset_index()

stats["daily_demand_std"] = (
    stats["daily_demand_std"].fillna(0)
)

stats["zero_demand_days"] = (
    stats["days_observed"] - stats["active_days"]
)

# --------------------------------------------------
# 5. Calculate safety stock and reorder point
# --------------------------------------------------

z_score = norm.ppf(SERVICE_LEVEL)

stats["lead_time_demand"] = (
    stats["average_daily_demand"] * LEAD_TIME_DAYS
)

# Assumes fixed lead time and independent daily demand.
stats["safety_stock"] = (
    z_score
    * stats["daily_demand_std"]
    * np.sqrt(LEAD_TIME_DAYS)
)

stats["reorder_point"] = (
    stats["lead_time_demand"] + stats["safety_stock"]
).apply(np.ceil).astype(int)

stats["safety_stock"] = (
    stats["safety_stock"].apply(np.ceil).astype(int)
)

stats["lead_time_demand"] = (
    stats["lead_time_demand"].apply(np.ceil).astype(int)
)

# Periodic-review order-up-to level covers both lead time
# and the time until the next inventory review.
protection_period = LEAD_TIME_DAYS + REVIEW_PERIOD_DAYS

stats["order_up_to_level"] = (
    stats["average_daily_demand"] * protection_period
    + z_score
    * stats["daily_demand_std"]
    * np.sqrt(protection_period)
).apply(np.ceil).astype(int)


# --------------------------------------------------
# 6. Add explicit simulated inventory inputs
# --------------------------------------------------

# Replace these columns with actual inventory data when available.
stats["on_hand"] = DEFAULT_ON_HAND
stats["on_order"] = DEFAULT_ON_ORDER

stats["inventory_position"] = (
    stats["on_hand"] + stats["on_order"]
)

stats["recommended_order_qty"] = (
    stats["order_up_to_level"] - stats["inventory_position"]
).clip(lower=0).astype(int)

stats["reorder_required"] = (
    stats["inventory_position"] <= stats["reorder_point"]
)

stats["recommendation"] = np.select(
    [
        stats["reorder_required"],
        stats["recommended_order_qty"] > 0,
    ],
    [
        "Reorder now",
        "Below order-up-to level",
    ],
    default="No order needed",
)

# --------------------------------------------------
# 7. Add assumptions and save report
# --------------------------------------------------

stats["lead_time_days"] = LEAD_TIME_DAYS
stats["review_period_days"] = REVIEW_PERIOD_DAYS
stats["target_service_level"] = SERVICE_LEVEL
stats["training_cutoff"] = TRAIN_END
stats["demand_target"] = "Net-adjusted recorded sales"

output_path = REPORT_DIR / "inventory_recommendations.csv"

stats = stats.sort_values(
    ["reorder_required", "recommended_order_qty"],
    ascending=[False, False],
)

stats.to_csv(output_path, index=False)

print("StockPilot Inventory Planning Engine")
print("=" * 60)
print(f"Demand target: Net-adjusted recorded sales")
print(f"Training cutoff: {TRAIN_END}")
print(f"Products analyzed: {len(stats):,}")
print(f"Supplier lead time: {LEAD_TIME_DAYS} days")
print(f"Review period: {REVIEW_PERIOD_DAYS} days")
print(f"Target service level: {SERVICE_LEVEL:.0%}")
print(f"Reorder point formula: lead-time demand + safety stock")
print(
    "Products flagged for reorder: "
    f"{stats['reorder_required'].sum():,}"
)
print(
    "Total recommended units: "
    f"{stats['recommended_order_qty'].sum():,}"
)

print("\nTop 15 recommendations:")
print(
    stats[
        [
            "StockCode",
            "average_daily_demand",
            "safety_stock",
            "reorder_point",
            "order_up_to_level",
            "recommended_order_qty",
            "recommendation",
        ]
    ]
    .head(15)
    .to_string(index=False)
)

print(f"\nSaved report: {output_path}")

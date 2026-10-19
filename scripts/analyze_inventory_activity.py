
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "comparable_net_daily_demand.csv"
OUTPUT_PATH = ROOT / "reports" / "inventory_activity_analysis.csv"

TRAIN_CUTOFF = pd.Timestamp("2011-05-01")
RECENT_WINDOW_DAYS = 28
INACTIVE_AFTER_DAYS = 90

df = pd.read_csv(
    DATA_PATH,
    dtype={"StockCode": str},
    low_memory=False,
)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["demand"] = pd.to_numeric(df["demand"], errors="coerce")

df = df.dropna(subset=["StockCode", "Date", "demand"])
df = df[df["Date"] <= TRAIN_CUTOFF].copy()

if (df["demand"] < 0).any():
    raise ValueError("The selected demand target must be non-negative.")

if df.duplicated(["StockCode", "Date"]).any():
    raise ValueError("Duplicate product-date rows detected.")

rows = []

for sku, product in df.groupby("StockCode", sort=False):
    product = product.sort_values("Date")
    demand = product["demand"]
    positive = product.loc[product["demand"] > 0]

    first_date = product["Date"].min()
    last_sale = positive["Date"].max() if not positive.empty else pd.NaT

    days_since_last_sale = (
        (TRAIN_CUTOFF - last_sale).days
        if pd.notna(last_sale)
        else np.nan
    )

    recent_start = TRAIN_CUTOFF - pd.Timedelta(
        days=RECENT_WINDOW_DAYS - 1
    )
    recent = product.loc[
        product["Date"].between(recent_start, TRAIN_CUTOFF)
    ]

    total = demand.sum()
    positive_sorted = demand[demand > 0].sort_values(ascending=False)

    rows.append({
        "StockCode": sku,
        "first_recorded_date": first_date,
        "last_sale_date": last_sale,
        "days_since_last_sale": days_since_last_sale,
        "recorded_days": len(product),
        "positive_days": int((demand > 0).sum()),
        "recent_28d_demand": recent["demand"].sum(),
        "recent_28d_positive_days": int((recent["demand"] > 0).sum()),
        "mean_daily_demand": demand.mean(),
        "max_daily_demand": demand.max(),
        "top_1_day_demand_share": (
            positive_sorted.head(1).sum() / total if total > 0 else 0
        ),
        "top_5_days_demand_share": (
            positive_sorted.head(5).sum() / total if total > 0 else 0
        ),
    })

result = pd.DataFrame(rows)

result["activity_status"] = np.select(
    [
        result["last_sale_date"].isna(),
        result["days_since_last_sale"] > INACTIVE_AFTER_DAYS,
        result["recent_28d_positive_days"] == 0,
    ],
    [
        "no_recorded_sales",
        "stale_over_90_days",
        "no_sales_in_recent_28_days",
    ],
    default="recently_active",
)

result["high_concentration"] = (
    result["top_5_days_demand_share"] >= 0.80
)

result.to_csv(OUTPUT_PATH, index=False)

print(f"Training cutoff: {TRAIN_CUTOFF.date()}")
print(f"Products analyzed: {len(result):,}")

print("\nActivity categories:")
print(result["activity_status"].value_counts().to_string())

print("\nProducts with concentrated demand:")
print(f"{result['high_concentration'].sum():,}")

print("\nActivity and demand summary:")
print(
    result.groupby("activity_status").agg(
        products=("StockCode", "size"),
        median_recent_28d_demand=("recent_28d_demand", "median"),
        median_days_since_last_sale=("days_since_last_sale", "median"),
        median_top_5_day_share=("top_5_days_demand_share", "median"),
    ).round(2).to_string()
)

print(f"\nSaved: {OUTPUT_PATH}")

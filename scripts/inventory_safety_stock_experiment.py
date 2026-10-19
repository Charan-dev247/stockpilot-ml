
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


# --------------------------------------------------
# Configuration
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "comparable_net_daily_demand.csv"
OUTPUT_PATH = ROOT / "reports" / "inventory_safety_stock_comparison.csv"

TRAIN_CUTOFF = pd.Timestamp("2011-05-01")

LEAD_TIME_DAYS = 7
REVIEW_PERIOD_DAYS = 7
SERVICE_LEVEL = 0.95

Z_SCORE = norm.ppf(SERVICE_LEVEL)


# --------------------------------------------------
# Load and validate data
# --------------------------------------------------
df = pd.read_csv(DATA_PATH)

# Accommodate common column-name variations.
date_candidates = ["Date", "date", "InvoiceDate"]
target_candidates = ["demand", "net_quantity", "quantity", "Quantity"]
product_candidates = ["StockCode", "stock_code", "Product"]

date_col = next((c for c in date_candidates if c in df.columns), None)
target_col = next((c for c in target_candidates if c in df.columns), None)
product_col = next((c for c in product_candidates if c in df.columns), None)

if not all([date_col, target_col, product_col]):
    raise ValueError(
        f"Could not identify required columns. Found: {df.columns.tolist()}"
    )

df = df[[product_col, date_col, target_col]].copy()
df.columns = ["StockCode", "Date", "demand"]

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["demand"] = pd.to_numeric(df["demand"], errors="coerce")

df = df.dropna(subset=["StockCode", "Date", "demand"])
df = df[df["Date"] <= TRAIN_CUTOFF].copy()

if (df["demand"] < 0).any():
    raise ValueError("Negative demand found. Check the net-demand target.")

if df.duplicated(["StockCode", "Date"]).any():
    raise ValueError("Duplicate product-date rows found in the input.")

df = df.sort_values(["StockCode", "Date"]).reset_index(drop=True)

print(f"Training data ends: {df['Date'].max().date()}")
print(f"Products available: {df['StockCode'].nunique():,}")


# --------------------------------------------------
# Calculate historical demand statistics
# --------------------------------------------------
grouped = df.groupby("StockCode", sort=False)["demand"]

stats = grouped.agg(
    average_daily_demand="mean",
    daily_demand_std="std",
    training_days="count",
    total_demand="sum",
).reset_index()

# Existing normal-approximation approach.
stats["normal_safety_stock"] = (
    Z_SCORE
    * stats["daily_demand_std"].fillna(0)
    * np.sqrt(LEAD_TIME_DAYS)
)

stats["normal_reorder_point"] = (
    stats["average_daily_demand"] * LEAD_TIME_DAYS
    + stats["normal_safety_stock"]
)

protection_period = LEAD_TIME_DAYS + REVIEW_PERIOD_DAYS

stats["normal_order_up_to"] = (
    stats["average_daily_demand"] * protection_period
    + Z_SCORE
    * stats["daily_demand_std"].fillna(0)
    * np.sqrt(protection_period)
)


# --------------------------------------------------
# Empirical lead-time and protection-period demand
# --------------------------------------------------
# Rolling sums use only past and current observations.
# A complete window is required; incomplete early windows
# are excluded from the empirical quantiles.


daily_frames = []

for stock_code, product_df in df.groupby("StockCode", sort=False):
    product_df = product_df.sort_values("Date").set_index("Date")

    # Reindex to consecutive calendar days.
    calendar = pd.date_range(
        start=product_df.index.min(),
        end=TRAIN_CUTOFF,
        freq="D",
    )

    daily_demand = product_df["demand"].reindex(calendar, fill_value=0)
    daily_demand.index.name = "Date"

    product_daily = pd.DataFrame({
        "StockCode": stock_code,
        "Date": calendar,
        "demand": daily_demand.to_numpy(),
    })

    product_daily["lead_time_demand"] = (
        product_daily["demand"]
        .rolling(LEAD_TIME_DAYS, min_periods=LEAD_TIME_DAYS)
        .sum()
    )

    product_daily["protection_period_demand"] = (
        product_daily["demand"]
        .rolling(protection_period, min_periods=protection_period)
        .sum()
    )

    daily_frames.append(product_daily)

complete_daily = pd.concat(daily_frames, ignore_index=True)

empirical = complete_daily.groupby("StockCode").agg(
    empirical_lead_time_p95=(
        "lead_time_demand",
        lambda s: s.quantile(0.95),
    ),
    empirical_protection_p95=(
        "protection_period_demand",
        lambda s: s.quantile(0.95),
    ),
    valid_lead_windows=("lead_time_demand", "count"),
    valid_protection_windows=("protection_period_demand", "count"),
).reset_index()

result = stats.merge(empirical, on="StockCode", how="left")

result["empirical_reorder_point"] = result["empirical_lead_time_p95"]

result["empirical_safety_stock"] = (
    result["empirical_reorder_point"]
    - result["average_daily_demand"] * LEAD_TIME_DAYS
).clip(lower=0)

result["empirical_order_up_to"] = result["empirical_protection_p95"]

result["empirical_available"] = (
    result["valid_lead_windows"].fillna(0).gt(0)
    & result["valid_protection_windows"].fillna(0).gt(0)
)


result.to_csv(OUTPUT_PATH, index=False)


# --------------------------------------------------
# Summary
# --------------------------------------------------
available = result[result["empirical_available"]].copy()

print("\n=== Safety-stock method comparison ===")
print(f"Products analyzed: {len(result):,}")
print(f"Products with empirical estimates: {len(available):,}")
print(f"Lead time: {LEAD_TIME_DAYS} days")
print(f"Review period: {REVIEW_PERIOD_DAYS} days")
print(f"Service-level quantile: {SERVICE_LEVEL:.0%}")

if not available.empty:
    print("\nAverage estimates across eligible products:")
    columns = [
        "normal_safety_stock",
        "empirical_safety_stock",
        "normal_reorder_point",
        "empirical_reorder_point",
        "normal_order_up_to",
        "empirical_order_up_to",
    ]
    print(available[columns].mean().round(2).to_string())

    print("\nProducts with the largest normal safety-stock estimates:")
    display_cols = [
        "StockCode",
        "average_daily_demand",
        "normal_safety_stock",
        "empirical_safety_stock",
        "normal_reorder_point",
        "empirical_reorder_point",
        "normal_order_up_to",
        "empirical_order_up_to",
    ]
    print(
        available.nlargest(10, "normal_safety_stock")[display_cols]
        .round(2)
        .to_string(index=False)
    )

print(f"\nSaved results to: {OUTPUT_PATH}")


from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "data" / "processed" / "comparable_net_daily_demand.csv"
)
ACTIVITY_PATH = ROOT / "reports" / "inventory_activity_analysis.csv"
REPORT_DIR = ROOT / "reports"

TRAIN_END = pd.Timestamp("2011-05-01")

LEAD_TIME_DAYS = 7
REVIEW_PERIOD_DAYS = 7
SERVICE_LEVEL = 0.95

# Demonstration assumptions only; these are NOT actual stock records.
DEFAULT_ON_HAND = 0
DEFAULT_ON_ORDER = 0

OUTPUT_PATH = REPORT_DIR / "inventory_recommendations.csv"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load and validate demand data
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Demand dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, dtype={"StockCode": str})
df.columns = df.columns.str.strip()

required_columns = {"StockCode", "Date", "demand"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Expected columns {required_columns}; "
        f"found {df.columns.tolist()}"
    )

df["Date"] = pd.to_datetime(df["Date"], errors="raise")
df["demand"] = pd.to_numeric(df["demand"], errors="raise")

if df[["StockCode", "Date", "demand"]].isna().any().any():
    raise ValueError("StockCode, Date, and demand must not contain missing values.")

if df.duplicated(["StockCode", "Date"]).any():
    raise ValueError("Duplicate product-date records found.")

if not np.isfinite(df["demand"]).all() or (df["demand"] < 0).any():
    raise ValueError("Demand must be finite and non-negative.")

# Use training-period observations only.
df = df[df["Date"] <= TRAIN_END].copy()

if df.empty:
    raise ValueError("No training-period demand data found.")


# --------------------------------------------------
# 3. Complete the daily calendar for each product
# --------------------------------------------------

products = sorted(df["StockCode"].unique())

calendar = pd.date_range(
    start=df["Date"].min(),
    end=TRAIN_END,
    freq="D",
)

full_index = pd.MultiIndex.from_product(
    [products, calendar],
    names=["StockCode", "Date"],
)

daily = (
    df.set_index(["StockCode", "Date"])["demand"]
    .reindex(full_index, fill_value=0)
    .rename("demand")
    .reset_index()
)


# --------------------------------------------------
# 4. Calculate historical demand statistics
# --------------------------------------------------

stats = (
    daily.groupby("StockCode")["demand"]
    .agg(
        average_daily_demand="mean",
        daily_demand_std="std",
        active_days=lambda x: int((x > 0).sum()),
        total_demand="sum",
        days_observed="count",
    )
    .reset_index()
)

stats["daily_demand_std"] = stats["daily_demand_std"].fillna(0)

stats["zero_demand_days"] = (
    stats["days_observed"] - stats["active_days"]
)


# --------------------------------------------------
# 5. Calculate demand activity using training data only
# --------------------------------------------------

recent_start = TRAIN_END - pd.Timedelta(days=27)
recent = daily[daily["Date"] >= recent_start].copy()

recent_stats = (
    recent.groupby("StockCode")
    .agg(
        demand_last_28_days=("demand", "sum"),
        positive_demand_days_last_28=(
            "demand",
            lambda x: int((x > 0).sum()),
        ),
    )
    .reset_index()
)

last_sale = (
    daily.loc[daily["demand"] > 0]
    .groupby("StockCode")["Date"]
    .max()
    .rename("last_recorded_sale_date")
    .reset_index()
)

stats = stats.merge(
    recent_stats,
    on="StockCode",
    how="left",
    validate="one_to_one",
)

stats = stats.merge(
    last_sale,
    on="StockCode",
    how="left",
    validate="one_to_one",
)

stats["demand_last_28_days"] = (
    stats["demand_last_28_days"].fillna(0)
)

stats["positive_demand_days_last_28"] = (
    stats["positive_demand_days_last_28"].fillna(0).astype(int)
)

stats["days_since_last_sale"] = (
    TRAIN_END - stats["last_recorded_sale_date"]
).dt.days

stats["activity_status"] = np.select(
    [
        stats["last_recorded_sale_date"].isna(),
        stats["days_since_last_sale"] > 90,
        stats["positive_demand_days_last_28"] == 0,
    ],
    [
        "no_recorded_sales",
        "stale_over_90_days",
        "no_sales_in_recent_28_days",
    ],
    default="recently_active",
)


# --------------------------------------------------
# 6. Merge existing activity concentration information
# --------------------------------------------------

if ACTIVITY_PATH.exists():
    activity = pd.read_csv(
        ACTIVITY_PATH,
        dtype={"StockCode": str},
    )
    activity.columns = activity.columns.str.strip()

    if "StockCode" not in activity.columns:
        raise ValueError(
            "Activity report exists but has no StockCode column."
        )

    concentration_columns = [
        column
        for column in activity.columns
        if column in {
            "top_five_share",
            "high_demand_concentration",
        }
    ]

    if activity["StockCode"].duplicated().any():
        raise ValueError(
            "Activity report contains duplicate StockCode records."
        )

    if concentration_columns:
        stats = stats.merge(
            activity[["StockCode", *concentration_columns]],
            on="StockCode",
            how="left",
            validate="one_to_one",
        )


# --------------------------------------------------
# 7. Calculate safety stock and reorder levels
# --------------------------------------------------

z_score = norm.ppf(SERVICE_LEVEL)

stats["lead_time_demand"] = (
    stats["average_daily_demand"] * LEAD_TIME_DAYS
)

stats["safety_stock"] = (
    z_score
    * stats["daily_demand_std"]
    * np.sqrt(LEAD_TIME_DAYS)
)

stats["reorder_point"] = np.ceil(
    stats["lead_time_demand"] + stats["safety_stock"]
).astype(int)

stats["safety_stock"] = (
    np.ceil(stats["safety_stock"]).astype(int)
)

stats["lead_time_demand"] = (
    np.ceil(stats["lead_time_demand"]).astype(int)
)

protection_period = LEAD_TIME_DAYS + REVIEW_PERIOD_DAYS

stats["order_up_to_level"] = np.ceil(
    stats["average_daily_demand"] * protection_period
    + z_score
    * stats["daily_demand_std"]
    * np.sqrt(protection_period)
).astype(int)


# --------------------------------------------------
# 8. Calculate inventory position and proposed quantity
# --------------------------------------------------

# Replace these assumptions with real inventory records when available.
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


# --------------------------------------------------
# 9. Assign recommendation categories
# --------------------------------------------------

stats["recommendation"] = np.select(
    [
        stats["activity_status"] == "no_recorded_sales",
        stats["activity_status"] == "stale_over_90_days",
        stats["activity_status"] == "no_sales_in_recent_28_days",
        stats["reorder_required"],
    ],
    [
        "Review inactive product",
        "Review inactive product",
        "Monitor",
        "Reorder now",
    ],
    default="No order needed",
)

# This is a suggested quantity, not an approved purchase order.
# Inactive and monitored products are excluded from automatic order quantities.
stats["actionable_order_qty"] = np.where(
    stats["recommendation"] == "Reorder now",
    stats["recommended_order_qty"],
    0,
).astype(int)


# --------------------------------------------------
# 10. Add assumptions and output metadata
# --------------------------------------------------

stats["lead_time_days"] = LEAD_TIME_DAYS
stats["review_period_days"] = REVIEW_PERIOD_DAYS
stats["target_service_level"] = SERVICE_LEVEL
stats["training_cutoff"] = TRAIN_END.strftime("%Y-%m-%d")
stats["demand_target"] = "Net-adjusted recorded sales"

stats["inventory_data_type"] = "Simulated assumptions"
stats["assumed_on_hand"] = DEFAULT_ON_HAND
stats["assumed_on_order"] = DEFAULT_ON_ORDER
stats["test_data_used_for_planning"] = False

stats["recommendation_priority"] = stats["recommendation"].map(
    {
        "Reorder now": 0,
        "Monitor": 1,
        "Review inactive product": 2,
        "No order needed": 3,
    }
)

# Sort by action priority first, then by largest proposed order quantity.
stats = stats.sort_values(
    ["recommendation_priority", "recommended_order_qty"],
    ascending=[True, False],
).reset_index(drop=True)

stats.to_csv(OUTPUT_PATH, index=False)


# --------------------------------------------------
# 11. Print summary
# --------------------------------------------------

print("StockPilot Inventory Planning Engine")
print("=" * 60)
print("Demand target: Net-adjusted recorded sales")
print(f"Training cutoff: {TRAIN_END.date()}")
print(f"Products analyzed: {len(stats):,}")
print(f"Supplier lead time: {LEAD_TIME_DAYS} days")
print(f"Review period: {REVIEW_PERIOD_DAYS} days")
print(f"Target service level: {SERVICE_LEVEL:.0%}")
print("Inventory inputs: simulated, not actual stock records")

print("\nRecommendation counts:")
print(
    stats["recommendation"]
    .value_counts()
    .reindex(
        [
            "Reorder now",
            "Monitor",
            "Review inactive product",
            "No order needed",
        ],
        fill_value=0,
    )
    .to_string()
)

print(
    "\nTotal proposed units across all products: "
    f"{stats['recommended_order_qty'].sum():,.0f}"
)

print(
    "Proposed units for immediate replenishment: "
    f"{stats['actionable_order_qty'].sum():,.0f}"
)

print("\nTop 15 recommendations:")
print(
    stats[
        [
            "StockCode",
            "activity_status",
            "average_daily_demand",
            "safety_stock",
            "reorder_point",
            "order_up_to_level",
            "recommended_order_qty",
            "actionable_order_qty",
            "recommendation",
        ]
    ]
    .head(15)
    .to_string(index=False)
)

print(f"\nSaved report: {OUTPUT_PATH}")

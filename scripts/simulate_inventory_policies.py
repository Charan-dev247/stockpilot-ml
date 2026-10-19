
from pathlib import Path
import heapq

import numpy as np
import pandas as pd
from scipy.stats import norm


# ============================================================
# Configuration
# ============================================================
ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "comparable_net_daily_demand.csv"
ACTIVITY_PATH = ROOT / "reports" / "inventory_activity_analysis.csv"

REPORTS_DIR = ROOT / "reports"
DETAIL_PATH = REPORTS_DIR / "inventory_simulation_product_results.csv"
SUMMARY_PATH = REPORTS_DIR / "inventory_simulation_summary.csv"
ACTIVITY_SUMMARY_PATH = (
    REPORTS_DIR / "inventory_simulation_activity_summary.csv"
)

TRAIN_CUTOFF = pd.Timestamp("2011-05-01")
VALIDATION_START = pd.Timestamp("2011-08-21")
VALIDATION_END = pd.Timestamp("2011-12-09")

LEAD_TIME_DAYS = 7
REVIEW_PERIOD_DAYS = 7
SERVICE_LEVEL = 0.95

Z_SCORE = norm.ppf(SERVICE_LEVEL)
PROTECTION_DAYS = LEAD_TIME_DAYS + REVIEW_PERIOD_DAYS


# ============================================================
# Load and validate data
# ============================================================
df = pd.read_csv(
    DATA_PATH,
    dtype={"StockCode": str},
    low_memory=False,
)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["demand"] = pd.to_numeric(df["demand"], errors="coerce")

df = df.dropna(subset=["StockCode", "Date", "demand"]).copy()

if (df["demand"] < 0).any():
    raise ValueError(
        "Negative values found in demand. Verify the selected target."
    )

if df.duplicated(["StockCode", "Date"]).any():
    raise ValueError("Duplicate product-date rows found.")

activity = pd.read_csv(
    ACTIVITY_PATH,
    dtype={"StockCode": str},
    low_memory=False,
)

required_cols = {"StockCode", "activity_status"}
if not required_cols.issubset(activity.columns):
    raise ValueError(
        f"Activity report must contain columns: {required_cols}"
    )

if activity["StockCode"].duplicated().any():
    raise ValueError("Duplicate StockCode values in activity report.")

activity_lookup = (
    activity.set_index("StockCode")["activity_status"].to_dict()
)


# ============================================================
# Estimate policy parameters using training data only
# ============================================================
train = df[df["Date"] <= TRAIN_CUTOFF].copy()

stats_rows = []

for sku, product in train.groupby("StockCode", sort=False):
    product = product.sort_values("Date").set_index("Date")

    # Consecutive calendar days from first recorded date to cutoff.
    calendar = pd.date_range(
        start=product.index.min(),
        end=TRAIN_CUTOFF,
        freq="D",
    )

    daily = (
        product["demand"]
        .reindex(calendar, fill_value=0)
        .astype(float)
    )

    mean_demand = float(daily.mean())
    daily_std = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
    daily_std = max(daily_std, 0.0)

    baseline_level = mean_demand * PROTECTION_DAYS

    safety_level = (
        mean_demand * PROTECTION_DAYS
        + Z_SCORE * daily_std * np.sqrt(PROTECTION_DAYS)
    )

    stats_rows.append({
        "StockCode": sku,
        "mean_daily_demand": mean_demand,
        "daily_demand_std": daily_std,
        "baseline_order_up_to": int(np.ceil(baseline_level)),
        "safety_order_up_to": int(np.ceil(safety_level)),
        "activity_status": activity_lookup.get(sku, "unknown"),
    })

stats = pd.DataFrame(stats_rows)

if stats.empty:
    raise ValueError("No products found in the training period.")

stats["activity_eligible"] = (
    stats["activity_status"] == "recently_active"
)

stats = stats.sort_values("StockCode").reset_index(drop=True)


# ============================================================
# Build validation demand matrix
# ============================================================
validation_dates = pd.date_range(
    start=VALIDATION_START,
    end=VALIDATION_END,
    freq="D",
)

validation = df[
    df["Date"].between(VALIDATION_START, VALIDATION_END)
    & df["StockCode"].isin(stats["StockCode"])
].copy()

demand_matrix = (
    validation.pivot(
        index="Date",
        columns="StockCode",
        values="demand",
    )
    .reindex(
        index=validation_dates,
        columns=stats["StockCode"],
        fill_value=0,
    )
    .fillna(0)
)

# Missing product-date observations are treated as zero recorded demand.
# They are not proof of zero unconstrained customer demand.


# ============================================================
# Inventory simulation
# ============================================================
def simulate_policy(policy_name, target_column, activity_aware=False):
    results = []

    target_lookup = stats.set_index("StockCode")[target_column]
    baseline_lookup = stats.set_index("StockCode")[
        "baseline_order_up_to"
    ]
    eligibility_lookup = stats.set_index("StockCode")[
        "activity_eligible"
    ]

    for row in stats.itertuples(index=False):
        sku = row.StockCode
        target = int(target_lookup.loc[sku])
        can_replenish = (
            bool(eligibility_lookup.loc[sku])
            if activity_aware
            else True
        )

        # Equal initial inventory for all policies:
        # baseline order-up-to stock, fixed before simulation starts.
        on_hand = int(baseline_lookup.loc[sku])

        # Heap entries: (arrival_day_index, quantity).
        outstanding_orders = []

        inventory_sum = 0.0
        total_demand = 0.0
        fulfilled_units = 0.0
        lost_units = 0.0
        stockout_days = 0
        order_count = 0
        units_ordered = 0

        for day_idx, date in enumerate(validation_dates):

            # Receive orders scheduled to arrive at the start of today.
            while (
                outstanding_orders
                and outstanding_orders[0][0] <= day_idx
            ):
                _, quantity = heapq.heappop(outstanding_orders)
                on_hand += quantity

            demand = float(demand_matrix.at[date, sku])
            total_demand += demand

            fulfilled_today = min(on_hand, demand)
            on_hand -= fulfilled_today

            lost_today = demand - fulfilled_today
            fulfilled_units += fulfilled_today
            lost_units += lost_today

            if lost_today > 0:
                stockout_days += 1

            # Review inventory after demand, every review-period days.
            if (
                can_replenish
                and (day_idx + 1) % REVIEW_PERIOD_DAYS == 0
            ):
                on_order = sum(
                    quantity
                    for _, quantity in outstanding_orders
                )
                inventory_position = on_hand + on_order

                if inventory_position < target:
                    order_quantity = int(
                        np.ceil(target - inventory_position)
                    )

                    if order_quantity > 0:
                        # Order placed after today's demand arrives at
                        # the start of the day 7 calendar days later.
                        arrival_idx = day_idx + LEAD_TIME_DAYS

                        heapq.heappush(
                            outstanding_orders,
                            (arrival_idx, order_quantity),
                        )

                        order_count += 1
                        units_ordered += order_quantity

            inventory_sum += on_hand

        fill_rate = (
            fulfilled_units / total_demand
            if total_demand > 0
            else np.nan
        )

        results.append({
            "StockCode": sku,
            "policy": policy_name,
            "activity_status": row.activity_status,
            "activity_eligible": can_replenish,
            "validation_demand": total_demand,
            "fulfilled_units": fulfilled_units,
            "lost_units": lost_units,
            "unit_fill_rate": fill_rate,
            "stockout_days": stockout_days,
            "inventory_unit_days": inventory_sum,
            "average_on_hand": inventory_sum / len(validation_dates),
            "orders_placed": order_count,
            "units_ordered": units_ordered,
            "order_up_to_level": target,
            "initial_on_hand": int(baseline_lookup.loc[sku]),
        })

    return pd.DataFrame(results)


# ============================================================
# Run all three policies
# ============================================================
baseline_results = simulate_policy(
    policy_name="baseline_average_demand",
    target_column="baseline_order_up_to",
)

safety_results = simulate_policy(
    policy_name="safety_stock",
    target_column="safety_order_up_to",
)

activity_results = simulate_policy(
    policy_name="activity_aware_safety_stock",
    target_column="safety_order_up_to",
    activity_aware=True,
)

details = pd.concat(
    [baseline_results, safety_results, activity_results],
    ignore_index=True,
)

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
details.to_csv(DETAIL_PATH, index=False)


# ============================================================
# Summary metrics
# ============================================================
def summarize(group):
    total_demand = group["validation_demand"].sum()
    fulfilled = group["fulfilled_units"].sum()

    return {
        "products": group["StockCode"].nunique(),
        "total_demand_units": total_demand,
        "fulfilled_units": fulfilled,
        "lost_units": group["lost_units"].sum(),
        "unit_fill_rate_pct": (
            100 * fulfilled / total_demand
            if total_demand > 0
            else np.nan
        ),
        "products_with_stockout": int(
            (group["lost_units"] > 0).sum()
        ),
        "total_stockout_days": group["stockout_days"].sum(),
        "average_on_hand_per_product": group["average_on_hand"].mean(),
        "total_orders": group["orders_placed"].sum(),
        "total_units_ordered": group["units_ordered"].sum(),
    }


overall_rows = []

for policy, group in details.groupby("policy", sort=False):
    overall_rows.append({
        "policy": policy,
        **summarize(group),
    })

summary = pd.DataFrame(overall_rows)
summary.to_csv(SUMMARY_PATH, index=False)


# ============================================================
# Breakdown by activity category
# ============================================================
activity_rows = []

for (policy, status), group in details.groupby(
    ["policy", "activity_status"],
    sort=False,
    dropna=False,
):
    activity_rows.append({
        "policy": policy,
        "activity_status": status,
        **summarize(group),
    })

activity_summary = pd.DataFrame(activity_rows)
activity_summary.to_csv(ACTIVITY_SUMMARY_PATH, index=False)


# ============================================================
# Print results
# ============================================================
print("\n=== Inventory simulation ===")
print(f"Training cutoff: {TRAIN_CUTOFF.date()}")
print(
    f"Validation: {VALIDATION_START.date()} "
    f"to {VALIDATION_END.date()}"
)
print(f"Lead time: {LEAD_TIME_DAYS} days")
print(f"Review period: {REVIEW_PERIOD_DAYS} days")
print(f"Service-level assumption: {SERVICE_LEVEL:.0%}")
print(f"Products in training cohort: {len(stats):,}")
print(
    "Activity-eligible products: "
    f"{int(stats['activity_eligible'].sum()):,}"
)
print("Initial inventory: baseline order-up-to level for every policy")

print("\n=== Overall policy comparison ===")
print(summary.round(2).to_string(index=False))

print("\n=== Policy comparison by activity category ===")
print(activity_summary.round(2).to_string(index=False))

print(f"\nSaved detailed results: {DETAIL_PATH}")
print(f"Saved overall summary: {SUMMARY_PATH}")
print(f"Saved activity summary: {ACTIVITY_SUMMARY_PATH}")

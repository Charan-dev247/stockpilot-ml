
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

TRAIN_CUTOFF = pd.Timestamp("2011-05-01")
VALIDATION_START = pd.Timestamp("2011-08-21")
VALIDATION_END = pd.Timestamp("2011-12-09")

LEAD_TIME_DAYS = 7
REVIEW_PERIOD_DAYS = 7
SERVICE_LEVEL = 0.95
Z_SCORE = norm.ppf(SERVICE_LEVEL)

PROTECTION_DAYS = LEAD_TIME_DAYS + REVIEW_PERIOD_DAYS


# ============================================================
# Load data
# ============================================================
df = pd.read_csv(
    DATA_PATH,
    dtype={"StockCode": str},
    low_memory=False,
)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["demand"] = pd.to_numeric(df["demand"], errors="coerce")

df = df.dropna(subset=["StockCode", "Date", "demand"])

if (df["demand"] < 0).any():
    raise ValueError(
        "The selected demand column must be non-negative. "
        "Check the target definition before simulating."
    )

if df.duplicated(["StockCode", "Date"]).any():
    raise ValueError("Duplicate product-date rows found.")

activity = pd.read_csv(
    ACTIVITY_PATH,
    dtype={"StockCode": str},
    low_memory=False,
)

required_activity_cols = {"StockCode", "activity_status"}
if not required_activity_cols.issubset(activity.columns):
    raise ValueError(
        f"Activity report must contain {required_activity_cols}"
    )

activity_lookup = activity.set_index("StockCode")["activity_status"].to_dict()


# ============================================================
# Estimate policy parameters from training data only
# ============================================================
train = df[df["Date"] <= TRAIN_CUTOFF].copy()

stats_rows = []

for sku, product in train.groupby("StockCode", sort=False):
    product = product.sort_values("Date").set_index("Date")

    # Use consecutive calendar days from first recorded date to cutoff.
    calendar = pd.date_range(
        product.index.min(),
        TRAIN_CUTOFF,
        freq="D",
    )
    daily = product["demand"].reindex(calendar, fill_value=0).astype(float)

    mean_demand = float(daily.mean())
    std_demand = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
    std_demand = max(std_demand, 0.0)

    baseline_level = mean_demand * PROTECTION_DAYS

    safety_level = (
        mean_demand * PROTECTION_DAYS
        + Z_SCORE * std_demand * np.sqrt(PROTECTION_DAYS)
    )

    status = activity_lookup.get(sku, "unknown")

    stats_rows.append({
        "StockCode": sku,
        "mean_daily_demand": mean_demand,
        "daily_demand_std": std_demand,
        "baseline_order_up_to": int(np.ceil(baseline_level)),
        "safety_order_up_to": int(np.ceil(safety_level)),
        "activity_status": status,
    })

stats = pd.DataFrame(stats_rows)

# The activity-aware policy suppresses routine replenishment for
# products not classified as recently active. This is an explicit
# policy assumption, not proof that the products are discontinued.
stats["activity_eligible"] = (
    stats["activity_status"] == "recently_active"
)


# ============================================================
# Build validation demand matrix
# ============================================================
validation_dates = pd.date_range(
    VALIDATION_START,
    VALIDATION_END,
    freq="D",
)

validation = df[
    df["Date"].between(VALIDATION_START, VALIDATION_END)
].copy()

validation = validation[
    validation["StockCode"].isin(stats["StockCode"])
]

demand_matrix = (
    validation.pivot(index="Date", columns="StockCode", values="demand")
    .reindex(index=validation_dates, columns=stats["StockCode"], fill_value=0)
    .fillna(0)
)

# Products missing a recorded date are treated as having zero recorded
# demand on that date. This is a proxy, not verified unconstrained demand.


# ============================================================
# Inventory simulation
# ============================================================
def simulate_policy(policy_name, order_up_to, eligible):
    results = []

    for row in stats.itertuples(index=False):
        sku = row.StockCode
        target = int(order_up_to[sku])
        can_replenish = bool(eligible[sku])

        # Start at the target for eligible products to avoid giving one
        # policy an arbitrary initial-stock advantage.
        on_hand = target if can_replenish else 0

        # Heap entries are (arrival_day_index, quantity).
        arrivals = []
        inventory_sum = 0.0
        total_demand = 0.0
        fulfilled = 0.0
        lost_units = 0.0
        stockout_days = 0
        order_count = 0
        units_ordered = 0

        for day_idx, date in enumerate(validation_dates):
            # Receive orders arriving today.
            while arrivals and arrivals[0][0] <= day_idx:
                _, quantity = heapq.heappop(arrivals)
                on_hand += quantity

            demand = float(demand_matrix.at[date, sku])
            total_demand += demand

            fulfilled_today = min(on_hand, demand)
            on_hand -= fulfilled_today

            lost_today = demand - fulfilled_today
            fulfilled += fulfilled_today
            lost_units += lost_today

            if lost_today > 0:
                stockout_days += 1

            # Review after today's demand. Inventory position includes
            # stock on hand and outstanding orders.
            if (
                can_replenish
                and (day_idx + 1) % REVIEW_PERIOD_DAYS == 0
            ):
                on_order = sum(qty for _, qty in arrivals)
                inventory_position = on_hand + on_order

                if inventory_position < target:
                    quantity = int(np.ceil(target - inventory_position))

                    if quantity > 0:
                        # Lead time of 7 days means an order placed on
                        # day t arrives at the start of day t + 7.
                        arrival_idx = day_idx + LEAD_TIME_DAYS + 1
                        heapq.heappush(arrivals, (arrival_idx, quantity))
                        order_count += 1
                        units_ordered += quantity

            inventory_sum += on_hand

        service_level = (
            fulfilled / total_demand if total_demand > 0 else np.nan
        )

        results.append({
            "StockCode": sku,
            "policy": policy_name,
            "activity_status": row.activity_status,
            "activity_eligible": can_replenish,
            "validation_demand": total_demand,
            "fulfilled_units": fulfilled,
            "lost_units": lost_units,
            "unit_fill_rate": service_level,
            "stockout_days": stockout_days,
            "average_on_hand": inventory_sum / len(validation_dates),
            "orders_placed": order_count,
            "units_ordered": units_ordered,
            "order_up_to_level": target,
        })

    return pd.DataFrame(results)


baseline_targets = stats.set_index("StockCode")["baseline_order_up_to"]
safety_targets = stats.set_index("StockCode")["safety_order_up_to"]

all_eligible = pd.Series(True, index=stats["StockCode"])
active_only = stats.set_index("StockCode")["activity_eligible"]

baseline_results = simulate_policy(
    "baseline_average_demand",
    baseline_targets,
    all_eligible,
)

safety_results = simulate_policy(
    "safety_stock",
    safety_targets,
    all_eligible,
)

activity_results = simulate_policy(
    "activity_aware_safety_stock",
    safety_targets,
    active_only,
)

details = pd.concat(
    [baseline_results, safety_results, activity_results],
    ignore_index=True,
)

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
details.to_csv(DETAIL_PATH, index=False)


# ============================================================
# Summaries
# ============================================================
def summarize(group):
    total_demand = group["validation_demand"].sum()
    total_fulfilled = group["fulfilled_units"].sum()

    return pd.Series({
        "products": group["StockCode"].nunique(),
        "total_demand_units": total_demand,
        "fulfilled_units": total_fulfilled,
        "lost_units": group["lost_units"].sum(),
        "unit_fill_rate_pct": (
            100 * total_fulfilled / total_demand
            if total_demand > 0 else np.nan
        ),
        "products_with_stockout": int((group["lost_units"] > 0).sum()),
        "total_stockout_days": group["stockout_days"].sum(),
        "average_on_hand_per_product": group["average_on_hand"].mean(),
        "total_orders": group["orders_placed"].sum(),
        "total_units_ordered": group["units_ordered"].sum(),
    })


summary = (
    details.groupby("policy", sort=False)
    .apply(summarize, include_groups=False)
    .reset_index()
)

summary.to_csv(SUMMARY_PATH, index=False)

print("\n=== Inventory simulation ===")
print(f"Training cutoff: {TRAIN_CUTOFF.date()}")
print(f"Validation: {VALIDATION_START.date()} to {VALIDATION_END.date()}")
print(f"Lead time: {LEAD_TIME_DAYS} days")
print(f"Review period: {REVIEW_PERIOD_DAYS} days")
print(f"Service-level assumption: {SERVICE_LEVEL:.0%}")
print(f"Products in training cohort: {len(stats):,}")
print(f"Activity-eligible products: {int(stats['activity_eligible'].sum()):,}")

print("\n=== Policy comparison ===")
print(summary.round(2).to_string(index=False))

print(f"\nSaved detailed results: {DETAIL_PATH}")
print(f"Saved summary: {SUMMARY_PATH}")

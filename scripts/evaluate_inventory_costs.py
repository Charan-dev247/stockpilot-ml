
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

INPUT_FILE = REPORTS / "inventory_simulation_product_results.csv"
OUTPUT_FILE = REPORTS / "inventory_cost_comparison.csv"

# Hypothetical costs, not actual retailer costs.
ORDER_COST_PER_ORDER = 5.00
LOST_UNIT_PENALTY = 2.00

# Test multiple holding-cost assumptions.
HOLDING_COSTS_PER_UNIT_DAY = [0.001, 0.01, 0.05]

df = pd.read_csv(INPUT_FILE)

required = {
    "StockCode",
    "policy",
    "validation_demand",
    "fulfilled_units",
    "lost_units",
    "stockout_days",
    "inventory_unit_days",
    "orders_placed",
    "units_ordered",
}

missing = required - set(df.columns)
if missing:
    raise ValueError(
        f"Missing columns: {sorted(missing)}. "
        "Run the updated inventory simulator first."
    )

if df.duplicated(["StockCode", "policy"]).any():
    raise ValueError("Duplicate product-policy rows found.")

summary = (
    df.groupby("policy", as_index=False)
    .agg(
        products=("StockCode", "nunique"),
        total_demand_units=("validation_demand", "sum"),
        fulfilled_units=("fulfilled_units", "sum"),
        lost_units=("lost_units", "sum"),
        stockout_days=("stockout_days", "sum"),
        total_orders=("orders_placed", "sum"),
        total_units_ordered=("units_ordered", "sum"),
        inventory_unit_days=("inventory_unit_days", "sum"),
    )
)

summary["unit_fill_rate_pct"] = (
    100 * summary["fulfilled_units"]
    / summary["total_demand_units"].replace(0, pd.NA)
)

summary["ordering_cost"] = (
    summary["total_orders"] * ORDER_COST_PER_ORDER
)

summary["lost_sales_penalty"] = (
    summary["lost_units"] * LOST_UNIT_PENALTY
)

summary["average_on_hand_per_product"] = (
    summary["inventory_unit_days"]
    / (summary["products"] * 111)
)

# Validation dates May 2-Aug 20, 2011: 111 calendar days.
# This is the mean end-of-day stock over products and days.
# Change the denominator if the simulation dates change.

scenario_results = []

for holding_cost in HOLDING_COSTS_PER_UNIT_DAY:
    result = summary.copy()

    result["holding_cost_per_unit_day"] = holding_cost
    result["holding_cost"] = (
        result["inventory_unit_days"] * holding_cost
    )

    result["total_simulated_cost"] = (
        result["ordering_cost"]
        + result["lost_sales_penalty"]
        + result["holding_cost"]
    )

    scenario_results.append(result)

comparison = pd.concat(scenario_results, ignore_index=True)

comparison = comparison.sort_values(
    ["holding_cost_per_unit_day", "total_simulated_cost"]
)

columns = [
    "holding_cost_per_unit_day",
    "policy",
    "unit_fill_rate_pct",
    "lost_units",
    "total_orders",
    "total_units_ordered",
    "average_on_hand_per_product",
    "inventory_unit_days",
    "ordering_cost",
    "lost_sales_penalty",
    "holding_cost",
    "total_simulated_cost",
]

comparison.to_csv(OUTPUT_FILE, index=False)

print("=== Inventory cost sensitivity analysis ===")
print(f"Order cost/order: {ORDER_COST_PER_ORDER:.2f}")
print(f"Lost-unit penalty: {LOST_UNIT_PENALTY:.2f}")
print("Holding costs tested:", HOLDING_COSTS_PER_UNIT_DAY)
print("All costs are hypothetical scenario values.")
print()

for holding_cost, group in comparison.groupby(
    "holding_cost_per_unit_day", sort=True
):
    print(f"\n--- Holding cost per unit-day: {holding_cost:.3f} ---")
    print(
        group[
            [
                "policy",
                "unit_fill_rate_pct",
                "lost_units",
                "ordering_cost",
                "lost_sales_penalty",
                "holding_cost",
                "total_simulated_cost",
            ]
        ].round(2).to_string(index=False)
    )

print(f"\nSaved comparison: {OUTPUT_FILE}")

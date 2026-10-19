
from pathlib import Path

import pandas as pd

REPORTS = Path("reports")

FILES = {
    "validation_simulation": REPORTS / "inventory_simulation_validation_summary.csv",
    "test_simulation": REPORTS / "inventory_simulation_test_summary.csv",
    "validation_cost": REPORTS / "inventory_cost_validation_comparison.csv",
    "test_cost": REPORTS / "inventory_cost_test_comparison.csv",
}

for name, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {name} report: {path}\n"
            "Check that the validation and test reports were backed up."
        )

validation = pd.read_csv(FILES["validation_simulation"])
test = pd.read_csv(FILES["test_simulation"])

required = {
    "policy",
    "unit_fill_rate_pct",
    "lost_units",
    "total_stockout_days",
    "average_on_hand_per_product",
}

for name, df in [("validation", validation), ("test", test)]:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} report is missing columns: {sorted(missing)}")

metrics = [
    "unit_fill_rate_pct",
    "lost_units",
    "total_stockout_days",
    "average_on_hand_per_product",
]

comparison = validation[["policy", *metrics]].merge(
    test[["policy", *metrics]],
    on="policy",
    how="inner",
    suffixes=("_validation", "_test"),
    validate="one_to_one",
)

if len(comparison) != len(validation) or len(comparison) != len(test):
    raise ValueError("Policy rows do not match between validation and test reports.")

comparison["fill_rate_change_pp"] = (
    comparison["unit_fill_rate_pct_test"]
    - comparison["unit_fill_rate_pct_validation"]
)

comparison["lost_units_change_pct"] = (
    (comparison["lost_units_test"] - comparison["lost_units_validation"])
    / comparison["lost_units_validation"].replace(0, pd.NA)
    * 100
)

comparison["stockout_days_change_pct"] = (
    (
        comparison["total_stockout_days_test"]
        - comparison["total_stockout_days_validation"]
    )
    / comparison["total_stockout_days_validation"].replace(0, pd.NA)
    * 100
)

comparison["fill_rate_target_pct"] = 85.0
comparison["test_meets_fill_rate_target"] = (
    comparison["unit_fill_rate_pct_test"] >= 85.0
)

output_path = REPORTS / "validation_test_comparison.csv"
comparison.to_csv(output_path, index=False)

print("\n=== Validation vs Test Comparison ===")
print(
    comparison[
        [
            "policy",
            "unit_fill_rate_pct_validation",
            "unit_fill_rate_pct_test",
            "fill_rate_change_pp",
            "lost_units_validation",
            "lost_units_test",
            "lost_units_change_pct",
            "total_stockout_days_validation",
            "total_stockout_days_test",
            "stockout_days_change_pct",
            "average_on_hand_per_product_validation",
            "average_on_hand_per_product_test",
            "test_meets_fill_rate_target",
        ]
    ].round(2).to_string(index=False)
)

# Compare simulated costs for the same policy and holding-cost assumption.
validation_cost = pd.read_csv(FILES["validation_cost"])
test_cost = pd.read_csv(FILES["test_cost"])

cost_keys = {"policy", "holding_cost_per_unit_day", "total_simulated_cost"}
for name, df in [
    ("validation cost", validation_cost),
    ("test cost", test_cost),
]:
    missing = cost_keys - set(df.columns)
    if missing:
        raise ValueError(
            f"{name} report is missing columns: {sorted(missing)}. "
            "Check the actual column names in that CSV."
        )

cost_comparison = validation_cost.merge(
    test_cost,
    on=["policy", "holding_cost_per_unit_day"],
    how="inner",
    suffixes=("_validation", "_test"),
    validate="one_to_one",
)

cost_comparison["cost_change_pct"] = (
    (
        cost_comparison["total_simulated_cost_test"]
        - cost_comparison["total_simulated_cost_validation"]
    )
    / cost_comparison["total_simulated_cost_validation"].replace(0, pd.NA)
    * 100
)

cost_output = REPORTS / "validation_test_cost_comparison.csv"
cost_comparison.to_csv(cost_output, index=False)

print("\n=== Validation vs Test Cost Comparison ===")
print(
    cost_comparison[
        [
            "policy",
            "holding_cost_per_unit_day",
            "total_simulated_cost_validation",
            "total_simulated_cost_test",
            "cost_change_pct",
        ]
    ].round(2).to_string(index=False)
)

print(f"\nSaved simulation comparison: {output_path}")
print(f"Saved cost comparison: {cost_output}")

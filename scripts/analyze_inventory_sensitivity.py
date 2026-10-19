
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

INPUT_FILE = REPORTS / "inventory_cost_comparison.csv"
OUTPUT_FILE = REPORTS / "inventory_service_cost_sensitivity.csv"

MINIMUM_FILL_RATES = [80, 85, 90, 95]

df = pd.read_csv(INPUT_FILE)

required = {
    "policy",
    "holding_cost_per_unit_day",
    "unit_fill_rate_pct",
    "total_simulated_cost",
}

missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

# These are validation results and hypothetical costs.
# Do not load or evaluate the held-out test period here.

rows = []

for holding_cost, scenario in df.groupby(
    "holding_cost_per_unit_day", sort=True
):
    for target in MINIMUM_FILL_RATES:
        eligible = scenario[
            scenario["unit_fill_rate_pct"] >= target
        ].sort_values("total_simulated_cost")

        if eligible.empty:
            rows.append({
                "holding_cost_per_unit_day": holding_cost,
                "minimum_fill_rate_pct": target,
                "eligible_policy_count": 0,
                "cheapest_eligible_policy": "No policy qualifies",
                "achieved_fill_rate_pct": None,
                "total_simulated_cost": None,
            })
        else:
            best = eligible.iloc[0]
            rows.append({
                "holding_cost_per_unit_day": holding_cost,
                "minimum_fill_rate_pct": target,
                "eligible_policy_count": len(eligible),
                "cheapest_eligible_policy": best["policy"],
                "achieved_fill_rate_pct": best["unit_fill_rate_pct"],
                "total_simulated_cost": best["total_simulated_cost"],
            })

result = pd.DataFrame(rows)
result.to_csv(OUTPUT_FILE, index=False)

print("=== Validation service-cost sensitivity ===")
print("Minimum fill-rate targets:", MINIMUM_FILL_RATES)
print("All costs are hypothetical.")
print()

for holding_cost, scenario in result.groupby(
    "holding_cost_per_unit_day", sort=True
):
    print(f"--- Holding cost per unit-day: {holding_cost:.3f} ---")
    print(
        scenario[
            [
                "minimum_fill_rate_pct",
                "eligible_policy_count",
                "cheapest_eligible_policy",
                "achieved_fill_rate_pct",
                "total_simulated_cost",
            ]
        ].round(2).to_string(index=False)
    )
    print()

print(f"Saved: {OUTPUT_FILE}")

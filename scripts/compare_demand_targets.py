
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

original_file = DATA / "regular_daily_demand.csv"
net_file = DATA / "comparable_net_daily_demand.csv"

original = pd.read_csv(
    original_file,
    usecols=["StockCode", "Date", "demand"],
    dtype={"StockCode": "string"},
    parse_dates=["Date"],
).rename(columns={"demand": "original_demand"})

net = pd.read_csv(
    net_file,
    usecols=["StockCode", "Date", "demand", "net_quantity"],
    dtype={"StockCode": "string"},
    parse_dates=["Date"],
).rename(columns={"demand": "net_demand"})

comparison = original.merge(
    net,
    on=["StockCode", "Date"],
    how="inner",
    validate="one_to_one",
)

if len(comparison) != len(original) or len(comparison) != len(net):
    raise ValueError("Datasets do not have identical product-date keys.")

comparison["target_difference"] = (
    comparison["original_demand"] - comparison["net_demand"]
)

print("=" * 68)
print("STOCKPILOT ML - DEMAND TARGET COMPARISON")
print("=" * 68)

print(f"Matched rows: {len(comparison):,}")
print(f"Products: {comparison['StockCode'].nunique():,}")
print(f"Dates: {comparison['Date'].nunique():,}")

print("\n--- Target statistics ---")
print(
    comparison[["original_demand", "net_demand"]]
    .describe(percentiles=[0.5, 0.9, 0.95, 0.99])
    .round(2)
)

changed = comparison["target_difference"] != 0
print(f"\nRows where targets differ: {changed.sum():,}")
print(f"Rows where original demand is greater: "
      f"{(comparison['target_difference'] > 0).sum():,}")
print(f"Rows where net demand is greater: "
      f"{(comparison['target_difference'] < 0).sum():,}")

print("\n--- Largest target differences ---")
print(
    comparison.loc[changed]
    .assign(
        absolute_difference=lambda x: x["target_difference"].abs()
    )
    .nlargest(15, "absolute_difference")[
        [
            "Date", "StockCode", "original_demand",
            "net_demand", "net_quantity", "target_difference"
        ]
    ]
    .to_string(index=False)
)

REPORTS.mkdir(parents=True, exist_ok=True)
output = REPORTS / "demand_target_comparison.csv"
comparison.to_csv(output, index=False)

print(f"\nSaved: {output}")

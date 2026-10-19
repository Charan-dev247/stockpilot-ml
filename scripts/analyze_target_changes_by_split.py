
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

original = pd.read_csv(
    DATA / "regular_daily_demand.csv",
    usecols=["StockCode", "Date", "demand"],
    dtype={"StockCode": "string"},
    parse_dates=["Date"],
).rename(columns={"demand": "original_demand"})

net = pd.read_csv(
    DATA / "comparable_net_daily_demand.csv",
    usecols=["StockCode", "Date", "demand"],
    dtype={"StockCode": "string"},
    parse_dates=["Date"],
).rename(columns={"demand": "net_demand"})

df = original.merge(
    net,
    on=["StockCode", "Date"],
    how="inner",
    validate="one_to_one",
)

if len(df) != len(original) or len(df) != len(net):
    raise ValueError("Product-date keys do not match.")

splits = {
    "TRAIN": ("2009-12-01", "2011-05-01"),
    "VALIDATION": ("2011-05-02", "2011-08-20"),
    "TEST": ("2011-08-21", "2011-12-09"),
}

print("=" * 68)
print("TARGET CHANGES BY CHRONOLOGICAL SPLIT")
print("=" * 68)

for name, (start, end) in splits.items():
    part = df[df["Date"].between(start, end)]
    changed = part["original_demand"] != part["net_demand"]

    print(f"\n--- {name}: {start} to {end} ---")
    print(f"Rows: {len(part):,}")
    print(f"Changed target rows: {changed.sum():,}")
    print(
        f"Original total demand: "
        f"{part['original_demand'].sum():,.0f}"
    )
    print(
        f"Net-adjusted total demand: "
        f"{part['net_demand'].sum():,.0f}"
    )
    print(
        f"Original positive-demand days: "
        f"{(part['original_demand'] > 0).sum():,}"
    )
    print(
        f"Net-adjusted positive-demand days: "
        f"{(part['net_demand'] > 0).sum():,}"
    )

print("\nAnalysis complete. No datasets were modified.")

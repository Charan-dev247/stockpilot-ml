
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

original_file = PROCESSED / "regular_daily_demand.csv"
alternative_file = PROCESSED / "regular_net_daily_demand.csv"
audit_file = REPORTS / "cancellation_audit_daily.csv"

original = pd.read_csv(
    original_file,
    usecols=["StockCode"],
    dtype={"StockCode": "string"},
)
alternative = pd.read_csv(
    alternative_file,
    usecols=["StockCode"],
    dtype={"StockCode": "string"},
)

original_products = set(original["StockCode"].dropna().unique())
alternative_products = set(alternative["StockCode"].dropna().unique())

common_products = original_products & alternative_products
only_original = original_products - alternative_products
only_alternative = alternative_products - original_products

print("=" * 68)
print("STOCKPILOT ML - PRODUCT POPULATION COMPARISON")
print("=" * 68)

print(f"Original products: {len(original_products):,}")
print(f"Alternative products: {len(alternative_products):,}")
print(f"Products in both: {len(common_products):,}")
print(f"Only in original: {len(only_original):,}")
print(f"Only in alternative: {len(only_alternative):,}")

if only_original:
    print("\n--- Products only in original calendar (first 30) ---")
    print(sorted(only_original)[:30])

if only_alternative:
    print("\n--- Products only in alternative calendar (first 30) ---")
    print(sorted(only_alternative)[:30])

# Investigate products found only in the alternative population.
audit = pd.read_csv(
    audit_file,
    dtype={"StockCode": "string"},
)

extra = audit[audit["StockCode"].isin(only_alternative)].copy()

if not extra.empty:
    summary = (
        extra.groupby("StockCode", as_index=False)
        .agg(
            positive_quantity=("positive_quantity", "sum"),
            negative_quantity=("negative_quantity", "sum"),
            net_quantity=("net_quantity", "sum"),
            transaction_lines=("transaction_lines", "sum"),
        )
    )

    summary["negative_quantity"] = summary["negative_quantity"].abs()

    summary = summary.sort_values(
        "transaction_lines", ascending=False
    )

    print("\n--- Extra products: transaction summary ---")
    print(summary.head(20).to_string(index=False))

    REPORTS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        REPORTS / "alternative_only_products.csv", index=False
    )
    print("\nSaved: reports/alternative_only_products.csv")

print("\nComparison complete.")

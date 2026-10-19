
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "online_retail_II.xlsx"
OUTPUT_DIR = ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sheets = ["Year 2009-2010", "Year 2010-2011"]
frames = []

for sheet in sheets:
    df = pd.read_excel(RAW_FILE, sheet_name=sheet)

    df["SourceSheet"] = sheet
    df["Invoice"] = df["Invoice"].astype("string").str.strip()
    df["StockCode"] = df["StockCode"].astype("string").str.strip()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    frames.append(df)

transactions = pd.concat(frames, ignore_index=True)

# Apply the same exact-duplicate policy used by preprocessing.
original_columns = [
    "Invoice", "StockCode", "Description", "Quantity",
    "InvoiceDate", "Price", "Customer ID", "Country"
]
transactions = transactions.drop_duplicates(
    subset=original_columns
).copy()

transactions["Date"] = transactions["InvoiceDate"].dt.normalize()
transactions["is_cancellation_invoice"] = (
    transactions["Invoice"].str.startswith("C", na=False)
)

# Summarize original transaction quantities by product and date.
daily = (
    transactions.groupby(["Date", "StockCode"], dropna=False)
    .agg(
        net_quantity=("Quantity", "sum"),
        positive_quantity=(
            "Quantity", lambda x: x[x > 0].sum()
        ),
        negative_quantity=(
            "Quantity", lambda x: x[x < 0].sum()
        ),
        transaction_lines=("Quantity", "size"),
        cancellation_invoice_lines=(
            "is_cancellation_invoice", "sum"
        ),
    )
    .reset_index()
)

daily["absolute_net_quantity"] = daily["net_quantity"].abs()
daily["positive_quantity"] = daily["positive_quantity"].astype("int64")
daily["negative_quantity"] = daily["negative_quantity"].astype("int64")

daily.to_csv(
    OUTPUT_DIR / "cancellation_audit_daily.csv", index=False
)

# Focus on extreme positive quantities in the current demand target.
positive_daily = daily.sort_values(
    "positive_quantity", ascending=False
)

print("=" * 68)
print("STOCKPILOT ML - CANCELLATION AUDIT")
print("=" * 68)

print("\n--- Overall transaction counts ---")
print(f"Deduplicated transaction rows: {len(transactions):,}")
print(
    "Cancellation-invoice rows:",
    f"{transactions['is_cancellation_invoice'].sum():,}"
)
print(
    "Negative-quantity rows:",
    f"{(transactions['Quantity'] < 0).sum():,}"
)

print("\n--- Largest positive product-day quantities ---")
print(
    positive_daily[
        [
            "Date", "StockCode", "positive_quantity",
            "negative_quantity", "net_quantity",
            "transaction_lines", "cancellation_invoice_lines"
        ]
    ].head(20).to_string(index=False)
)

print("\n--- Extreme product-days with net quantity zero ---")
extreme_net_zero = daily[
    (daily["positive_quantity"] >= 1000)
    & (daily["net_quantity"] == 0)
]

print(f"Count: {len(extreme_net_zero):,}")
print(
    extreme_net_zero.sort_values(
        "positive_quantity", ascending=False
    ).head(20).to_string(index=False)
)

print("\nSaved: reports/cancellation_audit_daily.csv")

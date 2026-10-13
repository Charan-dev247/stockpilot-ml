
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKBOOK = PROJECT_ROOT / "data" / "raw" / "online_retail_II.xlsx"


def main():
    if not WORKBOOK.exists():
        raise FileNotFoundError(f"Workbook not found: {WORKBOOK}")

    print("Loading both Excel sheets. Please wait...")
    sheets = pd.read_excel(
        WORKBOOK,
        sheet_name=None,
        engine="openpyxl",
    )

    frames = []

    for sheet_name, df in sheets.items():
        df["SourceSheet"] = sheet_name
        frames.append(df)

        print(f"\n{sheet_name}")
        print(f"Data rows: {len(df):,}")
        print(f"Columns: {list(df.columns)}")

    data = pd.concat(frames, ignore_index=True)
    original_columns = [
        col for col in data.columns if col != "SourceSheet"
    ]

    print("\n" + "=" * 70)
    print("FULL DATASET QUALITY REPORT")
    print("=" * 70)

    print(f"\nTotal records: {len(data):,}")

    print("\n1. MISSING VALUES")
    missing = data[original_columns].isna().sum()
    missing_report = pd.DataFrame({
        "missing_count": missing,
        "missing_percent": (missing / len(data) * 100).round(3),
    })
    print(missing_report.to_string())

    print("\n2. EXACT DUPLICATE ROWS")
    duplicates = data[original_columns].duplicated().sum()
    print(f"Duplicate rows: {duplicates:,}")

    print("\n3. DATE COVERAGE")
    data["InvoiceDate"] = pd.to_datetime(
        data["InvoiceDate"], errors="coerce"
    )
    print(f"Earliest date: {data['InvoiceDate'].min()}")
    print(f"Latest date:   {data['InvoiceDate'].max()}")
    print(f"Invalid dates: {data['InvoiceDate'].isna().sum():,}")

    print("\n4. QUANTITY DISTRIBUTION")
    print(data["Quantity"].describe().to_string())
    print(f"Zero quantity:     {(data['Quantity'] == 0).sum():,}")
    print(f"Negative quantity: {(data['Quantity'] < 0).sum():,}")
    print(f"Positive quantity: {(data['Quantity'] > 0).sum():,}")

    print("\n5. PRICE CHECKS")
    print(data["Price"].describe().to_string())
    print(f"Zero price:     {(data['Price'] == 0).sum():,}")
    print(f"Negative price: {(data['Price'] < 0).sum():,}")

    print("\n6. INVOICE IDENTIFIERS")
    invoice = data["Invoice"].astype("string").str.strip()
    cancellation_like = invoice.str.upper().str.startswith(
        "C", na=False
    )
    print(f"Cancellation-like rows: {cancellation_like.sum():,}")
    print(f"Unique invoice IDs: {invoice.nunique(dropna=True):,}")

    print("\n7. PRODUCT COVERAGE")
    print(f"Unique StockCodes: {data['StockCode'].nunique(dropna=True):,}")
    print(f"Missing StockCodes: {data['StockCode'].isna().sum():,}")

    print("\nTop 10 StockCodes by row count:")
    print(data["StockCode"].value_counts().head(10).to_string())

    print("\n8. COUNTRY COVERAGE")
    print(f"Unique countries: {data['Country'].nunique(dropna=True):,}")
    print("\nTop 10 countries by row count:")
    print(data["Country"].value_counts().head(10).to_string())

    print("\n9. ROW COUNTS BY SOURCE SHEET")
    print(data["SourceSheet"].value_counts().to_string())

    print("\n" + "=" * 70)
    print("Profiling completed. Original workbook remains unchanged.")
    print("No rows were removed and no model was trained.")
    print("=" * 70)


if __name__ == "__main__":
    main()

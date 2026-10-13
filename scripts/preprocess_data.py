
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WORKBOOK = ROOT / "data" / "raw" / "online_retail_II.xlsx"
PROCESSED_DIR = ROOT / "data" / "processed"


def main():
    if not WORKBOOK.exists():
        raise FileNotFoundError(f"Dataset not found: {WORKBOOK}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading workbook...")
    sheets = pd.read_excel(
        WORKBOOK,
        sheet_name=None,
        engine="openpyxl",
    )

    frames = []

    for sheet_name, frame in sheets.items():
        frame["SourceSheet"] = sheet_name
        frames.append(frame)

    raw = pd.concat(frames, ignore_index=True)
    original_count = len(raw)

    print(f"Original records: {original_count:,}")

    # 1. Remove exact duplicates using the original transaction columns.
    # SourceSheet is our added metadata, so it is excluded.
    original_columns = [
        col for col in raw.columns if col != "SourceSheet"
    ]

    data = raw.drop_duplicates(
        subset=original_columns,
        keep="first",
    ).copy()

    duplicate_count = original_count - len(data)

    print(f"Exact duplicate rows removed: {duplicate_count:,}")

    # 2. Convert and validate the fields required for forecasting.
    data["InvoiceDate"] = pd.to_datetime(
        data["InvoiceDate"], errors="coerce"
    )
    data["Quantity"] = pd.to_numeric(
        data["Quantity"], errors="coerce"
    )
    data["StockCode"] = data["StockCode"].astype("string").str.strip()

    invalid_required = (
        data["StockCode"].isna()
        | data["StockCode"].eq("")
        | data["InvoiceDate"].isna()
        | data["Quantity"].isna()
    )

    invalid_count = int(invalid_required.sum())
    data = data.loc[~invalid_required].copy()

    print(f"Rows excluded for invalid required fields: {invalid_count:,}")

    # 3. Report negative and zero quantities separately.
    negative_count = int((data["Quantity"] < 0).sum())
    zero_count = int((data["Quantity"] == 0).sum())

    print(f"Negative-quantity rows retained for reporting: {negative_count:,}")
    print(f"Zero-quantity rows retained for reporting: {zero_count:,}")

    # 4. Define initial target: positive recorded quantities only.
    sales = data.loc[data["Quantity"] > 0].copy()

    print(f"Positive-quantity rows for initial target: {len(sales):,}")

    # 5. Aggregate transaction quantities by product and calendar day.
    sales["Date"] = sales["InvoiceDate"].dt.normalize()

    daily = (
        sales.groupby(["Date", "StockCode"], as_index=False)
        .agg(
            demand=("Quantity", "sum"),
            transaction_lines=("Quantity", "size"),
        )
        .sort_values(["StockCode", "Date"])
        .reset_index(drop=True)
    )

    # 6. Save separate outputs; do not overwrite the raw workbook.
    cleaned_path = PROCESSED_DIR / "cleaned_transactions.csv"
    daily_path = PROCESSED_DIR / "daily_demand.csv"
    report_path = PROCESSED_DIR / "preprocessing_report.txt"

    sales.to_csv(cleaned_path, index=False)
    daily.to_csv(daily_path, index=False)

    report = [
        "STOCKPILOT ML - PREPROCESSING REPORT",
        "=" * 45,
        f"Original records: {original_count:,}",
        f"Exact duplicate rows removed: {duplicate_count:,}",
        f"Invalid required-field rows excluded: {invalid_count:,}",
        f"Negative-quantity rows excluded from initial target: {negative_count:,}",
        f"Zero-quantity rows excluded from initial target: {zero_count:,}",
        f"Positive-quantity transaction rows: {len(sales):,}",
        f"Unique products in daily dataset: {daily['StockCode'].nunique():,}",
        f"Daily product records: {len(daily):,}",
        f"First daily date: {daily['Date'].min()}",
        f"Last daily date: {daily['Date'].max()}",
        "",
        "Target definition:",
        "Sum of positive recorded quantities per product per calendar day.",
        "",
        "Important limitations:",
        "This target is a proxy for gross recorded sales, not true latent demand.",
        "Days without recorded sales are not yet explicitly represented.",
        "Inventory and stockout data are unavailable in the source dataset.",
    ]

    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Cleaned transactions: {cleaned_path}")
    print(f"Daily demand dataset: {daily_path}")
    print(f"Report: {report_path}")
    print("\nDaily dataset preview:")
    print(daily.head(10).to_string(index=False))
    print("\nThe original Excel workbook has not been modified.")


if __name__ == "__main__":
    main()

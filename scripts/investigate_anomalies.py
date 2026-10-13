
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WORKBOOK = ROOT / "data" / "raw" / "online_retail_II.xlsx"


def main():
    print("Loading dataset...")
    sheets = pd.read_excel(
        WORKBOOK,
        sheet_name=None,
        engine="openpyxl",
    )
    data = pd.concat(sheets.values(), ignore_index=True)

    invoice = data["Invoice"].astype("string").str.strip()
    cancellation_like = invoice.str.upper().str.startswith(
        "C", na=False
    )
    negative_qty = data["Quantity"] < 0
    positive_qty = data["Quantity"] > 0
    negative_price = data["Price"] < 0
    zero_price = data["Price"] == 0

    print("\n" + "=" * 65)
    print("TARGETED ANOMALY INVESTIGATION")
    print("=" * 65)

    # 1. Compare cancellation-like invoices with quantity signs.
    print("\n1. INVOICE PREFIX VS QUANTITY")
    print(pd.crosstab(
        cancellation_like.rename("Cancellation-like invoice"),
        data["Quantity"].lt(0).rename("Negative quantity"),
    ).to_string())

    print(
        "\nCancellation-like rows with positive quantities:",
        int((cancellation_like & positive_qty).sum()),
    )
    print(
        "Negative-quantity rows without a C prefix:",
        int((~cancellation_like & negative_qty).sum()),
    )

    # 2. Quantify exact duplicates and their quantity contribution.
    print("\n2. EXACT DUPLICATE IMPACT")
    duplicate_mask = data.duplicated(keep="first")
    print(f"Duplicate rows after first occurrence: {duplicate_mask.sum():,}")
    print(
        "Sum of quantities in duplicate rows:",
        int(data.loc[duplicate_mask, "Quantity"].sum()),
    )
    print("\nExample duplicate rows:")
    print(data.loc[duplicate_mask].head(10).to_string(index=False))

    # 3. Investigate unusually large quantities.
    print("\n3. LARGE QUANTITIES")
    for threshold in [100, 1000, 10000]:
        mask = data["Quantity"].abs() >= threshold
        print(f"|Quantity| >= {threshold}: {int(mask.sum()):,} rows")

    print("\nLargest absolute quantities:")
    large = data.loc[
        data["Quantity"].abs().nlargest(10).index
    ]
    print(
        large[
            ["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price"]
        ].to_string(index=False)
    )

    # 4. Investigate price anomalies.
    print("\n4. PRICE ANOMALIES")
    print(f"Zero-price rows: {int(zero_price.sum()):,}")
    print(f"Negative-price rows: {int(negative_price.sum()):,}")

    print("\nNegative-price examples:")
    print(
        data.loc[
            negative_price,
            ["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price"],
        ].head(10).to_string(index=False)
    )

    print("\nZero-price rows with positive quantities:")
    print(int((zero_price & positive_qty).sum()))

    # 5. Inspect missing product descriptions.
    print("\n5. MISSING DESCRIPTIONS")
    missing_desc = data["Description"].isna()
    print(f"Rows with missing descriptions: {int(missing_desc.sum()):,}")
    print(
        "Distinct stock codes among those rows:",
        data.loc[missing_desc, "StockCode"].nunique(),
    )
    print(
        "Rows with missing descriptions and valid stock codes:",
        int((missing_desc & data["StockCode"].notna()).sum()),
    )

    print("\n" + "=" * 65)
    print("Investigation complete. No data was modified.")
    print("=" * 65)


if __name__ == "__main__":
    main()

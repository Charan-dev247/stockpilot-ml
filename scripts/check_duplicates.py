
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WORKBOOK = ROOT / "data" / "raw" / "online_retail_II.xlsx"


def main():
    sheets = pd.read_excel(
        WORKBOOK,
        sheet_name=None,
        engine="openpyxl",
    )

    frames = []
    for sheet_name, frame in sheets.items():
        frame["SourceSheet"] = sheet_name
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    original_columns = [
        col for col in data.columns if col != "SourceSheet"
    ]

    print("Total rows:", len(data))
    print(
        "Duplicates ignoring SourceSheet:",
        data[original_columns].duplicated().sum(),
    )
    print(
        "Duplicates including SourceSheet:",
        data.duplicated().sum(),
    )

    # Count identical rows that appear in both source sheets.
    cross_sheet = (
        data.groupby(original_columns, dropna=False)["SourceSheet"]
        .nunique()
        .gt(1)
    )
    print(
        "Distinct row patterns appearing in both sheets:",
        int(cross_sheet.sum()),
    )


if __name__ == "__main__":
    main()

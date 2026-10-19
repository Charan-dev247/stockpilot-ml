
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"

ORIGINAL_FILE = REPORT_DIR / "random_forest_v2_validation_predictions.csv"
NET_FILE = REPORT_DIR / "random_forest_net_v2_validation_predictions.csv"


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(np.asarray(predicted, dtype=float), 0)

    total_actual = np.abs(actual).sum()

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "WAPE_percent": (
            np.abs(actual - predicted).sum() / total_actual * 100
            if total_actual > 0 else np.nan
        ),
    }


def main():
    original = pd.read_csv(
        ORIGINAL_FILE,
        parse_dates=["Date"],
        dtype={"StockCode": "string"},
    )

    net = pd.read_csv(
        NET_FILE,
        parse_dates=["Date"],
        dtype={"StockCode": "string"},
    )

    # Keep only common product-date rows.
    merged = original.merge(
        net,
        on=["StockCode", "Date"],
        how="inner",
        suffixes=("_original", "_net"),
        validate="one_to_one",
    )

    if merged.empty:
        raise ValueError("No matching product-date rows found.")

    # Confirm both models use the same validation dates and products.
    print("=" * 70)
    print("STOCKPILOT ML - CONTROLLED NET V2 COMPARISON")
    print("=" * 70)
    print(f"Original prediction rows: {len(original):,}")
    print(f"Net prediction rows:      {len(net):,}")
    print(f"Matched rows:             {len(merged):,}")
    print(f"Matched products:         {merged['StockCode'].nunique():,}")
    print(
        f"Validation dates:         "
        f"{merged['Date'].min().date()} to {merged['Date'].max().date()}"
    )

    # Each model's native-target metrics are useful, but not directly
    # comparable because the actual targets differ.
    results = []

    comparisons = [
        (
            "Original V2 evaluated on original target",
            "demand_original",
            "prediction_v2",
        ),
        (
            "Net V2 evaluated on net target",
            "demand_net",
            "prediction_net_v2",
        ),
        (
            "Original V2 evaluated on net target",
            "demand_net",
            "prediction_v2",
        ),
        (
            "Net V2 evaluated on original target",
            "demand_original",
            "prediction_net_v2",
        ),
    ]

    for label, actual_col, prediction_col in comparisons:
        metrics = calculate_metrics(
            merged[actual_col],
            merged[prediction_col],
        )
        results.append({"Comparison": label, **metrics})

    results_df = pd.DataFrame(results)

    print("\n--- Metrics ---")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    output_file = REPORT_DIR / "net_v2_controlled_comparison.csv"
    results_df.to_csv(output_file, index=False)

    print(f"\nSaved: {output_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

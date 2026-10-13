
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "regular_daily_demand.csv"
PREDICTIONS_FILE = ROOT / "reports" / "random_forest_validation_predictions.csv"
OUTPUT_FILE = ROOT / "reports" / "fair_model_comparison.csv"


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "WAPE_percent": (
            np.abs(actual - predicted).sum() / np.abs(actual).sum() * 100
            if np.abs(actual).sum() > 0 else np.nan
        ),
    }


def main():
    predictions = pd.read_csv(
        PREDICTIONS_FILE,
        parse_dates=["Date"],
        dtype={"StockCode": "string"},
    )

    data = pd.read_csv(
        DATA_FILE,
        parse_dates=["Date"],
        dtype={"StockCode": "string"},
    )

    lookup = data.set_index(["StockCode", "Date"])["demand"]

    # All models use exactly the same validation rows.
    actual = predictions["demand"].to_numpy()
    rf_prediction = predictions["prediction"].to_numpy()

    lag_dates = predictions["Date"] - pd.Timedelta(days=7)

    seasonal_prediction = np.array([
        lookup.get((product, date), 0)
        for product, date in zip(
            predictions["StockCode"], lag_dates
        )
    ], dtype=float)

    model_predictions = {
        "Zero baseline": np.zeros(len(predictions)),
        "Seasonal Naive": seasonal_prediction,
        "Random Forest": rf_prediction,
    }

    results = []

    positive_mask = actual > 0

    for model_name, predicted in model_predictions.items():
        row = {"Model": model_name}
        row.update(metrics(actual, predicted))

        row["Positive_day_MAE"] = (
            mean_absolute_error(
                actual[positive_mask],
                predicted[positive_mask],
            )
            if positive_mask.any() else np.nan
        )

        results.append(row)

    result_df = pd.DataFrame(results)

    print("=" * 65)
    print("FAIR MODEL COMPARISON - SAME VALIDATION ROWS")
    print("=" * 65)
    print(f"\nValidation rows: {len(predictions):,}")
    print(f"Products: {predictions['StockCode'].nunique():,}")
    print(f"Positive-demand rows: {positive_mask.sum():,}")

    print("\n--- Results (lower is better) ---")
    print(result_df.round(4).to_string(index=False))

    result_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

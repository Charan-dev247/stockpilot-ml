
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
DATA_FILE = ROOT / "data" / "processed" / "regular_daily_demand.csv"


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "WAPE_percent": (
            np.abs(actual - predicted).sum() / np.abs(actual).sum() * 100
            if np.abs(actual).sum() > 0 else np.nan
        ),
        "Positive_day_MAE": (
            mean_absolute_error(actual[actual > 0], predicted[actual > 0])
            if np.any(actual > 0) else np.nan
        ),
    }


def main():
    v1 = pd.read_csv(
        REPORTS / "random_forest_validation_predictions.csv",
        parse_dates=["Date"],
        dtype={"StockCode": "string"},
    )

    v2 = pd.read_csv(
        REPORTS / "random_forest_v2_validation_predictions.csv",
        parse_dates=["Date"],
        dtype={"StockCode": "string"},
    )

    common = v1.merge(
        v2[["StockCode", "Date", "demand", "prediction_v2"]],
        on=["StockCode", "Date"],
        how="inner",
        suffixes=("_v1", "_v2"),
        validate="one_to_one",
     )

    if common.empty:
        raise ValueError("No shared validation rows found.")

    # Confirm actual values agree across both prediction files.
    if not np.allclose(common["demand_v1"], common["demand_v2"]):
        raise ValueError("Actual demand differs between model versions.")

    actual = common["demand_v1"].to_numpy()

    data = pd.read_csv(
        DATA_FILE,
        parse_dates=["Date"],
        dtype={"StockCode": "string"},
    )

    lookup = data.set_index(["StockCode", "Date"])["demand"]

    lag_dates = common["Date"] - pd.Timedelta(days=7)
    seasonal = np.array([
        lookup.get((product, date), 0)
        for product, date in zip(common["StockCode"], lag_dates)
    ], dtype=float)

    predictions = {
        "Zero baseline": np.zeros(len(common)),
        "Seasonal Naive": seasonal,
        "Random Forest V1": common["prediction"].to_numpy(),
        "Random Forest V2": common["prediction_v2"].to_numpy(),
    }

    rows = []
    for name, pred in predictions.items():
        row = {"Model": name}
        row.update(calculate_metrics(actual, pred))
        rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(REPORTS / "rf_version_comparison.csv", index=False)

    print("=" * 65)
    print("RANDOM FOREST VERSION COMPARISON")
    print("=" * 65)
    print(f"\nShared validation rows: {len(common):,}")
    print(f"Products: {common['StockCode'].nunique():,}")
    print(f"Positive-demand rows: {(actual > 0).sum():,}")
    print("\n--- Results: lower is better ---")
    print(results.round(4).to_string(index=False))
    print("\nSaved: reports/rf_version_comparison.csv")


if __name__ == "__main__":
    main()

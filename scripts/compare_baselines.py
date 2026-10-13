
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data" / "processed" / "regular_daily_demand.csv"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    absolute_errors = np.abs(actual - predicted)
    total_actual = np.abs(actual).sum()

    return {
        "MAE": absolute_errors.mean(),
        "RMSE": np.sqrt(np.mean((actual - predicted) ** 2)),
        "WAPE_percent": (
            absolute_errors.sum() / total_actual * 100
            if total_actual > 0 else np.nan
        ),
    }


def main():
    data = pd.read_csv(
        DATA_FILE,
        parse_dates=["Date"],
        dtype={"StockCode": "string"},
    )

    first_date = data["Date"].min()
    total_days = (data["Date"].max() - first_date).days + 1

    train_end = first_date + pd.Timedelta(
        days=int(total_days * 0.70) - 1
    )
    validation_end = first_date + pd.Timedelta(
        days=int(total_days * 0.85) - 1
    )

    # Select products using training data only.
    products = sorted(
        data.loc[data["Date"] <= train_end, "StockCode"].unique()
    )

    validation_dates = pd.date_range(
        train_end + pd.Timedelta(days=1),
        validation_end,
        freq="D",
    )

    # Build every product-date combination in validation.
    panel = pd.MultiIndex.from_product(
        [products, validation_dates],
        names=["StockCode", "Date"],
    ).to_frame(index=False)

    # Map actual demand and fill absent product-days with zero.
    demand_lookup = data.set_index(
        ["StockCode", "Date"]
    )["demand"]

    panel["actual"] = [
        demand_lookup.get((product, date), 0)
        for product, date in zip(panel["StockCode"], panel["Date"])
    ]

    # Baseline 1: always predict zero.
    panel["zero_prediction"] = 0.0

    # Baseline 2: use actual demand seven days earlier.
    panel["lag_date"] = panel["Date"] - pd.Timedelta(days=7)

    panel["seasonal_prediction"] = [
        demand_lookup.get((product, date), 0)
        for product, date in zip(panel["StockCode"], panel["lag_date"])
    ]

    results = []

    for name, column in [
        ("Zero baseline", "zero_prediction"),
        ("Seasonal naive (lag 7)", "seasonal_prediction"),
    ]:
        metrics = calculate_metrics(panel["actual"], panel[column])
        metrics["Model"] = name
        results.append(metrics)

    results_df = pd.DataFrame(results)[
        ["Model", "MAE", "RMSE", "WAPE_percent"]
    ]

    print("=" * 65)
    print("STOCKPILOT ML - FAIR BASELINE COMPARISON")
    print("=" * 65)
    print(f"\nTraining cutoff: {train_end.date()}")
    print(
        f"Validation: {validation_dates.min().date()} to "
        f"{validation_dates.max().date()}"
    )
    print(f"Products selected from training: {len(products):,}")
    print(f"Validation product-days: {len(panel):,}")

    print("\n--- Baseline Metrics ---")
    print(results_df.round(4).to_string(index=False))

    results_df.to_csv(
        REPORT_DIR / "baseline_comparison.csv",
        index=False,
    )

    panel.to_csv(
        REPORT_DIR / "baseline_comparison_predictions.csv",
        index=False,
    )

    print("\nSaved:")
    print("reports/baseline_comparison.csv")
    print("reports/baseline_comparison_predictions.csv")
    print("\nNo source data was modified.")


if __name__ == "__main__":
    main()

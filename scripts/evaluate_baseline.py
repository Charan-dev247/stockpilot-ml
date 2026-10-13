
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

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    # WAPE is useful for comparing aggregate forecast errors.
    total_actual = np.sum(np.abs(actual))
    wape = (
        np.sum(np.abs(actual - predicted)) / total_actual * 100
        if total_actual > 0
        else np.nan
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE_percent": wape,
    }


def main():
    data = pd.read_csv(
        DATA_FILE,
        parse_dates=["Date"],
        dtype={"StockCode": "string"},
    )

    first_date = data["Date"].min()
    last_date = data["Date"].max()
    total_days = (last_date - first_date).days + 1

    train_end = first_date + pd.Timedelta(
        days=int(total_days * 0.70) - 1
    )
    validation_end = first_date + pd.Timedelta(
        days=int(total_days * 0.85) - 1
    )

    # Start with products that have recorded history during training.
    train_products = data.loc[
        data["Date"] <= train_end, "StockCode"
    ].unique()

    # Evaluate only products with training history and at least
    # one positive-demand observation in validation.
    validation_products = set(
        data.loc[
            (data["Date"] > train_end)
            & (data["Date"] <= validation_end)
            & (data["demand"] > 0),
            "StockCode",
        ].unique()
    )

    eligible_products = sorted(
        set(train_products) & validation_products
    )

    if not eligible_products:
        raise ValueError("No eligible products found.")

    # Build a complete product-day panel for the evaluation dates.
    validation_dates = pd.date_range(
        train_end + pd.Timedelta(days=1),
        validation_end,
        freq="D",
    )

    panel = pd.MultiIndex.from_product(
        [eligible_products, validation_dates],
        names=["StockCode", "Date"],
    ).to_frame(index=False)

    # Include training history so that each forecast can use t - 7.
    history_start = validation_dates.min() - pd.Timedelta(days=7)

    history = data.loc[
        data["StockCode"].isin(eligible_products)
        & (data["Date"] >= history_start)
        & (data["Date"] <= validation_end),
        ["StockCode", "Date", "demand"],
    ]

    panel = panel.merge(
        history,
        on=["StockCode", "Date"],
        how="left",
    )

    panel["demand"] = panel["demand"].fillna(0)

    # Shift demand by seven calendar days for each product.
    lookup = history.set_index(
        ["StockCode", "Date"]
    )["demand"]

    panel["lag_date"] = panel["Date"] - pd.Timedelta(days=7)

    panel["prediction"] = [
        lookup.get((product, date), 0)
        for product, date in zip(
            panel["StockCode"], panel["lag_date"]
        )
    ]

    # The first validation date uses demand from the final
    # training week. Later dates use observed validation demand
    # from seven days earlier (rolling one-step-ahead evaluation).
    metrics = calculate_metrics(
        panel["demand"],
        panel["prediction"],
    )

    print("=" * 65)
    print("STOCKPILOT ML - SEASONAL-NAIVE BASELINE")
    print("=" * 65)

    print(f"\nTraining cutoff: {train_end.date()}")
    print(f"Validation period: {validation_dates.min().date()} "
          f"to {validation_dates.max().date()}")
    print(f"Eligible products: {len(eligible_products):,}")
    print(f"Validation product-days: {len(panel):,}")

    print("\n--- Validation Metrics ---")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    panel.to_csv(
        REPORT_DIR / "baseline_validation_predictions.csv",
        index=False,
    )

    pd.DataFrame([metrics]).to_csv(
        REPORT_DIR / "baseline_validation_metrics.csv",
        index=False,
    )

    print("\nSaved:")
    print("reports/baseline_validation_predictions.csv")
    print("reports/baseline_validation_metrics.csv")
    print("\nBaseline evaluation completed.")


if __name__ == "__main__":
    main()

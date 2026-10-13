
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data" / "processed" / "regular_daily_demand.csv"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    total_actual = np.abs(actual).sum()
    wape = (
        np.abs(actual - predicted).sum() / total_actual * 100
        if total_actual > 0 else np.nan
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
    total_days = (data["Date"].max() - first_date).days + 1

    train_end = first_date + pd.Timedelta(
        days=int(total_days * 0.70) - 1
    )
    validation_end = first_date + pd.Timedelta(
        days=int(total_days * 0.85) - 1
    )

    # Select products using training data only.
    train_products = sorted(
        data.loc[data["Date"] <= train_end, "StockCode"].unique()
    )

    # Sample products to keep the first experiment manageable.
    rng = np.random.default_rng(42)
    if len(train_products) > 500:
        products = sorted(
            rng.choice(train_products, size=500, replace=False)
        )
    else:
        products = train_products

    # Build a complete daily panel for selected products.
    all_dates = pd.date_range(
        first_date,
        validation_end,
        freq="D",
    )

    panel = pd.MultiIndex.from_product(
        [products, all_dates],
        names=["StockCode", "Date"],
    ).to_frame(index=False)

    lookup = data.set_index(["StockCode", "Date"])["demand"]

    panel["demand"] = [
        lookup.get((product, date), 0)
        for product, date in zip(panel["StockCode"], panel["Date"])
    ]

    # Create features from past demand.
    panel = panel.sort_values(["StockCode", "Date"])
    grouped = panel.groupby("StockCode", sort=False)["demand"]

    panel["lag_1"] = grouped.shift(1)
    panel["lag_7"] = grouped.shift(7)

    panel["rolling_mean_7"] = grouped.transform(
        lambda s: s.shift(1).rolling(7, min_periods=7).mean()
    )

    panel["day_of_week"] = panel["Date"].dt.dayofweek
    panel["month"] = panel["Date"].dt.month

    feature_columns = [
        "lag_1",
        "lag_7",
        "rolling_mean_7",
        "day_of_week",
        "month",
    ]

    panel = panel.dropna(subset=feature_columns).copy()

    train = panel.loc[panel["Date"] <= train_end]
    validation = panel.loc[
        (panel["Date"] > train_end)
        & (panel["Date"] <= validation_end)
    ].copy()

    X_train = train[feature_columns]
    y_train = train["demand"]

    X_validation = validation[feature_columns]
    y_validation = validation["demand"]

    print("=" * 65)
    print("STOCKPILOT ML - RANDOM FOREST")
    print("=" * 65)

    print(f"\nTraining cutoff: {train_end.date()}")
    print(f"Validation: {validation['Date'].min().date()} to "
          f"{validation['Date'].max().date()}")
    print(f"Selected products: {len(products):,}")
    print(f"Training rows: {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Features: {feature_columns}")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_validation)
    predictions = np.maximum(predictions, 0)

    metrics = calculate_metrics(y_validation, predictions)

    print("\n--- Validation Metrics ---")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    validation["prediction"] = predictions

    validation[
        ["StockCode", "Date", "demand", "prediction"]
    ].to_csv(
        REPORT_DIR / "random_forest_validation_predictions.csv",
        index=False,
    )

    pd.DataFrame([metrics]).to_csv(
        REPORT_DIR / "random_forest_validation_metrics.csv",
        index=False,
    )

    importance = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance.to_csv(
        REPORT_DIR / "random_forest_feature_importance.csv",
        index=False,
    )

    print("\nSaved:")
    print("reports/random_forest_validation_predictions.csv")
    print("reports/random_forest_validation_metrics.csv")
    print("reports/random_forest_feature_importance.csv")
    print("\nTraining completed.")


if __name__ == "__main__":
    main()

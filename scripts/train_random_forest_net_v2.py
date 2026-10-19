
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parent.parent

# Use the cancellation-adjusted target with the original product cohort.
DATA_FILE = (
    ROOT / "data" / "processed" / "comparable_net_daily_demand.csv"
)

REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    total_actual = np.abs(actual).sum()

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "WAPE_percent": (
            np.abs(actual - predicted).sum() / total_actual * 100
            if total_actual > 0
            else np.nan
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

    train_products = sorted(
        data.loc[data["Date"] <= train_end, "StockCode"].unique()
    )

    rng = np.random.default_rng(42)

    if len(train_products) > 500:
        products = sorted(
            rng.choice(train_products, size=500, replace=False)
        )
    else:
        products = train_products

    # Build a complete calendar through the validation period.
    all_dates = pd.date_range(
        first_date, validation_end, freq="D"
    )

    panel = pd.MultiIndex.from_product(
        [products, all_dates],
        names=["StockCode", "Date"],
    ).to_frame(index=False)

    demand_lookup = data.set_index(
        ["StockCode", "Date"]
    )["demand"]

    panel["demand"] = [
        demand_lookup.get((product, date), 0)
        for product, date in zip(panel["StockCode"], panel["Date"])
    ]

    panel = panel.sort_values(
        ["StockCode", "Date"]
    ).reset_index(drop=True)

    grouped = panel.groupby("StockCode", sort=False)["demand"]

    # Lag features use previous observations only.
    panel["lag_1"] = grouped.shift(1)
    panel["lag_7"] = grouped.shift(7)
    panel["lag_14"] = grouped.shift(14)

    # Shift before rolling to avoid using the target day's demand.
    panel["rolling_mean_7"] = grouped.transform(
        lambda s: s.shift(1).rolling(7, min_periods=7).mean()
    )

    panel["rolling_mean_28"] = grouped.transform(
        lambda s: s.shift(1).rolling(28, min_periods=28).mean()
    )

    panel["rolling_std_7"] = grouped.transform(
        lambda s: s.shift(1).rolling(7, min_periods=7).std()
    )

    panel["active_days_28"] = grouped.transform(
        lambda s: (
            (s.shift(1) > 0)
            .rolling(28, min_periods=28)
            .sum()
        )
    )

    panel["day_of_week"] = panel["Date"].dt.dayofweek
    panel["month"] = panel["Date"].dt.month

    feature_columns = [
        "lag_1",
        "lag_7",
        "lag_14",
        "rolling_mean_7",
        "rolling_mean_28",
        "rolling_std_7",
        "active_days_28",
        "day_of_week",
        "month",
    ]

    # Omit early dates without sufficient feature history.
    panel = panel.dropna(subset=feature_columns).copy()

    train = panel.loc[panel["Date"] <= train_end].copy()

    validation = panel.loc[
        (panel["Date"] > train_end)
        & (panel["Date"] <= validation_end)
    ].copy()

    X_train = train[feature_columns]
    y_train = train["demand"]

    X_validation = validation[feature_columns]
    y_validation = validation["demand"]

    print("=" * 68)
    print("STOCKPILOT ML - RANDOM FOREST V2 (NET-ADJUSTED TARGET)")
    print("=" * 68)

    print(f"\nInput dataset: {DATA_FILE.name}")
    print(f"Training cutoff: {train_end.date()}")
    print(
        f"Validation: {validation['Date'].min().date()} to "
        f"{validation['Date'].max().date()}"
    )
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

    print("\nTraining Random Forest V2 on net-adjusted demand...")
    model.fit(X_train, y_train)

    predictions = np.maximum(model.predict(X_validation), 0)

    metrics = calculate_metrics(y_validation, predictions)

    print("\n--- Validation Metrics: Net-Adjusted Target ---")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    validation["prediction_net_v2"] = predictions

    validation[
        ["StockCode", "Date", "demand", "prediction_net_v2"]
    ].to_csv(
        REPORT_DIR / "random_forest_net_v2_validation_predictions.csv",
        index=False,
    )

    pd.DataFrame([metrics]).to_csv(
        REPORT_DIR / "random_forest_net_v2_validation_metrics.csv",
        index=False,
    )

    importance = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance.to_csv(
        REPORT_DIR / "random_forest_net_v2_feature_importance.csv",
        index=False,
    )

    print("\nSaved:")
    print("reports/random_forest_net_v2_validation_predictions.csv")
    print("reports/random_forest_net_v2_validation_metrics.csv")
    print("reports/random_forest_net_v2_feature_importance.csv")
    print("\nTraining completed.")


if __name__ == "__main__":
    main()

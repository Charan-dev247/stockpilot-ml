
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"

ORIGINAL_DATA = (
    ROOT / "data" / "processed" / "regular_daily_demand.csv"
)
NET_DATA = (
    ROOT / "data" / "processed" / "comparable_net_daily_demand.csv"
)

OUTPUT_FILE = REPORT_DIR / "final_test_comparison.csv"

RANDOM_STATE = 42
N_PRODUCTS = 500

FEATURES = [
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


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(
        np.asarray(predicted, dtype=float), 0
    )

    total = np.abs(actual).sum()

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "WAPE_percent": (
            np.abs(actual - predicted).sum() / total * 100
            if total > 0 else np.nan
        ),
    }


def build_panel(data, products, start_date, end_date):
    """Build a complete product-date panel without target leakage."""

    dates = pd.date_range(start_date, end_date, freq="D")

    index = pd.MultiIndex.from_product(
        [products, dates],
        names=["StockCode", "Date"],
    )

    lookup = data.set_index(["StockCode", "Date"])["demand"]

    panel = lookup.reindex(index, fill_value=0).rename(
        "demand"
    ).reset_index()

    panel = panel.sort_values(
        ["StockCode", "Date"]
    ).reset_index(drop=True)

    grouped = panel.groupby("StockCode", sort=False)["demand"]

    panel["lag_1"] = grouped.shift(1)
    panel["lag_7"] = grouped.shift(7)
    panel["lag_14"] = grouped.shift(14)

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
        lambda s: (s.shift(1) > 0).rolling(
            28, min_periods=28
        ).sum()
    )

    panel["day_of_week"] = panel["Date"].dt.dayofweek
    panel["month"] = panel["Date"].dt.month

    return panel


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    for file_path in [ORIGINAL_DATA, NET_DATA]:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing dataset: {file_path}"
            )

    original = pd.read_csv(
        ORIGINAL_DATA,
        parse_dates=["Date"],
        dtype={"StockCode": "string"},
    )

    net = pd.read_csv(
        NET_DATA,
        parse_dates=["Date"],
        dtype={"StockCode": "string"},
    )

    first_date = original["Date"].min()
    last_date = original["Date"].max()

    total_days = (last_date - first_date).days + 1

    train_end = first_date + pd.Timedelta(
        days=int(total_days * 0.70) - 1
    )
    validation_end = first_date + pd.Timedelta(
        days=int(total_days * 0.85) - 1
    )

    test_start = validation_end + pd.Timedelta(days=1)

    # Select products using training-period information only.
    train_products = sorted(
        original.loc[
            original["Date"] <= train_end, "StockCode"
        ].unique()
    )

    rng = np.random.default_rng(RANDOM_STATE)

    if len(train_products) > N_PRODUCTS:
        products = sorted(
            rng.choice(
                train_products,
                size=N_PRODUCTS,
                replace=False,
            )
        )
    else:
        products = train_products

    # Train on the original target using the existing V2 settings.
    # Test dates are never used to fit the model.
    train_panel = build_panel(
        original,
        products,
        first_date,
        train_end,
    )

    train_panel = train_panel.dropna(
        subset=FEATURES
    )

    # Use the same model settings as Random Forest V2.
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("Training original-target V2 model...")
    model.fit(
        train_panel[FEATURES],
        train_panel["demand"],
    )

    # Build test features from historical data through the test end.
    # Actual past demand is used for one-step-ahead forecasts.
    original_test_panel = build_panel(
        original,
        products,
        first_date,
        last_date,
    )

    net_test_panel = build_panel(
        net,
        products,
        first_date,
        last_date,
    )

    original_test = original_test_panel.loc[
        original_test_panel["Date"] >= test_start
    ].copy()

    net_test = net_test_panel.loc[
        net_test_panel["Date"] >= test_start
    ].copy()

    original_test = original_test.dropna(
        subset=FEATURES
    )

    net_test = net_test.dropna(
        subset=FEATURES
    )

    # Train a separate net-target model using only the training period.
    net_train = build_panel(
        net,
        products,
        first_date,
        train_end,
    ).dropna(subset=FEATURES)

    net_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("Training net-target V2 model...")
    net_model.fit(
        net_train[FEATURES],
        net_train["demand"],
    )

    # Align evaluation rows and use the same net-adjusted target.
    original_predictions = original_test[
        ["StockCode", "Date"]
    ].copy()

    original_predictions["prediction_original_v2"] = (
        np.maximum(
            model.predict(original_test[FEATURES]), 0
        )
    )

    net_predictions = net_test[
        ["StockCode", "Date", "demand"]
    ].copy()

    net_predictions["prediction_net_v2"] = np.maximum(
        net_model.predict(net_test[FEATURES]), 0
    )

    evaluation = original_predictions.merge(
        net_predictions,
        on=["StockCode", "Date"],
        how="inner",
        validate="one_to_one",
    )

    # Zero baseline and seasonal naive use the same net target.
    evaluation["zero_prediction"] = 0.0

    net_lookup = net.set_index(
        ["StockCode", "Date"]
    )["demand"]

    prior_keys = pd.MultiIndex.from_arrays([
        evaluation["StockCode"],
        evaluation["Date"] - pd.Timedelta(days=7),
    ])

    evaluation["seasonal_prediction"] = (
        net_lookup.reindex(prior_keys).to_numpy()
    )

    # Missing prior dates represent zero recorded net sales.
    evaluation["seasonal_prediction"] = (
        evaluation["seasonal_prediction"].fillna(0)
    )

    actual = evaluation["demand"].to_numpy()
    positive_mask = actual > 0

    prediction_columns = {
        "Zero baseline": "zero_prediction",
        "Seasonal naive": "seasonal_prediction",
        "Original-target RF V2": "prediction_original_v2",
        "Net-target RF V2": "prediction_net_v2",
    }

    results = []

    for name, column in prediction_columns.items():
        prediction = evaluation[column].to_numpy()

        all_metrics = metrics(actual, prediction)

        positive_mae = (
            mean_absolute_error(
                actual[positive_mask],
                prediction[positive_mask],
            )
            if positive_mask.any() else np.nan
        )

        results.append({
            "Model": name,
            "Rows": len(evaluation),
            **all_metrics,
            "Positive_day_MAE": positive_mae,
        })

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("STOCKPILOT ML - FINAL TEST RESULTS")
    print("=" * 70)
    print(f"Training cutoff: {train_end.date()}")
    print(f"Validation ends: {validation_end.date()}")
    print(
        f"Test period: {evaluation['Date'].min().date()} "
        f"to {evaluation['Date'].max().date()}"
    )
    print(f"Products: {evaluation['StockCode'].nunique():,}")
    print(f"Test rows: {len(evaluation):,}")
    print(f"Positive-demand rows: {positive_mask.sum():,}")

    print("\nAll models evaluated against the same net target:\n")
    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    results_df.to_csv(OUTPUT_FILE, index=False)

    evaluation.to_csv(
        REPORT_DIR / "final_test_predictions.csv",
        index=False,
    )

    print(f"\nSaved: {OUTPUT_FILE}")
    print("Saved: reports/final_test_predictions.csv")


if __name__ == "__main__":
    main()


from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"

DATA_FILE = (
    ROOT / "data" / "processed" / "comparable_net_daily_demand.csv"
)

ORIGINAL_FILE = (
    REPORT_DIR / "random_forest_v2_validation_predictions.csv"
)

NET_FILE = (
    REPORT_DIR / "random_forest_net_v2_validation_predictions.csv"
)

VALIDATION_START = pd.Timestamp("2011-05-02")
VALIDATION_END = pd.Timestamp("2011-08-20")


# ============================================================
# 2. METRIC CALCULATION
# ============================================================

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(
        np.asarray(predicted, dtype=float), 0
    )

    total_actual = np.abs(actual).sum()

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "WAPE_percent": (
            np.abs(actual - predicted).sum()
            / total_actual * 100
            if total_actual > 0
            else np.nan
        ),
    }


# ============================================================
# 3. LOAD DATA
# ============================================================

def main():
    print("=" * 75)
    print("STOCKPILOT ML - FAIR VALIDATION COMPARISON")
    print("=" * 75)

    required_files = [
        DATA_FILE,
        ORIGINAL_FILE,
        NET_FILE,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    target = pd.read_csv(
        DATA_FILE,
        usecols=["StockCode", "Date", "demand"],
        parse_dates=["Date"],
        dtype={"StockCode": "string"},
    )

    original = pd.read_csv(
        ORIGINAL_FILE,
        usecols=["StockCode", "Date", "prediction_v2"],
        parse_dates=["Date"],
        dtype={"StockCode": "string"},
    )

    net = pd.read_csv(
        NET_FILE,
        usecols=["StockCode", "Date", "prediction_net_v2"],
        parse_dates=["Date"],
        dtype={"StockCode": "string"},
    )

    # Validate product-date uniqueness.
    for name, frame in [
        ("Target", target),
        ("Original predictions", original),
        ("Net predictions", net),
    ]:
        if frame.duplicated(["StockCode", "Date"]).any():
            raise ValueError(
                f"{name} contains duplicate product-date rows."
            )

    # ========================================================
    # 4. BUILD COMPLETE TARGET FOR THE MODEL COHORT
    # ========================================================

    # Match the common product cohort used by both models.
    common_products = sorted(
        set(original["StockCode"].dropna())
        & set(net["StockCode"].dropna())
    )

    if not common_products:
        raise ValueError("No common products found between models.")

    # Build a complete daily calendar for these products.
    # Missing dates represent zero recorded net sales under
    # the project's current target definition.
    validation_dates = pd.date_range(
        VALIDATION_START,
        VALIDATION_END,
        freq="D",
    )

    complete_index = pd.MultiIndex.from_product(
        [common_products, validation_dates],
        names=["StockCode", "Date"],
    )

    # Restrict to the required products and validation dates.
    target_subset = target[
        target["StockCode"].isin(common_products)
        & target["Date"].between(
            VALIDATION_START, VALIDATION_END
        )
    ]

    target_calendar = (
        target_subset
        .set_index(["StockCode", "Date"])["demand"]
        .reindex(complete_index, fill_value=0)
        .rename("demand")
        .reset_index()
    )

    # ========================================================
    # 5. ALIGN BOTH MODELS
    # ========================================================

    predictions = original.merge(
        net,
        on=["StockCode", "Date"],
        how="inner",
        validate="one_to_one",
    )

    evaluation = predictions.merge(
        target_calendar,
        on=["StockCode", "Date"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    missing = evaluation["_merge"] != "both"

    print("\n--- Target alignment ---")
    print(f"Original prediction rows: {len(original):,}")
    print(f"Net prediction rows:      {len(net):,}")
    print(f"Matched prediction rows:  {len(evaluation):,}")
    print(f"Missing target rows:      {missing.sum():,}")

    if missing.any():
        print(
            evaluation.loc[
                missing, ["StockCode", "Date"]
            ].head(10).to_string(index=False)
        )
        raise ValueError(
            "Target alignment failed. Check product/date keys."
        )

    evaluation = evaluation.drop(columns="_merge")

    if evaluation.empty:
        raise ValueError("No common validation rows found.")

    # ========================================================
    # 6. CREATE BASELINES
    # ========================================================

    # Zero baseline.
    evaluation["zero_prediction"] = 0.0

    # Seasonal-naive baseline: actual target from seven days
    # earlier. Build it using the complete calendar, including
    # the seven days before validation starts.
    history_start = VALIDATION_START - pd.Timedelta(days=7)

    history_dates = pd.date_range(
        history_start,
        VALIDATION_END,
        freq="D",
    )

    history_index = pd.MultiIndex.from_product(
        [common_products, history_dates],
        names=["StockCode", "Date"],
    )

    history_target = (
        target[
            target["StockCode"].isin(common_products)
            & target["Date"].between(
                history_start, VALIDATION_END
            )
        ]
        .set_index(["StockCode", "Date"])["demand"]
        .reindex(history_index, fill_value=0)
        .rename("demand")
        .reset_index()
    )

    lagged = history_target.copy()
    lagged["Date"] += pd.Timedelta(days=7)
    lagged = lagged.rename(
        columns={"demand": "seasonal_prediction"}
    )

    evaluation = evaluation.merge(
        lagged[
            ["StockCode", "Date", "seasonal_prediction"]
        ],
        on=["StockCode", "Date"],
        how="left",
        validate="one_to_one",
    )

    if evaluation["seasonal_prediction"].isna().any():
        raise ValueError(
            "Missing lag-7 baseline values."
        )

    # ========================================================
    # 7. EVALUATE ALL MODELS ON THE SAME TARGET
    # ========================================================

    actual = evaluation["demand"].to_numpy(dtype=float)
    positive_mask = actual > 0

    model_predictions = {
        "Zero baseline": "zero_prediction",
        "Seasonal naive (lag 7)": "seasonal_prediction",
        "Random Forest V2 - original training target":
            "prediction_v2",
        "Random Forest V2 - net training target":
            "prediction_net_v2",
    }

    results = []

    for model_name, prediction_column in model_predictions.items():
        predicted = evaluation[
            prediction_column
        ].to_numpy(dtype=float)

        overall = calculate_metrics(actual, predicted)

        if positive_mask.any():
            positive = calculate_metrics(
                actual[positive_mask],
                predicted[positive_mask],
            )
        else:
            positive = {
                "MAE": np.nan,
                "RMSE": np.nan,
            }

        results.append({
            "Model": model_name,
            "Rows": len(evaluation),
            "Positive_demand_rows": int(positive_mask.sum()),
            "MAE": overall["MAE"],
            "RMSE": overall["RMSE"],
            "WAPE_percent": overall["WAPE_percent"],
            "Positive_day_MAE": positive["MAE"],
            "Positive_day_RMSE": positive["RMSE"],
        })

    results_df = (
        pd.DataFrame(results)
        .sort_values("MAE")
        .reset_index(drop=True)
    )

    # ========================================================
    # 8. DISPLAY RESULTS
    # ========================================================

    print("\n--- Validation details ---")
    print(f"Products: {evaluation['StockCode'].nunique():,}")
    print(
        f"Dates: {evaluation['Date'].min().date()} to "
        f"{evaluation['Date'].max().date()}"
    )
    print(f"Validation rows: {len(evaluation):,}")
    print(f"Positive-demand rows: {positive_mask.sum():,}")

    print("\n--- Model comparison ---")
    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # ========================================================
    # 9. SAVE REPORT
    # ========================================================

    output_file = (
        REPORT_DIR / "final_validation_comparison.csv"
    )

    results_df.to_csv(output_file, index=False)

    print(f"\nSaved report: {output_file}")
    print("\nValidation comparison completed successfully.")


if __name__ == "__main__":
    main()

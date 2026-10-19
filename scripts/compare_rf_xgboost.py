
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"

rf_path = REPORT_DIR / "random_forest_v2_validation_predictions.csv"
xgb_path = REPORT_DIR / "xgboost_validation_predictions.csv"

rf = pd.read_csv(
    rf_path,
    dtype={"StockCode": str},
    parse_dates=["Date"],
)

xgb = pd.read_csv(
    xgb_path,
    dtype={"StockCode": str},
    parse_dates=["Date"],
)

keys = ["StockCode", "Date"]

if rf.duplicated(keys).any() or xgb.duplicated(keys).any():
    raise ValueError("Duplicate product-date keys found.")

rf = rf.rename(columns={"demand": "actual_rf"})
xgb = xgb.rename(columns={"demand": "actual_xgb"})

comparison = rf.merge(
    xgb,
    on=keys,
    how="inner",
    validate="one_to_one",
)

if len(comparison) != len(rf) or len(comparison) != len(xgb):
    raise ValueError("Prediction files do not have identical product-date rows.")

if not np.allclose(
    comparison["actual_rf"],
    comparison["actual_xgb"],
):
    raise ValueError("Actual demand differs between models.")

actual = comparison["actual_rf"].to_numpy(dtype=float)

models = {
    "Zero baseline": np.zeros(len(comparison)),
    "Seasonal naive": comparison["lag_7"].to_numpy(dtype=float)
    if "lag_7" in comparison.columns
    else None,
    "Random Forest V2": comparison["prediction_v2"].to_numpy(dtype=float),
    "XGBoost": comparison["xgboost_prediction"].to_numpy(dtype=float),
}

# The RF file does not contain lag_7, so use the existing
# seasonal-naive predictions from the XGBoost file.
models["Seasonal naive"] = comparison[
    "seasonal_naive_prediction"
].to_numpy(dtype=float)

positive = actual > 0
results = []

for name, prediction in models.items():
    if prediction is None:
        continue

    prediction = np.maximum(prediction, 0)
    errors = actual - prediction
    absolute_errors = np.abs(errors)

    results.append({
        "Model": name,
        "MAE": absolute_errors.mean(),
        "RMSE": np.sqrt(np.mean(errors ** 2)),
        "WAPE_percent": (
            100 * absolute_errors.sum() / actual.sum()
            if actual.sum() else np.nan
        ),
        "Positive_day_MAE": absolute_errors[positive].mean(),
    })

results_df = pd.DataFrame(results).sort_values("MAE")

output_path = REPORT_DIR / "rf_xgboost_validation_comparison.csv"
results_df.to_csv(output_path, index=False)

print(f"Common validation rows: {len(comparison):,}")
print(f"Products: {comparison['StockCode'].nunique():,}")
print(f"Positive-demand rows: {positive.sum():,}")
print("\nControlled validation comparison")
print(results_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"\nSaved: {output_path}")

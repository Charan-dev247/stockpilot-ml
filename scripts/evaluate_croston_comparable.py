
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "regular_daily_demand.csv"
REPORT_DIR = ROOT / "reports"

RF_PATH = REPORT_DIR / "random_forest_v2_validation_predictions.csv"
XGB_PATH = REPORT_DIR / "xgboost_validation_predictions.csv"

TRAIN_START = "2009-12-01"
TRAIN_END = "2011-05-01"
VAL_START = "2011-05-02"
VAL_END = "2011-08-20"

N_PRODUCTS = 500
RANDOM_STATE = 42
CROSTON_ALPHA = 0.10

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Croston SBA forecasting function
# --------------------------------------------------

def croston_sba_forecast(series, alpha=0.10):
    """
    Estimate average intermittent demand using Croston's
    method with the Syntetos-Boylan Approximation correction.

    Only nonzero training observations update demand size
    and the interval between nonzero demands.
    """
    values = np.asarray(series, dtype=float)
    nonzero_positions = np.flatnonzero(values > 0)

    if len(nonzero_positions) == 0:
        return 0.0

    first_pos = nonzero_positions[0]

    # Initialize demand size and interval.
    estimated_size = values[first_pos]
    estimated_interval = float(first_pos + 1)
    previous_pos = first_pos

    # Update estimates when another nonzero demand occurs.
    for pos in nonzero_positions[1:]:
        interval = float(pos - previous_pos)

        estimated_size = (
            alpha * values[pos]
            + (1 - alpha) * estimated_size
        )

        estimated_interval = (
            alpha * interval
            + (1 - alpha) * estimated_interval
        )

        previous_pos = pos

    # SBA bias correction.
    forecast = (
        (1 - alpha / 2)
        * estimated_size
        / estimated_interval
    )

    return max(0.0, float(forecast))


# --------------------------------------------------
# 3. Metrics
# --------------------------------------------------

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    errors = actual - predicted
    absolute_errors = np.abs(errors)
    positive_mask = actual > 0

    return {
        "MAE": absolute_errors.mean(),
        "RMSE": np.sqrt(np.mean(errors ** 2)),
        "WAPE_percent": (
            100 * absolute_errors.sum() / actual.sum()
            if actual.sum() > 0
            else np.nan
        ),
        "Positive_day_MAE": (
            absolute_errors[positive_mask].mean()
            if positive_mask.any()
            else np.nan
        ),
        "Positive_days": int(positive_mask.sum()),
    }


# --------------------------------------------------
# 4. Load original demand data
# --------------------------------------------------

df = pd.read_csv(
    DATA_PATH,
    dtype={"StockCode": str},
)

df.columns = df.columns.str.strip()

column_lookup = {
    col.lower().replace(" ", "_"): col
    for col in df.columns
}


def find_column(candidates, description):
    for candidate in candidates:
        if candidate in column_lookup:
            return column_lookup[candidate]

    raise ValueError(
        f"Cannot find {description}. "
        f"Available columns: {df.columns.tolist()}"
    )


product_col = find_column(
    ["stockcode", "stock_code", "product", "product_id"],
    "product column",
)

date_col = find_column(
    ["date", "invoicedate", "invoice_date"],
    "date column",
)

demand_col = find_column(
    ["demand", "quantity", "daily_demand"],
    "demand column",
)

df = df.rename(
    columns={
        product_col: "StockCode",
        date_col: "Date",
        demand_col: "demand",
    }
)

df["Date"] = pd.to_datetime(df["Date"])
df["demand"] = pd.to_numeric(
    df["demand"],
    errors="raise",
)

if df[["StockCode", "Date", "demand"]].isna().any().any():
    raise ValueError("Missing product, date, or demand values found.")

if (df["demand"] < 0).any():
    raise ValueError(
        "Negative demand found in regular_daily_demand.csv."
    )

df = df[["StockCode", "Date", "demand"]].copy()
df = df.sort_values(["StockCode", "Date"])

if df.duplicated(["StockCode", "Date"]).any():
    raise ValueError("Duplicate product-date records found.")

train_start = pd.Timestamp(TRAIN_START)
train_end = pd.Timestamp(TRAIN_END)
val_start = pd.Timestamp(VAL_START)
val_end = pd.Timestamp(VAL_END)


# --------------------------------------------------
# 5. Select exactly the same 500-product sample
# --------------------------------------------------

train_products = np.sort(
    df.loc[df["Date"] <= train_end, "StockCode"].unique()
)

if len(train_products) < N_PRODUCTS:
    raise ValueError(
        f"Only {len(train_products)} training products available."
    )

rng = np.random.default_rng(RANDOM_STATE)

selected_products = rng.choice(
    train_products,
    size=N_PRODUCTS,
    replace=False,
)

print("Croston SBA comparable evaluation")
print("-" * 60)
print(f"Available training products: {len(train_products):,}")
print(f"Selected products: {len(selected_products):,}")


# --------------------------------------------------
# 6. Build a complete daily calendar
# --------------------------------------------------

df = df[df["StockCode"].isin(selected_products)].copy()

all_dates = pd.date_range(
    start=TRAIN_START,
    end=VAL_END,
    freq="D",
)

full_index = pd.MultiIndex.from_product(
    [selected_products, all_dates],
    names=["StockCode", "Date"],
)

panel = (
    df.set_index(["StockCode", "Date"])["demand"]
    .reindex(full_index, fill_value=0)
    .rename("demand")
    .reset_index()
)

panel = panel.sort_values(
    ["StockCode", "Date"]
).reset_index(drop=True)

expected_rows = len(selected_products) * len(all_dates)

if len(panel) != expected_rows:
    raise ValueError(
        f"Expected {expected_rows} calendar rows, got {len(panel)}."
    )

train = panel[
    panel["Date"].between(train_start, train_end)
].copy()

validation = panel[
    panel["Date"].between(val_start, val_end)
].copy()

print(f"Calendar dates: {len(all_dates):,}")
print(f"Training rows: {len(train):,}")
print(f"Validation rows: {len(validation):,}")


# --------------------------------------------------
# 7. Fit Croston on training data only
# --------------------------------------------------

croston_forecasts = {}

for product, group in train.groupby("StockCode", sort=False):
    group = group.sort_values("Date")

    croston_forecasts[product] = croston_sba_forecast(
        group["demand"].to_numpy(),
        alpha=CROSTON_ALPHA,
    )

validation["croston_prediction"] = (
    validation["StockCode"].map(croston_forecasts)
)

validation["zero_prediction"] = 0.0

# Seasonal naive: demand seven calendar days earlier.
validation["seasonal_naive_prediction"] = (
    panel.groupby("StockCode")["demand"].shift(7)
    .loc[validation.index]
    .to_numpy()
)


# --------------------------------------------------
# 8. Load existing RF and XGBoost predictions
# --------------------------------------------------

for path in [RF_PATH, XGB_PATH]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required prediction file not found: {path}"
        )

rf = pd.read_csv(
    RF_PATH,
    dtype={"StockCode": str},
    parse_dates=["Date"],
)

xgb = pd.read_csv(
    XGB_PATH,
    dtype={"StockCode": str},
    parse_dates=["Date"],
)

keys = ["StockCode", "Date"]

for name, predictions in [
    ("Random Forest V2", rf),
    ("XGBoost", xgb),
]:
    if predictions.duplicated(keys).any():
        raise ValueError(
            f"Duplicate product-date rows in {name} predictions."
        )

# The existing RF and XGBoost files must match this sample.
expected_keys = validation[keys].sort_values(keys).reset_index(drop=True)

for name, predictions in [
    ("Random Forest V2", rf),
    ("XGBoost", xgb),
]:
    actual_keys = (
        predictions[keys]
        .sort_values(keys)
        .reset_index(drop=True)
    )

    if not expected_keys.equals(actual_keys):
        raise ValueError(
            f"{name} predictions do not match the selected "
            "products and validation dates. Check the sample "
            "selection and date ranges before comparing."
        )

# Rename columns so the merge is explicit.
rf = rf.rename(columns={"demand": "rf_actual"})
xgb = xgb.rename(columns={"demand": "xgb_actual"})

comparison = validation.merge(
    rf[keys + ["rf_actual", "prediction_v2"]],
    on=keys,
    how="left",
    validate="one_to_one",
)

comparison = comparison.merge(
    xgb[
        keys
        + ["xgb_actual", "xgboost_prediction"]
    ],
    on=keys,
    how="left",
    validate="one_to_one",
)

# Verify the actual target is identical across all models.
for actual_col in ["rf_actual", "xgb_actual"]:
    if comparison[actual_col].isna().any():
        raise ValueError(f"Missing actual values in {actual_col}.")

    if not np.allclose(
        comparison["demand"],
        comparison[actual_col],
    ):
        raise ValueError(
            f"Actual demand differs in column {actual_col}."
        )

print("\nPrediction alignment checks passed.")
print(f"Common validation rows: {len(comparison):,}")
print(f"Common products: {comparison['StockCode'].nunique():,}")


# --------------------------------------------------
# 9. Compare all models on the same target
# --------------------------------------------------

actual = comparison["demand"].to_numpy(dtype=float)

models = {
    "Zero baseline": comparison["zero_prediction"].to_numpy(),
    "Seasonal naive": comparison[
        "seasonal_naive_prediction"
    ].to_numpy(),
    "Croston SBA": comparison["croston_prediction"].to_numpy(),
    "Random Forest V2": comparison["prediction_v2"].to_numpy(),
    "XGBoost": comparison["xgboost_prediction"].to_numpy(),
}

results = []

for name, prediction in models.items():
    prediction = np.maximum(
        np.asarray(prediction, dtype=float),
        0,
    )

    metrics = calculate_metrics(actual, prediction)

    results.append({
        "Model": name,
        **metrics,
    })

results_df = (
    pd.DataFrame(results)
    .sort_values("MAE")
    .reset_index(drop=True)
)


# --------------------------------------------------
# 10. Save comparison and predictions
# --------------------------------------------------

metrics_path = (
    REPORT_DIR / "all_models_validation_comparison.csv"
)

predictions_path = (
    REPORT_DIR / "all_models_validation_predictions.csv"
)

results_df.to_csv(metrics_path, index=False)

prediction_columns = [
    "StockCode",
    "Date",
    "demand",
    "zero_prediction",
    "seasonal_naive_prediction",
    "croston_prediction",
    "prediction_v2",
    "xgboost_prediction",
]

comparison[prediction_columns].to_csv(
    predictions_path,
    index=False,
)

print("\nFINAL VALIDATION COMPARISON")
print("=" * 90)

print(
    results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

print("\nSaved reports:")
print(metrics_path)
print(predictions_path)

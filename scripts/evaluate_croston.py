
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "comparable_net_daily_demand.csv"
REPORT_DIR = ROOT / "reports"

TRAIN_START = "2009-12-01"
TRAIN_END = "2011-05-01"
VAL_START = "2011-05-02"
VAL_END = "2011-08-20"

N_PRODUCTS = 500
RANDOM_STATE = 42
CROSTON_ALPHA = 0.10

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load and validate the dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH, dtype={"StockCode": str}, low_memory=False)
df.columns = df.columns.str.strip()

# Handle common column-name variations.
column_lookup = {col.lower().replace(" ", "_"): col for col in df.columns}

def find_column(candidates, description):
    for candidate in candidates:
        if candidate in column_lookup:
            return column_lookup[candidate]
    raise ValueError(
        f"Could not identify {description}. "
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

target_col = find_column(
    ["demand_net", "net_demand", "demand", "quantity", "net_quantity"],
    "net-demand target column",
)

df = df.rename(
    columns={
        product_col: "StockCode",
        date_col: "Date",
        target_col: "demand",
    }
)

df["Date"] = pd.to_datetime(df["Date"])
df["demand"] = pd.to_numeric(df["demand"], errors="raise").clip(lower=0)

df = df[["StockCode", "Date", "demand"]].copy()
df = df.sort_values(["StockCode", "Date"])

if df.duplicated(["StockCode", "Date"]).any():
    raise ValueError(
        "Duplicate product-date rows found. "
        "Resolve duplicates before evaluating Croston."
    )

# --------------------------------------------------
# 3. Select the same 500-product sample
# --------------------------------------------------

products = np.sort(df["StockCode"].unique())
products = np.sort(df["StockCode"].dropna().unique())

if len(products) < N_PRODUCTS:
    raise ValueError(f"Only {len(products)} products are available.")

rng = np.random.RandomState(RANDOM_STATE)
selected_products = rng.choice(
    products, size=N_PRODUCTS, replace=False
)

df = df[df["StockCode"].isin(selected_products)].copy()

# Construct a complete daily calendar for each selected product.
# Missing dates are treated as zero recorded demand, consistent
# with the project's existing calendar assumption.
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

train_start = pd.Timestamp(TRAIN_START)
train_end = pd.Timestamp(TRAIN_END)
val_start = pd.Timestamp(VAL_START)
val_end = pd.Timestamp(VAL_END)

train = panel[
    panel["Date"].between(train_start, train_end)
].copy()

validation = panel[
    panel["Date"].between(val_start, val_end)
].copy()

print("Croston evaluation")
print("-" * 50)
print(f"Products: {len(selected_products):,}")
print(f"Training rows: {len(train):,}")
print(f"Validation rows: {len(validation):,}")
print(f"Training: {TRAIN_START} to {TRAIN_END}")
print(f"Validation: {VAL_START} to {VAL_END}")
print(f"Target column used: {target_col}")


# --------------------------------------------------
# 4. Croston's method with SBA correction
# --------------------------------------------------

def croston_sba_forecast(series, alpha=0.10):
    """
    Calculate a constant one-step-ahead forecast using
    Croston's method with the Syntetos-Boylan Approximation.

    Demand size and the interval between nonzero demands
    are updated only when a nonzero demand occurs.
    """
    values = np.asarray(series, dtype=float)
    nonzero_positions = np.flatnonzero(values > 0)

    if len(nonzero_positions) == 0:
        return 0.0

    first_pos = nonzero_positions[0]

    # Initialize from the first observed nonzero demand.
    z = values[first_pos]
    p = float(first_pos + 1)
    previous_pos = first_pos

    for pos in nonzero_positions[1:]:
        interval = float(pos - previous_pos)

        z = alpha * values[pos] + (1 - alpha) * z
        p = alpha * interval + (1 - alpha) * p

        previous_pos = pos

    # SBA bias correction.
    forecast = (1 - alpha / 2) * (z / p)

    return max(0.0, float(forecast))


# --------------------------------------------------
# 5. Generate validation predictions
# --------------------------------------------------

train_by_product = {
    product: group.sort_values("Date")
    for product, group in train.groupby("StockCode")
}

validation = validation.sort_values(["StockCode", "Date"]).copy()

croston_forecasts = {}

for product, group in train_by_product.items():
    croston_forecasts[product] = croston_sba_forecast(
        group["demand"].to_numpy(),
        alpha=CROSTON_ALPHA,
    )

validation["croston_prediction"] = (
    validation["StockCode"].map(croston_forecasts)
)

# Zero baseline
validation["zero_prediction"] = 0.0

# Seasonal naive: demand from seven days earlier.
# Uses observed previous-day history, as in the project's
# existing one-step-ahead evaluation.
panel["seasonal_naive_prediction"] = (
    panel.groupby("StockCode")["demand"].shift(7)
)

validation = validation.merge(
    panel[["StockCode", "Date", "seasonal_naive_prediction"]],
    on=["StockCode", "Date"],
    how="left",
    validate="one_to_one",
)

validation["seasonal_naive_prediction"] = (
    validation["seasonal_naive_prediction"].fillna(0)
)


# --------------------------------------------------
# 6. Evaluate models
# --------------------------------------------------

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    errors = actual - predicted
    absolute_errors = np.abs(errors)

    positive_mask = actual > 0

    return {
        "MAE": absolute_errors.mean(),
        "RMSE": np.sqrt(np.mean(errors**2)),
        "WAPE_percent": (
            absolute_errors.sum() / actual.sum() * 100
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


models = {
    "Zero baseline": "zero_prediction",
    "Croston SBA": "croston_prediction",
    "Seasonal naive (lag 7)": "seasonal_naive_prediction",
}

actual = validation["demand"].to_numpy()
results = []

for model_name, prediction_col in models.items():
    metrics = calculate_metrics(
        actual,
        validation[prediction_col].to_numpy(),
    )

    results.append({
        "Model": model_name,
        **metrics,
    })

results_df = pd.DataFrame(results).sort_values("MAE")

print("\nValidation results")
print("=" * 80)
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# --------------------------------------------------
# 7. Save reports
# --------------------------------------------------

metrics_path = REPORT_DIR / "croston_validation_comparison.csv"
predictions_path = REPORT_DIR / "croston_validation_predictions.csv"

results_df.to_csv(metrics_path, index=False)

validation[
    [
        "StockCode",
        "Date",
        "demand",
        "zero_prediction",
        "croston_prediction",
        "seasonal_naive_prediction",
    ]
].to_csv(predictions_path, index=False)

print("\nSaved reports:")
print(metrics_path)
print(predictions_path)

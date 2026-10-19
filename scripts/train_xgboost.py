
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBRegressor

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "regular_daily_demand.csv"
REPORT_DIR = ROOT / "reports"

TRAIN_START = "2009-12-01"
TRAIN_END = "2011-05-01"
VAL_START = "2011-05-02"
VAL_END = "2011-08-20"

N_PRODUCTS = 500
RANDOM_STATE = 42

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load and validate data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH, dtype={"StockCode": str})
df.columns = df.columns.str.strip()

lookup = {
    col.lower().replace(" ", "_"): col
    for col in df.columns
}


def get_column(candidates, description):
    for name in candidates:
        if name in lookup:
            return lookup[name]

    raise ValueError(
        f"Cannot find {description}. "
        f"Available columns: {df.columns.tolist()}"
    )


product_col = get_column(
    ["stockcode", "stock_code", "product", "product_id"],
    "product column",
)

date_col = get_column(
    ["date", "invoicedate", "invoice_date"],
    "date column",
)

demand_col = get_column(
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

df["Date"] = pd.to_datetime(df["Date"], errors="raise")
df["demand"] = pd.to_numeric(df["demand"], errors="raise")

if df["StockCode"].isna().any():
    raise ValueError("Missing product codes found.")

if df["demand"].isna().any():
    raise ValueError("Missing demand values found.")

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
# 3. Select 500 products from training cohort
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

print(f"Available training products: {len(train_products):,}")
print(f"Selected products: {len(selected_products):,}")


# --------------------------------------------------
# 4. Create complete daily product calendar
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

df = (
    df.set_index(["StockCode", "Date"])["demand"]
    .reindex(full_index, fill_value=0)
    .rename("demand")
    .reset_index()
)

df = df.sort_values(
    ["StockCode", "Date"]
).reset_index(drop=True)

expected_rows = len(selected_products) * len(all_dates)

if len(df) != expected_rows:
    raise ValueError(
        f"Expected {expected_rows} calendar rows, got {len(df)}."
    )

print(f"Calendar dates: {len(all_dates):,}")
print(f"Complete calendar rows: {len(df):,}")


# --------------------------------------------------
# 5. Engineer past-only features
# --------------------------------------------------

grouped = df.groupby("StockCode", sort=False)["demand"]

df["lag_1"] = grouped.shift(1)
df["lag_7"] = grouped.shift(7)
df["lag_14"] = grouped.shift(14)

df["rolling_mean_7"] = grouped.transform(
    lambda s: s.shift(1).rolling(
        window=7, min_periods=1
    ).mean()
)

df["rolling_mean_28"] = grouped.transform(
    lambda s: s.shift(1).rolling(
        window=28, min_periods=1
    ).mean()
)

df["rolling_std_7"] = grouped.transform(
    lambda s: s.shift(1).rolling(
        window=7, min_periods=2
    ).std()
)

df["active_days_28"] = grouped.transform(
    lambda s: s.shift(1).gt(0).rolling(
        window=28, min_periods=1
    ).sum()
)

df["day_of_week"] = df["Date"].dt.dayofweek
df["month"] = df["Date"].dt.month

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

df[FEATURES] = (
    df[FEATURES]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)


# --------------------------------------------------
# 6. Chronological train/validation split
# --------------------------------------------------

train = df[
    df["Date"].between(train_start, train_end)
].copy()

validation = df[
    df["Date"].between(val_start, val_end)
].copy()

X_train = train[FEATURES]
y_train = train["demand"]

X_val = validation[FEATURES]
y_val = validation["demand"]

print("\nXGBoost demand forecasting")
print("-" * 55)
print(f"XGBoost version: {xgboost.__version__}")
print(f"Training rows: {len(train):,}")
print(f"Validation rows: {len(validation):,}")
print(f"Features: {len(FEATURES)}")
print(f"Training: {TRAIN_START} to {TRAIN_END}")
print(f"Validation: {VAL_START} to {VAL_END}")


# --------------------------------------------------
# 7. Train XGBoost
# --------------------------------------------------

model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=5.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method="hist",
)

model.fit(X_train, y_train)

validation["xgboost_prediction"] = np.maximum(
    model.predict(X_val), 0
)

# Zero baseline.
validation["zero_prediction"] = 0.0

# Seven-calendar-day seasonal naive baseline.
validation["seasonal_naive_prediction"] = validation["lag_7"]


# --------------------------------------------------
# 8. Evaluate models on identical validation rows
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
            if actual.sum() > 0 else np.nan
        ),
        "Positive_day_MAE": (
            absolute_errors[positive_mask].mean()
            if positive_mask.any() else np.nan
        ),
        "Positive_days": int(positive_mask.sum()),
    }


results = []

for name, prediction_col in [
    ("Zero baseline", "zero_prediction"),
    ("Seasonal naive (lag 7)", "seasonal_naive_prediction"),
    ("XGBoost", "xgboost_prediction"),
]:
    results.append(
        {
            "Model": name,
            **calculate_metrics(
                validation["demand"],
                validation[prediction_col],
            ),
        }
    )

results_df = pd.DataFrame(results).sort_values("MAE")

print("\nValidation results")
print("=" * 85)
print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# --------------------------------------------------
# 9. Feature importance
# --------------------------------------------------

importance = pd.DataFrame(
    {
        "Feature": FEATURES,
        "Importance": model.feature_importances_,
    }
).sort_values("Importance", ascending=False)


# --------------------------------------------------
# 10. Save reports
# --------------------------------------------------

metrics_path = REPORT_DIR / "xgboost_validation_metrics.csv"
predictions_path = REPORT_DIR / "xgboost_validation_predictions.csv"
importance_path = REPORT_DIR / "xgboost_feature_importance.csv"

results_df.to_csv(metrics_path, index=False)

validation[
    [
        "StockCode",
        "Date",
        "demand",
        "zero_prediction",
        "seasonal_naive_prediction",
        "xgboost_prediction",
    ]
].to_csv(predictions_path, index=False)

importance.to_csv(importance_path, index=False)

print("\nFeature importance")
print(
    importance.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print("\nSaved reports:")
print(metrics_path)
print(predictions_path)
print(importance_path)

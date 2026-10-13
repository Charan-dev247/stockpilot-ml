
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data" / "processed" / "regular_daily_demand.csv"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    data = pd.read_csv(
        DATA_FILE,
        parse_dates=["Date"],
        dtype={"StockCode": "string"},
    )

    print("=" * 65)
    print("STOCKPILOT ML - DEMAND ANOMALY ANALYSIS")
    print("=" * 65)

    # 1. Find the largest product-day observations.
    print("\n--- Top 20 Product-Days by Recorded Demand ---")

    top_days = (
        data.loc[data["demand"] > 0]
        .nlargest(20, "demand")
        [["Date", "StockCode", "demand", "transaction_lines"]]
    )

    print(top_days.to_string(index=False))

    top_days.to_csv(
        REPORT_DIR / "extreme_product_days.csv",
        index=False,
    )

    # 2. Calculate product-level demand statistics.
    print("\n--- Product-Level Demand Statistics ---")

    product_stats = (
        data.groupby("StockCode")
        .agg(
            total_demand=("demand", "sum"),
            max_daily_demand=("demand", "max"),
            active_days=("demand", lambda x: (x > 0).sum()),
            calendar_days=("demand", "size"),
        )
    )

    product_stats["active_day_percentage"] = (
        product_stats["active_days"]
        / product_stats["calendar_days"]
        * 100
    )

    product_stats["max_day_share_pct"] = (
        product_stats["max_daily_demand"]
        / product_stats["total_demand"].replace(0, pd.NA)
        * 100
    )

    print("\nProducts whose largest day accounts for most of their total:")
    concentrated = (
        product_stats.loc[
            (product_stats["total_demand"] > 0)
            & (product_stats["max_day_share_pct"] >= 50)
        ]
        .sort_values("max_day_share_pct", ascending=False)
        .head(20)
    )

    print(concentrated.round(2).to_string())

    concentrated.to_csv(
        REPORT_DIR / "highly_concentrated_products.csv"
    )

    # 3. Summarize the demand distribution.
    print("\n--- Positive Daily Demand Percentiles ---")

    positive_demand = data.loc[data["demand"] > 0, "demand"]

    print(
        positive_demand.quantile(
            [0.50, 0.75, 0.90, 0.95, 0.99, 0.999]
        ).to_string()
    )

    # 4. Flag extreme product-days using a percentile threshold.
    threshold = positive_demand.quantile(0.99)

    extreme = data.loc[data["demand"] > threshold].copy()

    print(f"\n99th-percentile threshold: {threshold:,.2f}")
    print(f"Product-days above threshold: {len(extreme):,}")

    extreme.to_csv(
        REPORT_DIR / "extreme_demand_flags.csv",
        index=False,
    )

    print("\n--- Important interpretation ---")
    print(
        "These observations are flags for investigation, not confirmed errors."
    )
    print(
        "Do not delete, cap, or transform them automatically."
    )

    print("\nReports saved:")
    print("reports/extreme_product_days.csv")
    print("reports/highly_concentrated_products.csv")
    print("reports/extreme_demand_flags.csv")

    print("\nAnomaly analysis completed.")


if __name__ == "__main__":
    main()

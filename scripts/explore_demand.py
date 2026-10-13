
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data" / "processed" / "regular_daily_demand.csv"
FIGURES_DIR = ROOT / "reports" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def main():
    # --------------------------------------------------
    # 2. Load the validated dataset
    # --------------------------------------------------

    data = pd.read_csv(
        DATA_FILE,
        parse_dates=["Date"],
        dtype={"StockCode": "string"},
    )

    print("=" * 65)
    print("STOCKPILOT ML - EXPLORATORY DATA ANALYSIS")
    print("=" * 65)

    print(f"\nTotal product-day records: {len(data):,}")
    print(f"Unique products: {data['StockCode'].nunique():,}")
    print(f"Date range: {data['Date'].min().date()} to "
          f"{data['Date'].max().date()}")

    # --------------------------------------------------
    # 3. Overall demand statistics
    # --------------------------------------------------

    print("\n--- Overall Demand Statistics ---")
    print(data["demand"].describe().round(2))

    zero_percentage = (data["demand"] == 0).mean() * 100
    positive_percentage = (data["demand"] > 0).mean() * 100

    print(f"\nZero-demand days: {zero_percentage:.2f}%")
    print(f"Positive-demand days: {positive_percentage:.2f}%")

    # --------------------------------------------------
    # 4. Top 10 products by total recorded demand
    # --------------------------------------------------

    product_summary = (
        data.groupby("StockCode", as_index=False)
        .agg(
            total_demand=("demand", "sum"),
            active_days=("demand", lambda x: (x > 0).sum()),
            calendar_days=("demand", "size"),
            average_daily_demand=("demand", "mean"),
        )
        .sort_values("total_demand", ascending=False)
    )

    top_10 = product_summary.head(10)

    print("\n--- Top 10 Products by Total Recorded Demand ---")
    print(top_10.round(2).to_string(index=False))

    top_10.to_csv(
        ROOT / "reports" / "top_products.csv",
        index=False,
    )

    # --------------------------------------------------
    # 5. Plot top 10 products
    # --------------------------------------------------

    plt.figure(figsize=(12, 6))

    sns.barplot(
        data=top_10,
        x="total_demand",
        y="StockCode",
    )

    plt.title("Top 10 Products by Total Recorded Demand")
    plt.xlabel("Total Recorded Quantity Sold")
    plt.ylabel("Stock Code")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "top_10_products.png",
        dpi=150,
    )
    plt.close()

    # --------------------------------------------------
    # 6. Daily total demand trend
    # --------------------------------------------------

    daily_total = (
        data.groupby("Date", as_index=False)["demand"]
        .sum()
        .sort_values("Date")
    )

    plt.figure(figsize=(14, 6))

    sns.lineplot(
        data=daily_total,
        x="Date",
        y="demand",
    )

    plt.title("Total Recorded Demand Over Time")
    plt.xlabel("Date")
    plt.ylabel("Total Recorded Quantity Sold")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "daily_demand_trend.png",
        dpi=150,
    )
    plt.close()

    # --------------------------------------------------
    # 7. Weekday demand pattern
    # --------------------------------------------------

    data["Weekday"] = data["Date"].dt.day_name()

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    weekday_summary = (
        data.groupby("Weekday")["demand"]
        .mean()
        .reindex(weekday_order)
    )

    print("\n--- Average Demand per Product-Day by Weekday ---")
    print(weekday_summary.round(2).to_string())

    plt.figure(figsize=(10, 5))

    sns.barplot(
        x=weekday_summary.index,
        y=weekday_summary.values,
    )

    plt.title("Average Demand per Product-Day by Weekday")
    plt.xlabel("Weekday")
    plt.ylabel("Average Recorded Demand")
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "weekday_demand.png",
        dpi=150,
    )
    plt.close()

    # --------------------------------------------------
    # 8. Sparse-demand products
    # --------------------------------------------------

    product_summary["active_day_percentage"] = (
        product_summary["active_days"]
        / product_summary["calendar_days"]
        * 100
    )

    sparse_products = product_summary.sort_values(
        "active_day_percentage"
    ).head(20)

    print("\n--- 20 Products with Lowest Active-Day Percentage ---")
    print(
        sparse_products[
            [
                "StockCode",
                "total_demand",
                "active_days",
                "calendar_days",
                "active_day_percentage",
            ]
        ].round(2).to_string(index=False)
    )

    sparse_products.to_csv(
        ROOT / "reports" / "sparse_products.csv",
        index=False,
    )

    # --------------------------------------------------
    # 9. Save the overall daily demand summary
    # --------------------------------------------------

    daily_total.to_csv(
        ROOT / "reports" / "daily_total_demand.csv",
        index=False,
    )

    print("\n--- EDA Outputs ---")
    print("reports/top_products.csv")
    print("reports/sparse_products.csv")
    print("reports/daily_total_demand.csv")
    print("reports/figures/top_10_products.png")
    print("reports/figures/daily_demand_trend.png")
    print("reports/figures/weekday_demand.png")

    print("\nEDA completed successfully.")


if __name__ == "__main__":
    main()


from pathlib import Path

import pandas as pd
from fastapi import HTTPException


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RECOMMENDATIONS_PATH = (
    PROJECT_ROOT / "reports" / "inventory_recommendations.csv"
)


def load_inventory() -> pd.DataFrame:
    """Load the latest inventory recommendations from the project reports."""
    if not RECOMMENDATIONS_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Inventory recommendations are unavailable. "
                   "Run scripts/inventory_planning.py first.",
        )

    try:
        df = pd.read_csv(RECOMMENDATIONS_PATH)
    except (OSError, pd.errors.ParserError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not read the inventory recommendations report.",
        ) from exc

    required_columns = {
        "StockCode",
        "average_daily_demand",
        "safety_stock",
        "reorder_point",
        "order_up_to_level",
        "on_hand",
        "on_order",
        "inventory_position",
        "recommended_order_qty",
        "recommendation",
        "actionable_order_qty",
        "activity_status",
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise HTTPException(
            status_code=500,
            detail=f"Inventory report is missing columns: "
                   f"{sorted(missing_columns)}",
        )

    df["StockCode"] = df["StockCode"].astype(str)
    df = df.replace({float("inf"): None, float("-inf"): None})
    df = df.where(pd.notna(df), None)

    return df


def get_inventory_summary(df: pd.DataFrame) -> dict:
    """Calculate dashboard KPIs from the recommendation report."""
    recommendation_counts = (
        df["recommendation"].value_counts().to_dict()
    )

    return {
        "total_products": int(len(df)),
        "reorder_now": int(
            recommendation_counts.get("Reorder now", 0)
        ),
        "monitor": int(
            recommendation_counts.get("Monitor", 0)
        ),
        "review_inactive": int(
            recommendation_counts.get("Review inactive product", 0)
        ),
        "no_order_needed": int(
            recommendation_counts.get("No order needed", 0)
        ),
        "total_proposed_units": int(
            df["recommended_order_qty"].sum()
        ),
        "immediate_replenishment_units": int(
            df["actionable_order_qty"].sum()
        ),
        "inventory_data_type": (
            str(df["inventory_data_type"].iloc[0])
            if "inventory_data_type" in df.columns and not df.empty
            else "Unknown"
        ),
        "demand_target": (
            str(df["demand_target"].iloc[0])
            if "demand_target" in df.columns and not df.empty
            else "Unknown"
        ),
        "test_data_used_for_planning": (
            bool(df["test_data_used_for_planning"].any())
            if "test_data_used_for_planning" in df.columns
            else None
        ),
    }


def query_inventory(
    df: pd.DataFrame,
    search: str | None = None,
    recommendation: str | None = None,
    activity_status: str | None = None,
    sort_by: str = "recommended_order_qty",
    order: str = "desc",
    page: int = 1,
    page_size: int = 25,
) -> dict:
    """Search, filter, sort, and paginate inventory recommendations."""
    allowed_sort_columns = {
        "StockCode",
        "average_daily_demand",
        "safety_stock",
        "reorder_point",
        "recommended_order_qty",
        "actionable_order_qty",
        "days_since_last_sale",
        "total_demand",
    }

    if sort_by not in allowed_sort_columns:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported sort_by. Choose from: "
                   f"{sorted(allowed_sort_columns)}",
        )

    if order not in {"asc", "desc"}:
        raise HTTPException(
            status_code=400,
            detail="order must be 'asc' or 'desc'.",
        )

    result = df.copy()

    if search:
        result = result[
            result["StockCode"].str.contains(
                search, case=False, regex=False, na=False
            )
        ]

    if recommendation:
        result = result[
            result["recommendation"].str.casefold()
            == recommendation.casefold()
        ]

    if activity_status:
        result = result[
            result["activity_status"].str.casefold()
            == activity_status.casefold()
        ]

    result = result.sort_values(
        sort_by,
        ascending=(order == "asc"),
        na_position="last",
    )

    total = int(len(result))
    start = (page - 1) * page_size
    records = result.iloc[start:start + page_size]

    return {
        "items": records.to_dict(orient="records"),
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
    }


def get_product_detail(df: pd.DataFrame, stock_code: str) -> dict:
    """Return the recommendation record for one stock code."""
    product = df[df["StockCode"] == stock_code]

    if product.empty:
        raise HTTPException(
            status_code=404,
            detail=f"StockCode '{stock_code}' was not found.",
        )

    return product.iloc[0].to_dict()


from fastapi import FastAPI, Query

from app.services.inventory_service import (
    get_inventory_summary,
    get_product_detail,
    load_inventory,
    query_inventory,
)

from app.services.evaluation_service import (
    get_cost_evaluation,
    get_inventory_evaluation,
    get_model_evaluation,
    get_robustness_comparison,
)


app = FastAPI(
    title="StockPilot ML API",
    description=(
        "Inventory planning and replenishment recommendations "
        "based on historical retail sales."
    ),
    version="1.0.0",
)


@app.get("/health", tags=["System"])
def health_check():
    """Check whether the API is running."""
    return {"status": "healthy", "service": "StockPilot ML API"}


@app.get("/api/summary", tags=["Inventory"])
def inventory_summary():
    """Return inventory dashboard KPIs."""
    df = load_inventory()
    return get_inventory_summary(df)


@app.get("/api/inventory", tags=["Inventory"])
def list_inventory(
    search: str | None = Query(default=None, max_length=100),
    recommendation: str | None = Query(default=None, max_length=50),
    activity_status: str | None = Query(default=None, max_length=50),
    sort_by: str = Query(
        default="recommended_order_qty",
        max_length=50,
    ),
    order: str = Query(default="desc", pattern="^(asc|desc)$"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
):
    """Search and filter inventory recommendations."""
    df = load_inventory()

    return query_inventory(
        df=df,
        search=search,
        recommendation=recommendation,
        activity_status=activity_status,
        sort_by=sort_by,
        order=order,
        page=page,
        page_size=page_size,
    )


@app.get("/api/inventory/{stock_code}", tags=["Inventory"])
def inventory_product_detail(stock_code: str):
    """Return inventory planning details for one product."""
    df = load_inventory()
    return get_product_detail(df, stock_code)


@app.get("/api/evaluation/models", tags=["Evaluation"])
def model_evaluation():
    """Compare forecasting models using saved validation metrics."""
    return get_model_evaluation()


@app.get("/api/evaluation/inventory", tags=["Evaluation"])
def inventory_evaluation(
    period: str = Query(default="validation", pattern="^(validation|test)$"),
):
    """Compare inventory policies on validation or test."""
    return get_inventory_evaluation(period)


@app.get("/api/evaluation/costs", tags=["Evaluation"])
def inventory_cost_evaluation(
    period: str = Query(default="validation", pattern="^(validation|test)$"),
):
    """Return inventory policy cost sensitivity results."""
    return get_cost_evaluation(period)


@app.get("/api/evaluation/robustness", tags=["Evaluation"])
def inventory_robustness():
    """Compare inventory simulation performance between periods."""
    return get_robustness_comparison()


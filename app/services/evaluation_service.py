
from pathlib import Path

import pandas as pd
from fastapi import HTTPException


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"


REPORT_FILES = {
    "models": "all_models_validation_comparison.csv",
    "inventory_validation": "inventory_simulation_validation_summary.csv",
    "inventory_test": "inventory_simulation_test_summary.csv",
    "cost_validation": "inventory_cost_validation_comparison.csv",
    "cost_test": "inventory_cost_test_comparison.csv",
    "robustness": "validation_test_comparison.csv",
}


def load_report(report_key: str) -> list[dict]:
    """Load a known evaluation report and return JSON-safe records."""
    filename = REPORT_FILES.get(report_key)

    if filename is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown evaluation report: {report_key}",
        )

    path = REPORTS_DIR / filename

    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Evaluation report is unavailable: {filename}",
        )

    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read evaluation report: {filename}",
        ) from exc

    # Convert NaN/NaT values to JSON null.
    df = df.astype(object).where(pd.notna(df), None)
    return df.to_dict(orient="records")


def get_model_evaluation() -> dict:
    """Return controlled validation metrics for the evaluated models."""
    models = load_report("models")

    return {
        "evaluation_period": "Validation",
        "target": "Original recorded positive-sales demand",
        "rows_evaluated": 55500,
        "products_evaluated": 500,
        "positive_days": 7966,
        "metrics": models,
        "interpretation": {
            "best_overall_mae": "Zero baseline",
            "best_overall_wape": "Zero baseline",
            "best_rmse": "Random Forest V2",
            "caveat": (
                "No evaluated ML model beats the zero baseline on "
                "overall MAE or WAPE in this comparison. "
                "Positive-day and RMSE performance differ."
            ),
        },
    }


def get_inventory_evaluation(period: str) -> dict:
    """Return simulation results for validation or held-out test."""
    key = f"inventory_{period}"
    rows = load_report(key)

    return {
        "period": period,
        "data_type": "Simulated inventory policy evaluation",
        "fill_rate_target_pct": 85.0,
        "results": rows,
        "warning": (
            "Inventory quantities and costs are simulated. "
            "They are not live warehouse measurements."
        ),
    }


def get_cost_evaluation(period: str) -> dict:
    """Return hypothetical cost sensitivity results."""
    key = f"cost_{period}"
    rows = load_report(key)

    rates = sorted({
        float(row["holding_cost_per_unit_day"])
        for row in rows
        if row.get("holding_cost_per_unit_day") is not None
    })

    return {
        "period": period,
        "cost_type": "Hypothetical simulated cost; not currency",
        "cost_assumptions": {
            "ordering_cost_per_order": 5,
            "lost_unit_penalty": 2,
            "holding_cost_per_unit_day": rates,
        },
        "results": rows,
        "warning": (
            "These cost coefficients are illustrative assumptions. "
            "They are not measured business costs or monetary forecasts."
        ),
    }


def get_robustness_comparison() -> dict:
    """Return validation-versus-test inventory policy changes."""
    return {
        "validation_vs_test": load_report("robustness"),
        "interpretation": (
            "The test period is a held-out evaluation. "
            "Use it to report generalization, not to tune policies."
        ),
    }

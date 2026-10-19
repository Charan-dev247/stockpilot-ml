
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPORTS = Path("reports")
OUTPUT = REPORTS / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

comparison_path = REPORTS / "validation_test_comparison.csv"
cost_path = REPORTS / "validation_test_cost_comparison.csv"

if not comparison_path.exists() or not cost_path.exists():
    raise FileNotFoundError(
        "Run scripts/compare_validation_test.py first."
    )

results = pd.read_csv(comparison_path)
costs = pd.read_csv(cost_path)

# Consistent policy order across plots
policy_order = [
    "baseline_average_demand",
    "safety_stock",
    "activity_aware_safety_stock",
]

results["policy"] = pd.Categorical(
    results["policy"], categories=policy_order, ordered=True
)
results = results.sort_values("policy")

labels = {
    "baseline_average_demand": "Baseline",
    "safety_stock": "Safety stock",
    "activity_aware_safety_stock": "Activity-aware",
}
names = [labels[str(p)] for p in results["policy"]]


def save_bar_chart(values, title, ylabel, filename, validation_values=None):
    fig, ax = plt.subplots(figsize=(9, 5))

    x = range(len(names))
    width = 0.36 if validation_values is not None else 0.65

    if validation_values is not None:
        ax.bar(
            [i - width / 2 for i in x],
            validation_values,
            width=width,
            label="Validation",
        )
        ax.bar(
            [i + width / 2 for i in x],
            values,
            width=width,
            label="Test",
        )
        ax.legend()
    else:
        ax.bar(x, values, width=width)

    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUTPUT / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


save_bar_chart(
    results["unit_fill_rate_pct_test"],
    "Inventory Unit Fill Rate: Validation vs Test",
    "Unit fill rate (%)",
    "fill_rate_validation_vs_test.png",
    results["unit_fill_rate_pct_validation"],
)

save_bar_chart(
    results["lost_units_test"],
    "Lost Units: Validation vs Test",
    "Lost units",
    "lost_units_validation_vs_test.png",
    results["lost_units_validation"],
)

save_bar_chart(
    results["total_stockout_days_test"],
    "Stockout Days: Validation vs Test",
    "Stockout days",
    "stockout_days_validation_vs_test.png",
    results["total_stockout_days_validation"],
)

save_bar_chart(
    results["average_on_hand_per_product_test"],
    "Average On-Hand Inventory: Validation vs Test",
    "Average units per product",
    "average_inventory_validation_vs_test.png",
    results["average_on_hand_per_product_validation"],
)

# Cost sensitivity: one plot for each holding-cost assumption.
for rate in sorted(costs["holding_cost_per_unit_day"].unique()):
    subset = costs[
        costs["holding_cost_per_unit_day"].eq(rate)
    ].copy()

    subset["policy"] = pd.Categorical(
        subset["policy"], categories=policy_order, ordered=True
    )
    subset = subset.sort_values("policy")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(subset))
    width = 0.36

    ax.bar(
        [i - width / 2 for i in x],
        subset["total_simulated_cost_validation"],
        width=width,
        label="Validation",
    )
    ax.bar(
        [i + width / 2 for i in x],
        subset["total_simulated_cost_test"],
        width=width,
        label="Test",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels([labels[str(p)] for p in subset["policy"]])
    ax.set_ylabel("Hypothetical simulated cost")
    ax.set_title(f"Validation vs Test Cost (holding rate = {rate:.3f})")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()

    filename = f"cost_comparison_holding_{rate:.3f}.png"
    fig.savefig(OUTPUT / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)

print(f"Charts saved in: {OUTPUT.resolve()}")
for path in sorted(OUTPUT.glob("*.png")):
    print(path.name)

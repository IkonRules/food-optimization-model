"""Run the deterministic synthetic portfolio example from the repository root."""

from __future__ import annotations

import numpy as np
import pandas as pd

from food_optimizer.demo_data import DEMO_PROFILE, load_demo_bundle
from food_optimizer.optimization_module import (
    calculate_kcal_contribution,
    optimize_daily_intake,
    parse_constraints,
)


RANDOM_SEED = 7
MAX_FOODS = 8


def selected_foods(result: pd.DataFrame, db: pd.DataFrame) -> pd.Series:
    """Return positive food quantities from the first solution, largest first."""
    food_columns = [
        food
        for food in db["Livsmedelsnamn"]
        if food in result.columns and float(result.loc[0, food]) > 1e-3
    ]
    return result.loc[0, food_columns].astype(float).sort_values(ascending=False)


def verify_reported_solution(
    foods: pd.Series,
    db: pd.DataFrame,
    reference_values: dict,
) -> tuple[int, float, tuple[float, float]]:
    """Recalculate the public numeric constraints and raise on a violation."""
    constraints = parse_constraints(reference_values)
    indexed = db.set_index("Livsmedelsnamn")
    checks = 0

    for name, rule in constraints["absolute_constraints"].items():
        column = rule["db_name"]
        if column not in indexed.columns:
            raise RuntimeError(f"Missing model column for {name!r}: {column!r}")
        values = indexed[column].fillna(0)
        total = sum(
            grams * float(values.at[food]) / 100
            for food, grams in foods.items()
        )
        for bound_name in ("min", "max"):
            if bound_name not in rule:
                continue
            limit = float(rule[bound_name])
            tolerance = max(1e-6, abs(limit) * 1e-5)
            valid = total + tolerance >= limit if bound_name == "min" else total <= limit + tolerance
            if not valid:
                raise RuntimeError(
                    f"Reported solution violates {name} {bound_name}: {total} vs {limit}."
                )
            checks += 1

    energy_series = calculate_kcal_contribution(db).set_axis(db["Livsmedelsnamn"])
    energy = sum(grams * float(energy_series.at[food]) / 100 for food, grams in foods.items())
    energy_range = constraints["energy_kcal_range"]
    if energy_range is None or not energy_range[0] - 1e-3 <= energy <= energy_range[1] + 1e-3:
        raise RuntimeError(f"Reported solution violates the model energy range: {energy:.2f} kcal.")
    checks += 1

    macro_sources = {
        "Fett procent": ("Fett, totalt", 9),
        "Kolhydrater procent": ("Kolhydrater, tillgängliga", 4),
        "Protein procent": ("Protein", 4),
    }
    for macro, rule in constraints["macro_percent_constraints"].items():
        column, factor = macro_sources[macro]
        values = indexed[column].fillna(0)
        macro_energy = sum(
            grams * float(values.at[food]) * factor / 100
            for food, grams in foods.items()
        )
        share = 100 * macro_energy / energy
        lower, upper = rule["range"]
        if not lower - 1e-3 <= share <= upper + 1e-3:
            raise RuntimeError(f"Reported solution violates {macro}: {share:.2f}%.")
        checks += 1

    return checks, energy, energy_range


def main() -> None:
    """Load artificial inputs, solve the production MILP, and print a compact report."""
    np.random.seed(RANDOM_SEED)

    bundle = load_demo_bundle(DEMO_PROFILE)
    db = bundle.foods
    reference_values = bundle.reference_values
    result = optimize_daily_intake(
        db,
        reference_values,
        bundle.toxin_columns,
        num_solutions=1,
        prioritize="weight",
        max_foods=MAX_FOODS,
        prefer_sparse=True,
        plate_model_constraints=bundle.plate_model_constraints,
        realism_metadata=bundle.realism_metadata,
        realism_config=bundle.realism_config,
        enforce_realism=True,
    )

    foods = selected_foods(result, db)
    checks, model_energy, energy_range = verify_reported_solution(
        foods,
        db,
        reference_values,
    )

    table = foods.rename("grams").rename_axis("food").reset_index()
    print("Food Optimization Model — Synthetic Portfolio Demonstration")
    print("------------------------------------------------------------")
    print("This example uses artificial data included with the repository.")
    print("It does not use or redistribute the original source datasets.")
    print()
    print("Artificial profile: synthetic adult profile, age 30")
    print(table.to_string(index=False, formatters={"grams": "{:,.1f}".format}))
    print()
    print(f"Selected foods: {len(foods)} / {MAX_FOODS} maximum")
    print(f"Total food weight: {foods.sum():,.1f} g/day")
    print(
        f"Model energy: {model_energy:,.1f} kcal/day "
        f"(allowed {energy_range[0]:,.1f}-{energy_range[1]:,.1f})"
    )
    print(
        "Selected validation values: "
        f"protein {float(result.loc[0, 'Protein']):.1f}, "
        f"synthetic fiber {float(result.loc[0, 'Synthetic Fiber']):.1f}, "
        f"marker A {float(result.loc[0, 'Synthetic contaminant marker A']):.3f}"
    )
    print(f"Validation: PASS ({checks} nutrient, marker, energy, and macro checks)")


if __name__ == "__main__":
    main()

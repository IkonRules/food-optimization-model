"""User-facing entry point for running the food optimizer."""

import pickle
from pathlib import Path

import pandas as pd

from food_optimizer.config import DATA_DIR
from food_optimizer.get_ref_vals_dict import build_reference_values_dict, toxic_columns
from food_optimizer.get_user_input import get_opt_input_parameters, get_ref_val_input_parameters
from food_optimizer.optimization_module import OptimizationInfeasibleError, optimize_daily_intake
from food_optimizer.realism_constraints import (
    default_realism_config,
    load_food_realism_metadata,
)


FOOD_DATABASE_FILENAME = "df_foods_and_toxins_database.csv"
PLATE_MODEL_CONSTRAINTS_FILENAME = "plate_model_constraints.pkl"
REALISM_METADATA_FILENAME = "food_realism_metadata.csv"


def load_foods_and_toxins_database(path: Path | None = None) -> pd.DataFrame:
    """Load the processed food database with toxin columns."""
    path = path or DATA_DIR / FOOD_DATABASE_FILENAME
    return pd.read_csv(path)


def load_plate_model_constraints(path: Path | None = None) -> dict:
    """Load plate-model constraints from pickle."""
    path = path or DATA_DIR / PLATE_MODEL_CONSTRAINTS_FILENAME
    with open(path, "rb") as f:
        return pickle.load(f)


def run_optimizer(
    db: pd.DataFrame | None = None,
    plate_model_constraints: dict | None = None,
    realism_metadata: pd.DataFrame | None = None,
    enforce_realism: bool = True,
    max_pareto_solutions: int = 100,
) -> pd.DataFrame:
    """Run the full interactive food-optimization workflow."""
    db = db if db is not None else load_foods_and_toxins_database()
    plate_model_constraints = (
        plate_model_constraints
        if plate_model_constraints is not None
        else load_plate_model_constraints()
    )
    realism_metadata = (
        realism_metadata
        if realism_metadata is not None
        else load_food_realism_metadata(DATA_DIR / REALISM_METADATA_FILENAME)
    )

    ref_vals_input_parameters = get_ref_val_input_parameters()
    reference_values_dict = build_reference_values_dict(ref_vals_input_parameters, db)

    opt_input_parameters = get_opt_input_parameters()

    recommendations = optimize_daily_intake(
        db,
        reference_values_dict,
        toxic_columns,
        num_solutions=opt_input_parameters["num_solutions"],
        prioritize=opt_input_parameters["prioritize"],
        max_foods=opt_input_parameters["max_foods"],
        prefer_sparse=opt_input_parameters["prefer_sparse"],
        max_pareto_solutions=max_pareto_solutions,
        plate_model_constraints=plate_model_constraints,
        realism_metadata=realism_metadata,
        realism_config=default_realism_config(),
        enforce_realism=enforce_realism,
    )

    return recommendations

if __name__ == "__main__":
    from food_optimizer.display_results import plot_toxin_totals, solution_foods_table

    try:
        recommendations = run_optimizer()
    except OptimizationInfeasibleError as exc:
        print(exc)
    else:
        print(recommendations)
        print(solution_foods_table(recommendations).to_string(index=False))
        plot_toxin_totals(recommendations)

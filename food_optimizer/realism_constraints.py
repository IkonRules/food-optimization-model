"""Realism metadata and MILP constraints for practical daily food recommendations.

The nutrition optimizer can produce mathematically valid but unrealistic corner
solutions, such as very large quantities of spices or raw ingredients. This
module adds a separate realism layer based on food-role metadata.

The intended workflow is:
1. Load or generate ``food_realism_metadata.csv``.
2. Optionally review/edit the metadata manually.
3. Merge metadata into the food database.
4. Pre-filter excluded foods and add portion/count/total constraints to the MILP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pulp

from food_optimizer.config import DATA_DIR


REALISM_METADATA_FILENAME = "food_realism_metadata.csv"


@dataclass(frozen=True)
class RealismConfig:
    """Configuration for the default realism layer.

    Count bounds are applied to the number of selected foods in each count group.
    Total gram bounds are applied to the sum of grams in each realism group.
    Bounds can be relaxed by passing a custom config to the optimizer.
    """

    exclude_flag_column: str = "include_by_default"
    food_name_column: str = "Livsmedelsnamn"
    realism_group_column: str = "realism_group"
    count_group_column: str = "count_group"
    min_if_selected_column: str = "min_if_selected_g"
    max_if_selected_column: str = "max_if_selected_g"
    max_daily_column: str = "max_daily_g"
    allow_excluded_foods: bool = False
    apply_min_if_selected: bool = True
    apply_max_if_selected: bool = True
    apply_count_bounds: bool = True
    apply_group_gram_bounds: bool = True
    count_bounds: dict[str, tuple[int | None, int | None]] = field(default_factory=lambda: {
        # Conservative defaults: enough structure to look like a day of food,
        # but not so strict that the model immediately becomes infeasible.
        "main_protein": (1, 4),
        "staple_carbohydrate": (1, 4),
        "vegetable_fruit": (1, 8),
        "snack_nuts_seeds": (0, 2),
    })
    group_gram_bounds: dict[str, tuple[float | None, float | None]] = field(default_factory=lambda: {
        "seasoning_spice": (0, 10),
        "fat_oil": (0, 50),
        "snack": (0, 60),
        "nuts_seeds": (0, 80),
        "processed_or_organ_protein": (0, 150),
        "sauce_condiment": (0, 80),
        "beverage": (0, 1000),
    })


def default_realism_config() -> RealismConfig:
    """Return the default realism constraint configuration."""
    return RealismConfig()


def load_food_realism_metadata(path: Path | None = None) -> pd.DataFrame:
    """Load the food realism metadata CSV.

    Parameters
    ----------
    path:
        Optional explicit metadata path. Defaults to
        ``DATA_DIR / 'food_realism_metadata.csv'``.
    """
    path = path or DATA_DIR / REALISM_METADATA_FILENAME
    return pd.read_csv(path)


def merge_realism_metadata(
    db: pd.DataFrame,
    metadata: pd.DataFrame,
    food_name_column: str = "Livsmedelsnamn",
) -> pd.DataFrame:
    """Return ``db`` with realism metadata columns merged in by food name."""
    if food_name_column not in db.columns:
        raise KeyError(f"Food database is missing required column: {food_name_column!r}")
    if food_name_column not in metadata.columns:
        raise KeyError(f"Realism metadata is missing required column: {food_name_column!r}")

    metadata = metadata.drop_duplicates(subset=[food_name_column]).copy()
    metadata_columns = [col for col in metadata.columns if col != "Gruppering"]

    merged = db.merge(
        metadata[metadata_columns],
        on=food_name_column,
        how="left",
        validate="one_to_one",
    )

    missing = merged["realism_group"].isna().sum() if "realism_group" in merged.columns else len(merged)
    if missing:
        raise ValueError(
            f"Realism metadata is missing for {missing} foods. Regenerate or review "
            "food_realism_metadata.csv before optimizing with realism constraints."
        )

    return merged


def prepare_realism_database(
    db: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    config: RealismConfig | None = None,
) -> pd.DataFrame:
    """Merge metadata and remove foods excluded by default, unless disabled."""
    config = config or default_realism_config()
    metadata = metadata if metadata is not None else load_food_realism_metadata()
    db_with_metadata = merge_realism_metadata(db, metadata, config.food_name_column)

    if not config.allow_excluded_foods and config.exclude_flag_column in db_with_metadata.columns:
        include_mask = db_with_metadata[config.exclude_flag_column].fillna(True).astype(bool)
        db_with_metadata = db_with_metadata.loc[include_mask].copy()

    return db_with_metadata.reset_index(drop=True)


def needs_binary_use_vars(
    max_foods: int | None = None,
    realism_metadata: pd.DataFrame | None = None,
    realism_config: RealismConfig | None = None,
) -> bool:
    """Return whether binary food-use variables are needed."""
    if max_foods is not None:
        return True

    if realism_metadata is None:
        return False

    config = realism_config or default_realism_config()
    return (
        config.apply_min_if_selected
        or config.apply_max_if_selected
        or config.apply_count_bounds
        or config.apply_group_gram_bounds
    )


def add_realism_constraints(
    model: pulp.LpProblem,
    food_vars: dict[str, pulp.LpVariable],
    use_vars: dict[str, pulp.LpVariable],
    db: pd.DataFrame,
    config: RealismConfig | None = None,
) -> None:
    """Add portion, count, and category-total realism constraints to a MILP model."""
    if not use_vars:
        raise ValueError("Realism constraints require binary use variables.")

    config = config or default_realism_config()
    food_col = config.food_name_column

    db_meta = db.set_index(food_col, drop=False)

    # Per-food portion bounds.
    for food, row in db_meta.iterrows():
        if food not in food_vars or food not in use_vars:
            continue

        if config.apply_max_if_selected:
            upper_candidates = []
            for col in [config.max_if_selected_column, config.max_daily_column]:
                value = row.get(col)
                if pd.notna(value):
                    upper_candidates.append(float(value))
            if upper_candidates:
                upper = min(upper_candidates)
                model += food_vars[food] <= upper * use_vars[food], f"realism_max_g_{_safe_name(food)}"

        if config.apply_min_if_selected:
            lower = row.get(config.min_if_selected_column)
            if pd.notna(lower) and float(lower) > 0:
                model += food_vars[food] >= float(lower) * use_vars[food], f"realism_min_g_{_safe_name(food)}"

    # Count bounds by count group.
    if config.apply_count_bounds and config.count_group_column in db.columns:
        for group, (lower, upper) in config.count_bounds.items():
            foods = db.loc[db[config.count_group_column] == group, food_col].tolist()
            group_use = pulp.lpSum(use_vars[f] for f in foods if f in use_vars)
            if lower is not None:
                model += group_use >= int(lower), f"realism_count_min_{_safe_name(group)}"
            if upper is not None:
                model += group_use <= int(upper), f"realism_count_max_{_safe_name(group)}"

    # Gram total bounds by realism group.
    if config.apply_group_gram_bounds and config.realism_group_column in db.columns:
        for group, (lower, upper) in config.group_gram_bounds.items():
            foods = db.loc[db[config.realism_group_column] == group, food_col].tolist()
            group_weight = pulp.lpSum(food_vars[f] for f in foods if f in food_vars)
            if lower is not None:
                model += group_weight >= float(lower), f"realism_grams_min_{_safe_name(group)}"
            if upper is not None:
                model += group_weight <= float(upper), f"realism_grams_max_{_safe_name(group)}"


def summarize_realism_for_solution(
    solution: dict[str, float],
    db: pd.DataFrame,
    tolerance: float = 1e-3,
) -> dict[str, Any]:
    """Return selected-food counts and gram totals by realism group for one solution."""
    food_col = "Livsmedelsnamn"
    if "realism_group" not in db.columns:
        return {}

    db_meta = db.set_index(food_col)
    selected = {
        food: grams
        for food, grams in solution.items()
        if food in db_meta.index and grams > tolerance
    }

    result: dict[str, Any] = {
        "Selected foods": len(selected),
    }

    for group, foods in db_meta.groupby("realism_group"):
        names = set(foods.index)
        grams = sum(value for food, value in selected.items() if food in names)
        count = sum(1 for food in selected if food in names)
        if grams > tolerance or count:
            result[f"{group} count"] = count
            result[f"{group} grams"] = grams

    return result


def validate_realism_solution(
    solution: dict[str, float],
    db: pd.DataFrame,
    config: RealismConfig | None = None,
    tolerance: float = 1e-3,
) -> None:
    """Raise ``RuntimeError`` if an extracted solution violates realism metadata."""
    config = config or default_realism_config()
    food_col = config.food_name_column
    if "realism_group" not in db.columns:
        return

    db_meta = db.set_index(food_col)
    selected = {
        food: grams
        for food, grams in solution.items()
        if food in db_meta.index and grams > tolerance
    }

    for food, grams in selected.items():
        row = db_meta.loc[food]
        max_values = [
            row.get(config.max_if_selected_column),
            row.get(config.max_daily_column),
        ]
        max_values = [float(v) for v in max_values if pd.notna(v)]
        if max_values and grams > min(max_values) + tolerance:
            raise RuntimeError(f"{food} has {grams:.2f} g, above realism max {min(max_values):.2f} g.")

        min_value = row.get(config.min_if_selected_column)
        if pd.notna(min_value) and grams + tolerance < float(min_value):
            raise RuntimeError(f"{food} has {grams:.2f} g, below realism min {float(min_value):.2f} g.")

    for group, (lower, upper) in config.count_bounds.items():
        foods = set(db_meta.loc[db_meta[config.count_group_column] == group].index)
        count = sum(1 for food in selected if food in foods)
        if lower is not None and count < lower:
            raise RuntimeError(f"Realism count group {group!r} has {count}, below minimum {lower}.")
        if upper is not None and count > upper:
            raise RuntimeError(f"Realism count group {group!r} has {count}, above maximum {upper}.")

    for group, (lower, upper) in config.group_gram_bounds.items():
        foods = set(db_meta.loc[db_meta[config.realism_group_column] == group].index)
        grams = sum(value for food, value in selected.items() if food in foods)
        if lower is not None and grams + tolerance < lower:
            raise RuntimeError(f"Realism group {group!r} has {grams:.2f} g, below minimum {lower} g.")
        if upper is not None and grams > upper + tolerance:
            raise RuntimeError(f"Realism group {group!r} has {grams:.2f} g, above maximum {upper} g.")


def _safe_name(value: Any) -> str:
    """Return a PuLP-safe constraint-name fragment."""
    return "".join(ch if str(ch).isalnum() else "_" for ch in str(value))[:80]

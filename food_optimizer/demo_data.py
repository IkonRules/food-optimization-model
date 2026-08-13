"""Load the artificial dataset used by the public portfolio demonstration."""

from __future__ import annotations

from dataclasses import dataclass
import json

import pandas as pd

from food_optimizer.config import DATA_DIR
from food_optimizer.get_ref_vals_dict import get_kcal, validate_user_input
from food_optimizer.realism_constraints import RealismConfig


DEMO_DIR = DATA_DIR / "demo"
DEMO_PROFILE = {
    "sex": "man",
    "pregnant": None,
    "trimester": None,
    "breastfeeding": None,
    "phase": None,
    "age": 30,
}


@dataclass(frozen=True)
class DemoBundle:
    """Inputs for one run through the production optimizer."""

    foods: pd.DataFrame
    reference_values: dict
    toxin_columns: list[str]
    plate_model_constraints: dict
    realism_metadata: pd.DataFrame
    realism_config: RealismConfig
    profile: dict


def load_demo_energy_tables() -> dict[str, pd.DataFrame]:
    """Return artificial energy-reference tables in the research loader's shape."""
    profiles = pd.read_csv(DEMO_DIR / "synthetic_reference_profiles.csv")
    profiles["Åldersgrupp"] = list(zip(profiles["age_min"], profiles["age_max"]))
    profiles = profiles.rename(columns={"reference_weight_kg": "Referensvikt kg"})
    columns = ["Åldersgrupp", "Referensvikt kg", "kcal"]
    return {
        sex: profiles.loc[profiles["sex"] == sex, columns].reset_index(drop=True)
        for sex in ("women", "men")
    }


def build_demo_reference_values(profile: dict | None = None) -> dict:
    """Build artificial constraints while reusing the profile energy lookup logic."""
    profile = dict(profile or DEMO_PROFILE)
    validate_user_input(profile)

    with open(DEMO_DIR / "synthetic_constraints.json", encoding="utf-8") as file:
        config = json.load(file)

    energy = float(get_kcal(profile, energy_tables=load_demo_energy_tables()))
    reference_values = {
        name: {bound: float(value) for bound, value in bounds.items()}
        for name, bounds in config["absolute_constraints"].items()
    }
    reference_values["kcal"] = energy
    reference_values.update(
        {
            name: tuple(float(value) for value in bounds)
            for name, bounds in config["macro_percent_constraints"].items()
        }
    )
    reference_values.update(
        {name: float(value) for name, value in config["toxin_upper_bounds"].items()}
    )
    return reference_values


def load_demo_bundle(profile: dict | None = None) -> DemoBundle:
    """Load and validate all bundled synthetic demonstration inputs."""
    foods = pd.read_csv(DEMO_DIR / "synthetic_foods.csv")
    realism_metadata = pd.read_csv(DEMO_DIR / "synthetic_realism_metadata.csv")
    with open(DEMO_DIR / "synthetic_constraints.json", encoding="utf-8") as file:
        config = json.load(file)

    names = foods["Livsmedelsnamn"].astype(str)
    if not names.str.startswith("Synthetic ").all():
        raise ValueError("Every public demo food name must begin with 'Synthetic '.")
    if set(names) != set(realism_metadata["Livsmedelsnamn"].astype(str)):
        raise ValueError("Synthetic food and realism metadata names do not match.")

    count_bounds = {
        name: tuple(bounds)
        for name, bounds in config["realism"]["count_bounds"].items()
    }
    group_gram_bounds = {
        name: tuple(bounds)
        for name, bounds in config["realism"]["group_gram_bounds"].items()
    }
    profile = dict(profile or DEMO_PROFILE)
    return DemoBundle(
        foods=foods,
        reference_values=build_demo_reference_values(profile),
        toxin_columns=list(config["toxin_upper_bounds"]),
        plate_model_constraints=config["plate_model_constraints"],
        realism_metadata=realism_metadata,
        realism_config=RealismConfig(
            count_bounds=count_bounds,
            group_gram_bounds=group_gram_bounds,
        ),
        profile=profile,
    )

"""Linear and mixed-integer optimization routines for daily food intake."""

import pickle

import numpy as np
import pandas as pd
import pulp

from food_optimizer.config import DATA_DIR
from food_optimizer.realism_constraints import (
    RealismConfig,
    add_realism_constraints,
    default_realism_config,
    prepare_realism_database,
    summarize_realism_for_solution,
    validate_realism_solution,
)


class OptimizationInfeasibleError(ValueError):
    """Raised when no feasible food solution exists for the requested settings."""



input_path_db = DATA_DIR / "df_foods_and_toxins_database.csv"
input_path_plate_model_constraints = DATA_DIR / "plate_model_constraints.pkl"

def extract_macro_group_proportions(
    summary_df: pd.DataFrame,
    db: pd.DataFrame,
    plate_model_constraints: dict
) -> pd.DataFrame:
    """
    Extract the macro group proportions (as percentages of total weight)
    from each solution in the summary DataFrame.

    Parameters:
    - summary_df (pd.DataFrame): Result of `summarize_solutions()`.
    - db (pd.DataFrame): Food database with 'Livsmedelsnamn' and 'Gruppering'.
    - plate_model_constraints (dict): Mapping of macro groups to food groupings and target proportions.

    Returns:
    - pd.DataFrame: One row per solution, with percentage columns for each macro group.
    """
    # Clean up database for consistent matching
    db_clean = db[['Livsmedelsnamn', 'Gruppering']].copy()
    db_clean['Livsmedelsnamn'] = db_clean['Livsmedelsnamn'].str.strip()
    db_clean['Gruppering'] = db_clean['Gruppering'].str.strip()

    grouping_lookup = db_clean.set_index('Livsmedelsnamn')['Gruppering'].to_dict()

    # Reverse mapping: grouping → macro group
    grouping_to_macro = {}
    for macro, info in plate_model_constraints.items():
        for g in info['groupings']:
            grouping_to_macro[g.strip()] = macro

    # Initialize list for result rows
    result_rows = []

    for i, row in summary_df.iterrows():
        solution_id = row.get("Solution", f"Solution {i+1}")
        food_weights = row.drop(labels=["Solution"], errors="ignore")

        # Filter to only valid food items
        valid_food_weights = {
            food: weight for food, weight in food_weights.items()
            if food in grouping_lookup and weight > 0
        }

        total_weight = sum(valid_food_weights.values())
        macro_totals = {macro: 0.0 for macro in plate_model_constraints}

        for food, weight in valid_food_weights.items():
            grouping = grouping_lookup.get(food)
            macro = grouping_to_macro.get(grouping)
            if macro:
                macro_totals[macro] += weight

        # Calculate proportions
        macro_percentages = {
            macro: 100 * weight / total_weight if total_weight else 0
            for macro, weight in macro_totals.items()
        }

        result_rows.append({
            "Solution": solution_id,
            "Total weight (g)": total_weight,
            **macro_percentages
        })

    return pd.DataFrame(result_rows)

def add_macro_group_weight_constraints(
    model,
    food_vars: dict,
    db: pd.DataFrame,
    plate_model_constraints: dict,
    total_weight_expr
):
    """
    Adds plate model-style proportion constraints to a linear programming model.

    This enforces that the total weight of selected foods from each macro group
    (e.g., "Greens", "Carbohydrates", "Proteins") must lie within a specified 
    percentage range of the total food weight.

    Parameters:
    - model (pulp.LpProblem): The LP or MILP model to which constraints will be added.
    - food_vars (dict): Dictionary of pulp variables, mapping food names to decision variables (grams).
    - db (pd.DataFrame): Food database containing 'Livsmedelsnamn' and 'Gruppering' columns.
    - plate_model_constraints (dict): A dictionary where each key is a macro group name and 
      the value is another dict with:
        - 'groupings': list of food group names in the group
        - 'proportion': (min_pct, max_pct) tuple specifying allowed percentage range
    - total_weight_expr (pulp.LpAffineExpression): Expression representing the total weight of all selected foods.

    Example constraint added:
        45% ≤ (total grams of vegetables and fruits) / (total food weight) ≤ 55%
    """
    food_to_macro = map_foods_to_macro_groups(db, plate_model_constraints)
    for macro_group, info in plate_model_constraints.items():
        foods_in_group = [f for f, groups in food_to_macro.items() if macro_group in groups]
        if not foods_in_group:
            continue
        group_weight = pulp.lpSum(food_vars[f] for f in foods_in_group if f in food_vars)
        lower, upper = info['proportion']
        model += group_weight >= (lower / 100) * total_weight_expr, f"{macro_group}_min"
        model += group_weight <= (upper / 100) * total_weight_expr, f"{macro_group}_max"


def map_foods_to_macro_groups(db: pd.DataFrame, plate_model_constraints: dict) -> dict:
    """
    Maps individual foods in the database to their corresponding macro food groups.

    This function is used to determine which foods belong to which high-level 
    categories (e.g., "Greens", "Carbohydrates", "Proteins") based on the 
    'Gruppering' column in the food database and the 'groupings' list defined in
    the constraint dictionary.

    Parameters:
    - db (pd.DataFrame): Food database containing 'Livsmedelsnamn' and 'Gruppering' columns.
    - plate_model_constraints (dict): A dictionary where each key is a macro group name and 
      the value is another dict with a 'groupings' list of food category names.

    Returns:
    - dict: A mapping {food_name: [macro_group1, macro_group2, ...]}, indicating
            which macro group(s) each food belongs to.

    Notes:
    - Strips whitespace from both 'Gruppering' and 'Livsmedelsnamn' to ensure robust matching.
    - Allows foods to belong to multiple macro groups if definitions overlap.
    """
    food_to_macro_group = {}
    for macro_group, info in plate_model_constraints.items():
        for grouping in info['groupings']:
            matches = db[db['Gruppering'].str.strip() == grouping.strip()]['Livsmedelsnamn']
            for food in matches:
                food_to_macro_group.setdefault(food.strip(), []).append(macro_group)
    return food_to_macro_group


def parse_constraints(reference_values_dict: dict) -> dict:
    """
    Parses a dictionary of reference values into structured constraint components
    for use in linear programming (LP) diet optimization.

    Handles:
    1. Absolute constraints (min/max for nutrients or toxins in g/day)
    2. Macronutrient percentage constraints (e.g., 'Fett procent')
    3. Energy range (±10% of target kcal value)
    4. Optional upper limits (toxins, when only 'max' is given)

    Parameters:
    - reference_values_dict (dict): Dictionary where keys are nutrient names
      and values can be:
        - A tuple (min%, max%) for macronutrient percentages
        - A single float for upper limit constraints (toxins)
        - A dict with keys 'min' and/or 'max' for general constraints
        - A float or dict representing total energy in kcal

    Returns:
    - dict: A structured dictionary with three parts:
        {
            'absolute_constraints': {
                nutrient_name: {'db_name': str, 'min': float (optional), 'max': float (optional)}
            },
            'macro_percent_constraints': {
                macro_name: {'db_name': str, 'range': tuple(min_pct, max_pct)}
            },
            'energy_kcal_range': tuple(min_kcal, max_kcal) or None
        }
    """
    absolute_constraints = {}
    macro_percent_constraints = {}
    energy_kcal_range = None

    for key, val in reference_values_dict.items():
        # 1. Macronutrient % constraint
        if isinstance(val, tuple) and key.lower().endswith("procent"):
            macro_percent_constraints[key] = {
                "db_name": key,
                "range": val
            }

        # 2. Energy (total kcal)
        elif key.lower() in {"energi", "kcal"}:
            kcal = val if isinstance(val, (int, float)) else val.get("ref_value", 0)
            kcal_min = kcal * 0.9
            kcal_max = kcal * 1.1
            energy_kcal_range = (kcal_min, kcal_max)
            absolute_constraints[key] = {
                "db_name": key,
                "min": kcal_min,
                "max": kcal_max
            }

        # 3. Dict-style constraints with 'min' and/or 'max'
        elif isinstance(val, dict):
            constraint = {"db_name": key}
            if "min" in val:
                constraint["min"] = val["min"]
            if "max" in val:
                constraint["max"] = val["max"]
            absolute_constraints[key] = constraint

        # 4. Toxin-style constraint (only max value given)
        elif val is not None:
            absolute_constraints[key] = {
                "db_name": key,
                "max": val
            }

    return {
        "absolute_constraints": absolute_constraints,
        "macro_percent_constraints": macro_percent_constraints,
        "energy_kcal_range": energy_kcal_range
    }


def calculate_kcal_contribution(db: pd.DataFrame) -> pd.Series:
    """
    Calculates the energy contribution (kcal per 100g) for each food item
    based on macronutrient content.

    Assumptions:
    - Fat provides 9 kcal/g
    - Protein provides 4 kcal/g
    - Carbohydrates provide 4 kcal/g

    Parameters:
    - db (pd.DataFrame): The food composition database. Should contain columns:
        - 'Fett, totalt' (total fat in g/100g)
        - 'Protein' (in g/100g)
        - 'Kolhydrater, tillgängliga' (available carbs in g/100g)

    Returns:
    - pd.Series: A Series with kcal values per 100g for each food item.
    """
    fat = db.get("Fett, totalt", pd.Series(0, index=db.index))
    protein = db.get("Protein", pd.Series(0, index=db.index))
    carbs = db.get("Kolhydrater, tillgängliga", pd.Series(0, index=db.index))
    return fat * 9 + protein * 4 + carbs * 4


def build_lp_model(
    db: pd.DataFrame,
    constraints: dict,
    toxic_columns: list,
    prioritize: str = None,
    max_foods: int | None = 10,
    prefer_sparse: bool = True,
    plate_model_constraints: dict = None,
    realism_config: RealismConfig | None = None,
    enforce_realism: bool = False,
    big_m_grams: float = 5000.0,
):
    """
    Builds a mixed-integer linear programming (MILP) model for daily intake optimization.

    Parameters:
    - db (pd.DataFrame): Food composition database with nutrient values per 100g.
    - constraints (dict): Dictionary with nutrient constraints including:
        - "absolute_constraints": dict of nutrient min/max bounds.
        - "energy_kcal_range": tuple (min_kcal, max_kcal) or None.
        - "macro_percent_constraints": dict of macronutrient % targets.
    - toxic_columns (list): List of toxin-related column names to minimize.
    - prioritize (str or None): Optimization priority: 'toxins', 'weight', or None for combined objective.
    - max_foods (int): Maximum number of different foods allowed in the solution.
    - prefer_sparse (bool): Whether to add binary variables to enforce sparsity.

    Returns:
    - model (pulp.LpProblem): The linear programming model object.
    - food_vars (dict): Dictionary of pulp variables (food name → variable) representing food gram amounts.
    """
    model = pulp.LpProblem("Optimize_Intake", pulp.LpMinimize)
    food_items = db["Livsmedelsnamn"].tolist()

    # Decision variables
    food_vars = {
        food: pulp.LpVariable(f"x_{i}", lowBound=0, cat='Continuous')
        for i, food in enumerate(food_items)
    }

    # Food-count constraints
    #
    # The maximum number of foods must be enforced independently of the optional
    # sparse-solution preference.
    use_vars = {}
    needs_use_vars = max_foods is not None or enforce_realism
    if needs_use_vars:
        use_vars = {
            food: pulp.LpVariable(f"use_{i}", cat="Binary")
            for i, food in enumerate(food_items)
        }

        for food in food_items:
            model += food_vars[food] <= big_m_grams * use_vars[food], f"link_use_{food}"

    if max_foods is not None:
        max_foods = int(max_foods)
        if max_foods < 1:
            raise ValueError("max_foods must be at least 1, or None to disable the food-count constraint.")
        model += pulp.lpSum(use_vars.values()) <= max_foods, "max_number_of_foods"

    if enforce_realism:
        add_realism_constraints(model, food_vars, use_vars, db, realism_config)

    # Absolute nutrient constraints
    for name, rule in constraints["absolute_constraints"].items():
        col = rule["db_name"]
        nutrient_values = db[col].fillna(0)
        total = pulp.lpSum(food_vars[f] * nutrient_values.iloc[i] / 100 for i, f in enumerate(food_items))
        if "min" in rule:
            model += total >= rule["min"], f"{name}_min"
        if "max" in rule:
            model += total <= rule["max"], f"{name}_max"

    # Energy and macronutrient percent constraints
    total_weight = pulp.lpSum(food_vars.values())

    if constraints["energy_kcal_range"]:
        kcal_series = calculate_kcal_contribution(db)
        kcal_min, kcal_max = constraints["energy_kcal_range"]
        total_kcal = pulp.lpSum(food_vars[f] * kcal_series.iloc[i] / 100 for i, f in enumerate(food_items))
        model += total_kcal >= kcal_min
        model += total_kcal <= kcal_max

        macro_kcal_factors = {
            'Fett procent': 9,
            'Kolhydrater procent': 4,
            'Protein procent': 4,
        }

        macro_percent_source_map = {
            'Fett procent': 'Fett, totalt',
            'Kolhydrater procent': 'Kolhydrater, tillgängliga',
            'Protein procent': 'Protein',
        }

        for macro, rule in constraints["macro_percent_constraints"].items():
            col = macro_percent_source_map.get(macro, rule["db_name"])
            kcal_per_gram = macro_kcal_factors.get(macro, 4)
            macro_series = db[col].fillna(0)

            macro_energy = pulp.lpSum(
                food_vars[f] * macro_series.iloc[i] * kcal_per_gram / 100
                for i, f in enumerate(food_items)
            )
            model += macro_energy >= (rule["range"][0] / 100) * total_kcal, f"{macro}_min_kcal_pct"
            model += macro_energy <= (rule["range"][1] / 100) * total_kcal, f"{macro}_max_kcal_pct"

    # Add macro group constraints (plate model)
    if plate_model_constraints is not None:
        add_macro_group_weight_constraints(model, food_vars, db, plate_model_constraints, total_weight)

    # Objective function
    toxin_total = pulp.lpSum(
        food_vars[f] * db[t].fillna(0).iloc[i] / 100
        for t in toxic_columns
        for i, f in enumerate(food_items)
    )

    food_count_penalty = (1e-6 * pulp.lpSum(use_vars.values())) if (prefer_sparse and use_vars) else 0

    if prioritize == "toxins":
        model += toxin_total + 1e-6 * total_weight + food_count_penalty
    elif prioritize == "weight":
        model += total_weight + 1e-6 * toxin_total + food_count_penalty
    else:
        model += total_weight + toxin_total + food_count_penalty

    return model, food_vars


def extract_solution(model, food_vars, db, constraints, toxic_columns):
    """
    Extracts the selected food items and their weights from a solved LP model,
    and calculates total nutrient and toxin intake based on those weights.

    Parameters:
    - model (pulp.LpProblem): The solved optimization model (not used directly but included for context).
    - food_vars (dict): Dictionary mapping food names to their pulp variable objects.
    - db (pd.DataFrame): Food composition database indexed by food names.
    - constraints (dict): Dictionary containing absolute nutrient constraints (with db column names).
    - toxic_columns (list): List of toxin column names to aggregate in the solution.

    Returns:
    - dict: A dictionary with keys:
        - Food names → float (grams selected)
        - Nutrient names → float (total amount consumed)
        - Toxin names → float (total amount consumed)
    """
    # Extract selected foods and weights
    solution = {f: v.varValue or 0 for f, v in food_vars.items() if v.varValue and v.varValue > 1e-3}
    food_series = pd.Series(solution)
    db_indexed = db.set_index("Livsmedelsnamn")

    # Nutrients (values assumed per 100g, already scaled)
    nutrient_totals = {}
    for key, rule in constraints["absolute_constraints"].items():
        col = rule["db_name"]
        if col in db_indexed.columns:
            col_vals = db_indexed[col].fillna(0)
            nutrient_totals[key] = (food_series * col_vals / 100).sum()

    # Toxins (also per 100g)
    toxin_totals = {}
    for toxin in toxic_columns:
        if toxin in db_indexed.columns:
            col_vals = db_indexed[toxin].fillna(0)
            toxin_totals[toxin] = (food_series * col_vals / 100).sum()

    return {
        **solution,
        **nutrient_totals,
        **toxin_totals,
    }


def count_selected_foods(solution: dict, food_items: list[str], tolerance: float = 1e-3) -> int:
    """Count foods with positive gram amounts in an extracted solution."""
    return sum(1 for food in food_items if solution.get(food, 0) > tolerance)


def validate_solution_food_count(
    solution: dict,
    food_items: list[str],
    max_foods: int | None,
    tolerance: float = 1e-3,
) -> None:
    """Raise an error if a solution contains more foods than allowed."""
    if max_foods is None:
        return

    selected_food_count = count_selected_foods(solution, food_items, tolerance=tolerance)
    if selected_food_count > max_foods:
        raise RuntimeError(
            f"Internal optimization error: solution contains {selected_food_count} foods, "
            f"but max_foods={max_foods}."
        )


def generate_pareto_solutions(
    db,
    constraints,
    toxic_columns,
    max_foods=10,
    prefer_sparse=True,
    max_solutions=100,
    weight_steps=101,
    max_random_attempts=200,
    uniqueness_round_decimals=2,
    plate_model_constraints=None,
    realism_config: RealismConfig | None = None,
    enforce_realism: bool = False,
):
    """
    Generates up to `max_solutions` Pareto-optimal solutions by scalarizing two conflicting objectives:
    - Minimizing total food weight
    - Minimizing total toxin intake

    Parameters:
    - db (pd.DataFrame): Food database with nutrient and toxin columns.
    - constraints (dict): Parsed LP constraints from `parse_constraints`.
    - toxic_columns (list): List of toxin column names to minimize.
    - max_foods (int): Maximum number of foods allowed in a solution.
    - prefer_sparse (bool): If True, uses binary indicators to enforce sparsity.
    - max_solutions (int): Maximum number of unique Pareto-optimal solutions to generate.
    - weight_steps (int): Number of evenly spaced weight combinations to try first.
    - max_random_attempts (int): Maximum random scalarization attempts if grid is exhausted.
    - uniqueness_round_decimals (int): Decimal rounding used to identify unique solutions.

    Returns:
    - list of dict: List of Pareto-optimal solutions with food quantities and nutrient/toxin totals.
    """
    def canonical_solution(sol, decimals=2):
        return frozenset((k, round(v, decimals)) for k, v in sol.items())

    weights = [(w / (weight_steps - 1), 1 - w / (weight_steps - 1)) for w in range(weight_steps)]
    results = []
    seen = set()

    # CBC solver with strict tolerances
    solver = pulp.PULP_CBC_CMD(msg=False, options=["-ratio", "0.0", "-integerTolerance", "1e-12"])

    # First: regular grid of weights
    for w_weight, w_tox in weights:
        if len(results) >= max_solutions:
            break
        model, food_vars = build_lp_model(
            db, constraints, toxic_columns,
            prioritize=None,
            max_foods=max_foods,
            prefer_sparse=prefer_sparse,
            plate_model_constraints=plate_model_constraints,
            realism_config=realism_config,
            enforce_realism=enforce_realism,
        )
        model.setObjective(
            w_weight * pulp.lpSum(food_vars.values()) +
            w_tox * pulp.lpSum(
                food_vars[f] * db[t].fillna(0).iloc[i] / 100 / 100
                for t in toxic_columns
                for i, f in enumerate(db["Livsmedelsnamn"])
            )
        )
        model.solve(solver)
        if model.status == 1:
            sol = extract_solution(model, food_vars, db, constraints, toxic_columns)
            validate_solution_food_count(sol, db["Livsmedelsnamn"].tolist(), max_foods)
            if enforce_realism:
                validate_realism_solution(sol, db, realism_config)
                sol.update(summarize_realism_for_solution(sol, db))
            key = canonical_solution(sol, uniqueness_round_decimals)
            if key not in seen:
                seen.add(key)
                results.append(sol)
                print(f"{len(results)} solutions generated...")
        else:
            print(f"Skipped: Status {model.status} for weights ({w_weight:.2f}, {w_tox:.2f})")

    # Second: randomized weights if needed
    attempts = 0
    while len(results) < max_solutions and attempts < max_random_attempts:
        w_weight = np.random.uniform(0, 1)
        w_tox = 1 - w_weight

        model, food_vars = build_lp_model(
            db, constraints, toxic_columns,
            prioritize=None,
            max_foods=max_foods,
            prefer_sparse=prefer_sparse,
            plate_model_constraints=plate_model_constraints,
            realism_config=realism_config,
            enforce_realism=enforce_realism,
        )
        model.setObjective(
            w_weight * pulp.lpSum(food_vars.values()) +
            w_tox * pulp.lpSum(
                food_vars[f] * db[t].fillna(0).iloc[i] / 100 / 100
                for t in toxic_columns
                for i, f in enumerate(db["Livsmedelsnamn"])
            )
        )
        model.solve(solver)
        attempts += 1

        if model.status == 1:
            sol = extract_solution(model, food_vars, db, constraints, toxic_columns)
            validate_solution_food_count(sol, db["Livsmedelsnamn"].tolist(), max_foods)
            if enforce_realism:
                validate_realism_solution(sol, db, realism_config)
                sol.update(summarize_realism_for_solution(sol, db))
            key = canonical_solution(sol, uniqueness_round_decimals)
            if key not in seen:
                seen.add(key)
                results.append(sol)
                print(f"{len(results)} solutions generated...")
        else:
            print(f"Random attempt {attempts}: infeasible.")

    return results

def extend_pareto_fronts(
    db,
    constraints,
    toxic_columns,
    primary_solutions,
    all_weights,
    max_solutions,
    max_foods,
    prefer_sparse,
    plate_model_constraints=None,
    realism_config: RealismConfig | None = None,
    enforce_realism: bool = False,
):
    """
    Extends an existing Pareto front with additional weight combinations
    to reach up to `max_solutions` if the initial set is too small.

    Parameters:
    - db (pd.DataFrame): Food database with nutrient and toxin columns.
    - constraints (dict): Parsed LP constraints from `parse_constraints`.
    - toxic_columns (list): List of toxin column names to minimize.
    - primary_solutions (list): Existing Pareto-optimal solutions to build upon.
    - all_weights (list of tuples): Weight combinations to iterate over.
    - max_solutions (int): Desired maximum number of solutions.
    - max_foods (int): Maximum number of foods per solution.
    - prefer_sparse (bool): Whether to enforce sparsity with binary variables.

    Returns:
    - list of dict: Combined list of initial and extended Pareto-optimal solutions.
    """
    seen = [frozenset(sol.items()) for sol in primary_solutions]
    extended = []

    # CBC solver with strict tolerances
    solver = pulp.PULP_CBC_CMD(msg=False, options=["-ratio", "0.0", "-integerTolerance", "1e-12"])

    for w_weight, w_tox in all_weights:
        if len(primary_solutions) + len(extended) >= max_solutions:
            break

        model, food_vars = build_lp_model(
            db, constraints, toxic_columns,
            prioritize=None,
            max_foods=max_foods,
            prefer_sparse=prefer_sparse,
            plate_model_constraints=plate_model_constraints,
            realism_config=realism_config,
            enforce_realism=enforce_realism,
        )
        model.setObjective(
            w_weight * pulp.lpSum(food_vars.values()) +
            w_tox * pulp.lpSum(
                food_vars[f] * db[t].fillna(0).iloc[i] / 100 / 100
                for t in toxic_columns
                for i, f in enumerate(db["Livsmedelsnamn"])
            )
        )
        model.solve(solver)
        if model.status == 1:
            sol = extract_solution(model, food_vars, db, constraints, toxic_columns)
            validate_solution_food_count(sol, db["Livsmedelsnamn"].tolist(), max_foods)
            if frozenset(sol.items()) not in seen:
                seen.append(frozenset(sol.items()))
                extended.append(sol)

    return primary_solutions + extended


def summarize_solutions(solution_list: list) -> pd.DataFrame:
    """
    Converts a list of solution dictionaries into a well-structured summary DataFrame.

    Each dictionary represents a solution from the optimization process, containing food weights,
    nutrient totals, and toxin totals. The function:
    - Fills missing values with 0.
    - Adds a "Solution" column as an identifier.
    - Reorders columns into three logical groups: food items, reference nutrients, and toxins.

    Parameters:
    - solution_list (list): A list of dictionaries, where each dict represents a solution 
                            with food weights and optional nutrient/toxin totals.

    Returns:
    - pd.DataFrame: A summary DataFrame with one row per solution and organized columns.
    """
    if not solution_list:
        return pd.DataFrame()

    df = pd.DataFrame(solution_list).fillna(0)
    df.insert(0, "Solution", [f"solution {i+1}" for i in range(len(df))])

    all_columns = set(df.columns) - {"Solution"}

    ref_order = [
        'Fett procent', 'Kolhydrater procent', 'Protein procent', 'kcal',
        'Vitamin A', 'Vitamin D', 'Tiamin', 'Riboflavin', 'Niacin', 'Vitamin B6',
        'Folat', 'Vitamin C', 'Vitamin E', 'Vitamin K', 'Vitamin B12',
        'Kalcium', 'Järn', 'Zink', 'Fosfor', 'Kalium', 'Magnesium',
        'Jod', 'Selen', 'Fluor', 'Natrium', 'Salt', 'Protein',
        'Total weight (g)'
    ]

    toxin_order = [
        'Acrylamide', 'Brominated flame retardants', 'Fluoride', 'Glycidol', 'Metals',
        'Mycotoxins 1', 'Mycotoxins 2', 'Organochlorine pesticides', 'PAHs', 'PCAs',
        'PCBs and Dioxins', 'PFAS', 'PFRs', 'Plasticizers'
    ]

    ref_columns = [col for col in ref_order if col in all_columns]
    toxin_columns = [col for col in toxin_order if col in all_columns]
    food_columns = sorted(list(all_columns - set(ref_columns) - set(toxin_columns)))

    final_columns = ["Solution"] + food_columns + ref_columns + toxin_columns
    return df[final_columns]


def optimize_daily_intake(
    db: pd.DataFrame,
    search_dict: dict,
    toxic_columns: list,
    num_solutions: int = None,
    prioritize: str = None,  # 'weight', 'toxins', or None
    max_foods: int = 10,
    prefer_sparse: bool = True,
    max_pareto_solutions: int = 100,
    plate_model_constraints: dict = None,
    realism_metadata: pd.DataFrame | None = None,
    realism_config: RealismConfig | None = None,
    enforce_realism: bool = True,
) -> pd.DataFrame:
    """
    Generates optimized daily food intake solutions based on nutritional requirements
    and toxin limits using linear programming.

    Parameters:
    - db (pd.DataFrame): Food composition database (must include nutrient and toxin columns).
    - search_dict (dict): Dictionary of constraint specifications (from reference values).
    - toxic_columns (list): List of column names representing toxin compounds.
    - num_solutions (int, optional): Number of solutions to return. Default is 1 for prioritized search, or 100 for Pareto.
    - prioritize (str, optional): Optimization goal, one of 'weight', 'toxins', or None.
    - max_foods (int): Maximum number of foods allowed in each solution.
    - prefer_sparse (bool): If True, encourages sparse solutions via binary-use constraints.
    - max_pareto_solutions (int): Max solutions when generating Pareto front (ignored if `prioritize` is set).
    - plate_model_constraints (dict, optional): Macro food category constraints to enforce (as proportion ranges).

    Returns:
    - pd.DataFrame: Summary of optimized solutions, including nutrient and toxin content.
    """
    constraints = parse_constraints(search_dict)

    if enforce_realism:
        realism_config = realism_config or default_realism_config()
        db = prepare_realism_database(db, metadata=realism_metadata, config=realism_config)

    if prioritize in ['weight', 'toxins']:
        results = []
        food_items = db["Livsmedelsnamn"].tolist()

        solver = pulp.PULP_CBC_CMD(msg=False, options=["-ratio", "0.0", "-integerTolerance", "1e-12"])

        for _ in range(num_solutions or 1):
            model, food_vars = build_lp_model(
                db, constraints, toxic_columns,
                prioritize=None,
                max_foods=max_foods,
                prefer_sparse=prefer_sparse,
                plate_model_constraints=plate_model_constraints,
                realism_config=realism_config,
                enforce_realism=enforce_realism,
            )

            weights = np.random.uniform(0.98, 1.02, size=len(food_items))

            if prioritize == "weight":
                objective = pulp.lpSum(weights[i] * food_vars[f] for i, f in enumerate(food_items))
            elif prioritize == "toxins":
                objective = pulp.lpSum(
                    weights[i] * food_vars[f] * db[t].fillna(0).iloc[i] / 100
                    for t in toxic_columns
                    for i, f in enumerate(food_items)
                )

            model.setObjective(objective)
            model.solve(solver)

            if model.status == 1:
                solution = extract_solution(model, food_vars, db, constraints, toxic_columns)
                validate_solution_food_count(solution, food_items, max_foods)
                if enforce_realism:
                    validate_realism_solution(solution, db, realism_config)
                    solution.update(summarize_realism_for_solution(solution, db))

                total_weight = sum(v for k, v in solution.items() if k in food_items)
                solution['Total weight (g)'] = total_weight

                db_indexed = db.set_index("Livsmedelsnamn")
                fat_kcal = 9 * sum(solution.get(f, 0) * db_indexed.get("Fett, totalt", pd.Series(0)).get(f, 0) / 100 for f in food_items)
                carb_kcal = 4 * sum(solution.get(f, 0) * db_indexed.get("Kolhydrater, tillgängliga", pd.Series(0)).get(f, 0) / 100 for f in food_items)
                protein_kcal = 4 * sum(solution.get(f, 0) * db_indexed.get("Protein", pd.Series(0)).get(f, 0) / 100 for f in food_items)
                total_kcal = fat_kcal + carb_kcal + protein_kcal

                if total_kcal > 0:
                    solution['Fett procent'] = 100 * fat_kcal / total_kcal
                    solution['Kolhydrater procent'] = 100 * carb_kcal / total_kcal
                    solution['Protein procent'] = 100 * protein_kcal / total_kcal
                else:
                    solution['Fett procent'] = 0
                    solution['Kolhydrater procent'] = 0
                    solution['Protein procent'] = 0

                results.append(solution)
            else:
                break

    else:
        results = generate_pareto_solutions(
            db=db,
            constraints=constraints,
            toxic_columns=toxic_columns,
            max_foods=max_foods,
            prefer_sparse=prefer_sparse,
            max_solutions=num_solutions or max_pareto_solutions,
            plate_model_constraints=plate_model_constraints,
            realism_config=realism_config,
            enforce_realism=enforce_realism,
        )

    if not results:
        raise OptimizationInfeasibleError(
            f"No feasible solution found with max_foods={max_foods}. "
            "Try increasing the maximum number of foods, relaxing the realism/plate-model constraints, "
            "or checking whether nutrient/toxin limits are too restrictive."
        )

    return summarize_solutions(results)


# Only run interactively if script is called directly
if __name__ == "__main__":
    from food_optimizer.get_user_input import get_ref_val_input_parameters
    from food_optimizer.get_ref_vals_dict import build_reference_values_dict, toxic_columns

    db = pd.read_csv(input_path_db)
    with open(input_path_plate_model_constraints, "rb") as file:
        plate_model_constraints = pickle.load(file)

    user_input = get_ref_val_input_parameters()
    reference_values_dict = build_reference_values_dict(user_input, db)
    result = optimize_daily_intake(db,                     # Database with foods and toxins.
                               reference_values_dict,      # Dictionary w. col. mapping and max/min ref. values.
                               toxic_columns,              # Columns in db containing toxins.
                               num_solutions=1,            # Number of solutions to be generated.
                               prioritize='toxins',        # Prioritize obj. func. ('weight', 'toxins', or None)
                               max_foods=40,               # Max nr foods to be used in a solution.
                               prefer_sparse=True,         # Prefer solutions with fewer foods.
                               max_pareto_solutions=100,   # Cap on pareto sol. if prio is None.
                               plate_model_constraints=plate_model_constraints # Dict with macro constraints.
                              )
    print(result)

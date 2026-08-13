"""Build nutrient, energy, macronutrient, and toxin reference-value dictionaries."""

import ast
import pickle
import re
from pathlib import Path

import pandas as pd

from food_optimizer.config import DATA_DIR


input_path_db = DATA_DIR / "df_foods_and_toxins_database.csv"
input_path_toxins_thresh = DATA_DIR / "toxins_thresholds.csv"
input_path_salt_thresh = DATA_DIR / "salt_thresholds.csv"
input_path_ref_vals = DATA_DIR / "dict_dfs_NNR_tables.pkl"

# Full-research tables are loaded only when that workflow is used. This keeps
# imports, the public synthetic demo, and the public tests independent of the
# locally retained source-derived runtime files.
db = None
toxins_thresholds = None
salt_thresholds = None
dict_dfs_NNR_tables = None
df_intake_nutrients_women = None
df_intake_nutrients_men = None
df_intake_energy_women = None
df_intake_energy_men = None
df_intake_proportions = None


def _load_research_runtime_data() -> None:
    """Load the local full-research tables on first use."""
    global db, toxins_thresholds, salt_thresholds, dict_dfs_NNR_tables
    global df_intake_nutrients_women, df_intake_nutrients_men
    global df_intake_energy_women, df_intake_energy_men, df_intake_proportions

    if dict_dfs_NNR_tables is not None:
        return

    paths = (
        input_path_db,
        input_path_toxins_thresh,
        input_path_salt_thresh,
        input_path_ref_vals,
    )
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            "Full research mode requires local source-derived runtime files that "
            "are intentionally excluded from the public repository. Missing: "
            + ", ".join(missing)
        )

    db = pd.read_csv(input_path_db)
    toxins_thresholds = pd.read_csv(input_path_toxins_thresh)
    salt_thresholds = pd.read_csv(input_path_salt_thresh)
    with open(input_path_ref_vals, "rb") as file:
        dict_dfs_NNR_tables = pickle.load(file)

    nutrient_tables = dict_dfs_NNR_tables["dfs_dict_nutrients_by_group"]
    energy_tables = dict_dfs_NNR_tables["dfs_dict_energy_by_group"]
    proportion_tables = dict_dfs_NNR_tables["dfs_dict_proportions_by_group"]
    df_intake_nutrients_women = nutrient_tables["df_intake_nutrients_women"]
    df_intake_nutrients_men = nutrient_tables["df_intake_nutrients_men"]
    df_intake_energy_women = energy_tables["df_intake_energy_women"]
    df_intake_energy_men = energy_tables["df_intake_energy_men"]
    df_intake_proportions = proportion_tables["df_intake_proportions"]

# Toxic compound columns in food- and toxins database.
toxic_columns = ['Acrylamide', 'Brominated flame retardants', 'Fluoride', 'Glycidol', 'Metals', 
                 'Mycotoxins 1', 'Mycotoxins 2', 'Organochlorine pesticides', 'PAHs', 'PCAs', 'PCBs and Dioxins',
                 'PFAS', 'PFRs', 'Plasticizers']

PROFILE_KEYS = (
    "sex",
    "pregnant",
    "trimester",
    "breastfeeding",
    "phase",
    "age",
)


def _profile_values(input_parameters):
    """Return profile values in the model's expected order.

    Explicit key access keeps programmatic profiles independent of dictionary
    insertion order while preserving the ordering used by the interactive UI.
    """
    return tuple(input_parameters[key] for key in PROFILE_KEYS)

def move_column(df, col_name, new_index):
    """
    Moves a column to a new position in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame.
    - col_name (str): The name of the column to move.
    - new_index (int): The target index to move the column to.

    Returns:
    - pd.DataFrame: A DataFrame with the column repositioned.
    """
    cols = df.columns.tolist()
    cols.insert(new_index, cols.pop(cols.index(col_name)))
    return df[cols]

def find_age_group_row(age, df):
    """
    Finds the appropriate row in a reference table based on the given age.

    Parameters:
    - age (float): The age to find the reference group for.
    - df (pd.DataFrame): The reference DataFrame with an 'Åldersgrupp' column.

    Returns:
    - pd.Series: The matching row.
    """
    for _, row in df.iterrows():
        group = row['Åldersgrupp']
        try:
            group = ast.literal_eval(group) if isinstance(group, str) and group.startswith("(") else group
            if isinstance(group, tuple) and group[0] <= age <= group[1]:
                return row
        except:
            continue

def complete_nutrients_row(incomplete_row, age, pregnant, trimester, breastfeeding, phase):
    """
    Adds additional protein to a nutrient row based on pregnancy or breastfeeding status.

    Parameters:
    - incomplete_row (pd.Series): The original nutrient values.
    - age (float): Age of the individual.
    - pregnant (bool): Whether the individual is pregnant.
    - trimester (int): Trimester of pregnancy.
    - breastfeeding (bool): Whether the individual is breastfeeding.
    - phase (int): Phase of breastfeeding.

    Returns:
    - pd.Series: The completed nutrient row.
    """
    protein_value = incomplete_row['Protein'] + calculate_extra_protein(age, pregnant, trimester, breastfeeding, phase)
    incomplete_row['Protein'] = protein_value
    return incomplete_row.copy()

def complete_energy_row(pregnant, breastfeeding, women_energy_table=None):
    """
    Returns the correct energy row for pregnant or breastfeeding women.

    Parameters:
    - pregnant (bool): Whether the individual is pregnant.
    - breastfeeding (bool): Whether the individual is breastfeeding.

    Returns:
    - pd.Series: A single row with energy values.
    """
    if women_energy_table is None:
        _load_research_runtime_data()
        women_energy_table = df_intake_energy_women

    if pregnant:
        incomplete_row = women_energy_table[women_energy_table['Åldersgrupp'] == 'Gravida']
    elif breastfeeding:
        incomplete_row = women_energy_table[women_energy_table['Åldersgrupp'] == 'Ammande']
    else:
        raise ValueError('Error in complete_energy_row function: specify pregnant or breastfeeding.')

    return incomplete_row.iloc[0]

def get_reference_weight(input_parameters):
    """
    Retrieves the reference weight from the energy reference tables based on input parameters.

    Parameters:
    - input_parameters (dict): Dictionary containing 'sex', 'age', 'pregnant', 'trimester', 'breastfeeding', and 'phase'.

    Returns:
    - float: Reference weight in kilograms.
    """
    _load_research_runtime_data()
    sex, pregnant, trimester, breastfeeding, phase, age = _profile_values(input_parameters)
    if sex == 'woman':
        energy_row = find_age_group_row(age, df_intake_energy_women)
        if pregnant or breastfeeding:
            energy_row = complete_energy_row(pregnant, breastfeeding)
    elif sex == 'man':
        energy_row = find_age_group_row(age, df_intake_energy_men)
    else:
        raise ValueError("Invalid sex input.")

    return energy_row['Referensvikt kg']

def get_kcal(input_parameters, energy_tables=None):
    """
    Retrieves the kcal energy requirement from the energy reference tables.

    Parameters:
    - input_parameters (dict): Dictionary containing 'sex', 'age', 'pregnant', 'trimester', 'breastfeeding', and 'phase'.
    - energy_tables (dict, optional): Explicit ``women`` and ``men`` tables.
      When omitted, the local full-research NNR tables are loaded lazily.

    Returns:
    - float: Energy requirement in kilocalories.
    """
    if energy_tables is None:
        _load_research_runtime_data()
        women_energy_table = df_intake_energy_women
        men_energy_table = df_intake_energy_men
    else:
        women_energy_table = energy_tables["women"]
        men_energy_table = energy_tables["men"]

    sex, pregnant, trimester, breastfeeding, phase, age = _profile_values(input_parameters)
    if sex == 'woman':
        energy_row = find_age_group_row(age, women_energy_table)
        if pregnant or breastfeeding:
            energy_row = complete_energy_row(pregnant, breastfeeding, women_energy_table)
    elif sex == 'man':
        energy_row = find_age_group_row(age, men_energy_table)
    else:
        raise ValueError("Invalid sex input.")

    return energy_row['kcal']

def rename_columns(df, name_pairs):
    """
    Renames DataFrame columns based on a list of preferred and existing name pairs and fixed mappings.

    Parameters:
    - df (pd.DataFrame): The DataFrame to rename.
    - name_pairs (list of tuples): List of (preferred_name, existing_name) pairs.

    Returns:
    - pd.DataFrame: A new DataFrame with renamed columns.
    """
    rename_map = {
        existing: preferred for preferred, existing in name_pairs
        if existing in df.columns and preferred != existing
    }
    fixed_map = {'Energi': 'kcal'}
    for existing, preferred in fixed_map.items():
        if existing in df.columns and preferred != existing:
            rename_map[existing] = preferred

    return df.rename(columns=rename_map)

def match_nutrients(reference_dict, db_columns, verbose=False):
    """
    Matches keys from the reference dictionary to the closest matching column in the database.

    Parameters:
    - reference_dict (dict): Dictionary of reference nutrient values.
    - db_columns (list of str): List of column names from the food database.

    Returns:
    - matches (dict): Mapping of reference name -> matched db column.
    - unmatched (list of str): Reference names with no matching db column.
    """
    matches = {}
    unmatched = []
    for ref_key in reference_dict:
        match = next((col for col in db_columns if ref_key.lower() in col.lower()), None)
        if match:
            matches[ref_key] = match
        else:
            unmatched.append(ref_key)

    # Optional diagnostic output for unmatched nutrient labels.
    if verbose:
        print(matches, unmatched)
    
    return matches, unmatched

def filter_reference_dict(reference_dict, matches):
    """
    Filters the reference dictionary to include only keys that have a matching column.

    Parameters:
    - reference_dict (dict): Original reference dictionary.
    - matches (dict): Mapping of reference name -> db column name.

    Returns:
    - dict: Filtered reference dictionary.
    """
    return {k: v for k, v in reference_dict.items() if k in matches}


def validate_user_input(user_input):
    """
    Validates the structure and values of user input parameters.

    Parameters:
    - user_input (dict): Dictionary of user inputs.

    Raises:
    - ValueError: If any input is missing or invalid.
    """
    required_keys = {'sex', 'age', 'pregnant', 'trimester', 'breastfeeding', 'phase'}
    if not required_keys.issubset(user_input):
        raise ValueError("Missing one or more required keys in input.")

    if user_input['sex'] not in ['man', 'woman']:
        raise ValueError(f"Invalid sex: {user_input['sex']}")

    if not isinstance(user_input['age'], (int, float)) or not (1 <= user_input['age'] <= 120):
        raise ValueError(f"Invalid age: {user_input['age']}")

    if user_input['pregnant'] not in [True, False, None]:
        raise ValueError(f"Invalid pregnant value: {user_input['pregnant']}")

    if user_input['trimester'] not in [1, 2, 3, None]:
        raise ValueError(f"Invalid trimester: {user_input['trimester']}")

    if user_input['breastfeeding'] not in [True, False, None]:
        raise ValueError(f"Invalid breastfeeding value: {user_input['breastfeeding']}")

    if user_input['phase'] not in [1, 2, None]:
        raise ValueError(f"Invalid phase: {user_input['phase']}")


def calculate_extra_protein(age, pregnant, trimester, breastfeeding, phase):
    """
    Calculates the additional protein needed based on pregnancy or breastfeeding status.

    Parameters:
    - age (float): Age of the individual.
    - pregnant (bool): Whether the individual is pregnant.
    - trimester (int): Trimester of pregnancy (1, 2, or 3).
    - breastfeeding (bool): Whether the individual is breastfeeding.
    - phase (int): Breastfeeding phase (1 or 2).

    Returns:
    - float: Additional protein in grams.
    """
    extra_protein = 0
    extra_protein_pregnant = {1: 1, 2: 9, 3: 28}
    extra_protein_breastfeeding = {1: 13, 2: 19}

    if pregnant:
        extra_protein = extra_protein_pregnant[trimester]
    elif breastfeeding:
        extra_protein = extra_protein_breastfeeding[phase]
    else:
        print('Error in calculate_extra_protein function...')

    return extra_protein

def compute_niacin_intake(input_parameters):
    """
    Computes niacin intake based on energy needs.

    Parameters:
    - input_parameters (dict): Dictionary containing user input.

    Returns:
    - float: Niacin requirement in grams.
    """
    energy_kcal = get_kcal(input_parameters)
    niacin_per_mj = 1.6  # NE/MJ
    energy_mj = energy_kcal / 239
    niacin = niacin_per_mj * energy_mj / 1000  # convert mg to grams
    return niacin

def compute_tiamin_intake(input_parameters):
    """
    Computes thiamin intake based on energy needs.

    Parameters:
    - input_parameters (dict): Dictionary containing user input.

    Returns:
    - float: Thiamin requirement in grams.
    """
    energy_kcal = get_kcal(input_parameters)
    tiamin_per_mj = 0.1  # mg/MJ
    energy_mj = energy_kcal / 239
    tiamin = tiamin_per_mj * energy_mj / 1000  # convert mg to grams
    return tiamin


def add_multiplied_upper_bounds(lower_bounds_dict, factor=2.5):
    """
    Generates a dictionary of upper bounds by multiplying each lower bound by a factor.

    Parameters:
    - lower_bounds_dict (dict): Dictionary of nutrient lower bounds.
    - factor (float): The multiplier to apply (default is 2.5).

    Returns:
    - dict: Dictionary of upper bounds.
    """
    upper_bounds_dict = {nutrient: (ref_value * factor) for nutrient, ref_value in lower_bounds_dict.items()}
    return upper_bounds_dict

def compute_salt_threshold(age):
    """
    Retrieves salt and sodium upper limits for a specific age from the salt thresholds table.

    Parameters:
    - age (float): Age of the individual.

    Returns:
    - dict: {'ref_val_salt': max salt intake, 'ref_val_natrium': max sodium intake}
    """
    _load_research_runtime_data()
    salt_row = find_age_group_row(age, salt_thresholds)
    ref_val_salt = salt_row['Maxintag salt']
    ref_val_natrium = salt_row['Maxintag natrium']
    salt_threshold_dict = {'ref_val_salt': ref_val_salt, 'ref_val_natrium': ref_val_natrium}
    return salt_threshold_dict

def add_upper_bounds_by_func(incomplete_dict, input_parameters):
    """
    Adds nutrient-specific upper bounds based on reference functions and user parameters.

    Parameters:
    - incomplete_dict (dict): Dictionary of initial upper bounds (e.g., from multiplication).
    - input_parameters (dict): Dictionary containing 'sex', 'age', 'pregnant', 'trimester', 'breastfeeding', and 'phase'.

    Returns:
    - dict: Updated upper bounds dictionary with function-based limits applied.
    """
    sex, pregnant, trimester, breastfeeding, phase, age = _profile_values(input_parameters)
    return_dict = incomplete_dict.copy()

    # Add upper bounds for Salt and Natrium.
    return_dict['Natrium'] = compute_salt_threshold(age)['ref_val_natrium']
    return_dict['Salt'] = compute_salt_threshold(age)['ref_val_salt']

    return return_dict


def compute_fluoride_threshold(input_parameters, toxins_thresholds_df):
    """
    Computes the upper intake threshold for fluoride based on body weight.

    Parameters:
    - input_parameters (dict): Dictionary containing user parameters.
    - toxins_thresholds_df (pd.DataFrame): DataFrame containing toxin threshold values (mg/kg).

    Returns:
    - float: Fluoride intake limit in grams.
    """
    weight = get_reference_weight(input_parameters)
    fluoride_per_kg = toxins_thresholds_df.loc[
        toxins_thresholds_df['Compound'] == 'Fluoride', 'Limit value'
    ].values[0]  # mg/kg
    mg_to_g = 1000
    fluoride_ref_value = weight * fluoride_per_kg / mg_to_g
    return float(fluoride_ref_value)

def compute_mycotox1_threshold(input_parameters, toxins_thresholds_df):
    """
    Computes the upper intake threshold for Mycotoxins 1 based on body weight.

    Parameters:
    - input_parameters (dict): Dictionary containing user parameters.
    - toxins_thresholds_df (pd.DataFrame): DataFrame containing toxin threshold values (µg/kg).

    Returns:
    - float: Mycotoxins 1 intake limit in grams.
    """
    weight = get_reference_weight(input_parameters)
    mycotox1_per_kg = toxins_thresholds_df.loc[
        toxins_thresholds_df['Compound'] == 'Mycotoxins 1', 'Limit value'
    ].values[0]  # µg/kg
    µg_to_g = 1_000_000
    mycotox1_ref_value = weight * mycotox1_per_kg / µg_to_g
    return float(mycotox1_ref_value)

def compute_mycotox2_threshold(input_parameters, toxins_thresholds_df):
    """
    Computes the upper intake threshold for Mycotoxins 2 based on body weight.

    Parameters:
    - input_parameters (dict): Dictionary containing user parameters.
    - toxins_thresholds_df (pd.DataFrame): DataFrame containing toxin threshold values (µg/kg).

    Returns:
    - float: Mycotoxins 2 intake limit in grams.
    """
    weight = get_reference_weight(input_parameters)
    mycotox2_per_kg = toxins_thresholds_df.loc[
        toxins_thresholds_df['Compound'] == 'Mycotoxins 2', 'Limit value'
    ].values[0]  # µg/kg
    µg_to_g = 1_000_000
    mycotox2_ref_value = weight * mycotox2_per_kg / µg_to_g
    return float(mycotox2_ref_value)

def compute_PCB_dioxin_threshold(input_parameters, toxins_thresholds_df):
    """
    Computes the upper intake threshold for PCBs and Dioxins based on body weight.

    Parameters:
    - input_parameters (dict): Dictionary containing user parameters.
    - toxins_thresholds_df (pd.DataFrame): DataFrame containing toxin threshold values (pg/kg).

    Returns:
    - float: PCBs and Dioxins intake limit in grams.
    """
    weight = get_reference_weight(input_parameters)
    PCB_dioxin_per_kg = toxins_thresholds_df.loc[
        toxins_thresholds_df['Compound'] == 'PCBs and Dioxins', 'Limit value'
    ].values[0]  # pg/kg
    pg_to_g = 1_000_000_000_000
    PCB_dioxin_ref_value = weight * PCB_dioxin_per_kg / pg_to_g
    return float(PCB_dioxin_ref_value)

def compute_PFAS_threshold(input_parameters, toxins_thresholds_df):
    """
    Computes the upper intake threshold for PFAS based on body weight.

    Parameters:
    - input_parameters (dict): Dictionary containing user parameters.
    - toxins_thresholds_df (pd.DataFrame): DataFrame containing toxin threshold values (pg/kg).

    Returns:
    - float: PFAS intake limit in grams.
    """
    weight = get_reference_weight(input_parameters)
    PFAS_per_kg = toxins_thresholds_df.loc[
        toxins_thresholds_df['Compound'] == 'PFAS', 'Limit value'
    ].values[0]  # pg/kg
    ng_to_g = 1_000_000_000
    PFAS_ref_value = weight * PFAS_per_kg / ng_to_g
    return float(PFAS_ref_value)


def get_nutrients_energy_dict(input_parameters):
    """
    Retrieves the energy-related nutrient reference values based on user parameters.

    Parameters:
    - input_parameters (dict): Dictionary containing user input:
        'sex', 'age', 'pregnant', 'trimester', 'breastfeeding', and 'phase'.

    Returns:
    - pd.Series: A row containing energy-related nutrient values, excluding 'Referensvikt kg'.
    """
    _load_research_runtime_data()
    sex, pregnant, trimester, breastfeeding, phase, age = _profile_values(input_parameters)

    if sex == 'woman':
        # Use general or special case row for pregnant/breastfeeding women
        energy_row = find_age_group_row(age, df_intake_energy_women)
        if pregnant or breastfeeding:
            energy_row = complete_energy_row(pregnant, breastfeeding)
    elif sex == 'man':
        energy_row = find_age_group_row(age, df_intake_energy_men)
    else:
        print('Error in sex assignment...')
        return

    # Convert row values to reference dict and remove weight field
    nutrients_energy_dict = energy_row.iloc[1:].to_dict()
    del nutrients_energy_dict['Referensvikt kg']
    
    return nutrients_energy_dict

def get_nutrients_proportions_dict(input_parameters):
    """
    Retrieves the macronutrient proportion reference values based on user parameters.

    Parameters:
    - input_parameters (dict): Dictionary containing user input:
        'sex', 'age', 'pregnant', 'trimester', 'breastfeeding', and 'phase'.

    Returns:
    - dict: Dictionary of macronutrient proportion reference values (e.g., % of energy from fat, protein, carbs).
    """
    _load_research_runtime_data()
    sex, pregnant, trimester, breastfeeding, phase, age = _profile_values(input_parameters)

    if sex == 'woman':
        proportions_row = find_age_group_row(age, df_intake_proportions)
    elif sex == 'man':
        proportions_row = find_age_group_row(age, df_intake_proportions)
    else:
        print('Error in sex assignment...')
        return

    nutrients_proportions_dict = proportions_row.iloc[1:].to_dict()
    return nutrients_proportions_dict


def get_nutrients_lower_bounds_dict(input_parameters):
    """
    Retrieves the lower bound reference values for essential nutrients based on user parameters.
    Automatically adjusts protein values for pregnancy or breastfeeding and recalculates 
    niacin and thiamin based on energy requirements.

    Parameters:
    - input_parameters (dict): Dictionary containing user input:
        'sex', 'age', 'pregnant', 'trimester', 'breastfeeding', and 'phase'.

    Returns:
    - dict: Dictionary of nutrient lower bounds in grams or milligrams, depending on the nutrient.
    """
    _load_research_runtime_data()
    sex, pregnant, trimester, breastfeeding, phase, age = _profile_values(input_parameters)

    if sex == 'woman':
        nutrients_row = find_age_group_row(age, df_intake_nutrients_women)
        if pregnant or breastfeeding:
            nutrients_row = complete_nutrients_row(nutrients_row, age, pregnant, trimester, breastfeeding, phase)
    elif sex == 'man':
        nutrients_row = find_age_group_row(age, df_intake_nutrients_men)
    else:
        print('Error in sex assignment...')
        return

    nutrients_lower_bounds_dict = nutrients_row.iloc[1:].to_dict()

    # Add intake values calculated from energy requirement
    nutrients_lower_bounds_dict['Niacin'] = compute_niacin_intake(input_parameters)
    nutrients_lower_bounds_dict['Tiamin'] = compute_tiamin_intake(input_parameters)

    return nutrients_lower_bounds_dict


def get_nutrients_upper_bounds_dict(lower_bounds_dict, input_parameters):
    """
    Generates upper bound reference values for nutrients based on lower bounds and user-specific adjustments.
    Applies a default multiplier to all lower bounds, then overrides selected values using custom logic
    (e.g., salt and sodium based on age).

    Parameters:
    - lower_bounds_dict (dict): Dictionary of nutrient lower bounds.
    - input_parameters (dict): Dictionary containing user input:
        'sex', 'age', 'pregnant', 'trimester', 'breastfeeding', and 'phase'.

    Returns:
    - dict: Dictionary of nutrient upper bounds.
    """
    multiplied_dict = add_multiplied_upper_bounds(lower_bounds_dict)
    nutrients_upper_bounds_dict = add_upper_bounds_by_func(multiplied_dict, input_parameters)
    return nutrients_upper_bounds_dict


def get_toxins_upper_bounds_dict(toxins_thresholds_df, input_parameters):
    """
    Computes upper intake thresholds for selected environmental toxins based on body weight.

    Parameters:
    - toxins_thresholds_df (pd.DataFrame): DataFrame containing toxin threshold values (per kg body weight).
    - input_parameters (dict): Dictionary containing user input:
        'sex', 'age', 'pregnant', 'trimester', 'breastfeeding', and 'phase'.

    Returns:
    - dict: Dictionary of toxin names mapped to their upper intake thresholds (in grams).
    """
    # Initialize dictionary with toxin names
    toxins_upper_bounds_dict = {key: None for key in toxins_thresholds_df['Compound']}
    
    # Compute weight-adjusted upper thresholds for selected toxins
    toxins_upper_bounds_dict['Fluoride'] = compute_fluoride_threshold(input_parameters, toxins_thresholds_df)
    toxins_upper_bounds_dict['Mycotoxins 1'] = compute_mycotox1_threshold(input_parameters, toxins_thresholds_df)
    toxins_upper_bounds_dict['Mycotoxins 2'] = compute_mycotox2_threshold(input_parameters, toxins_thresholds_df)
    toxins_upper_bounds_dict['PCBs and Dioxins'] = compute_PCB_dioxin_threshold(input_parameters, toxins_thresholds_df)
    toxins_upper_bounds_dict['PFAS'] = compute_PFAS_threshold(input_parameters, toxins_thresholds_df)

    return toxins_upper_bounds_dict


def build_reference_values_dict(user_input, db, print_unmatched=False):
    """
    Constructs a comprehensive reference dictionary for dietary optimization based on user-specific parameters.
    
    This dictionary includes:
    - Minimum and maximum intake bounds for nutrients (only those present in the food database),
    - Energy requirements (e.g., kcal),
    - Macronutrient proportion targets (e.g., % energy from fat, carbs, protein),
    - Upper intake thresholds for environmental toxins.
    
    Additionally, it optionally prints any nutrients from the reference set that were not found in the food database.

    Parameters:
    ----------
    user_input : dict
        A dictionary with user-specific parameters:
        - 'sex' (str): 'man' or 'woman'
        - 'age' (float): Age in years
        - 'pregnant' (bool): Whether the individual is pregnant
        - 'trimester' (int or None): Trimester (1–3) if pregnant
        - 'breastfeeding' (bool): Whether the individual is breastfeeding
        - 'phase' (int or None): Breastfeeding phase (1 or 2)

    db : pd.DataFrame
        The food composition database containing columns for nutrients and toxins.

    return_db : bool, optional (default=True)
        If True, returns both the reference dictionary and the renamed database.
        If False, returns only the reference dictionary.

    print_unmatched : bool, optional (default=False)
        If True, prints a list of nutrients from the reference set that were not found in the food database.

    Returns:
    -------
    dict or (dict, pd.DataFrame)
        - reference_values_dict : dict
            A unified dictionary containing:
            - Per-nutrient min/max intake bounds: {nutrient: {'min': val, 'max': val}}
            - Energy targets (e.g., 'kcal': value)
            - Macronutrient proportions (e.g., '% Protein': value)
            - Toxin intake upper limits (e.g., 'PFAS': value)
        - db_renamed : pd.DataFrame (only if return_db=True)
            Food database with renamed columns matching the reference dictionary.

    Raises:
    ------
    ValueError:
        If any required key in the user_input dictionary is missing or invalid.
    """
    validate_user_input(user_input)
    _load_research_runtime_data()

    # Step 1: Get raw reference values
    nutrients_energy_dict = get_nutrients_energy_dict(user_input)
    nutrients_proportions_dict = get_nutrients_proportions_dict(user_input)
    nutrients_lower_bounds_dict = get_nutrients_lower_bounds_dict(user_input)
    nutrients_upper_bounds_dict = get_nutrients_upper_bounds_dict(nutrients_lower_bounds_dict, user_input)
    toxins_upper_bounds_dict = get_toxins_upper_bounds_dict(toxins_thresholds, user_input)

    # Step 2: Match nutrient names with food DB
    matches, unmatched = match_nutrients(nutrients_lower_bounds_dict, db.columns.tolist())
    if print_unmatched:
        print(unmatched)
    filtered_lower_bounds = filter_reference_dict(nutrients_lower_bounds_dict, matches)
    filtered_upper_bounds = filter_reference_dict(nutrients_upper_bounds_dict, matches)

    # Step 3: Build bounds dictionary
    nutrients_bounds_dict = {
        nutrient: {
            'min': filtered_lower_bounds[nutrient],
            'max': filtered_upper_bounds[nutrient]
        } for nutrient in filtered_lower_bounds
    }

    # Step 4: Merge all into one reference dictionary
    excluded_keys = {'Åldersgrupp', 'Referensvikt kg'}
    reference_values_dict = {k: v for k, v in {
        **nutrients_bounds_dict,
        **nutrients_energy_dict,
        **nutrients_proportions_dict,
        **toxins_upper_bounds_dict
    }.items() if k not in excluded_keys}
    
    return reference_values_dict


# Only run interactively if script is called directly
if __name__ == "__main__":
    from food_optimizer.get_user_input import get_ref_val_input_parameters
    _load_research_runtime_data()
    user_input = get_ref_val_input_parameters()
    ref_vals_dict = build_reference_values_dict(user_input, db)
    print(ref_vals_dict)

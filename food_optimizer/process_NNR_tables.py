"""Clean Nordic Nutrition Recommendations tables and export structured reference data."""

import ast
import re

import pandas as pd

from food_optimizer.config import DATA_DIR

# Build full input file path.
input_path = DATA_DIR / 'get_NNR_tables.xlsx'

# Load excel file into dataframes.
all_dfs = pd.read_excel(input_path, sheet_name=['intag_vitaminer_åldersgrupper', 
                                                        'intag_mineraler_åldersgrupper',
                                                        'intag_salt_åldersgrupper',
                                                        'intag_protein_åldersgrupper',
                                                        'intag_näring_6_till_23_m', 
                                                        'intag_näring_2_till_vux',
                                                        'intag_energi_barn',
                                                        'intag_energi_vux'])

# Dataframes with bsolute intakes of vitamins, minerals, salt and protein.
intake_vitamins = all_dfs['intag_vitaminer_åldersgrupper']
intake_minerals = all_dfs['intag_mineraler_åldersgrupper']
intake_salt = all_dfs['intag_salt_åldersgrupper']
intake_protein = all_dfs['intag_protein_åldersgrupper']
intake_dfs = [intake_vitamins, intake_minerals, intake_salt, intake_protein] # Group list.

# Dataframes with proportional intakes of macro nutrients.
proportions_6_23_m = all_dfs['intag_näring_6_till_23_m']
proportions_2_plus = all_dfs['intag_näring_2_till_vux']
proportions_df = [proportions_6_23_m, proportions_2_plus] # Group list.

# Dataframes with absolute intakes of energy.
intake_energy_1_17 = all_dfs['intag_energi_barn']
intake_energy_18_plus = all_dfs['intag_energi_vux']
energy_dfs = [intake_energy_1_17, intake_energy_18_plus]

# Conversion factors (add/edit as needed)
unit_to_gram = {
    "mg": 1e-3,
    "µg": 1e-6,
    "g": 1,
    "NE/MJ": 1e-3,
    "mg/MJ": 1e-3,
    "µg/MJ": 1e-6,
    "RE": 1e-6,
    "α-TE": 1e-3,  # Optional special cases
    }

def remove_superscript_reference(text):
    """
    Removes superscript references (e.g., ^{1}) from a string.

    Parameters:
    - text (str): Input string.

    Returns:
    - str: Cleaned string.
    """
    if isinstance(text, str):
        return re.sub(r'\^\{\d+\}', '', text).strip()
    return text

def add_unit_to_label(df):
    """
    Appends the unit (from the first row) to each column label and removes the first row.

    Parameters:
    - df (pd.DataFrame): DataFrame where first row contains unit labels.

    Returns:
    - pd.DataFrame: Updated DataFrame with units added to column names.
    """
    new_columns = {col: f"{col} {df.iloc[0][col]}" for col in df.columns}
    df = df.rename(columns=new_columns)
    df = df.drop(index=0).reset_index(drop=True)
    return df

def fix_åldersgrupp_label(df):
    """
    Renames any column labeled with 'Ålder...nan' to 'Åldersgrupp'.

    Parameters:
    - df (pd.DataFrame): The DataFrame to process.

    Returns:
    - pd.DataFrame: Updated DataFrame with corrected column name.
    """
    new_columns = {}
    for col in df.columns:
        if re.search(r'Ålder.*nan', str(col), re.IGNORECASE):
            new_columns[col] = 'Åldersgrupp'
    return df.rename(columns=new_columns)

def replace_with_zero(df, chars_to_replace_list):
    """
    Replaces specified characters or strings with 0 throughout the DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - chars_to_replace_list (list): Characters or strings to replace.

    Returns:
    - pd.DataFrame: Updated DataFrame with replacements applied.
    """
    for char in chars_to_replace_list:
        df = df.replace(char, 0)
    return df

def convert_columns_to_grams(df, unit_to_gram, clean_colnames=True):
    """
    Converts columns in a DataFrame to grams using a unit-to-gram conversion dictionary.

    Parameters:
    - df (pd.DataFrame): DataFrame with values to convert.
    - unit_to_gram (dict): Dictionary mapping units to conversion factors.
    - clean_colnames (bool): Whether to remove units from column names.

    Returns:
    - pd.DataFrame: Converted DataFrame.
    """
    df = df.copy()
    new_columns = {}

    for col in df.columns:
        matches = [unit for unit in unit_to_gram if re.search(rf'\b{re.escape(unit)}\b', col)]
        if matches:
            unit = matches[0]
            factor = unit_to_gram[unit]
            cleaned_col = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(cleaned_col, errors='coerce') * factor
            if clean_colnames:
                new_col = re.sub(rf'\s*{re.escape(unit)}\b', '', col).strip()
                new_columns[col] = new_col

    if clean_colnames:
        df.rename(columns=new_columns, inplace=True)

    return df

def remove_regex_match(text, pattern, repl=''):
    """
    Removes or replaces text in a string using a regex pattern.

    Parameters:
    - text (str): Input string.
    - pattern (str): Regex pattern.
    - repl (str): Replacement string.

    Returns:
    - str: Processed string.
    """
    if isinstance(text, str):
        return re.sub(pattern, repl, text)
    return text

def fix_column_names(df):
    """
    Cleans DataFrame column names by removing superscripts and units like '/MJ'.

    Parameters:
    - df (pd.DataFrame): DataFrame with column names to clean.

    Returns:
    - pd.DataFrame: Updated DataFrame with cleaned column names.
    """
    column_names = df.columns.tolist()
    clean_names = []
    for name in column_names:
        clean_name = remove_superscript_reference(name)
        clean_name = remove_regex_match(clean_name, '/MJ', repl='') 
        clean_names.append(clean_name)
    df.columns = clean_names
    return df

def split_df_on_markers(df, marker_values, name_prefix, first_group, column='Åldersgrupp'):
    """
    Splits a DataFrame into multiple sub-DataFrames based on marker values in a specific column.

    Parameters:
    - df (pd.DataFrame): DataFrame to split.
    - marker_values (list): Values that indicate a new group.
    - name_prefix (str): Prefix for each group name.
    - first_group (str): Name for the first group before the first marker.
    - column (str): Column to check for marker values.

    Returns:
    - dict: Dictionary of {group_name: sub-DataFrame}.
    """
    dfs = {}
    start_idx = 0
    df_group_name = f'{name_prefix}_{first_group}'

    for marker in marker_values:
        try:
            end_idx = df[df[column] == marker].index[0]
        except IndexError:
            print(f"Warning: marker '{marker}' not found, skipping.")
            continue

        dfs[df_group_name] = df.iloc[start_idx:end_idx].copy()
        start_idx = end_idx + 1
        df_group_name = f'{name_prefix}_{marker}'

    if start_idx < len(df):
        dfs[df_group_name] = df.iloc[start_idx:].copy()

    return dfs

def create_split_df(source_df, name, range_values):
    """
    Creates a single split DataFrame using a range of row indices.

    Parameters:
    - source_df (pd.DataFrame): Original DataFrame.
    - name (str): Name for the output key.
    - range_values (tuple): Start and end row indices (inclusive).

    Returns:
    - dict: {name: split DataFrame}
    """
    row_indices = range(range_values[0], range_values[1] + 1)
    split_df = source_df.loc[row_indices].copy()
    split_df.columns = source_df.columns
    return {name: split_df}

def cell_contains(cell, words, match_type="any", match_whole_words=True):
    """
    Checks if a string cell contains any or all specified words.

    Parameters:
    - cell (str): Text to search.
    - words (list): List of words to match.
    - match_type (str): 'any' or 'all' (default: 'any').
    - match_whole_words (bool): Whether to match whole words only.

    Returns:
    - bool: True if match conditions are met, else False.
    """
    if not isinstance(cell, str):
        return False

    cell = cell.lower()
    if match_whole_words:
        matches = [re.search(rf"\b{re.escape(word)}\b", cell) for word in words]
    else:
        matches = [re.search(re.escape(word), cell) for word in words]

    return any(matches) if match_type == "any" else all(matches)

def convert_aldersgrupp_column(df, column='Åldersgrupp'):
    """
    Converts age group labels from months to a consistent year-based format.

    Parameters:
    - df (pd.DataFrame): DataFrame with age group labels.
    - column (str): Column containing age group strings.

    Returns:
    - pd.DataFrame: Updated DataFrame with converted age labels.
    """
    def convert_label(label):
        if label in ['mån', 'månader']:
            nums = list(map(float, re.findall(r'\d+', label)))
            nums_in_years = [round(n / 12, 2) for n in nums]
            if len(nums_in_years) == 1:
                return f"≤ {nums_in_years[0]} år"
            else:
                return f"{nums_in_years[0]}–{nums_in_years[1]} år"
        return label

    df[column] = df[column].apply(convert_label)
    return df

def parse_age_range(label):
    """
    Parses a Swedish age range label and returns a numeric tuple or special group string.

    Parameters:
    - label (str): Label such as '0.5–0.92 år', '< 1 år', '≥ 70 år', '70+', 'Gravida', 'Ammande'.

    Returns:
    - tuple | str | None: (min, max) tuple for age ranges, capitalized special group, or None if unrecognized.
    """
    label = label.strip().lower()

    if label in ['gravida', 'ammande']:
        return label.capitalize()

    match = re.match(r'(\d+(?:[\.,]\d+)?)\s*[–-]\s*(\d+(?:[\.,]\d+)?)', label)
    if match:
        val1, val2 = match.groups()
        return (float(val1.replace(',', '.')), float(val2.replace(',', '.')))

    match = re.match(r'(≤|<)\s*(\d+(?:[\.,]\d+)?)', label)
    if match:
        val = float(match.group(2).replace(',', '.'))
        return (0, val)

    match = re.match(r'(≥|>|)?\s*(\d+(?:[\.,]\d+)?)(\+|)', label)
    if match and (match.group(1) in ['≥', '>'] or match.group(3) == '+'):
        val = float(match.group(2).replace(',', '.'))
        return (val + 1, 120)

    return None

def overwrite_aldersgrupp_with_tuples(df, column='Åldersgrupp'):
    """
    Converts human-readable age group labels in a DataFrame to tuples.

    Parameters:
    - df (pd.DataFrame): DataFrame with age group column.
    - column (str): Name of the age group column.

    Returns:
    - pd.DataFrame: Updated DataFrame with tuple-formatted age groups.
    """
    df[column] = df[column].apply(parse_age_range)
    return df

def pivot_macro_table(df):
    """
    Pivots a macro table so that nutrient labels become column headers for a single age group.

    Parameters:
    - df (pd.DataFrame): DataFrame where row 0 has age group and others are nutrient-energy pairs.

    Returns:
    - pd.DataFrame: Pivoted DataFrame with one row per age group and nutrients as columns.
    """
    alder_tuple = df.loc[0, 'Energiprocent']
    reshaped = df.iloc[1:].set_index('Näringsämne')['Energiprocent'].T.to_frame().T
    reshaped.columns.name = None
    reshaped.insert(0, 'Åldersgrupp', [alder_tuple])
    return reshaped.reset_index(drop=True)

def convert_range_to_tuple(value):
    """
    Converts a string representing a numeric range to a tuple.

    Parameters:
    - value (str): A string like '2–4' or '1.5 - 3.0'.

    Returns:
    - tuple | original: Tuple of (min, max) or original value if not parseable.
    """
    if isinstance(value, str):
        match = re.match(r'^\s*(\d+(?:[.,]\d+)?)\s*[–-]\s*(\d+(?:[.,]\d+)?)\s*$', value)
        if match:
            val1, val2 = match.groups()
            return (float(val1.replace(',', '.')), float(val2.replace(',', '.')))
    return value

def find_age_group_row(age, df):
    """
    Finds the row in the DataFrame that matches a given age within its 'Åldersgrupp' range.

    Parameters:
    - age (float): Age to match.
    - df (pd.DataFrame): DataFrame with a column 'Åldersgrupp' containing ranges.

    Returns:
    - pd.Series: Matching row if found, else None.
    """
    for _, row in df.iterrows():
        group = row['Åldersgrupp']
        try:
            group = ast.literal_eval(group) if isinstance(group, str) and group.startswith("(") else group
            if isinstance(group, tuple) and group[0] <= age <= group[1]:
                return row
        except:
            continue

def calculate_protein(df):
    """
    Calculates the total protein requirement based on weight and protein per kg.

    Parameters:
    - df (pd.DataFrame): DataFrame row with 'Referensvikt kg' and 'Protein/kg' columns.

    Returns:
    - float: Calculated protein amount in grams.
    """
    weight = df['Referensvikt kg']
    protein_per_kg = df['Protein/kg']
    return weight * protein_per_kg

intake_dfs = [df.map(remove_superscript_reference) for df in intake_dfs]
intake_dfs = [add_unit_to_label(df) for df in intake_dfs]
intake_dfs = [fix_åldersgrupp_label(df) for df in intake_dfs]
intake_dfs = [replace_with_zero(df, ['-']) for df in intake_dfs]
intake_dfs = [df.fillna(0) for df in intake_dfs]
intake_dfs = [convert_columns_to_grams(df, unit_to_gram) for df in intake_dfs]
intake_dfs = [fix_column_names(df) for df in intake_dfs]

# Name dfs based on compound.
intake_vitamins = intake_dfs[0]
intake_minerals = intake_dfs[1]
intake_salt = intake_dfs[2]
intake_protein = intake_dfs[3]

# Process intake minerals df.
intake_minerals = intake_minerals.drop('Fluor', axis=1) # Not essential acc. to gov.

# Process intake protein df.
intake_protein = intake_protein.drop('AR', axis=1)
intake_protein = intake_protein.rename(columns={'RI': 'Protein/kg'})

# Create dict with compound dfs for easier name processing.
intake_dfs_dict = {'intake_vitamins': intake_vitamins,
                   'intake_minerals': intake_minerals,
                   'intake_salt': intake_salt,
                   'intake_protein': intake_protein}

# Split into specific dataframes based on groups.
intake_dfs_all_dicts = {compound_name: 
                        split_df_on_markers(compound_dict, ['BARN', 'KVINNOR', 'MÄN'], compound_name, 'SPÄDBARN') 
                        for compound_name, compound_dict in intake_dfs_dict.items()}

# Create single layer dict.
intake_dfs_all_dicts = {
    key: value
    for inner in intake_dfs_all_dicts.copy().values()
    for key, value in inner.items()
}

# Create ranges for age groups.
for df_name, df in intake_dfs_all_dicts.items():
    convert_aldersgrupp_column(df)
    overwrite_aldersgrupp_with_tuples(df)

# Concat dfs for final database structure and ad to dict.
dfs_dict_nutrients_by_group = {}
for sex in ['women', 'men']:
    list_name = f'intake_nutrients_{sex}'
    df_name = f'df_intake_nutrients_{sex}'
    list_name =  []
    for compound in ['vitamins', 'minerals', 'salt', 'protein']:
        dfs_to_concat = []
        if sex == 'women':
            dfs_to_concat = [intake_dfs_all_dicts[f'intake_{compound}_SPÄDBARN'],
                             intake_dfs_all_dicts[f'intake_{compound}_BARN'],
                             intake_dfs_all_dicts[f'intake_{compound}_KVINNOR']]
        elif sex == 'men':
            dfs_to_concat = [intake_dfs_all_dicts[f'intake_{compound}_SPÄDBARN'],
                             intake_dfs_all_dicts[f'intake_{compound}_BARN'],
                             intake_dfs_all_dicts[f'intake_{compound}_MÄN']]
        else:
            print('Error in group identity.')
        list_name.append(pd.concat(dfs_to_concat, join='outer'))
    df = pd.concat(list_name, axis=1, join='outer')
    df = df.loc[:, ~df.columns.duplicated()]

    # Manually convert months to years.
    df.loc[0:0, 'Åldersgrupp'] = df.loc[0:0, 'Åldersgrupp'].apply(lambda x: (0.0, 0.5))
    df.loc[1:1, 'Åldersgrupp'] = df.loc[1:1, 'Åldersgrupp'].apply(lambda x: (0.6, 0.9))
    dfs_dict_nutrients_by_group[df_name] = df

# Split and clean column names
proportions_6_11_m = proportions_6_23_m[['Näringsämne', 'Energiprocent']].copy()
proportions_12_23_m = proportions_6_23_m[['Näringsämne', 'Energiprocent2']].copy()
proportions_12_23_m = proportions_12_23_m.rename(columns={'Energiprocent2': 'Energiprocent'})

# Add age span tuples. Cast the target column to object first because
# modern pandas treats tuple assignment through .iloc as iterable assignment.
proportions_6_11_m['Energiprocent'] = proportions_6_11_m['Energiprocent'].astype(object)
proportions_12_23_m['Energiprocent'] = proportions_12_23_m['Energiprocent'].astype(object)
proportions_6_11_m.at[proportions_6_11_m.index[0], 'Energiprocent'] = (0.5, 0.999)
proportions_12_23_m.at[proportions_12_23_m.index[0], 'Energiprocent'] = (1, 1.999)

# Process before pivoting to new format.
proportions_2_plus = pd.concat([pd.DataFrame(
    {'Näringsämne': [None], 'Energiprocent': [(2, 120)]}), 
    proportions_2_plus], ignore_index=True).reset_index(drop=True)


# Pivot to new format.
proportions_6_11 = pivot_macro_table(proportions_6_11_m)
proportions_12_23 = pivot_macro_table(proportions_12_23_m)
proportions_2_plus = pivot_macro_table(proportions_2_plus)

# Create the list
proportions_df = [proportions_6_11, proportions_12_23, proportions_2_plus]

# Clean each df (only cell-wise if safe)
proportions_df = [df.map(remove_superscript_reference) for df in proportions_df]
proportions_df = [df.rename(columns=lambda col: remove_superscript_reference(col))
              for df in proportions_df]

df_intake_proportions = pd.concat(proportions_df, join='outer').reset_index()
df_intake_proportions = df_intake_proportions[['Åldersgrupp', 'FETT', 'KOLHYDRATER', 'PROTEIN']]
df_intake_proportions = df_intake_proportions.map(convert_range_to_tuple)
df_intake_proportions = df_intake_proportions.rename(columns={'FETT': 'Fett procent',
                                                              'KOLHYDRATER': 'Kolhydrater procent',
                                                              'PROTEIN': 'Protein procent'})

# Put df in a dict to fit larger structure.
dfs_dict_proportions_by_group = {'df_intake_proportions': df_intake_proportions}

intake_energy_1_17 = intake_energy_1_17.map(remove_superscript_reference)
intake_energy_1_17 = fix_åldersgrupp_label(intake_energy_1_17)
intake_energy_1_17 = replace_with_zero(intake_energy_1_17, ['-'])
intake_energy_1_17 = intake_energy_1_17.fillna(0)
intake_energy_1_17.columns = [remove_superscript_reference(col) for col in intake_energy_1_17.columns]

# Split dataframe into multiple based on group.
intake_energy_dfs_children = split_df_on_markers(intake_energy_1_17, 
                                        ['FLICKOR', 'POJKAR'], 
                                        'intake_energy', 
                                        'BARN <= 10', 
                                         column='Åldersgrupp')

# Concat the child intake df with the sexes for complete dfs and update dict.
intake_energy_dfs_children['intake_energy_FLICKOR'] = pd.concat(
    [intake_energy_dfs_children['intake_energy_BARN <= 10'], intake_energy_dfs_children['intake_energy_FLICKOR']],
    join='outer')
intake_energy_dfs_children['intake_energy_POJKAR'] = pd.concat(
    [intake_energy_dfs_children['intake_energy_BARN <= 10'], intake_energy_dfs_children['intake_energy_POJKAR']],
    join='outer')
del intake_energy_dfs_children['intake_energy_BARN <= 10']

# Process the new dfs.
for key, df in intake_energy_dfs_children.items():
    df.drop('REE MJ/d', axis=1, inplace=True)
    for col in ['Referensvikt kg', 'Skattat energibehov MJ/d']:
        df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
    df['Skattat energibehov MJ/d'] = df['Skattat energibehov MJ/d'] * 239.005736
    df.rename(columns={'Skattat energibehov MJ/d': 'Skattat energibehov kcal/d'}, inplace=True)
    df = overwrite_aldersgrupp_with_tuples(df)

# Clean and organize.
intake_energy_18_plus = intake_energy_18_plus.map(
    remove_superscript_reference)
intake_energy_18_plus = add_unit_to_label(
    intake_energy_18_plus)
intake_energy_18_plus = fix_åldersgrupp_label(
    intake_energy_18_plus)
intake_energy_18_plus = replace_with_zero(
    intake_energy_18_plus, ['-'])
intake_energy_18_plus = intake_energy_18_plus.fillna(0)
intake_energy_18_plus.columns = [
    remove_superscript_reference(col) for col in intake_energy_18_plus.columns]
intake_energy_18_plus = overwrite_aldersgrupp_with_tuples(
    intake_energy_18_plus)
intake_energy_18_plus = intake_energy_18_plus.drop(                                # Use close to median values.
    ['REE MJ/d', 'Medelvärde PAL 1,4 MJ/d', 'Aktiv PAL 1,8 MJ/d'], axis=1)
intake_energy_18_plus = intake_energy_18_plus.rename(
    columns={'Medelvärde PAL2 1,6 MJ/d': 'Skattat energibehov kcal/d'})
intake_energy_18_plus['Skattat energibehov kcal/d'] = intake_energy_18_plus['Skattat energibehov kcal/d'].apply(
    lambda x: x * 239.005736)


# Create dataframes.
intake_energy_KVINNOR = create_split_df(
    intake_energy_18_plus, 'intake_energy_KVINNOR', [1, 4])['intake_energy_KVINNOR']
intake_energy_MÄN = create_split_df(
    intake_energy_18_plus, 'intake_energy_MÄN', [6, 9])['intake_energy_MÄN']
intake_energy_GRAVIDA = create_split_df(
    intake_energy_18_plus, 'intake_energy_GRAVIDA', [11, 11])['intake_energy_GRAVIDA']
intake_energy_AMMANDE = create_split_df(
    intake_energy_18_plus, 'intake_energy_AMMANDE', [13, 13])['intake_energy_AMMANDE']

# Concat dfs for final database structure and ad to dict.
df_intake_energy_women = pd.concat([intake_energy_dfs_children['intake_energy_FLICKOR'],
                                    intake_energy_KVINNOR,
                                    intake_energy_GRAVIDA,
                                    intake_energy_AMMANDE],
                                    join='outer', ignore_index=True)
df_intake_energy_women.loc[[9, 10], 'Åldersgrupp'] = ['Gravida', 'Ammande']

df_intake_energy_men = pd.concat([intake_energy_dfs_children['intake_energy_POJKAR'],
                                    intake_energy_MÄN],
                                    join='outer')
dfs_dict_energy_by_group = {'df_intake_energy_women': df_intake_energy_women,
                            'df_intake_energy_men': df_intake_energy_men}

ref_weight_women = [None, None, 13.6, 20.7, 30.8, 46.5, 57.8, 64.2, 64.1, 62.5, 60.6, 76.4, 62.4]
ref_weight_men = [None, None, 13.6, 20.7, 30.8, 48.2, 65.6, 75.2, 74.8, 73.0, 70.6]
dfs_dict_nutrients_by_group['df_intake_nutrients_women']['Referensvikt kg'] = ref_weight_women
dfs_dict_nutrients_by_group['df_intake_nutrients_men']['Referensvikt kg'] = ref_weight_men

# These values are calculated later from the profile's energy requirement.
dfs_dict_nutrients_by_group['df_intake_nutrients_women']['Niacin'] = None
dfs_dict_nutrients_by_group['df_intake_nutrients_men']['Niacin'] = None
dfs_dict_nutrients_by_group['df_intake_nutrients_women']['Tiamin'] = None
dfs_dict_nutrients_by_group['df_intake_nutrients_men']['Tiamin'] = None


for key, df in dfs_dict_nutrients_by_group.items():
    df['Protein'] = calculate_protein(df)
    del df['Referensvikt kg']
    del df['Protein/kg']

for key, df in dfs_dict_energy_by_group.items():
    dfs_dict_energy_by_group[key] = df.rename(columns={'Skattat energibehov kcal/d': 'kcal'})

import pickle

# Create dict.
dict_dfs_NNR_tables = {'dfs_dict_nutrients_by_group': dfs_dict_nutrients_by_group,
                       'dfs_dict_proportions_by_group': dfs_dict_proportions_by_group,
                       'dfs_dict_energy_by_group': dfs_dict_energy_by_group}

# Save the dict.
filename = 'dict_dfs_NNR_tables.pkl'
full_output_path = DATA_DIR / filename
with open(full_output_path, 'wb') as f:
    pickle.dump(dict_dfs_NNR_tables, f)

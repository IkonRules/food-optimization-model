"""Clean Swedish Market Basket Study tables and export toxin datasets.

This module processes the Excel workbook produced from the Swedish Market
Basket Study tables. It creates:
- a product lookup table,
- an unconverted compound table,
- a converted mean-compound table,
- a toxin-only database summed by compound category.

The module is designed for package-style execution from the project root, e.g.:

    python -m food_optimizer.process_matkorgen_tables
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

from food_optimizer.config import DATA_DIR


INPUT_FILENAME = "get_matkorgen_tables.xlsx"

COMPOUNDS_TOTAL_UNCONVERTED_FILENAME = "df_compounds_total_unconverted.csv"
COMPOUNDS_SINGLE_UNIT_MEANS_FILENAME = "df_compounds_single_unit_means.csv"
MATKORGEN_TOXINS_DATABASE_FILENAME = "df_matkorgen_toxins_database.csv"
MATKORGEN_PRODUCTS_FILENAME = "df_matkorgen_products.csv"

PRODUCT_TYPES = [
    "Cereal products",
    "Pastries",
    "Meat",
    "Lean fish",
    "Fatty fish",
    "Meat substitutes",
    "Lean dairy products",
    "Fatty dairy products",
    "Plant-based drinks",
    "Eggs",
    "Fats and oils",
    "Vegetables",
    "Fruits",
    "Potatoes",
    "Sugar and sweets",
    "Beverages",
    "Coffee and tea",
    "Processed meat",
    "Pizza, hand pie",
]

NAMES_TO_CHANGE = {
    "Meat substitutes^{1}": "Meat substitutes",
    "Meat\nsubstitutes": "Meat substitutes",
    "Fatty dairy": "Fatty dairy products",
    "Fatty dairy\nproducts": "Fatty dairy products",
    "Fatty fish^{2}": "Fatty fish",
    "Fats and\noils": "Fats and oils",
    "Lean dairy\nproducts": "Lean dairy products",
    "Lean fish^{1}": "Lean fish",
    "Potatoes^{2}": "Potatoes",
    "Fruits^{3}": "Fruits",
    "Coffee and\ntea": "Coffee and tea",
    "Cereal\nproducts": "Cereal products",
    "Plant-": "Plant-based drinks",
    "Plant-based\ndrinks": "Plant-based drinks",
    "Meat3": "Meat",
    "Sugar and": "Sugar and sweets",
    "substitutes": "empty_rows",
    "products": "empty_rows",
    "based": "empty_rows",
    "drinks": "empty_rows",
    "sweets": "empty_rows",
    "products4": "empty_rows",
    "Column1": "Compound",
    "Column2": "Parameter",
}

UNIT_CONVERSION = {
    "g/kg": 1 / 10,
    "mg/kg": 1 / 10000,
    "μg/kg": 1 / 10000000,
    "µg/kg": 1 / 10000000,
    "ug/kg": 1 / 10000000,
    "ng/kg": 1 / 1000000000,
    "kcal/kg": 1 / 10,
    "MJ/kg": 23.9005736,
    "g/MJ": 1 / 23900.5736,
    "pg TEQ/kg": 1e-13,
    "RE/kg": 1 / 10,
    "%": 1,
}

UNIT_UPDATE = {
    "g/100g": ["g/kg", "mg/kg", "μg/kg", "µg/kg", "ug/kg", "ng/kg", "pg TEQ/kg"],
    "kcal/100g": ["kcal/kg", "MJ/kg"],
    "100g/kcal": ["g/MJ"],
    "RE/100g": ["RE/kg"],
    "%": ["%"],
}

NON_TOXIN_COMPOUND_TYPES = {
    "Macros",
    "Carbohydrates",
    "Minerals",
    "Vitamins",
    "Fatty Acids",
}


def move_column(df: pd.DataFrame, col_name: str, new_index: int) -> pd.DataFrame:
    """Return a copy of ``df`` with ``col_name`` moved to ``new_index``."""
    cols = df.columns.tolist()
    cols.insert(new_index, cols.pop(cols.index(col_name)))
    return df.loc[:, cols]


def extract_unit(compound: str) -> str | None:
    """Extract the text inside parentheses from a column or compound label."""
    if not isinstance(compound, str):
        return None

    match = re.search(r"\((.*?)\)", compound)
    if match:
        return match.group(1).strip()

    return None


def remove_parentheses(text):
    """Remove parenthetical content from a string."""
    if isinstance(text, str):
        return re.sub(r"\s*\(.*?\)", "", text).strip()
    return text


def remove_superscript_reference(text):
    """Remove superscript-style references such as ``^{1}`` from a string."""
    if isinstance(text, str):
        return re.sub(r"\^\{\d+\}", "", text).strip()
    return text


def process_whitespace(text):
    """Normalize repeated whitespace to single spaces."""
    if isinstance(text, str):
        return re.sub(r"\s+", " ", text).strip()
    return text


def get_non_float_unique_values_excluding(
    df: pd.DataFrame,
    exclude_columns: list[str] | set[str],
) -> set:
    """Return unique non-float values outside the excluded columns."""
    non_float_values = set()

    for col in df.columns.difference(exclude_columns):
        for val in df[col].unique():
            if pd.notna(val) and not isinstance(val, float):
                non_float_values.add(val)

    return non_float_values


def average_range(val):
    """Convert string ranges such as ``'2 - 6'`` to their midpoint."""
    if isinstance(val, str) and re.match(r"^-?\d+(\.\d+)?\s*-\s*-?\d+(\.\d+)?$", val):
        parts = re.split(r"\s*-\s*", val)
        return (float(parts[0]) + float(parts[1])) / 2

    try:
        return float(val)
    except (TypeError, ValueError):
        return val


def convert_less_than_values(df: pd.DataFrame) -> pd.DataFrame:
    """Convert strings such as ``'<5'`` or ``'< 0.02'`` to numeric approximations."""
    def parse_cell(cell):
        if isinstance(cell, str) and cell.strip().startswith("<"):
            match = re.search(r"<\s*(\d+\.?\d*)", cell)
            if match:
                value_str = match.group(1)
                value = float(value_str)

                if "." in value_str:
                    decimal_places = len(value_str.split(".")[1])
                    adjustment = 10 ** (-(decimal_places + 1))
                else:
                    adjustment = 0.1

                return value - adjustment

        return cell

    return df.map(parse_cell)


def find_id_variable_with_targets(
    df: pd.DataFrame,
    targets: list,
    column: str,
) -> list:
    """Return unique values from ``column`` for rows containing target values."""
    mask = df.isin(targets).any(axis=1)
    return df.loc[mask, column].unique().tolist()


def find_nan_locations(df: pd.DataFrame) -> list[tuple]:
    """Return ``(row_index, column_name)`` locations of NaN values."""
    nan_mask = df.isna()
    nan_locations = nan_mask.stack()
    return nan_locations[nan_locations].index.tolist()


def convert_units(value, unit):
    """Convert one numeric cell using ``UNIT_CONVERSION``."""
    factor = UNIT_CONVERSION.get(unit)

    if pd.isna(value) or factor is None:
        return value

    try:
        if isinstance(value, str) and value.strip().startswith("<"):
            match = re.search(r"<\s*(\d+\.?\d*)", value)
            if match:
                value = float(match.group(1)) - 1
                return value * factor
            return value

        if isinstance(value, (int, float, np.integer, np.floating)):
            return value * factor

        return value

    except Exception:
        return value


def update_unit_labels(unit_label):
    """Map an original unit label to a standardized output unit label."""
    for updated_label, original_labels in UNIT_UPDATE.items():
        if unit_label in original_labels:
            return updated_label

    return unit_label


def load_matkorgen_workbook(input_path: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load the Swedish Market Basket Study workbook into a sheet dictionary."""
    input_path = input_path or DATA_DIR / INPUT_FILENAME
    return pd.read_excel(input_path, sheet_name=None)


def process_products_table(all_sheets_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Extract and clean the product table from the workbook sheet dictionary."""
    if "Products" not in all_sheets_dict:
        raise KeyError("Expected a sheet named 'Products' in the Matkorgen workbook.")

    df_matkorgen_products = all_sheets_dict["Products"].copy()

    df_matkorgen_products = df_matkorgen_products.rename(columns={"Column1": "Product"})
    df_matkorgen_products["Product type"] = df_matkorgen_products["Product"].apply(
        lambda x: x if x in PRODUCT_TYPES else None
    )
    df_matkorgen_products["Product type"] = df_matkorgen_products["Product type"].ffill()

    df_matkorgen_products = df_matkorgen_products.drop(
        df_matkorgen_products[df_matkorgen_products["Product"] == "Total"].index
    )
    df_matkorgen_products = df_matkorgen_products.drop(
        df_matkorgen_products[
            df_matkorgen_products["Product"] == df_matkorgen_products["Product type"]
        ].index
    )

    return df_matkorgen_products.reset_index(drop=True)


def build_compounds_table(all_sheets_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a cleaned compound table from all non-product sheets."""
    compound_sheets = {
        sheet_name: df.copy()
        for sheet_name, df in all_sheets_dict.items()
        if sheet_name != "Products"
    }

    for sheet_name, df in compound_sheets.items():
        clean_sheet_name = remove_superscript_reference(sheet_name)
        df["Compound type"] = clean_sheet_name

        df.rename(
            columns={col: NAMES_TO_CHANGE[col] for col in df.columns if col in NAMES_TO_CHANGE},
            inplace=True,
        )

    dfs = list(compound_sheets.values())
    df_compounds = pd.concat(dfs, axis=0, join="outer", ignore_index=True)

    df_compounds = move_column(df_compounds, "Compound type", 0)
    df_compounds["Compound"] = df_compounds["Compound"].ffill()

    df_compounds["Unit"] = df_compounds["Compound"].apply(extract_unit)
    df_compounds["Unit"] = df_compounds["Unit"].bfill()
    df_compounds = move_column(df_compounds, "Unit", 2)

    df_compounds["Compound"] = df_compounds["Compound"].apply(remove_parentheses)
    df_compounds["Compound"] = df_compounds["Compound"].replace("", pd.NA)
    df_compounds["Compound"] = df_compounds["Compound"].ffill()
    df_compounds["Compound"] = df_compounds["Compound"].apply(process_whitespace)

    # This row is a metadata row in the original workbook export. Use a guarded
    # drop so the module still works if the workbook changes row positions.
    df_compounds = df_compounds.drop(index=660, errors="ignore")

    return df_compounds


def clean_compound_values(df_compounds: pd.DataFrame) -> pd.DataFrame:
    """Clean non-numeric markers and convert compound values to numeric values where possible."""
    df_compounds = convert_less_than_values(df_compounds.copy())

    non_product_columns = ["Compound type", "Compound", "Unit", "Parameter"]

    for col in df_compounds.columns.difference(non_product_columns):
        df_compounds[col] = df_compounds[col].apply(average_range)
        df_compounds[col] = df_compounds[col].apply(remove_parentheses)
        df_compounds[col] = df_compounds[col].apply(remove_superscript_reference)
        df_compounds[col] = df_compounds[col].apply(
            lambda x: float(0) if x in ["0*", "<LOD", "<LOQ", "nd"] else x
        )
        df_compounds[col] = df_compounds[col].apply(
            lambda x: float(x) if isinstance(x, int) else x
        )
        df_compounds[col] = df_compounds[col].apply(
            lambda x: float(x) if isinstance(x, str) and x.replace(".", "", 1).isdigit() else x
        )
        df_compounds[col] = df_compounds[col].fillna(0)

    return df_compounds


def convert_compound_units(df_compounds: pd.DataFrame) -> pd.DataFrame:
    """Convert compound values to standardized per-100g units."""
    df_compounds = df_compounds.copy()
    value_columns = df_compounds.columns.difference(
        ["Compound type", "Compound", "Unit", "Parameter"]
    )

    for idx, row in df_compounds.iterrows():
        unit = row["Unit"]
        for col in value_columns:
            df_compounds.at[idx, col] = convert_units(row[col], unit)

    df_compounds["Unit"] = df_compounds["Unit"].apply(update_unit_labels)

    return df_compounds


def build_toxins_database(df_compounds_single_unit_means: pd.DataFrame) -> pd.DataFrame:
    """Aggregate converted mean values into a toxin-only database."""
    columns_to_sum = list(
        df_compounds_single_unit_means.columns.difference(
            ["Compound type", "Compound", "Unit", "Parameter"]
        )
    )

    toxins_database = (
        df_compounds_single_unit_means
        .groupby(["Compound type", "Unit"])[columns_to_sum]
        .sum()
        .reset_index()
    )

    toxins_database = toxins_database[
        ~toxins_database["Compound type"].isin(NON_TOXIN_COMPOUND_TYPES)
    ]

    return toxins_database.reset_index(drop=True)


def process_matkorgen_tables(
    input_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Process the Matkorgen workbook and write cleaned CSV outputs.

    Parameters
    ----------
    input_path:
        Optional path to the source Excel workbook. Defaults to
        ``DATA_DIR / INPUT_FILENAME``.
    output_dir:
        Optional directory where CSV outputs should be saved. Defaults to
        ``DATA_DIR``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary containing all generated output DataFrames.
    """
    output_dir = output_dir or DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    all_sheets_dict = load_matkorgen_workbook(input_path)

    df_matkorgen_products = process_products_table(all_sheets_dict)
    df_compounds = build_compounds_table(all_sheets_dict)
    df_compounds = clean_compound_values(df_compounds)

    df_compounds_total_unconverted = df_compounds.copy()

    df_compounds = convert_compound_units(df_compounds)
    df_compounds_single_unit_means = df_compounds[
        df_compounds["Parameter"] == "Mean"
    ].copy()

    df_matkorgen_toxins_database = build_toxins_database(
        df_compounds_single_unit_means
    )

    outputs = {
        COMPOUNDS_TOTAL_UNCONVERTED_FILENAME: df_compounds_total_unconverted,
        COMPOUNDS_SINGLE_UNIT_MEANS_FILENAME: df_compounds_single_unit_means,
        MATKORGEN_TOXINS_DATABASE_FILENAME: df_matkorgen_toxins_database,
        MATKORGEN_PRODUCTS_FILENAME: df_matkorgen_products,
    }

    for filename, df in outputs.items():
        df.to_csv(output_dir / filename, index=False)

    return {
        "df_compounds_total_unconverted": df_compounds_total_unconverted,
        "df_compounds_single_unit_means": df_compounds_single_unit_means,
        "df_matkorgen_toxins_database": df_matkorgen_toxins_database,
        "df_matkorgen_products": df_matkorgen_products,
    }


if __name__ == "__main__":
    results = process_matkorgen_tables()
    print("Saved Matkorgen outputs to:", DATA_DIR)
    for name, df in results.items():
        print(f"{name}: {df.shape}")

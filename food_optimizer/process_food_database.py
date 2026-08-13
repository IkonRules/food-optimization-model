"""Clean the Livsmedelsverket food database and export a processed CSV."""

from pathlib import Path
import re

import pandas as pd

from food_optimizer.config import DATA_DIR

INPUT_FILENAME = "LivsmedelsDB.xlsx"
OUTPUT_FILENAME = "df_food_database.csv"

UNIT_TO_GRAM_PER_100G = {
    "NE/mg": 1e-3,
    "RE/µg": 1e-6,
    "mg": 1e-3,
    "µg": 1e-6,
    "g": 1,
    "kcal": 1,
}


def remove_parentheses(text):
    """
    Remove text enclosed in parentheses from a string.

    Examples
    --------
    "Protein (g)" -> "Protein"
    """
    if isinstance(text, str):
        return re.sub(r"\s*\(.*?\)", "", text).strip()
    return text


def extract_unit(compound):
    """
    Extract the unit from a column label.

    Examples
    --------
    "Protein (g)" -> "g"
    "Vitamin D (µg)" -> "µg"
    """
    if not isinstance(compound, str):
        return None

    match = re.search(r"\((.*?)\)", compound)
    if match:
        return match.group(1).strip()

    return None


def load_food_database(input_path: Path | None = None) -> pd.DataFrame:
    """
    Load the raw Livsmedelsverket Excel food database.

    Parameters
    ----------
    input_path:
        Optional explicit path to the raw Excel file. If omitted, the file is
        loaded from DATA_DIR / INPUT_FILENAME.

    Returns
    -------
    pd.DataFrame
        Raw food database.
    """
    input_path = input_path or DATA_DIR / INPUT_FILENAME
    return pd.read_excel(input_path, skiprows=2)


def clean_food_database(db: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize the raw Livsmedelsverket food database.

    The function removes unused columns, converts nutrient columns to grams
    per 100 g where possible, and removes unit suffixes from column names.
    """
    db = db.copy()

    remove_columns = [
        "Livsmedelsnummer",
        "Energi (kJ)",
        "Avfall (skal etc.) (%)",
    ]

    db = db.drop(columns=remove_columns, errors="ignore")

    label_unit_dict = {
        label: extract_unit(label)
        for label in db.columns.tolist()
    }

    label_unit_dict["EPA (C20:5) (g)"] = "g"
    label_unit_dict["DPA (C22:5) (g)"] = "g"
    label_unit_dict["DHA (C22:6) (g)"] = "g"

    non_nutrient_columns = {"Livsmedelsnamn", "Gruppering"}

    for col, unit in label_unit_dict.items():
        if unit is None:
            continue

        if col not in db.columns:
            continue

        if col in non_nutrient_columns:
            continue

        factor = UNIT_TO_GRAM_PER_100G.get(unit)

        if factor is None:
            print(f"No conversion factor for unit '{unit}' in column '{col}'")
            continue

        db[col] = pd.to_numeric(db[col], errors="coerce") * factor

    db.columns = [remove_parentheses(col) for col in db.columns.tolist()]

    return db


def export_food_database(
    db: pd.DataFrame,
    output_path: Path | None = None,
) -> Path:
    """
    Save the cleaned food database as a CSV file.

    Parameters
    ----------
    db:
        Cleaned food database.
    output_path:
        Optional explicit output path. If omitted, the file is written to
        DATA_DIR / OUTPUT_FILENAME.

    Returns
    -------
    Path
        Path to the written CSV file.
    """
    output_path = output_path or DATA_DIR / OUTPUT_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)

    db.to_csv(output_path, index=False)

    return output_path


def process_food_database(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Load, clean, export, and return the Livsmedelsverket food database.
    """
    raw_db = load_food_database(input_path)
    cleaned_db = clean_food_database(raw_db)
    export_food_database(cleaned_db, output_path)

    return cleaned_db


if __name__ == "__main__":
    output_path = DATA_DIR / OUTPUT_FILENAME
    process_food_database(output_path=output_path)
    print(f"Saved cleaned food database to: {output_path}")

"""Utilities for displaying and plotting optimizer results.

The plotting functions in this module use seaborn's dark-grid theme and
vertical bar charts with diagonal x-axis labels for readability.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_TOXIN_COLUMNS = [
    "Acrylamide",
    "Brominated flame retardants",
    "Fluoride",
    "Glycidol",
    "Metals",
    "Mycotoxins 1",
    "Mycotoxins 2",
    "Organochlorine pesticides",
    "PAHs",
    "PCAs",
    "PCBs and Dioxins",
    "PFAS",
    "PFRs",
    "Plasticizers",
]

DEFAULT_REFERENCE_COLUMNS = [
    "Fett procent",
    "Kolhydrater procent",
    "Protein procent",
    "kcal",
    "Vitamin A",
    "Vitamin D",
    "Tiamin",
    "Riboflavin",
    "Niacin",
    "Vitamin B6",
    "Folat",
    "Vitamin C",
    "Vitamin E",
    "Vitamin K",
    "Vitamin B12",
    "Kalcium",
    "Järn",
    "Zink",
    "Fosfor",
    "Kalium",
    "Magnesium",
    "Jod",
    "Selen",
    "Fluor",
    "Natrium",
    "Salt",
    "Protein",
    "Total weight (g)",
]


def apply_plot_theme() -> None:
    """Apply the plotting theme used by all display functions."""
    sns.set_theme(style="darkgrid", context="notebook")


def _format_x_labels(ax, rotation: int = 45) -> None:
    """Rotate x-axis labels diagonally and right-align them."""
    ax.tick_params(axis="x", labelrotation=rotation)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")


def _save_if_requested(fig, save_path: Path | None) -> None:
    """Save a matplotlib figure if ``save_path`` is provided."""
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")


def split_solution_columns(
    recommendations: pd.DataFrame,
    toxin_columns: list[str] | None = None,
    reference_columns: list[str] | None = None,
) -> dict[str, list[str]]:
    """Split a recommendations table into food, nutrient, toxin, and realism columns."""
    toxin_columns = toxin_columns or DEFAULT_TOXIN_COLUMNS
    reference_columns = reference_columns or DEFAULT_REFERENCE_COLUMNS

    cols = list(recommendations.columns)
    toxin_cols = [c for c in toxin_columns if c in cols]
    nutrient_cols = [c for c in reference_columns if c in cols]
    realism_cols = [
        c for c in cols
        if c.endswith(" count") or c.endswith(" grams") or c == "Selected foods"
    ]

    non_food = set(["Solution"] + toxin_cols + nutrient_cols + realism_cols)
    food_cols = [c for c in cols if c not in non_food]

    return {
        "foods": food_cols,
        "nutrients": nutrient_cols,
        "toxins": toxin_cols,
        "realism": realism_cols,
    }


def solution_foods_table(
    recommendations: pd.DataFrame,
    solution_index: int = 0,
    min_grams: float = 1e-3,
) -> pd.DataFrame:
    """Return selected foods and gram amounts for one solution."""
    groups = split_solution_columns(recommendations)
    row = recommendations.iloc[solution_index]
    foods = row[groups["foods"]]
    foods = foods[foods > min_grams].sort_values(ascending=False)

    return foods.rename("grams").reset_index().rename(columns={"index": "food"})


def solution_nutrients_table(
    recommendations: pd.DataFrame,
    solution_index: int = 0,
) -> pd.DataFrame:
    """Return nutrient totals for one solution.

    Most nutrient columns are expressed in grams per day after preprocessing.
    The exceptions are columns such as kcal and percentage columns.
    """
    groups = split_solution_columns(recommendations)
    row = recommendations.iloc[solution_index]

    return (
        row[groups["nutrients"]]
        .rename("amount")
        .reset_index()
        .rename(columns={"index": "nutrient"})
    )


def solution_toxins_table(
    recommendations: pd.DataFrame,
    solution_index: int = 0,
) -> pd.DataFrame:
    """Return toxin totals for one solution.

    Toxin totals are expected to be grams per day because the processed toxin
    columns are converted to gram-based units before optimization.
    """
    groups = split_solution_columns(recommendations)
    row = recommendations.iloc[solution_index]

    return (
        row[groups["toxins"]]
        .rename("amount")
        .reset_index()
        .rename(columns={"index": "toxin"})
    )


def solution_realism_table(
    recommendations: pd.DataFrame,
    solution_index: int = 0,
) -> pd.DataFrame:
    """Return realism-count and realism-weight summary for one solution."""
    groups = split_solution_columns(recommendations)
    row = recommendations.iloc[solution_index]

    return (
        row[groups["realism"]]
        .rename("value")
        .reset_index()
        .rename(columns={"index": "metric"})
    )


def plot_food_weights(
    recommendations: pd.DataFrame,
    solution_index: int = 0,
    min_grams: float = 1e-3,
    save_path: Path | None = None,
):
    """Plot selected food amounts as vertical bars in grams per day."""
    apply_plot_theme()

    table = solution_foods_table(recommendations, solution_index, min_grams)
    fig_width = max(10, 0.55 * len(table))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    sns.barplot(data=table, x="food", y="grams", ax=ax)

    ax.set_xlabel("Food")
    ax.set_ylabel("Amount (g/day)")
    ax.set_title(f"Food weights — solution {solution_index + 1}")
    _format_x_labels(ax)

    fig.tight_layout()
    _save_if_requested(fig, save_path)

    return fig, ax


def plot_toxin_totals(
    recommendations: pd.DataFrame,
    solution_index: int = 0,
    save_path: Path | None = None,
):
    """Plot toxin totals as vertical bars in grams per day."""
    apply_plot_theme()

    table = solution_toxins_table(recommendations, solution_index)
    table = table[table["amount"] > 0].sort_values("amount", ascending=False)

    fig_width = max(10, 0.6 * len(table))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    sns.barplot(data=table, x="toxin", y="amount", ax=ax)

    ax.set_xlabel("Toxin")
    ax.set_ylabel("Amount (g/day)")
    ax.set_title(f"Toxin totals — solution {solution_index + 1}")
    _format_x_labels(ax)

    fig.tight_layout()
    _save_if_requested(fig, save_path)

    return fig, ax


def plot_macro_percentages(
    recommendations: pd.DataFrame,
    solution_index: int = 0,
    save_path: Path | None = None,
):
    """Plot kcal percentage from fat, carbohydrates, and protein."""
    apply_plot_theme()

    macro_cols = ["Fett procent", "Kolhydrater procent", "Protein procent"]
    row = recommendations.iloc[solution_index]
    values = row[[c for c in macro_cols if c in recommendations.columns]]

    table = (
        values.rename("percent")
        .reset_index()
        .rename(columns={"index": "macro"})
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=table, x="macro", y="percent", ax=ax)

    ax.set_xlabel("Macronutrient")
    ax.set_ylabel("Energy share (%)")
    ax.set_title(f"Macronutrient energy distribution — solution {solution_index + 1}")
    _format_x_labels(ax)

    fig.tight_layout()
    _save_if_requested(fig, save_path)

    return fig, ax


def plot_nutrient_totals(
    recommendations: pd.DataFrame,
    solution_index: int = 0,
    exclude_energy_and_percentages: bool = True,
    save_path: Path | None = None,
):
    """Plot nutrient totals as vertical bars.

    Parameters
    ----------
    recommendations:
        Optimizer output table.
    solution_index:
        Row index of the solution to plot.
    exclude_energy_and_percentages:
        If True, exclude kcal and macronutrient percentage columns so the
        y-axis can be interpreted as grams per day.
    save_path:
        Optional path where the figure should be saved.
    """
    apply_plot_theme()

    table = solution_nutrients_table(recommendations, solution_index)

    if exclude_energy_and_percentages:
        excluded = {"kcal", "Fett procent", "Kolhydrater procent", "Protein procent"}
        table = table[~table["nutrient"].isin(excluded)]

    table = table[table["amount"] > 0].sort_values("amount", ascending=False)

    fig_width = max(10, 0.55 * len(table))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    sns.barplot(data=table, x="nutrient", y="amount", ax=ax)

    ax.set_xlabel("Nutrient")
    ax.set_ylabel("Amount (g/day)")
    ax.set_title(f"Nutrient totals — solution {solution_index + 1}")
    _format_x_labels(ax)

    fig.tight_layout()
    _save_if_requested(fig, save_path)

    return fig, ax


def plot_realism_summary(
    recommendations: pd.DataFrame,
    solution_index: int = 0,
    save_path: Path | None = None,
):
    """Plot realism summary metrics as vertical bars."""
    apply_plot_theme()

    table = solution_realism_table(recommendations, solution_index)
    table = table[table["value"] > 0].sort_values("value", ascending=False)

    fig_width = max(10, 0.6 * len(table))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    sns.barplot(data=table, x="metric", y="value", ax=ax)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.set_title(f"Realism summary — solution {solution_index + 1}")
    _format_x_labels(ax)

    fig.tight_layout()
    _save_if_requested(fig, save_path)

    return fig, ax

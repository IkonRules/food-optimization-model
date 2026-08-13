# Food Optimization Model

A Python constrained-optimization model that generates daily food combinations subject to nutritional, energy, toxicological, and practical eating constraints. The original research workflow combines Swedish food-composition and market-basket data with Nordic nutrition reference values. The public repository runs the same LP/MILP optimizer with a small, entirely artificial demonstration dataset.

> This is an exploratory quantitative-modelling project, not a clinical nutrition or diet-recommendation system.

## Synthetic public demonstration

The deterministic public example minimizes total food weight while enforcing artificial nutrient, energy, contaminant-marker, plate-model, food-count, portion, and realism constraints. Every bundled food name and numeric value was invented for this repository and is not derived from an official dataset.

| Example metric | Result |
| --- | ---: |
| Synthetic foods available | 14 |
| Selected foods | 6 |
| Total food weight | 583.7 g/day |
| Artificial model energy | 1,980.0 kcal/day |
| Maximum foods allowed | 8 |

The command makes the data boundary explicit and independently rechecks the public numeric constraints:

```text
Food Optimization Model — Synthetic Portfolio Demonstration
------------------------------------------------------------
This example uses artificial data included with the repository.
It does not use or redistribute the original source datasets.

Selected foods: 6 / 8 maximum
Total food weight: 583.7 g/day
Model energy: 1,980.0 kcal/day (allowed 1,980.0-2,420.0)
Validation: PASS (16 nutrient, marker, energy, and macro checks)
```

The synthetic values exist solely to demonstrate the software path; they have no nutritional or toxicological meaning.

### Example result from the full research dataset

A previously verified run using the author's local full research dataset selected 14 foods, totalled 663.2 g/day, produced approximately 2,430.7 kcal/day, and passed all 55 independently recalculated nutrient, toxin, energy, and macronutrient checks. That result is retained as project evidence; the underlying source-derived runtime datasets are not bundled with the public repository.

Both outputs are mathematical solutions, not suggested menus. Feasibility does not imply clinical or practical dietary suitability; see [Limitations](#limitations).

## Overview

The model asks whether a set of food quantities can simultaneously meet a person's reference intakes, keep total energy and macronutrient shares within ranges, stay below selected toxicological thresholds, and resemble a plausible day of food. The result is one or more feasible combinations expressed as grams of each selected food, together with calculated nutrient and contaminant totals.

This is an optimization problem because many constraints compete. A food that efficiently supplies one nutrient can increase weight, energy, another nutrient, or a mapped contaminant. The solver searches this feasible region rather than applying a sequence of independent food rules.

## Project context

The model was developed iteratively over several years of university study. This portfolio cleanup preserves that implementation and its scientific assumptions rather than replacing it with a newly designed demonstration. The work represented in the repository spans source-data cleaning, cross-dataset mapping, user-specific reference construction, LP/MILP formulation, realism constraints, validation, and result presentation.

## Key features

- Preserves a full local workflow integrating Swedish food composition, NNR 2023 reference values, and Swedish Market Basket Study contaminant groups.
- Includes an openly distributable synthetic workflow that exercises the same optimizer without redistributing source-derived research data.
- Builds age- and sex-specific nutrient, energy, macronutrient, salt, and selected body-weight-adjusted toxicological bounds.
- Supports LP objectives and MILP food-use variables.
- Enforces nutrient lower/upper bounds, an energy range, and macronutrient energy-share ranges.
- Applies contaminant upper bounds where a numerical threshold is available and can minimize total mapped contaminants.
- Supports food-count limits, plate-model weight shares, per-food portion limits, category counts, and category gram caps.
- Can generate weight-prioritized, contaminant-prioritized, or scalarized Pareto-style solutions.
- Includes six synthetic constraint tests and a deterministic, non-interactive synthetic example.

## Computational approach

For food item `i`, the main continuous decision variable is:

```text
x_i = grams of food i selected for one day
```

Binary `use_i` variables are added when food-count or realism constraints are enabled. Nutrient and contaminant data are stored per 100 g, so a daily total for component `j` is a linear expression:

```text
sum_i x_i * amount_ij / 100
```

The model combines:

- lower and upper nutrient bounds;
- a ±10% interval around the profile's reference energy;
- linearized fat, carbohydrate, and protein energy-share bounds;
- upper limits for contaminants with quantified thresholds;
- plate-model bounds on food-group shares of total weight;
- binary food-count and conditional portion constraints; and
- total gram/count caps for selected realism groups.

The objective can prioritize total food weight, mapped contaminant load, or weighted combinations used to explore trade-offs. PuLP formulates the problem and its bundled CBC solver handles the LP/MILP solve.

## Data sources

- [Swedish Food Composition Database](https://www.livsmedelsverket.se/livsmedel-och-innehall/naringsamne/livsmedelsdatabasen/sok-naringsinnehall/) — individual-food nutrient composition.
- [Nordic Nutrition Recommendations 2023](https://www.norden.org/en/publication/nordic-nutrition-recommendations-2023) — reference intakes, energy values, and macronutrient ranges.
- [Swedish Market Basket Study 2022](https://www.livsmedelsverket.se/om-oss/publikationer/artiklar/2024/l-2024-nr-08-swedish-market-basket-study-2022) — grouped contaminant measurements.

These sources use different classifications. The project maps Market Basket food groups onto individual Swedish foods using rule-based category assignments. That makes the contaminant component useful for exploratory modelling, but it does not turn group means into food-specific laboratory measurements. See [data/README.md](data/README.md) for the runtime files, provenance inputs, attribution requirements, and redistribution caution.

The public repository does not redistribute the processed research datasets. Users who want to reproduce the full workflow must obtain or construct the required local inputs in accordance with the respective source terms. The bundled synthetic data are original artificial fixtures created only to demonstrate the code and optimization workflow.

## Project structure

```text
food-optimization-model/
├── food_optimizer/                 # Core model and data-processing modules
│   ├── optimization_module.py      # LP/MILP formulation and solution generation
│   ├── realism_constraints.py      # Portion, count, and food-group realism layer
│   ├── get_ref_vals_dict.py        # User-specific reference-value construction
│   ├── demo_data.py                # Explicit synthetic-demo loading boundary
│   ├── process_*.py                # Source-table cleaning workflows
│   ├── mapping_toxins_to_foods.py  # Cross-dataset category mapping
│   ├── display_results.py          # Tables and plots
│   └── main.py                     # Interactive entry point
├── data/demo/                      # Bundled artificial demonstration inputs
├── data/README.md                  # Provenance and local research-data guide
├── examples/run_example.py         # Deterministic, non-interactive workflow
├── tests/                          # Focused, discoverable unit tests
├── pyproject.toml                  # Package metadata and dependencies
└── requirements.txt                # One-command editable installation
```

The package remains intentionally flat: the existing modules already separate reference-value construction, data preparation, optimization, realism, and presentation clearly enough for a project of this size.

## Installation

Python 3.10 or newer is required. From a clone or downloaded copy of the repository:

```bash
cd food-optimization-model
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Then install the package and its documented data/plot extras:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PuLP supplies the CBC solver used by the model. The optional local full-research NNR pickle requires PyArrow to deserialize its pandas-backed values; both are declared dependencies.

The supported workflow runs from an editable repository checkout so the top-level `data/` directory remains available. The project is not currently distributed as a standalone wheel or PyPI package.

## Usage

Run the deterministic portfolio example from the repository root:

```bash
python examples/run_example.py
```

The example uses a fixed NumPy seed so the small objective perturbation is repeatable. It loads only `data/demo/`, selects an artificial reference profile through the same profile-energy lookup logic, runs the production MILP optimizer, prints selected synthetic foods, and recalculates the public bounds before reporting success.

The original interactive full-research workflow remains available when the five local source-derived runtime files listed in [data/README.md](data/README.md) are present:

```bash
python -m food_optimizer.main
```

It asks for a profile and optimization settings. Those local files are loaded lazily, so they are not needed to import the package, run the public example, or run the tests. For automated use, call `build_reference_values_dict(...)` and `optimize_daily_intake(...)` directly.

## Testing

Run the focused test suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

The tests use the bundled synthetic fixtures and demonstrate that:

- nutrient lower and upper bounds are respected;
- the energy interval and contaminant upper bound are respected;
- maximum food count, conditional portions, and realism count groups are enforced;
- impossible configurations raise `OptimizationInfeasibleError`;
- invalid food-count settings and user profiles are rejected; and
- programmatic profiles do not depend on dictionary insertion order.

## Limitations

- Market Basket contaminant values are group-level means mapped to individual foods, not measurements of those individual foods.
- Food-group mapping is rule-based and some classifications are inherently ambiguous.
- Numerical thresholds are available for only a subset of contaminant groups; groups without an established limit can affect the objective but do not receive a hard upper bound.
- Several nutrient upper bounds are modelling assumptions derived by multiplying lower bounds; they are not all official tolerable upper intake levels.
- Plate-model proportions and realism metadata are simplified representations of eating behaviour. They improve mathematical solutions but do not guarantee an appetizing or meal-ready menu.
- The model does not represent taste, allergies, cost, cooking loss, bioavailability, meal timing, food interactions, or individual medical needs.
- Reference energy is based on NNR profile tables rather than a personalized measurement of body composition or activity.
- Source compatibility, manual table extraction, and the date/version of source databases affect reproducibility.
- This project must not be used as clinical nutrition guidance.

## Future work

- Review the 102 realism-metadata rows currently flagged for manual attention.
- Replace group-mean contaminant mapping where food-specific analytical data becomes available.
- Add optional preference, allergy, price, and meal-structure constraints without changing the core formulation.
- Store reference tables in a transparent non-pickle format after validating a migration against current results.

## Technologies

Python · pandas · NumPy · PuLP/CBC · PyArrow · openpyxl · matplotlib · seaborn · `unittest`

## License

No code license has been selected yet. An MIT license could later cover the author's source code and bundled synthetic demonstration data, but it must not be presented as licensing third-party publications or locally retained source-derived datasets. Those materials remain governed by their respective terms; see [data/README.md](data/README.md).

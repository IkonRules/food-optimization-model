# Data guide

The public and full-research workflows have an explicit data boundary. A clean
public checkout contains artificial demonstration inputs, while source-derived
research runtime data remain local unless their redistribution status is
established separately.

## 1. Bundled synthetic demonstration data

The files under [`data/demo/`](demo/) were created specifically for this
repository. Their names, categories, constraints, profiles, and numeric values
are artificial; they are not copied, transformed, sampled, or approximated from
the official sources listed below.

| File | Demonstration role |
| --- | --- |
| `demo/synthetic_foods.csv` | Fourteen invented foods with artificial per-100 g model values. |
| `demo/synthetic_reference_profiles.csv` | Invented adult energy-profile rows. |
| `demo/synthetic_constraints.json` | Artificial nutrient, marker, macro, plate, count, and gram bounds. |
| `demo/synthetic_realism_metadata.csv` | Artificial portions, food roles, and selection groups. |

These fixtures may be used as part of the repository's eventual source-code
license. They exist only to demonstrate loading, profile selection, LP/MILP
optimization, binary food-use constraints, realism validation, and reporting.
They are not nutrition or toxicology data.

## 2. Original external sources

The local full-research workflow was constructed from three independently
structured sources:

- [Swedish Food Composition Database — Livsmedelsverket](https://www.livsmedelsverket.se/livsmedel-och-innehall/naringsamne/livsmedelsdatabasen/sok-naringsinnehall/)
- [Nordic Nutrition Recommendations 2023 — Nordic Council of Ministers](https://www.norden.org/en/publication/nordic-nutrition-recommendations-2023)
- [Swedish Market Basket Study 2022 — Livsmedelsverket report L 2024 nr 08](https://www.livsmedelsverket.se/om-oss/publikationer/artiklar/2024/l-2024-nr-08-swedish-market-basket-study-2022)

NNR-derived reference values should cite:

> Blomhoff, R., Andersen, R., Arnesen, E.K., Christensen, J.J., Eneroth, H.,
> Erkkola, M., Gudanaviciene, I., Halldorsson, T.I., Høyer-Lund, A., Lemming,
> E.W., Meltzer, H.M., Pitsi, T., Schwab, U., Siksna, I., Thorsdottir, I., &
> Trolle, E. *Nordic Nutrition Recommendations 2023*. Nordic Council of
> Ministers, 2023.

The source publications and downloaded databases remain governed by their
publishers' terms. Attribution does not itself establish permission to
redistribute transformed data or imply endorsement. Record the exact version
and download date of the Swedish Food Composition Database when maintaining the
local research workflow.

The Market Basket source reports grouped contaminant measurements. The project
maps those groups onto individual foods using rule-based categories; the mapped
values are not food-specific laboratory measurements. Manual extraction and
mapping introduce additional uncertainty.

## 3. Local full-research runtime data

These processed source-derived datasets are intentionally excluded from the
public repository because redistribution rights for the transformed data have
not been established:

- `df_foods_and_toxins_database.csv`
- `dict_dfs_NNR_tables.pkl`
- `toxins_thresholds.csv`
- `salt_thresholds.csv`
- `food_realism_metadata.csv`

When supplied locally under `data/`, these files continue to support the
interactive full-research workflow:

```bash
python -m food_optimizer.main
```

They are loaded only when that workflow is selected. The public example and
tests do not access them. The pickle should be loaded only from a trusted local
source; Python pickle is unsafe for untrusted files.

Raw source PDFs, `LivsmedelsDB.xlsx`, `get_NNR_tables.xlsx`,
`get_matkorgen_tables.xlsx`, and rebuildable preprocessing intermediates are
also ignored. The manually prepared workbooks can contain Microsoft Office
author metadata and should remain local or be sanitized before any separate
distribution.

No repository license has been selected. A future source-code license may cover
the author's code and synthetic fixtures, but it does not automatically cover
third-party source material or source-derived local datasets.

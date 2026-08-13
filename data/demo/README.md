# Synthetic demonstration data

Every file in this directory was created specifically for the public portfolio
demonstration. The food names, category labels, reference profiles, constraints,
and numeric values are artificial. They are not copied, transformed, sampled,
or approximated from the Swedish Food Composition Database, Nordic Nutrition
Recommendations, or Swedish Market Basket Study.

These fixtures exist only to exercise the repository's real loading,
reference-profile selection, LP/MILP optimization, realism, validation, and
reporting code. They must not be interpreted as nutrition or toxicology data.

- `synthetic_foods.csv` contains 14 invented foods and per-100 g model values.
- `synthetic_reference_profiles.csv` supplies invented energy-profile rows.
- `synthetic_constraints.json` supplies invented nutrient, energy-share,
  contaminant-marker, plate-model, count, and group-gram bounds.
- `synthetic_realism_metadata.csv` supplies invented portions and food roles.

"""Deterministic tests of the optimizer using bundled artificial inputs."""

import unittest

import numpy as np

from food_optimizer.demo_data import load_demo_bundle
from food_optimizer.optimization_module import (
    OptimizationInfeasibleError,
    optimize_daily_intake,
    parse_constraints,
)


MAX_FOODS = 8


class TestOptimizerConstraints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_demo_bundle()

    def setUp(self):
        np.random.seed(7)

    def solve_demo(self):
        return optimize_daily_intake(
            self.bundle.foods,
            self.bundle.reference_values,
            self.bundle.toxin_columns,
            num_solutions=1,
            prioritize="weight",
            max_foods=MAX_FOODS,
            prefer_sparse=True,
            plate_model_constraints=self.bundle.plate_model_constraints,
            realism_metadata=self.bundle.realism_metadata,
            realism_config=self.bundle.realism_config,
            enforce_realism=True,
        )

    def test_solution_respects_core_bounds(self):
        result = self.solve_demo()
        row = result.loc[0]
        constraints = parse_constraints(self.bundle.reference_values)
        selected = {
            food: float(row.get(food, 0))
            for food in self.bundle.foods["Livsmedelsnamn"]
            if float(row.get(food, 0)) > 1e-3
        }

        self.assertLessEqual(len(selected), MAX_FOODS)
        for name, rule in constraints["absolute_constraints"].items():
            value = float(row[name])
            if "min" in rule:
                self.assertGreaterEqual(value, float(rule["min"]) - 1e-4, name)
            if "max" in rule:
                self.assertLessEqual(value, float(rule["max"]) + 1e-4, name)

        for name, rule in constraints["macro_percent_constraints"].items():
            lower, upper = rule["range"]
            self.assertGreaterEqual(float(row[name]), lower - 1e-4, name)
            self.assertLessEqual(float(row[name]), upper + 1e-4, name)

        group_lookup = self.bundle.foods.set_index("Livsmedelsnamn")["Gruppering"]
        total_weight = sum(selected.values())
        for name, rule in self.bundle.plate_model_constraints.items():
            group_weight = sum(
                grams
                for food, grams in selected.items()
                if group_lookup[food] in rule["groupings"]
            )
            share = 100 * group_weight / total_weight
            lower, upper = rule["proportion"]
            self.assertGreaterEqual(share, lower - 1e-4, name)
            self.assertLessEqual(share, upper + 1e-4, name)

    def test_realism_portions_and_counts_are_enforced(self):
        row = self.solve_demo().loc[0]
        metadata = self.bundle.realism_metadata.set_index("Livsmedelsnamn")
        selected = {
            food: float(row.get(food, 0))
            for food in metadata.index
            if float(row.get(food, 0)) > 1e-3
        }

        for food, grams in selected.items():
            limits = metadata.loc[food]
            self.assertGreaterEqual(grams, float(limits["min_if_selected_g"]) - 1e-4)
            self.assertLessEqual(grams, float(limits["max_if_selected_g"]) + 1e-4)
            self.assertLessEqual(grams, float(limits["max_daily_g"]) + 1e-4)

        for group, (lower, upper) in self.bundle.realism_config.count_bounds.items():
            count = sum(metadata.loc[food, "count_group"] == group for food in selected)
            if lower is not None:
                self.assertGreaterEqual(count, lower, group)
            if upper is not None:
                self.assertLessEqual(count, upper, group)

        for group, (lower, upper) in self.bundle.realism_config.group_gram_bounds.items():
            grams = sum(
                amount
                for food, amount in selected.items()
                if metadata.loc[food, "realism_group"] == group
            )
            if lower is not None:
                self.assertGreaterEqual(grams, lower - 1e-4, group)
            if upper is not None:
                self.assertLessEqual(grams, upper + 1e-4, group)

    def test_impossible_food_count_raises_clear_error(self):
        with self.assertRaises(OptimizationInfeasibleError):
            optimize_daily_intake(
                self.bundle.foods,
                self.bundle.reference_values,
                self.bundle.toxin_columns,
                num_solutions=1,
                prioritize="weight",
                max_foods=3,
                prefer_sparse=True,
                plate_model_constraints=self.bundle.plate_model_constraints,
                realism_metadata=self.bundle.realism_metadata,
                realism_config=self.bundle.realism_config,
                enforce_realism=True,
            )

    def test_zero_max_foods_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "max_foods must be at least 1"):
            optimize_daily_intake(
                self.bundle.foods,
                self.bundle.reference_values,
                self.bundle.toxin_columns,
                num_solutions=1,
                prioritize="weight",
                max_foods=0,
                enforce_realism=False,
            )


if __name__ == "__main__":
    unittest.main()

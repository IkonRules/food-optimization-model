"""Focused tests for programmatic user profiles and validation."""

import unittest

from food_optimizer.demo_data import load_demo_energy_tables
from food_optimizer.get_ref_vals_dict import get_kcal, validate_user_input


class TestReferenceProfiles(unittest.TestCase):
    def test_profile_key_order_does_not_change_reference_values(self):
        profile = {
            "sex": "man",
            "pregnant": None,
            "trimester": None,
            "breastfeeding": None,
            "phase": None,
            "age": 30,
        }
        reordered = {key: profile[key] for key in reversed(profile)}

        energy_tables = load_demo_energy_tables()
        self.assertEqual(
            get_kcal(profile, energy_tables=energy_tables),
            get_kcal(reordered, energy_tables=energy_tables),
        )

    def test_invalid_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid sex"):
            validate_user_input(
                {
                    "sex": "robot",
                    "pregnant": None,
                    "trimester": None,
                    "breastfeeding": None,
                    "phase": None,
                    "age": 30,
                }
            )


if __name__ == "__main__":
    unittest.main()

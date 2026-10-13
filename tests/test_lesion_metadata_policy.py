import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lesion_metadata_policy import SEX_CODES, SITE_CODES, scenario_metadata, select_operating_point, wilson_interval


class MetadataPolicyTest(unittest.TestCase):
    def test_source_codes_are_not_ui_positions(self):
        self.assertEqual(0, SEX_CODES["male"])
        self.assertEqual(1, SEX_CODES["female"])
        self.assertEqual(3, SITE_CODES["torso"])
        self.assertNotEqual(SITE_CODES["upper_extremity"], SITE_CODES["palms_soles"])

    def test_missing_values_remain_training_sentinels(self):
        self.assertEqual([-0.01, -1.0, -1 / 7], scenario_metadata(45, 0, 3, "no_context"))
        self.assertEqual([-0.01, -1.0, 3 / 7], scenario_metadata(45, 0, 3, "site_only"))

    def test_operating_point_respects_ties_and_minimizes_false_positives(self):
        policy = select_operating_point([1, 1, 0, 0], [0.8, 0.5, 0.5, 0.1], 1.0)
        self.assertEqual(0.5, policy["threshold"])
        self.assertEqual(1.0, policy["validation_sensitivity"])
        self.assertEqual(0.5, policy["validation_specificity"])
        with self.assertRaises(ValueError):
            select_operating_point([0, 1], [0.1, 0.9], 0.85, split="test")

    def test_invalid_scores_and_single_class_cannot_define_policy(self):
        for labels, values in [([0, 1], [0.1, float("nan")]), ([0, 0], [0.1, 0.2])]:
            with self.assertRaises(ValueError):
                select_operating_point(labels, values, 0.85)

    def test_small_cohorts_have_visible_uncertainty(self):
        low, high = wilson_interval(0, 20)
        self.assertEqual(0.0, low)
        self.assertGreater(high, 0.1)

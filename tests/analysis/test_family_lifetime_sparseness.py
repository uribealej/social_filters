"""Family sparseness geometry, missing data, identities, and fish weighting."""

import copy
import unittest

import numpy as np
import pandas as pd

from src.several_fish_diagnostics import (
    build_family_lifetime_sparseness, build_response_breadth_profiles,
    build_response_similarity_comparison,
)
from tests.analysis.test_recruitment_containment import prepared_analysis


def prepared_profiles():
    analysis = prepared_analysis()
    return build_response_breadth_profiles(analysis, build_response_similarity_comparison(analysis))


class FamilyLifetimeSparsenessTests(unittest.TestCase):
    def setUp(self):
        self.profiles = prepared_profiles()
        self.grid = [0, .25, .5, .75, 1]

    def test_equal_single_and_two_responses_have_known_sparseness(self):
        panel = self.profiles["profiles"]["fish_b"]["left"]["all_kept"]
        panel["responses"].loc[:, ["R1", "R2", "R3"]] = [[1, 1, 1], [3, 0, 0], [1, 1, 0]]
        result = build_family_lifetime_sparseness(self.profiles, self.grid)
        rows = result["tables"]["family_sparseness_neurons"].query(
            "fish_id == 'fish_b' and side == 'left' and population == 'all_kept' and family == 'rocking'")
        np.testing.assert_allclose(rows["lifetime_sparseness"], [0, 1, .5], atol=1e-15)
        self.assertEqual(rows["global_neuron_id"].tolist(), panel["responses"].index.tolist())
        self.assertEqual(rows["source_neuron_id"].tolist(), panel["row_metadata"]["source_neuron_id"].tolist())

    def test_negatives_nonpositive_and_missing_profiles_are_explicit(self):
        panel = self.profiles["profiles"]["fish_b"]["left"]["all_kept"]
        panel["responses"].loc[:, ["F1", "F2", "F3"]] = [[2, -1, 0], [-1, -2, 0], [1, np.nan, 0]]
        before = copy.deepcopy(panel)
        result = build_family_lifetime_sparseness(self.profiles, self.grid)
        rows = result["tables"]["family_sparseness_neurons"].query(
            "fish_id == 'fish_b' and side == 'left' and population == 'all_kept' and family == 'flicker'")
        np.testing.assert_allclose(rows["lifetime_sparseness"], [1, np.nan, np.nan], equal_nan=True)
        self.assertEqual(rows["sparseness_status"].tolist(), ["defined", "no_positive_response", "incomplete_profile"])
        self.assertEqual(rows["signed_response_F2"].iloc[0], -1)
        self.assertEqual(rows["positive_response_F2"].iloc[0], 0)
        summary = result["tables"]["family_sparseness_per_fish"].query(
            "fish_id == 'fish_b' and side == 'left' and population == 'all_kept' and family == 'flicker'").iloc[0]
        self.assertEqual(summary.n_defined_neurons, 1)
        self.assertEqual(summary.n_no_positive_neurons, 1)
        self.assertEqual(summary.n_incomplete_neurons, 1)
        pd.testing.assert_frame_equal(panel["responses"], before["responses"])

    def test_fish_means_and_cdfs_have_equal_weight_despite_neuron_counts(self):
        for fid, values in (("fish_b", [1, 0, 0]), ("fish_a", [1, 1, 1])):
            self.profiles["profiles"][fid]["left"]["all_kept"]["responses"].loc[:, ["R1", "R2", "R3"]] = values
        result = build_family_lifetime_sparseness(self.profiles, self.grid)
        summary = result["tables"]["family_sparseness_summary"].query(
            "side == 'left' and population == 'all_kept' and family == 'rocking'").iloc[0]
        self.assertAlmostEqual(summary.mean_sparseness, .5)
        self.assertNotAlmostEqual(summary.mean_sparseness, 3 / 5)
        self.assertEqual(summary.valid_fish, 2)
        self.assertEqual(summary.n_defined_neurons, 5)
        cdf = result["tables"]["family_sparseness_mean_cdf"].query(
            "side == 'left' and population == 'all_kept' and family == 'rocking'")
        np.testing.assert_allclose(cdf.cumulative_fraction, [.5, .5, .5, .5, 1])

    def test_missing_fish_and_empty_core_are_not_zero_sparseness(self):
        for population in ("all_kept", "all_rocking"):
            panel = self.profiles["profiles"]["fish_b"]["left"][population]
            if population == "all_kept":
                panel["responses"].loc[:, ["R1", "R2", "R3"]] = np.nan
            else:
                panel["responses"] = panel["responses"].iloc[:0]
                panel["row_metadata"] = panel["row_metadata"].iloc[:0]
        result = build_family_lifetime_sparseness(self.profiles, self.grid)
        summary = result["tables"]["family_sparseness_per_fish"]
        rows = summary.query("fish_id == 'fish_b' and side == 'left'")
        self.assertTrue(rows.query("population == 'all_rocking'").mean_sparseness.isna().all())
        mean = result["tables"]["family_sparseness_summary"].query(
            "side == 'left' and family == 'rocking' and population == 'all_kept'").iloc[0]
        self.assertEqual(mean.valid_fish, 1)

    def test_control_is_excluded_and_positive_rescaling_is_invariant(self):
        result = build_family_lifetime_sparseness(self.profiles, self.grid)
        for fid in self.profiles["fish_ids"]:
            for side in ("left", "right"):
                for population in self.profiles["populations"]:
                    response = self.profiles["profiles"][fid][side][population]["responses"]
                    response.loc[:, "Control"] = 1e9
                    response.loc[:, ["R1", "R2", "R3", "F1", "F2", "F3"]] *= 10
        scaled = build_family_lifetime_sparseness(self.profiles, self.grid)
        np.testing.assert_allclose(result["tables"]["family_sparseness_neurons"].lifetime_sparseness,
                                   scaled["tables"]["family_sparseness_neurons"].lifetime_sparseness, equal_nan=True)

    def test_bad_grid_and_neuron_alignment_raise(self):
        for grid in ([0, .5, .5, 1], [0, .5], [0, np.nan, 1]):
            with self.assertRaisesRegex(ValueError, "cdf_grid"):
                build_family_lifetime_sparseness(self.profiles, grid)
        panel = self.profiles["profiles"]["fish_b"]["left"]["all_kept"]
        panel["row_metadata"] = panel["row_metadata"].iloc[::-1]
        with self.assertRaisesRegex(ValueError, "identities differ"):
            build_family_lifetime_sparseness(self.profiles, self.grid)


if __name__ == "__main__":
    unittest.main()

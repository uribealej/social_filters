"""Known memberships, fish denominators, signed profiles, and identity checks."""

import copy
import unittest

import numpy as np
import pandas as pd

from src.stimulus_specificity import build_active_membership_table, build_family_responsive_membership_fractions
from src.neuron_ordering import build_response_profile_sort_order
from src.several_fish_diagnostics import (
    build_response_breadth_profiles, build_response_similarity_comparison,
)
from tests.analysis.test_recruitment_containment import prepared_analysis, CONDITIONS


class ActiveMembershipTests(unittest.TestCase):
    def test_responsive_fractions_exclude_none_and_normalize_each_group(self):
        rows = []
        for fid, family, counts in (("small", "rocking", [90, 9, 1, 0]),
                                    ("large", "rocking", [0, 100, 900, 0]),
                                    ("small", "flicker", [96, 1, 1, 2]),
                                    ("empty", "rocking", [20, 0, 0, 0])):
            for n_active, count in enumerate(counts):
                rows.append(dict(fish_id=fid, side="left", family=family,
                                 n_active_stimuli=n_active, n_neurons=count))
        original = pd.DataFrame(rows)
        before = original.copy(deep=True)
        result = build_family_responsive_membership_fractions(original)
        self.assertTrue((result.n_active_stimuli > 0).all())
        for fid, family, expected in (("small", "rocking", [.9, .1, 0]),
                                       ("large", "rocking", [.1, .9, 0]),
                                       ("small", "flicker", [.25, .25, .5])):
            values = result.loc[(result.fish_id == fid) & (result.family == family), "fraction_family_responsive"]
            np.testing.assert_allclose(values, expected)
            self.assertAlmostEqual(values.sum(), 1)
        self.assertTrue(result.loc[result.fish_id == "empty", "fraction_family_responsive"].isna().all())
        means = result.loc[result.family == "rocking"].groupby("n_active_stimuli").fraction_family_responsive.mean()
        np.testing.assert_allclose(means, [.5, .5, 0])
        pd.testing.assert_frame_equal(original, before)

    def test_all_eight_patterns_and_complete_counts(self):
        patterns = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                    [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
        active = pd.DataFrame(patterns, columns=[4, 8, 9], index=list(range(20, 28)))
        result = build_active_membership_table(active, [4, 8, 9], ["A", "B", "C"])
        self.assertEqual(result["neurons"].index.tolist(), active.index.tolist())
        self.assertEqual(result["neurons"]["membership"].tolist(),
                         ["None", "A", "B", "C", "A + B", "A + C", "B + C", "A + B + C"])
        np.testing.assert_array_equal(result["breadth"]["n_neurons"], [1, 3, 3, 1])
        np.testing.assert_allclose(result["combinations"]["fraction_kept"], 1 / 8)
        for name in ("breadth", "combinations"):
            self.assertAlmostEqual(result[name]["fraction_kept"].sum(), 1)
        grouped = result["combinations"].groupby("n_active_stimuli")["n_neurons"].sum()
        np.testing.assert_array_equal(grouped, result["breadth"]["n_neurons"])

    def test_absent_categories_empty_population_and_nonfinite(self):
        active = pd.DataFrame([[1, 1, 1]], columns=[1, 2, 3])
        result = build_active_membership_table(active, [1, 2, 3], ["A", "B", "C"])
        np.testing.assert_array_equal(result["breadth"]["n_neurons"], [0, 0, 0, 1])
        self.assertEqual(len(result["combinations"]), 8)
        empty = build_active_membership_table(active.iloc[:0], [1, 2, 3], ["A", "B", "C"])
        for name in ("breadth", "combinations"):
            self.assertTrue(empty[name]["fraction_kept"].isna().all())
            self.assertEqual(empty[name]["n_neurons"].sum(), 0)
        active.loc[0, 2] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            build_active_membership_table(active, [1, 2, 3], ["A", "B", "C"])

    def test_sort_uses_signed_values_configured_ties_and_missing_last(self):
        responses = pd.DataFrame({"Control": [999] * 7,
                                  "A": [-2, 2, 2, np.nan, 0, np.nan, 2],
                                  "B": [-3, 2, 1, 5, np.nan, np.nan, 1]})
        order = build_response_profile_sort_order(responses, ["A", "B"])
        np.testing.assert_array_equal(order, [1, 2, 6, 4, 0, 3, 5])
        self.assertEqual(build_response_profile_sort_order(responses.iloc[:0], ["A", "B"]).size, 0)


class ResponseBreadthProfileTests(unittest.TestCase):
    def setUp(self):
        self.analysis = prepared_analysis()
        self.similarity = build_response_similarity_comparison(self.analysis)
        self.result = build_response_breadth_profiles(self.analysis, self.similarity)

    def test_unequal_fish_mean_denominators_and_complete_tables(self):
        table = self.result["tables"]["response_breadth"]
        self.assertEqual(len(table), 2 * 2 * 2 * 4)
        combinations = self.result["tables"]["response_combinations"]
        self.assertEqual(len(combinations), 2 * 2 * 2 * 8)
        for _, rows in table.groupby(["fish_id", "side", "family"]):
            self.assertAlmostEqual(rows["fraction_kept"].sum(), 1)
            self.assertEqual(rows["n_neurons"].sum(), rows["n_kept_neurons"].iloc[0])
        summary = self.result["tables"]["response_breadth_summary"]
        row = summary.query("side == 'left' and family == 'rocking' and n_active_stimuli == 3").iloc[0]
        self.assertAlmostEqual(row["fraction_kept"], (1 / 3 + 1 / 2) / 2)
        self.assertNotAlmostEqual(row["fraction_kept"], 2 / 5)
        self.assertEqual(row["valid_fish"], 2)
        grouped = combinations.groupby(["fish_id", "side", "family", "n_active_stimuli"])["n_neurons"].sum()
        expected = table.set_index(["fish_id", "side", "family", "n_active_stimuli"])["n_neurons"].sort_index()
        pd.testing.assert_series_equal(grouped.sort_index(), expected)

    def test_profiles_retain_signed_values_source_identities_and_shared_range(self):
        for fid in self.result["fish_ids"]:
            for side in ("left", "right"):
                for population in self.result["populations"]:
                    panel = self.result["profiles"][fid][side][population]
                    original = self.similarity["response_vectors"][fid][side][population]
                    pd.testing.assert_frame_equal(panel["responses"], original.loc[panel["responses"].index])
                    self.assertEqual(panel["responses"].index.tolist(), panel["row_metadata"]["global_neuron_id"].tolist())
                    ids = panel["row_metadata"]["source_neuron_id"].tolist()
                    self.assertEqual(set(ids), set(self.similarity["row_metadata"][fid][side][population]["source_neuron_id"]))
        self.assertEqual(self.result["profile_color_limit"], 8)
        self.assertLess(self.result["profiles"]["fish_b"]["left"]["all_kept"]["responses"].min().min(), 0)

    def test_side_specific_core_and_empty_core(self):
        ids = self.analysis["side_selections"]["right"]["groups"]["rocking"]
        self.analysis["active_matrices"]["fish_b"].loc[:, ids] = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
        self.analysis["active_matrices"]["fish_a"].loc[:, ids] = 0
        similarity = build_response_similarity_comparison(self.analysis)
        result = build_response_breadth_profiles(self.analysis, similarity)
        self.assertEqual(result["profiles"]["fish_b"]["left"]["all_rocking"]["row_metadata"]["source_neuron_id"].tolist(), [3])
        self.assertEqual(result["profiles"]["fish_b"]["right"]["all_rocking"]["row_metadata"]["source_neuron_id"].tolist(), [1])
        self.assertTrue(result["profiles"]["fish_a"]["right"]["all_rocking"]["responses"].empty)

    def test_missing_profiles_stay_missing_and_inputs_unchanged(self):
        table = self.analysis["tables"]["neuron_responses"]
        table.loc[(table.fish_id == "fish_b") & (table.global_neuron_id == 0), "normalized_auc"] = np.nan
        similarity = build_response_similarity_comparison(self.analysis)
        before = copy.deepcopy(similarity)
        before_table = table.copy(deep=True)
        result = build_response_breadth_profiles(self.analysis, similarity)
        panel = result["profiles"]["fish_b"]["left"]["all_kept"]
        self.assertEqual(panel["responses"].index[-1], 0)
        self.assertTrue(panel["responses"].iloc[-1].isna().all())
        pd.testing.assert_frame_equal(table, before_table)
        pd.testing.assert_frame_equal(similarity["response_vectors"]["fish_b"]["left"]["all_kept"],
                                      before["response_vectors"]["fish_b"]["left"]["all_kept"])

    def test_empty_kept_population_has_missing_fractions_and_one_valid_fish(self):
        fid = "fish_b"
        self.analysis["active_matrices"][fid] = self.analysis["active_matrices"][fid].iloc[:0]
        for table in (self.analysis["responses"], self.analysis["tables"]):
            key = "row_metadata" if table is self.analysis["responses"] else "neuron_responses"
            table[key] = table[key].loc[table[key]["fish_id"] != fid].copy()
        for side in ("left", "right"):
            for population in ("all_kept", "all_rocking"):
                self.similarity["row_metadata"][fid][side][population] = self.similarity["row_metadata"][fid][side][population].iloc[:0]
                self.similarity["response_vectors"][fid][side][population] = self.similarity["response_vectors"][fid][side][population].iloc[:0]
        result = build_response_breadth_profiles(self.analysis, self.similarity)
        rows = result["tables"]["response_breadth"].query("fish_id == 'fish_b'")
        self.assertTrue(rows["fraction_kept"].isna().all())
        self.assertTrue((result["tables"]["response_breadth_summary"]["valid_fish"] == 1).all())

    def test_stale_or_reordered_cached_responses_raise(self):
        vectors = self.similarity["response_vectors"]["fish_b"]["left"]["all_kept"]
        vectors.iloc[0, 0] += 1
        with self.assertRaisesRegex(ValueError, "values differ"):
            build_response_breadth_profiles(self.analysis, self.similarity)
        self.similarity = build_response_similarity_comparison(self.analysis)
        self.similarity["response_vectors"]["fish_b"]["left"]["all_kept"] = vectors.iloc[::-1]
        with self.assertRaisesRegex(ValueError, "identities/order"):
            build_response_breadth_profiles(self.analysis, self.similarity)


if __name__ == "__main__":
    unittest.main()

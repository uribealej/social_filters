"""Known response geometry, identity, missing-value, and population checks."""

import copy
import unittest

import numpy as np
import pandas as pd

from src.several_fish_diagnostics import (
    build_recruitment_containment_analysis, build_response_similarity_comparison,
)
from tests.analysis.test_recruitment_containment import (
    ACTIVE, CONDITIONS, TIMING, prepared_analysis, synthetic_cohort,
)


def vector_example():
    """Three neurons with scaled, offset, reversed, zero, constant, and missing vectors."""
    analysis = prepared_analysis()
    columns = np.array([
        [-2, 0, 2], [-4, 0, 4], [3, 5, 7], [2, 0, -2],
        [0, 0, 0], [3, 3, 3], [1, np.nan, 2],
    ], dtype=float).T
    table = analysis["tables"]["neuron_responses"]
    for index, condition in enumerate(CONDITIONS):
        mask = (table["fish_id"] == "fish_b") & (table["side"] == "left") & (table["condition"] == condition)
        table.loc[mask, "normalized_auc"] = columns[:, index]
    return analysis


class ResponseSimilarityComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = vector_example()
        cls.result = build_response_similarity_comparison(cls.analysis)

    def test_identical_scaled_offset_and_reversed_vectors_preserve_signs(self):
        matrices = self.result["per_fish_matrices"]["fish_b"]["left"]["all_kept"]
        pearson, cosine = matrices["pearson"], matrices["cosine"]
        self.assertAlmostEqual(pearson.loc["Control", "Control"], 1)
        self.assertAlmostEqual(pearson.loc["Control", "R1"], 1)
        self.assertAlmostEqual(cosine.loc["Control", "R1"], 1)
        self.assertAlmostEqual(pearson.loc["Control", "R2"], 1)
        self.assertAlmostEqual(cosine.loc["Control", "R2"], 8 / np.sqrt(8 * 83))
        self.assertAlmostEqual(pearson.loc["Control", "R3"], -1)
        self.assertAlmostEqual(cosine.loc["Control", "R3"], -1)

    def test_constants_zeros_and_missing_values_keep_canonical_semantics(self):
        matrices = self.result["per_fish_matrices"]["fish_b"]["left"]["all_kept"]
        self.assertTrue(matrices["pearson"].loc["F1"].isna().all())
        self.assertTrue(matrices["cosine"].loc["F1"].isna().all())
        self.assertTrue(matrices["pearson"].loc["F2"].isna().all())
        self.assertAlmostEqual(matrices["cosine"].loc["F2", "F2"], 1)
        self.assertAlmostEqual(matrices["pearson"].loc["Control", "F3"], 1)
        self.assertAlmostEqual(matrices["cosine"].loc["Control", "F3"], 2 / np.sqrt(8 * 5))
        pair = self.result["tables"]["response_similarity_pairs"].query(
            "fish_id == 'fish_b' and side == 'left' and population == 'all_kept' "
            "and source_condition == 'Control' and target_condition == 'F3'").iloc[0]
        self.assertEqual(pair["n_population_neurons"], 3)
        self.assertEqual(pair["n_valid_response_neurons"], 2)

    def test_equal_fish_means_skip_undefined_and_record_counts(self):
        mean = self.result["mean_matrices"]["all_kept"]["left"]
        self.assertAlmostEqual(mean["pearson"].loc["Control", "R3"], 0)
        self.assertAlmostEqual(mean["cosine"].loc["Control", "R3"], 0)
        summary = self.result["tables"]["response_similarity_summary"].query(
            "population == 'all_kept' and side == 'left' and source_condition == 'Control' "
            "and target_condition == 'F1'").iloc[0]
        self.assertEqual(summary["pearson_valid_fish"], 1)
        self.assertEqual(summary["cosine_valid_fish"], 1)
        self.assertAlmostEqual(summary["pearson"], 1)

    def test_all_kept_jaccard_agrees_with_stage_one_and_rocking_core_is_conditional(self):
        for side in ("left", "right"):
            pd.testing.assert_frame_equal(
                self.result["mean_matrices"]["all_kept"][side]["jaccard"],
                self.analysis["mean_matrices"][side]["jaccard"],
            )
            for fid in self.result["fish_ids"]:
                core = self.result["per_fish_matrices"][fid][side]["all_rocking"]
                np.testing.assert_allclose(core["jaccard"].loc[["R1", "R2", "R3"], ["R1", "R2", "R3"]], 1)
                self.assertTrue(core["pearson"].isna().all().all())  # One-neuron cores.

    def test_source_order_is_recovered_even_when_response_table_is_shuffled(self):
        analysis = vector_example()
        analysis["tables"]["neuron_responses"] = analysis["tables"]["neuron_responses"].sample(frac=1, random_state=2)
        result = build_response_similarity_comparison(analysis)
        metadata = result["row_metadata"]["fish_b"]["left"]["all_kept"]
        self.assertEqual(metadata["source_neuron_id"].tolist(), [3, 1, 0])
        vectors = result["response_vectors"]["fish_b"]["left"]["all_kept"]
        self.assertEqual(vectors.index.tolist(), [0, 1, 2])
        self.assertEqual(vectors.columns.tolist(), CONDITIONS)
        np.testing.assert_array_equal(vectors["Control"], [-2, 0, 2])

    def test_core_is_separate_for_each_side_and_preserves_source_identity(self):
        data, fish_ids, groups = synthetic_cohort()
        for name in groups["right"]["rocking"]:
            stimulus = data["fish_b"]["stimuli_id_map"][name]
            raster = data["fish_b"]["trial_aligned_traces_raster"][stimulus]
            raster[:] = 0
            raster[1] = 1
        analysis = build_recruitment_containment_analysis(data, fish_ids, groups, TIMING, ACTIVE, CONDITIONS)
        result = build_response_similarity_comparison(analysis)
        self.assertEqual(result["row_metadata"]["fish_b"]["left"]["all_rocking"]["source_neuron_id"].tolist(), [3])
        self.assertEqual(result["row_metadata"]["fish_b"]["right"]["all_rocking"]["source_neuron_id"].tolist(), [1])

    def test_empty_core_remains_undefined(self):
        data, fish_ids, groups = synthetic_cohort()
        name = groups["left"]["rocking"][-1]
        for fid in fish_ids:
            stimulus = data[fid]["stimuli_id_map"][name]
            data[fid]["trial_aligned_traces_raster"][stimulus][:] = 0
        analysis = build_recruitment_containment_analysis(data, fish_ids, groups, TIMING, ACTIVE, CONDITIONS)
        result = build_response_similarity_comparison(analysis)
        for matrix in result["mean_matrices"]["all_rocking"]["left"].values():
            self.assertTrue(matrix.isna().all().all())
        rows = result["tables"]["response_similarity_population_counts"].query(
            "population == 'all_rocking' and side == 'left'")
        self.assertEqual(rows["n_population_neurons"].tolist(), [0, 0])

    def test_stage_one_inputs_are_unchanged_and_missing_rows_fail(self):
        analysis = vector_example()
        before = copy.deepcopy(analysis)
        build_response_similarity_comparison(analysis)
        for name, table in before["tables"].items():
            pd.testing.assert_frame_equal(analysis["tables"][name], table)
        for fid in before["fish_ids"]:
            pd.testing.assert_frame_equal(analysis["active_matrices"][fid], before["active_matrices"][fid])
        analysis["tables"]["neuron_responses"] = analysis["tables"]["neuron_responses"].iloc[1:]
        with self.assertRaisesRegex(ValueError, "exactly one response"):
            build_response_similarity_comparison(analysis)


if __name__ == "__main__":
    unittest.main()

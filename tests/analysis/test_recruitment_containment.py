"""Known-output checks for mirrored recruitment and directional containment."""

import copy
import unittest

import numpy as np
import pandas as pd

from src.active_neuron_analysis import compute_active_neuron_containment
from src.several_fish_selection import build_active_group_masks
from src.several_fish_diagnostics import build_recruitment_containment_analysis


CONDITIONS = ["Control", "R1", "R2", "R3", "F1", "F2", "F3"]
TIMING = {"fps_2p": 1.0, "t_pre_s": 1.0, "motion_onset_s": 1.0, "tau_s": 1.0}
ACTIVE = {"active_fraction_threshold": 0.10, "min_epoch_s": 1.0,
          "min_active_reps": 2, "expected_reps": 4, "require_expected_reps": True}


def synthetic_cohort():
    """Two unequal fish; full continuous rows, permuted kept rows, filtered rasters."""
    labels = [f"{condition}_{side}" for side in ("L", "R") for condition in CONDITIONS]
    ids = list(range(11, 11 + len(labels)))
    groups = {
        side: {"control": f"Control_{suffix}",
               "rocking": [f"R{i}_{suffix}" for i in range(1, 4)],
               "flicker": [f"F{i}_{suffix}" for i in range(1, 4)]}
        for side, suffix in (("left", "L"), ("right", "R"))
    }
    pattern = np.array([[1, 1, 1, 1, 1, 0, 0],
                        [1, 1, 0, 1, 0, 1, 0],
                        [1, 0, 1, 0, 0, 0, 0]])
    data = {}
    for fid, values, kept in (("fish_a", [-2., 1., 3.], [2, 0]),
                              ("fish_b", [2., -1., 0., 4.], [3, 1, 0])):
        zscore, raster = {}, {}
        for index, stimulus in enumerate(ids):
            scale = 1 if index < 7 else 2
            zscore[stimulus] = np.broadcast_to(
                np.asarray(values)[:, None, None] * scale, (len(values), 8, 4)).copy()
            raster[stimulus] = np.broadcast_to(
                pattern[:len(kept), index % 7, None, None], (len(kept), 8, 4)).copy()
        data[fid] = {
            "trial_aligned_traces_z_core": zscore, "trial_aligned_traces_raster": raster,
            "kept_neuron_indices": np.array(kept), "stimuli_id_map": dict(zip(labels, ids)),
            "stimuli_durations": {label: {"motion_sec": 10.0 if label.startswith("R1") else 2.0}
                                  for label in labels},
        }
    return data, ["fish_b", "fish_a"], groups


def prepared_analysis():
    data, fish_ids, groups = synthetic_cohort()
    return build_recruitment_containment_analysis(data, fish_ids, groups, TIMING, ACTIVE, CONDITIONS)


class ContainmentCalculationTests(unittest.TestCase):
    def test_identical_nested_partial_disjoint_and_empty_sets(self):
        active = pd.DataFrame({
            "C": [1, 1, 1, 0], "A": [1, 1, 0, 0], "A_copy": [1, 1, 0, 0],
            "B": [0, 1, 1, 1], "D": [0, 0, 0, 1], "empty": [0, 0, 0, 0],
        })
        original = active.copy(deep=True)
        order = ["A", "C", "B", "D", "empty", "A_copy"]
        result = compute_active_neuron_containment(active, order)
        self.assertEqual(result.index.tolist(), order)
        self.assertEqual(result.loc["A", "A_copy"], 1)
        self.assertEqual(result.loc["A", "C"], 1)
        self.assertAlmostEqual(result.loc["C", "A"], 2 / 3)
        self.assertEqual(result.loc["A", "B"], 0.5)
        self.assertAlmostEqual(result.loc["B", "A"], 1 / 3)
        self.assertEqual(result.loc["A", "D"], 0)
        self.assertEqual(result.loc["A", "empty"], 0)
        self.assertTrue(result.loc["empty"].isna().all())
        pd.testing.assert_frame_equal(active, original)

    def test_missing_decisions_raise_instead_of_becoming_active(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            compute_active_neuron_containment(pd.DataFrame({"A": [np.nan]}), ["A"])

    def test_all_group_mask_uses_intersection_and_preserves_rows(self):
        active = pd.DataFrame({11: [1, 1, 0], 12: [1, 0, 1], 13: [1, 1, 0]})
        result = build_active_group_masks(active, {"rocking": [11, 12, 13]})
        np.testing.assert_array_equal(result["rocking"]["all"], [True, False, False])
        np.testing.assert_array_equal(result["rocking"]["any"], [True, True, True])
        with self.assertRaisesRegex(ValueError, "nonempty"):
            build_active_group_masks(active, {"rocking": []})


class RecruitmentAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = prepared_analysis()

    def test_constant_signed_normalization_and_source_neuron_order(self):
        neurons = self.result["tables"]["neuron_responses"]
        rows = neurons.query("fish_id == 'fish_b' and stimulus == 'Control_L'")
        self.assertEqual(rows["source_neuron_id"].tolist(), [3, 1, 0])
        self.assertEqual(rows["global_neuron_id"].tolist(), [0, 1, 2])
        np.testing.assert_allclose(rows["normalized_auc"], [4, -1, 2])
        np.testing.assert_allclose(rows["raw_auc"], [12, -3, 6])
        long_window = neurons.query("fish_id == 'fish_b' and stimulus == 'R1_L'")
        np.testing.assert_allclose(long_window["normalized_auc"], [4, -1, 2])
        np.testing.assert_allclose(long_window["raw_auc"], [20, -5, 10])
        summary = self.result["tables"]["recruitment"].query(
            "fish_id == 'fish_b' and stimulus == 'Control_L'").iloc[0]
        self.assertAlmostEqual(summary["mean_normalized_auc"], 5 / 3)
        self.assertEqual(summary["n_neurons"], 3)
        self.assertEqual(summary["n_active"], 3)

    def test_clipped_window_metadata_records_integration_span(self):
        window = self.result["tables"]["response_windows"].query(
            "fish_id == 'fish_b' and stimulus == 'R1_L'").iloc[0]
        self.assertEqual(window["requested_duration_s"], 12)
        self.assertEqual(window["observed_duration_s"], 6)
        self.assertEqual(window["integration_span_s"], 5)
        self.assertTrue(window["window_clipped"])
        self.assertEqual(window["n_response_frames"], 6)

    def test_matrix_means_weight_fish_equally_and_keep_pair_order(self):
        self.assertEqual(self.result["fish_ids"], ["fish_b", "fish_a"])
        matrix = self.result["mean_matrices"]["left"]["containment"]
        self.assertEqual(matrix.index.tolist(), CONDITIONS)
        self.assertEqual(matrix.columns.tolist(), CONDITIONS)
        self.assertAlmostEqual(matrix.loc["Control", "R1"], (2 / 3 + 1) / 2)
        self.assertNotAlmostEqual(matrix.loc["Control", "R1"], 4 / 5)
        self.assertTrue(matrix.loc["F3"].isna().all())
        self.assertTrue(np.isnan(self.result["mean_matrices"]["left"]["jaccard"].loc["F3", "F3"]))
        pd.testing.assert_frame_equal(matrix, self.result["mean_matrices"]["right"]["containment"])

    def test_all_rocking_intersection_empty_sources_and_exception_counts(self):
        table = self.result["tables"]["containment_summary"].query(
            "fish_id == 'fish_b' and side == 'left' and summary_type == 'to_all_rocking'")
        self.assertEqual(table["condition"].tolist(), ["F1", "F2", "F3"])
        np.testing.assert_allclose(table["fraction_contained"].iloc[:2], [1, 0])
        self.assertTrue(np.isnan(table["fraction_contained"].iloc[2]))
        self.assertEqual(table["n_not_contained"].tolist(), [0, 1, 0])
        np.testing.assert_allclose(table["fraction_not_contained"].iloc[:2], [0, 1])

    def test_alignment_errors_are_not_silently_repaired(self):
        data, fish_ids, groups = synthetic_cohort()
        data["fish_b"]["trial_aligned_traces_raster"][11] = np.zeros((4, 8, 4))
        with self.assertRaisesRegex(ValueError, "kept rows"):
            build_recruitment_containment_analysis(data, fish_ids, groups, TIMING, ACTIVE, CONDITIONS)
        data, fish_ids, groups = synthetic_cohort()
        data["fish_b"]["kept_neuron_indices"] = np.array([3, 3, 0])
        with self.assertRaisesRegex(ValueError, "unique vector"):
            build_recruitment_containment_analysis(data, fish_ids, groups, TIMING, ACTIVE, CONDITIONS)

    def test_bad_groups_fail_and_success_does_not_mutate_inputs(self):
        data, fish_ids, groups = synthetic_cohort()
        original_groups = copy.deepcopy(groups)
        original = data["fish_b"]["trial_aligned_traces_z_core"][11].copy()
        build_recruitment_containment_analysis(data, fish_ids, groups, TIMING, ACTIVE, CONDITIONS)
        self.assertEqual(groups, original_groups)
        np.testing.assert_array_equal(data["fish_b"]["trial_aligned_traces_z_core"][11], original)
        groups["right"]["rocking"] = groups["right"]["rocking"][:2]
        with self.assertRaisesRegex(ValueError, "matching"):
            build_recruitment_containment_analysis(data, fish_ids, groups, TIMING, ACTIVE, CONDITIONS)


if __name__ == "__main__":
    unittest.main()

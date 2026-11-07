"""Known-output control onset transitions, timing boundaries and fish weights."""

import unittest
import numpy as np

from src.active_neuron_analysis import build_control_onset_comparison
from src.response_classification import first_persistent_response_onset


def fixture():
    control = np.zeros((6, 12, 4))
    control[:4, 4:6] = 1
    control[5, 9:11] = 1  # active, but persistence crosses the onset-window end
    comparison = np.zeros_like(control)
    for neuron, start in enumerate([2, 4, 6]):
        comparison[neuron, start:start + 2, :2] = 1
    def fish(a, b, kept):
        return dict(trial_aligned_traces_raster={1: a, 2: b},
            kept_neuron_indices=np.asarray(kept), stimuli_id_map={"control": 1, "test": 2},
            stimuli_durations={k: dict(motion_sec=6, static_before_sec=0) for k in ["control", "test"]})
    return {"a": fish(control, comparison, [9, 7, 5, 3, 1, 11]),
            "b": fish(control[:1], np.zeros_like(comparison[:1]), [42])}


def analyze(data=None, **kwargs):
    return build_control_onset_comparison(fixture() if data is None else data, ["a", "b"],
        {"right": dict(control="control", stimuli=["control", "test"])},
        np.arange(6), fps_2p=2, t_pre_s=0, motion_onset_s=0, tau_s=0, **kwargs)


class ControlOnsetTests(unittest.TestCase):
    def test_first_sustained_onset_and_threshold(self):
        time = np.arange(-1, 3, 0.5)
        self.assertEqual(first_persistent_response_onset([1, 1, 0.5, 0.5, 0, 0, 0, 0], time), 0)
        self.assertEqual(first_persistent_response_onset([0, 0, 1, 0, 0.5, 0.5, 0, 0], time), 1)
        self.assertTrue(np.isinf(first_persistent_response_onset([0.5, np.nan], [0, 0.5])))
        self.assertTrue(np.isinf(first_persistent_response_onset([np.inf, 1], [0, 0.5])))

    def test_transitions_and_equal_fish_denominators(self):
        result = analyze()
        rows = result["fish_bins"]
        row = rows[(rows.fish_id == "a") & (rows.stimulus == "test") & (rows.control_bin == 2)].iloc[0]
        self.assertEqual((row.n_control, row.n_shared, row.n_same, row.n_earlier,
                          row.n_later, row.n_no_onset), (4, 3, 1, 1, 1, 1))
        summary = result["bin_summary"]
        row = summary[(summary.stimulus == "test") & (summary.control_bin == 2)].iloc[0]
        self.assertEqual(row.fraction_shared, 0.375)
        self.assertEqual(row.n_fish, 2)
        self.assertEqual(row.n_control_total, 5)
        self.assertTrue(summary[summary.control_bin == 0].fraction_shared.isna().all())
        self.assertTrue((summary[summary.control_bin == 0].n_fish == 0).all())
        transitions = result["fish_transitions"]
        present = transitions[transitions.n_control > 0]
        np.testing.assert_allclose(present.groupby(["fish_id", "stimulus", "control_bin"]).fraction.sum(), 1)
        self_rows = rows[(rows.stimulus == "control") & (rows.n_control > 0)]
        self.assertTrue((self_rows.fraction_same == 1).all())
        identities = result["neuron_table"]
        self.assertEqual(identities[identities.fish_id == "a"].neuron_id.unique().tolist(), [9, 7, 5, 3])
        self.assertEqual(result["audit_table"].iloc[0].n_control_active_no_onset, 1)

    def test_bad_data_are_not_absent_responses(self):
        for invalid in [np.nan, 0.5]:
            data = fixture()
            data["a"]["trial_aligned_traces_raster"][2][0, 3, 0] = invalid
            with self.assertRaisesRegex(ValueError, "finite binary"):
                analyze(data)
        data = fixture()
        data["a"]["kept_neuron_indices"] = np.arange(3)
        with self.assertRaisesRegex(ValueError, "retained rows"):
            analyze(data)
        data = fixture()
        data["a"]["trial_aligned_traces_raster"][1] = data["a"]["trial_aligned_traces_raster"][1][:, :4]
        with self.assertRaisesRegex(ValueError, "complete onset window"):
            analyze(data)

    def test_empty_population_and_string_stimulus_keys(self):
        data = fixture()
        for fish in data.values():
            fish["trial_aligned_traces_raster"] = {str(k): np.zeros_like(v)
                for k, v in fish["trial_aligned_traces_raster"].items()}
        result = analyze(data)
        self.assertTrue(result["bin_summary"].fraction_shared.isna().all())
        self.assertTrue(result["neuron_table"].empty)
        self.assertTrue((result["fish_bins"].n_control == 0).all())


if __name__ == "__main__":
    unittest.main()

"""Known-count same-time sharing, repeated activity and empty-bin semantics."""

import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.active_neuron_analysis import build_control_bin_overlap
from src.plotting_onset_comparison import plot_control_bin_overlap


def fixture():
    control = np.zeros((5, 12, 4))
    comparison = np.zeros_like(control)
    control[:4, 2:4, :2] = 1
    control[0, 4:6, 2:] = 1  # the same neuron contributes in a later bin
    control[0, 9:11, :2] = 1  # epoch crosses the final edge: not active in bin 3
    control[3, 8, :] = 1  # a single significant frame is insufficient
    comparison[:2, 2:4, :2] = 1
    comparison[4, 2:4, :2] = 1  # stimulus-only recruitment is not in the numerator
    comparison[3, 2, :2] = 1
    comparison[3, 3, 2:] = 1  # consensus persistence without any persistent trial
    comparison[0, 4:6, 1:3] = 1
    comparison[2, 4:8, :2] = 1  # shifted response cannot count as same-bin sharing

    def fish(a, b, kept):
        return dict(trial_aligned_traces_raster={1: a, 2: b},
            kept_neuron_indices=np.asarray(kept), stimuli_id_map={"Control": 1, "Test": 2},
            fps_2p=2)

    return {"a": fish(control, comparison, [91, 7, 3, 22, 42]),
            "b": fish(control[:1].copy(), np.zeros_like(comparison[:1]), [100])}


def analyze(data=None, **kwargs):
    settings = dict(fps_2p=2, t_pre_s=0, program_onset_s=1)
    settings.update(kwargs)
    return build_control_bin_overlap(fixture() if data is None else data, ["a", "b"],
        {"left": dict(control="Control", stimuli=["Control", "Test"])},
        np.arange(5), **settings)


class ControlBinOverlapTests(unittest.TestCase):
    def test_counts_time_identity_and_equal_fish_percentages(self):
        result = analyze()
        rows = result["fish_bins"]
        first = rows[(rows.fish_id == "a") & (rows.stimulus == "Test") & (rows.bin_index == 0)].iloc[0]
        self.assertEqual((first.n_control, first.n_stimulus, first.n_shared,
                          first.n_control_only, first.n_stimulus_only), (4, 3, 2, 2, 1))
        summary = result["bin_summary"].query("stimulus == 'Test'").set_index("bin_index")
        self.assertEqual(summary.loc[0, "fraction_shared"], 0.25)  # not pooled 2/5
        self.assertEqual(summary.loc[0, "sem_shared"], 0.25)
        self.assertEqual(summary.loc[0, "n_fish"], 2)
        self.assertEqual(summary.loc[0, "n_control_total"], 5)
        self.assertEqual(summary.loc[0, "n_shared_total"], 2)
        self.assertEqual(summary.loc[1, "fraction_shared"], 0.5)
        self.assertTrue(summary.loc[2:, "fraction_shared"].isna().all())
        self.assertTrue((summary.loc[2:, "n_fish"] == 0).all())
        masks = result["bin_activity"]["left"]["a"]
        np.testing.assert_array_equal(result["neuron_ids"]["a"], [91, 7, 3, 22, 42])
        np.testing.assert_array_equal(masks["Control"][0], [True, True, False, False])
        self.assertFalse(masks["Test"][3, 0])
        self_rows = rows[(rows.stimulus == "Control") & (rows.n_control > 0)]
        self.assertTrue((self_rows.fraction_shared == 1).all())
        self.assertTrue((rows.n_shared <= rows.n_control).all())
        self.assertTrue((rows.n_shared <= rows.n_stimulus).all())
        windows = result["bin_windows"]
        self.assertTrue((windows.n_frames == 2).all())
        np.testing.assert_allclose(windows.first_sample_s, windows.bin_start_s)

    def test_empty_controls_and_empty_neuron_populations(self):
        data = fixture()
        for fish in data.values():
            fish["kept_neuron_indices"] = np.array([], dtype=int)
            fish["trial_aligned_traces_raster"] = {
                str(key): value[:0] for key, value in fish["trial_aligned_traces_raster"].items()}
        result = analyze(data)
        self.assertTrue(result["bin_summary"].fraction_shared.isna().all())
        self.assertTrue((result["bin_summary"].n_fish == 0).all())
        self.assertEqual(result["bin_activity"]["left"]["a"]["Control"].shape, (0, 4))

    def test_one_eligible_fish_has_no_sem(self):
        data = fixture()
        data["b"]["trial_aligned_traces_raster"][1][:] = 0
        summary = analyze(data)["bin_summary"].query("stimulus == 'Test'")
        self.assertEqual(summary.iloc[0].n_fish, 1)
        self.assertTrue(np.isnan(summary.iloc[0].sem_shared))

    def test_bad_inputs_raise_instead_of_becoming_zero_overlap(self):
        for invalid in [np.nan, 0.5]:
            data = fixture()
            data["a"]["trial_aligned_traces_raster"][2][0, 3, 0] = invalid
            with self.assertRaisesRegex(ValueError, "finite binary"):
                analyze(data)
        data = fixture()
        data["a"]["trial_aligned_traces_raster"][2] = np.zeros((5, 12, 3))
        with self.assertRaisesRegex(ValueError, "repetitions"):
            analyze(data)
        data = fixture()
        data["a"]["trial_aligned_traces_raster"][1] = np.zeros((5, 8, 4))
        with self.assertRaisesRegex(ValueError, "complete bin window"):
            analyze(data)
        data = fixture()
        data["a"]["kept_neuron_indices"][:] = 1
        with self.assertRaisesRegex(ValueError, "unique"):
            analyze(data)
        with self.assertRaisesRegex(ValueError, "shorter than the activity epoch"):
            analyze(min_epoch_s=1.5)
        with self.assertRaisesRegex(ValueError, "integer repetitions"):
            analyze(min_active_reps=5)

    def test_plot_consumes_cached_percentages_and_counts(self):
        result = analyze()
        plotted = plot_control_bin_overlap(result, "left", label_map={"Test": "Example"})
        try:
            main, counts = plotted["axes"]
            np.testing.assert_allclose(main.lines[0].get_ydata(), [25, 50, np.nan, np.nan], equal_nan=True)
            np.testing.assert_allclose(counts.lines[-1].get_ydata(), [2.5, 1, 0, 0])
            self.assertEqual(main.lines[0].get_label(), "Example")
            plotted["figure"].canvas.draw()
        finally:
            plt.close(plotted["figure"])


if __name__ == "__main__":
    unittest.main()

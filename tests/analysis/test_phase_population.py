"""Known-output phase measurements, reviewed cache validation and fish weighting."""
import json
from pathlib import Path, PureWindowsPath
import shutil
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import data_loading, response_classification, response_metrics, trial_alignment
from src import several_fish_loading, several_fish_diagnostics, several_fish_figures, several_fish_reporting
from tests.timing import test_phase_alignment as phase_tests


class PhasePopulationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = phase_tests.PhaseAlignmentTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.addCleanup(plt.close, "all")
        self.measurements = dict(bout_windows={"bout": (0, 3.6), "bout_recovery": (0, 8.6)},
            active_fraction_threshold=0.1, min_epoch_s=1,
            min_active_reps=1, expected_reps=2)

    def synthetic_cohort(self, recovery_s=10):
        fish = {}
        for fid, ids, level, active in [("first", [3, 8], 2, 1), ("second", [1, 2, 3, 4, 5], 10, 0)]:
            z = np.full((320, len(ids)), level, dtype=float)
            raster = np.full_like(z, active)
            data = dict(kept_neuron_indices=np.array(ids), trials=self.fixture.trials,
                        fps_2p=2, stimuli_ids=[1, 2], stimuli_id_map={"C1_Bout": 1, "S1_Overlap": 2},
                        z_traces=np.full((320, max(ids) + 1), level, dtype=float), raster=raster)
            data["aligned"] = {anchor: {kind: trial_alignment.build_event_aligned_traces(values,
                self.fixture.trials, 2, alignment=anchor, recovery_s=recovery_s)
                for kind, values in [("zscore", z), ("raster", raster)]}
                for anchor in ["sequence", "bout"]}
            fish[fid] = data
        return dict(fish_ids=list(fish), all_fish_data=fish)

    def test_explicit_auc_peak_and_missing_values(self):
        values = np.array([-2., -1., 0., 1.])[None, :, None]
        result = response_metrics.compute_explicit_window_metrics(values, [0, 1, 2, 3],
            np.arange(4) / 2, 2, baseline=np.array([[2.]]))
        self.assertEqual(result["auc_z_s"][0, 0], -0.75)
        self.assertEqual(result["peak_z"][0, 0], 1)
        self.assertEqual(result["mean_z"][0, 0], -0.5)
        self.assertEqual(result["delta_auc_z_s"][0, 0], -3.75)
        values[0, 1, 0] = np.nan
        result = response_metrics.compute_explicit_window_metrics(values, [0, 1, 2, 3], np.arange(4) / 2, 2)
        self.assertTrue(np.isnan(result["auc_z_s"]).all())
        self.assertTrue(np.isnan(result["peak_z"]).all())

    def test_native_window_boundaries_and_activity_rule(self):
        result = trial_alignment.select_event_window(np.arange(6) / 2 - 0.1, 0, 1)
        np.testing.assert_array_equal(result["frame_indices"], [1, 2])
        self.assertEqual(result["first_sample_s"], 0.4)
        self.assertEqual(result["last_sample_s"], 0.9)
        self.assertEqual(trial_alignment.select_event_window(np.arange(4), 0, 0)["n_samples"], 0)
        raster = np.array([[[1, 1], [1, 0], [0, 1], [0, 0]],
                           [[0, 1], [0, 1], [0, np.nan], [0, 0]]])
        result = response_classification.classify_trial_raster_activity(raster, 2, 0.5, 1)
        np.testing.assert_array_equal(result[0], [1, 0])
        self.assertEqual(result[1, 0], 0)
        self.assertTrue(np.isnan(result[1, 1]))

    def test_equal_fish_weight_and_original_neuron_order(self):
        cohort = self.synthetic_cohort()
        display = several_fish_diagnostics.build_phase_population_display(cohort, sort_mode="unsorted")
        values = display["fish_means"][1]
        np.testing.assert_allclose(values[0], 2)
        np.testing.assert_allclose(values[1], 10)
        self.assertEqual(values.mean(axis=0)[0], 6)
        self.assertNotEqual(values.mean(axis=0)[0], display["matrices"]["zscore", "mean"][1].mean(axis=0)[0])
        self.assertEqual(display["displayed_row_metadata"].neuron_id.tolist(), [3, 8, 1, 2, 3, 4, 5])
        self.assertEqual(display["matrices"]["raster", "concat"][1].shape, (7, 76))
        fig, axes = several_fish_figures.plot_phase_cohort_means(display, figsize=(8, 5))
        black = next(line for line in axes[0, 0].lines if line.get_label() == "Equal-fish mean")
        np.testing.assert_allclose(black.get_ydata(), 6)
        with self.assertRaisesRegex(ValueError, "No reviewed fish"):
            several_fish_diagnostics.build_phase_population_display(dict(fish_ids=[]))

    def test_pooled_phase_panels_reuse_cell06_rows_and_saved_trial_means(self):
        cohort, trial_metrics = self.recruitment_fixture()
        # This recruitment fixture expands traces from two to four trials;
        # complete its alignment metadata for the general population adapter.
        for fish in cohort['all_fish_data'].values():
            for aligned in fish['aligned'].values():
                for bundle in aligned.values():
                    for key in ('actual_time_s', 'onsets_by_id', 'alignment_error_s'):
                        bundle[key] = {sid: np.repeat(values[..., :1], 4, axis=-1)
                                       for sid, values in bundle[key].items()}
                    bundle['trials'] = fish['trials']
        conditions = {'left': ['C1', 'S1']}
        recruitment = several_fish_diagnostics.build_phase_recruitment_analysis(
            cohort, trial_metrics, conditions, (0, 3.6))
        displays = several_fish_diagnostics.build_phase_stimulus_panel_displays(
            cohort, conditions, repetitions=(2, 3, 4), sort_mode='cell06', recruitment=recruitment)
        order = recruitment['displays']['left']['neuron_order']
        for kind in ('zscore', 'raster'):
            prepared = displays[kind]['left']
            np.testing.assert_array_equal(prepared['neuron_order'], order)
            self.assertEqual(prepared['row_metadata'].neuron_id.tolist(), [3, 8, 1, 2, 3, 4, 5])
            np.testing.assert_array_equal(prepared['matrices'][1], 0)
            self.assertEqual(prepared['repetition_numbers'][1], [2, 3, 4])
            result = several_fish_figures.plot_all_fish_stimulus_figures(prepared,
                {'left': [1, 2]}, layouts=('combined',), mean_mode='pooled_neurons',
                show_fish_traces=False, z_limits=(0, 1))
            axes = result['axes']['left_combined']
            self.assertEqual(axes.shape, (2, 2))
            np.testing.assert_array_equal(axes[0, 0].images[0].get_array(), prepared['matrices'][1][order])
            curve = next(line for line in axes[1, 0].lines if line.get_label() == 'Pooled-neuron mean')
            np.testing.assert_array_equal(curve.get_ydata(), 0)
            self.assertEqual([x.get_text() for x in result['figures']['left_combined'].legends[0].get_texts()],
                             ['Preconditioning', 'Blank delay', 'Bout'])
            self.assertEqual(result['figures']['left_combined']._supxlabel.get_text(), 'From bout onset (s)')
        source = recruitment['displays']['left']['row_metadata']
        source.loc[0, 'neuron_id'] = 999
        with self.assertRaisesRegex(ValueError, 'identities'):
            several_fish_diagnostics.build_phase_stimulus_panel_displays(
                cohort, conditions, sort_mode='cell06', recruitment=recruitment)

    def test_sequence_panels_preserve_cached_rows_repetitions_delay_and_recovery(self):
        cohort = self.synthetic_cohort()
        for fish in cohort['all_fish_data'].values():
            ids = fish['kept_neuron_indices']
            fish['z_traces'] = np.arange(320)[:, None] + 100 * np.arange(max(ids) + 1)[None, :]
            fish['raster'] = (fish['z_traces'][:, ids] % 3 == 0).astype(float)
            fish['aligned'] = {anchor: {kind: trial_alignment.build_event_aligned_traces(
                values, fish['trials'], 2, alignment=anchor)
                for kind, values in [('zscore', fish['z_traces'][:, ids]), ('raster', fish['raster'])]}
                for anchor in ('sequence', 'bout')}
        cached = several_fish_diagnostics.build_phase_stimulus_panel_displays(
            cohort, {'left': ['C1', 'S1']}, repetitions=(2,), sort_mode='unsorted')
        for sides in cached.values():
            sides['left']['neuron_order'] = np.arange(7)[::-1]
        extended = several_fish_diagnostics.build_phase_sequence_panel_displays(cohort, cached)
        for kind in ('zscore', 'raster'):
            before, after = cached[kind]['left'], extended[kind]['left']
            np.testing.assert_array_equal(after['neuron_order'], np.arange(7)[::-1])
            self.assertEqual(after['repetition_numbers'], {1: [2], 2: [2]})
            self.assertEqual(after['alignment'], 'sequence')
            self.assertEqual(after['time_s'][2][0], -5)
            self.assertEqual(after['time_s'][2][-1], 26.5)  # 12 s sequence + 15 s tail, half-open.
            self.assertEqual(after['time_s'][1][-1], 18.5)  # 3.6 s bout + 15 s tail.
            self.assertAlmostEqual(after['phase_trials'][2].bout_onset_s - after['phase_trials'][2].pre_end_s, 0.3)
            expected = []
            for fish in cohort['all_fish_data'].values():
                continuous = (fish['z_traces'][:, fish['kept_neuron_indices']]
                              if kind == 'zscore' else fish['raster'])
                expected.append(continuous[226:290].T)  # Second S1 anchor rounds to frame 236.
            np.testing.assert_array_equal(after['matrices'][2], np.concatenate(expected))
            np.testing.assert_allclose(after['fish_means'][2], [v.mean(axis=0) for v in expected])
            self.assertLess(before['matrices'][2].shape[1], after['matrices'][2].shape[1])
            self.assertEqual(cohort['all_fish_data']['first']['aligned']['sequence'][kind]
                             ['time_s'][2][-1], 21.5)  # Original 10 s cache is untouched.
            result = several_fish_figures.plot_all_fish_stimulus_figures(after,
                {'left': [1, 2]}, layouts=('combined',), proportional_widths=True)
            axes = result['axes']['left_combined']
            for row in axes:
                np.testing.assert_allclose(row[0].get_xlim(), (-5, 18.6))
                np.testing.assert_allclose(row[1].get_xlim(), (-5, 27))
                seconds_scale = [ax.get_position().width / np.diff(ax.get_xlim())[0] for ax in row]
                self.assertAlmostEqual(*seconds_scale)
            self.assertLess(axes[0, 0].get_position().width, axes[0, 1].get_position().width)
            self.assertEqual(axes[0, 0].get_position().width, axes[1, 0].get_position().width)
        with self.assertRaisesRegex(ValueError, 'block boundary'):
            several_fish_diagnostics.build_phase_sequence_panel_displays(cohort, cached, recovery_s=100)
        cached['zscore']['left']['row_metadata'].loc[0, 'neuron_id'] = 99
        with self.assertRaisesRegex(ValueError, 'identities'):
            several_fish_diagnostics.build_phase_sequence_panel_displays(cohort, cached)

    def test_sequence_repetition_displays_keep_single_trials_rows_and_grid(self):
        from src import plotting_all_fish
        cohort = self.synthetic_cohort()
        for fish in cohort['all_fish_data'].values():
            ids = fish['kept_neuron_indices']
            fish['z_traces'] = np.arange(320)[:, None] + 100 * np.arange(max(ids) + 1)[None, :]
            fish['raster'] = (fish['z_traces'][:, ids] % 3 == 0).astype(float)
        cached = several_fish_diagnostics.build_phase_stimulus_panel_displays(
            cohort, {'left': ['C1', 'S1']}, repetitions=(2,), sort_mode='unsorted')
        for sides in cached.values():
            sides['left']['neuron_order'] = np.arange(7)[::-1]
        per_rep = several_fish_diagnostics.build_phase_sequence_repetition_displays(
            cohort, cached, repetitions=(1, 2))
        averaged = several_fish_diagnostics.build_phase_sequence_panel_displays(
            cohort, {kind: {'left': dict(sides['left'], repetition_numbers={1: [1, 2], 2: [1, 2]})}
                     for kind, sides in cached.items()})
        for kind in ('zscore', 'raster'):
            reps = per_rep[kind]['left']
            self.assertEqual(list(reps), [1, 2])
            second = reps[2]
            np.testing.assert_array_equal(second['neuron_order'], np.arange(7)[::-1])
            self.assertEqual(second['repetition_numbers'], {1: [2], 2: [2]})
            self.assertEqual(second['repetition'], 2)
            self.assertEqual(second['time_s'][2][0], -5)
            self.assertEqual(second['time_s'][2][-1], 26.5)
            np.testing.assert_allclose(second['time_bounds_s'][1], (-5.0, 18.6))
            expected = [(fish['z_traces'][:, fish['kept_neuron_indices']] if kind == 'zscore'
                         else fish['raster'])[226:290].T for fish in cohort['all_fish_data'].values()]
            np.testing.assert_array_equal(second['matrices'][2], np.concatenate(expected))
            np.testing.assert_allclose(second['fish_means'][2], [v.mean(axis=0) for v in expected])
            # Averaging the single-trial displays reproduces the Cell 04c trial mean.
            for sid in (1, 2):
                np.testing.assert_allclose((reps[1]['matrices'][sid] + reps[2]['matrices'][sid]) / 2,
                                           averaged[kind]['left']['matrices'][sid])
            if kind == 'raster':
                self.assertTrue(set(np.unique(second['matrices'][2])) <= {0.0, 1.0})
        self.assertEqual(cached['zscore']['left']['repetition_numbers'], {1: [2], 2: [2]})
        for bad in [(), (1, 1), (0,), (True,)]:
            with self.assertRaisesRegex(ValueError, 'one-based'):
                several_fish_diagnostics.build_phase_sequence_repetition_displays(cohort, cached, bad)
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            several_fish_diagnostics.build_phase_sequence_repetition_displays(cohort, cached, (1, 3))
        rows = per_rep['raster']['left']
        fig, axes = plotting_all_fish.plot_stimulus_response_panel_grid(
            [rows[1], rows[2]], ['Rep 1\n(excluded from analyses)', 'Rep 2'], [1, 2],
            show_traces=True, z_limits=(0, 1), proportional_widths=True)
        self.assertEqual(axes.shape, (4, 2))
        order = rows[2]['neuron_order']
        np.testing.assert_array_equal(axes[2, 1].images[0].get_array(), rows[2]['matrices'][2][order])
        curve = next(line for line in axes[3, 1].lines if line.get_label() == 'Pooled-neuron mean')
        np.testing.assert_allclose(curve.get_ydata(), rows[2]['matrices'][2].mean(axis=0))
        self.assertEqual([axes[0, 0].get_ylabel(), axes[2, 0].get_ylabel()],
                         ['Rep 1\n(excluded from analyses)', 'Rep 2'])
        self.assertEqual(axes[1, 0].get_ylabel(), 'Fraction\nactive')
        self.assertEqual(fig.axes[-1].get_ylabel(), 'Significant activity (0/1)')
        np.testing.assert_array_equal(fig.axes[-1].get_yticks(), [0, 1])  # Binary colorbar.
        for row in axes:
            np.testing.assert_allclose(row[0].get_xlim(), (-5, 18.6))
            np.testing.assert_allclose(row[1].get_xlim(), (-5, 27))
        self.assertEqual([x.get_text() for x in fig.legends[0].get_texts()],
                         ['Preconditioning', 'Blank delay', 'Bout'])
        fig, axes = plotting_all_fish.plot_stimulus_response_panel_grid(
            [rows[1], rows[2]], ['Rep 1', 'Rep 2'], show_traces=False, z_limits=(0, 1),
            show_legend=False)
        self.assertEqual(axes.shape, (2, 2))
        self.assertEqual(fig.legends, [])
        with self.assertRaisesRegex(ValueError, 'one label per row'):
            plotting_all_fish.plot_stimulus_response_panel_grid([rows[1]], ['a', 'b'])
        with self.assertRaisesRegex(ValueError, 'Grid rows must share'):
            plotting_all_fish.plot_stimulus_response_panel_grid(
                [rows[1], dict(rows[2], neuron_order=np.arange(7))], ['a', 'b'])

    def test_phase_panel_means_weights_and_signed_control_order(self):
        cohort = self.synthetic_cohort()
        displays = several_fish_diagnostics.build_phase_stimulus_panel_displays(
            cohort, {'left': ['C1'], 'right': ['S1']}, sort_mode='left_right_index',
            left_right_controls=['C1_Bout', 'S1_Overlap'], sort_window_s=(0, 3.6))
        left = displays['zscore']['left']
        np.testing.assert_array_equal(left['neuron_order'], np.arange(7))
        np.testing.assert_allclose(left['fish_means'][1][:, 0], [2, 10])
        np.testing.assert_allclose(left['matrices'][1].mean(axis=0), 54/7)
        result = several_fish_figures.plot_all_fish_stimulus_figures(left, {'left': [1]},
            layouts=('combined',), mean_mode='pooled_neurons', show_fish_traces=False)
        curve = next(line for line in result['axes']['left_combined'][1, 0].lines
                     if line.get_label() == 'Pooled-neuron mean')
        np.testing.assert_allclose(curve.get_ydata(), 54/7)
        # Negative right preference first, balanced responses next, positive
        # left preference last, using the configured control order.
        for aligned in cohort['all_fish_data']['first']['aligned'].values():
            aligned['zscore']['trial_aligned_traces'][1][:] = np.array([4, 0])[:, None, None]
            aligned['zscore']['trial_aligned_traces'][2][:] = np.array([0, 4])[:, None, None]
        ordered = several_fish_diagnostics.build_phase_stimulus_panel_displays(
            cohort, {'left': ['C1'], 'right': ['S1']}, sort_mode='left_right_index',
            left_right_controls=['C1_Bout', 'S1_Overlap'], sort_window_s=(0, 3.6))
        np.testing.assert_array_equal(ordered['zscore']['left']['neuron_order'], [1, 2, 3, 4, 5, 6, 0])
        with self.assertRaisesRegex(ValueError, 'requires the prepared'):
            several_fish_diagnostics.build_phase_stimulus_panel_displays(
                cohort, {'left': ['C1']}, sort_mode='cell06')

    def test_measurements_and_absent_phase(self):
        result = several_fish_diagnostics.build_phase_population_analysis(self.synthetic_cohort(),
            self.measurements)
        self.assertEqual(set(result), {"trial_metrics", "neuron_metrics", "fish_summary", "window_coverage"})
        self.assertEqual(set(result["trial_metrics"].window), {"bout", "bout_recovery", "preconditioning"})
        fish = result["fish_summary"]
        pre_control = fish.loc[fish.condition.eq("C1") & fish.window.eq("preconditioning")]
        self.assertTrue(pre_control.auc_z_s.isna().all())
        self.assertTrue(pre_control.active_fraction.isna().all())
        short = fish.loc[fish.condition.eq("C1") & fish.window.eq("bout")].set_index("fish_id")
        self.assertEqual(short.loc["first", "auc_z_s"], 6)
        self.assertEqual(short.loc["first", "mean_z"], 2)
        self.assertEqual(short.loc["first", "active_fraction"], 1)
        self.assertEqual(short.loc["second", "active_fraction"], 0)
        self.assertEqual(short.loc["first", "n_neurons"], 2)
        self.assertEqual(short.loc["second", "n_neurons"], 5)
        self.assertNotIn("delta_auc_z_s", fish)
        self.assertNotIn("early_late_change_z", fish)

    def test_measurement_windows_require_aligned_coverage(self):
        result = trial_alignment.select_event_window(np.arange(6) / 2, 0, 3, sample_period_s=0.5)
        self.assertEqual(result["n_samples"], 6)
        for start, stop in [(0, 3.1), (-0.5, 1)]:
            with self.subTest(window=(start, stop)), self.assertRaisesRegex(ValueError, "exceeds"):
                trial_alignment.select_event_window(np.arange(6) / 2, start, stop, sample_period_s=0.5)
        self.assertEqual(trial_alignment.select_event_window(np.arange(6) / 2, 0, 3.1)["n_samples"], 6)
        settings = dict(self.measurements, bout_windows={"bout_10s": (0, 10), "bout_15s": (0, 15)})
        with self.assertRaisesRegex(ValueError, "bout_15s.*exceeds"):
            several_fish_diagnostics.build_phase_population_analysis(self.synthetic_cohort(), settings)
        longer = several_fish_loading.align_phase_cohort(self.synthetic_cohort(), 5, 15)
        result = several_fish_diagnostics.build_phase_population_analysis(longer, settings)
        coverage = result["window_coverage"].groupby("window").n_samples.agg(["min", "max"])
        self.assertEqual(coverage.loc["bout_10s"].tolist(), [20, 20])
        self.assertEqual(coverage.loc["bout_15s"].tolist(), [30, 30])
        bout = result["fish_summary"].set_index(["fish_id", "condition", "window"])
        self.assertEqual(bout.loc[("first", "C1", "bout_15s"), "auc_z_s"], 2 * 14.5)

    def test_align_phase_cohort_returns_new_cohort_without_reloading(self):
        cohort = self.synthetic_cohort()
        cohort.update(readiness=pd.DataFrame(dict(fish_id=cohort["fish_ids"])), settings=dict(recovery_s=10))
        original = cohort["all_fish_data"]["first"]["aligned"]
        longer = several_fish_loading.align_phase_cohort(cohort, 5, 15)
        self.assertEqual(longer["fish_ids"], ["first", "second"])
        self.assertEqual(longer["settings"]["recovery_s"], 15)
        self.assertEqual(cohort["settings"]["recovery_s"], 10)
        self.assertIs(cohort["all_fish_data"]["first"]["aligned"], original)
        self.assertIs(longer["all_fish_data"]["first"]["z_traces"], cohort["all_fish_data"]["first"]["z_traces"])
        self.assertEqual(original["bout"]["zscore"]["time_s"][1][-1], 13.5)
        self.assertEqual(longer["all_fish_data"]["first"]["aligned"]["bout"]["zscore"]["time_s"][1][-1], 18.5)
        self.assertEqual(longer["all_fish_data"]["second"]["aligned"]["bout"]["raster"]
                         ["trial_aligned_traces"][2].shape, (5, 48, 2))
        with self.assertRaisesRegex(ValueError, "first: .*block boundary"):
            several_fish_loading.align_phase_cohort(cohort, 5, 100)

    def test_display_last_three_trials_and_grayscale_labels(self):
        cohort = self.synthetic_cohort()
        for fish in cohort["all_fish_data"].values():
            for anchor in fish["aligned"].values():
                for kind, aligned in anchor.items():
                    for sid, traces in aligned["trial_aligned_traces"].items():
                        four = np.repeat(traces[:, :, :1], 4, axis=2)
                        four[:, :, 0] = 100 if kind == "zscore" else 1
                        if kind == "raster":
                            four[:, :, 1:] = np.array([0, 1, 0])
                        aligned["trial_aligned_traces"][sid] = four
                        for key in ("actual_time_s", "onsets_by_id", "alignment_error_s"):
                            aligned[key][sid] = np.repeat(aligned[key][sid][..., :1], 4, axis=-1)
                    aligned["trials"] = pd.concat([aligned["trials"], aligned["trials"]], ignore_index=True)
        selected = several_fish_diagnostics.build_phase_population_display(
            cohort, sort_mode="unsorted", sort_scope="pooled", repetitions=(2, 3, 4))
        all_trials = several_fish_diagnostics.build_phase_population_display(cohort, sort_mode="unsorted")
        np.testing.assert_allclose(selected["matrices"]["raster", "mean"][1], 1 / 3)
        np.testing.assert_allclose(all_trials["matrices"]["raster", "mean"][1], 1 / 2)
        np.testing.assert_allclose(selected["fish_means"][1].mean(axis=0), 6)
        np.testing.assert_allclose(all_trials["fish_means"][1].mean(axis=0), 29.5)
        self.assertEqual(selected["matrices"]["raster", "concat"][1].shape, (7, 114))
        np.testing.assert_array_equal(cohort["all_fish_data"]["first"]["aligned"]["sequence"]
            ["raster"]["trial_aligned_traces"][1][0, 0], [1, 0, 1, 0])
        fig, axes = several_fish_figures.plot_phase_cohort_rasters(selected, "raster", "concat", figsize=(8, 5))
        self.assertEqual(axes[0, 0].images[0].get_cmap().name, "Greys")
        self.assertEqual([text.get_text() for text in axes[0, 0].texts], ["2", "3", "4"])
        self.assertEqual(axes[0, 0].get_ylabel(), "Pooled neurons (sorted)")
        self.assertFalse(any(line.get_color() == "cyan" for line in axes[0, 0].lines))
        for invalid in [[], [0], [2, 2], [5], [True], [1.5]]:
            with self.subTest(repetitions=invalid), self.assertRaises(ValueError):
                several_fish_diagnostics.build_phase_population_display(cohort, repetitions=invalid)

    def test_pooled_correlation_order_is_fixed_across_alignment_and_selection(self):
        from src.neuron_ordering import compute_single_sort_order
        cohort = self.synthetic_cohort()
        rng = np.random.default_rng(36)
        profiles = []
        for fish in cohort["all_fish_data"].values():
            parts = []
            for sid, traces in fish["aligned"]["sequence"]["zscore"]["trial_aligned_traces"].items():
                values = rng.normal(size=traces.shape)
                traces[:] = values
                parts.append(values.mean(axis=2))
            profiles.append(np.concatenate(parts, axis=1))
        expected = compute_single_sort_order(np.concatenate(profiles).T, "corravg")
        full = several_fish_diagnostics.build_phase_population_display(cohort, sort_scope="pooled")
        subset = several_fish_diagnostics.build_phase_population_display(
            cohort, "bout", sort_scope="pooled", repetitions=[2])
        np.testing.assert_array_equal(full["neuron_order"], expected)
        np.testing.assert_array_equal(subset["neuron_order"], expected)
        self.assertEqual(full["displayed_row_metadata"].neuron_id.tolist(),
                         full["row_metadata"].iloc[expected].neuron_id.tolist())
        within = several_fish_diagnostics.build_phase_population_display(cohort)
        self.assertFalse(np.array_equal(within["neuron_order"], expected))

    def test_c1_bout_only_reference_ignores_other_conditions_and_times(self):
        from src.neuron_ordering import compute_single_sort_order
        cohort = self.synthetic_cohort()
        rng = np.random.default_rng(91)
        profiles = []
        for fish in cohort["all_fish_data"].values():
            aligned = fish["aligned"]["bout"]["zscore"]
            values = aligned["trial_aligned_traces"][1]
            values[:] = rng.normal(size=values.shape)
            mask = (aligned["time_s"][1] >= 0) & (aligned["time_s"][1] < 3.6)
            profiles.append(values[:, mask, :].mean(axis=2))
        options = dict(sort_scope="pooled", sort_stimulus_keys=["C1_Bout"],
                       sort_alignment="bout", sort_window_s=(0, 3.6))
        before = several_fish_diagnostics.build_phase_population_display(cohort, **options)
        expected = compute_single_sort_order(np.concatenate(profiles).T, "corravg")
        np.testing.assert_array_equal(before["neuron_order"], expected)
        for fish in cohort["all_fish_data"].values():
            aligned = fish["aligned"]["bout"]["zscore"]
            values = aligned["trial_aligned_traces"][1]
            mask = (aligned["time_s"][1] >= 0) & (aligned["time_s"][1] < 3.6)
            values[:, ~mask, :] = rng.normal(size=values[:, ~mask, :].shape) * 1e6
            aligned["trial_aligned_traces"][2][:] = 1e9
        after = several_fish_diagnostics.build_phase_population_display(
            cohort, "sequence", repetitions=[2], **options)
        np.testing.assert_array_equal(after["neuron_order"], expected)
        fig, axes = several_fish_figures.plot_phase_cohort_rasters(
            before, zscore_cmap="Greys", figsize=(8, 5))
        self.assertEqual(axes[0, 0].images[0].get_cmap().name, "Greys")
        self.assertEqual(axes[0, 0].images[0].get_clim(), (-1, 4))
        for invalid in [dict(sort_stimulus_keys=[]), dict(sort_stimulus_keys=["missing"]),
                        dict(sort_window_s=(1, 0)), dict(sort_window_s=(100, 101))]:
            with self.subTest(options=invalid), self.assertRaises(ValueError):
                several_fish_diagnostics.build_phase_population_display(cohort, **(options | invalid))

    def recruitment_fixture(self, recovery_s=10):
        cohort = self.synthetic_cohort(recovery_s)
        rows = []
        for fid, fish in cohort["all_fish_data"].items():
            for aligned in fish["aligned"].values():
                for bundle in aligned.values():
                    for sid, traces in bundle["trial_aligned_traces"].items():
                        four = np.repeat(traces[:, :, :1], 4, axis=2)
                        four[:, :, 0], four[:, :, 1:] = 1, 0
                        bundle["trial_aligned_traces"][sid] = four
            fish["trials"] = pd.concat([fish["trials"], fish["trials"]], ignore_index=True)
            for j, neuron_id in enumerate(fish["kept_neuron_indices"]):
                for condition in ["C1", "S1"]:
                    for window in ["preconditioning", "bout"]:
                        flag = (j % 2 == 0) if condition == "C1" else (j == 0 if window == "preconditioning" else True)
                        for rep in range(4):
                            active = float(flag if rep > 0 else not flag)
                            if condition == "S1" and j == 1 and rep == 3:
                                active = float(window == "preconditioning")  # 2/3 bout, 1/3 pre.
                            if condition == "C1" and window == "preconditioning":
                                active = np.nan
                            rows.append(dict(fish_id=fid, neuron_id=neuron_id, condition=condition,
                                window=window, repetition=rep, active_trial=active))
        return cohort, pd.DataFrame(rows)

    def test_recruitment_counts_last_three_and_missing_classifications(self):
        cohort, trials = self.recruitment_fixture()
        conditions = {"left": ["C1", "S1"]}
        result = several_fish_diagnostics.build_phase_recruitment_analysis(
            cohort, trials, conditions, (0, 3.6))
        summary = result["fish_summary"].set_index(["fish_id", "condition"])
        self.assertEqual(summary.loc[("first", "S1"), "n_shared_c1"], 1)
        self.assertEqual(summary.loc[("second", "S1"), "n_shared_c1"], 3)
        self.assertEqual(summary.loc[("first", "S1"), "n_new_after_pre"], 1)
        self.assertEqual(summary.loc[("second", "S1"), "n_new_after_pre"], 4)
        self.assertTrue(np.isnan(summary.loc[("first", "C1"), "n_new_after_pre"]))
        self.assertEqual(summary.loc[("first", "S1"), "fraction_c1_retained"], 1)
        display = result["displays"]["left"]
        np.testing.assert_array_equal(display["matrices"]["preconditioning"], 0)
        np.testing.assert_array_equal(display["matrices"]["S1"], 0)
        self.assertEqual(len(display["row_metadata"]), 7)
        self.assertEqual(display["membership_codes"]["S1"].tolist(), [1, 2, 1, 2, 3, 2, 3])
        self.assertEqual(result["union_summary"].n_any_new_after_pre.tolist(), [1, 4])
        fig, axes = several_fish_figures.plot_phase_recruitment_rasters(result, "left", figsize=(10, 6))
        self.assertEqual(len(axes), 3)
        self.assertEqual(axes[0].images[0].get_cmap().name, "Greys")
        binary_fig, binary_axes = several_fish_figures.plot_phase_recruitment_decisions(result, "left", figsize=(10, 6))
        np.testing.assert_array_equal(binary_axes[1].images[0].get_array(),
            display["decision_matrix"][display["neuron_order"]])
        self.assertEqual(set(np.unique(display["decision_matrix"])), {0, 1})
        fig, axes = several_fish_figures.plot_phase_recruitment_counts(result, cohort["fish_ids"], conditions)
        self.assertEqual(axes[0, 0].patches[1].get_height(), 2)
        self.assertEqual(axes[1, 0].patches[1].get_height(), 2.5)
        _, proportion_axes = several_fish_figures.plot_phase_recruitment_proportions(
            result, cohort["fish_ids"], conditions)
        # New-after-pre: 1/2 retained neurons in one fish, 4/5 in the other.
        self.assertEqual(summary.loc[("first", "S1"), "fraction_new_after_pre"], 0.5)
        self.assertEqual(summary.loc[("second", "S1"), "fraction_new_after_pre"], 0.8)
        self.assertAlmostEqual(proportion_axes[1, 0].patches[1].get_height(), 0.65)
        self.assertNotAlmostEqual(proportion_axes[1, 0].patches[1].get_height(), 5 / 7)
        self.assertIn("all retained neurons", proportion_axes[1, 0].get_ylabel())
        self.assertIn("N/A", [text.get_text() for text in proportion_axes[1, 0].texts])
        # A missing selected repetition must not classify the neuron as inactive.
        mask = trials.fish_id.eq("first") & trials.neuron_id.eq(3) & trials.condition.eq("S1")
        trials.loc[mask & trials.window.eq("bout") & trials.repetition.eq(2), "active_trial"] = np.nan
        missing = several_fish_diagnostics.build_phase_recruitment_analysis(cohort, trials, conditions, (0, 3.6))
        first = missing["fish_summary"].query("fish_id == 'first' and condition == 'S1'").iloc[0]
        self.assertTrue(np.isnan(first.n_shared_c1))
        self.assertEqual(first.n_valid_shared_c1, 1)
        self.assertTrue(np.isnan(first.n_new_after_pre))
        self.assertTrue(np.isnan(first.fraction_c1_retained))
        self.assertTrue(np.isnan(first.fraction_new_after_pre))

    def test_c1_proportions_use_control_pool_and_equal_fish_weights(self):
        cohort, trials = self.recruitment_fixture()
        conditions = {"left": ["C1", "S1"]}
        # First fish retains its single C1 neuron; second retains 1 of 3.
        # Stimulus-only neurons stay active but cannot enter the numerator.
        control_ids = cohort["all_fish_data"]["second"]["kept_neuron_indices"][[2, 4]]
        mask = (trials.fish_id.eq("second") & trials.condition.eq("S1")
                & trials.window.eq("bout") & trials.neuron_id.isin(control_ids))
        trials.loc[mask, "active_trial"] = 0
        result = several_fish_diagnostics.build_phase_recruitment_analysis(
            cohort, trials, conditions, (0, 3.6))
        table = result["fish_summary"].set_index(["fish_id", "condition"])
        self.assertEqual(table.loc[("first", "S1"), "n_shared_c1"], 1)
        self.assertEqual(table.loc[("second", "S1"), "n_shared_c1"], 1)
        self.assertEqual(table.loc[("first", "S1"), "fraction_c1_retained"], 1)
        self.assertAlmostEqual(table.loc[("second", "S1"), "fraction_c1_retained"], 1 / 3)
        fig, axes = several_fish_figures.plot_phase_recruitment_proportions(
            result, cohort["fish_ids"], conditions)
        self.assertEqual(axes.shape, (2, 1))
        self.assertEqual(axes[0, 0].patches[0].get_height(), 1)
        self.assertAlmostEqual(axes[0, 0].patches[1].get_height(), 2 / 3)
        self.assertEqual(axes[0, 0].get_yticklabels()[-1].get_text(), "100%")
        trials.loc[trials.condition.eq("S1") & trials.window.eq("bout"), "active_trial"] = 0
        zero = several_fish_diagnostics.build_phase_recruitment_analysis(
            cohort, trials, conditions, (0, 3.6))
        self.assertTrue(zero["fish_summary"].loc[lambda t: t.condition.eq("S1")].fraction_c1_retained.eq(0).all())
        # No C1-responsive neurons means undefined, rather than 0% overlap.
        trials.loc[trials.condition.eq("C1") & trials.window.eq("bout"), "active_trial"] = 0
        empty = several_fish_diagnostics.build_phase_recruitment_analysis(
            cohort, trials, conditions, (0, 3.6))
        self.assertTrue(empty["fish_summary"].fraction_c1_retained.isna().all())
        self.assertTrue(empty["fish_summary"].loc[lambda t: t.condition.eq("S1")].fraction_new_after_pre.eq(0).all())
        _, empty_axes = several_fish_figures.plot_phase_recruitment_proportions(
            empty, cohort["fish_ids"], conditions)
        self.assertIn("N/A", [text.get_text() for text in empty_axes[0, 0].texts])

    def test_preconditioning_mean_across_conditions_and_unique_union(self):
        cohort, trials = self.recruitment_fixture()
        extra = trials.loc[trials.condition.eq("S1")].copy()
        extra["condition"] = "S2"
        trials = pd.concat([trials, extra], ignore_index=True)
        for fish in cohort["all_fish_data"].values():
            more = fish["trials"].loc[fish["trials"].logged_name.eq("S1")].copy()
            more["logged_name"], more["stimulus_id"] = "S2", 3
            fish["trials"] = pd.concat([fish["trials"], more], ignore_index=True)
            for aligned in fish["aligned"].values():
                for bundle in aligned.values():
                    bundle["trial_aligned_traces"][3] = np.ones_like(bundle["trial_aligned_traces"][2])
                    bundle["time_s"][3] = bundle["time_s"][2].copy()
        result = several_fish_diagnostics.build_phase_recruitment_analysis(
            cohort, trials, {"left": ["C1", "S1", "S2"]}, (0, 3.6))
        np.testing.assert_allclose(result["displays"]["left"]["matrices"]["preconditioning"], 0.5)
        # The same new neuron in two conditions is counted only once in the union.
        self.assertEqual(result["union_summary"].n_any_new_after_pre.tolist(), [1, 4])

    def test_recruitment_longer_bout_window_and_coverage(self):
        cohort, trials = self.recruitment_fixture()
        with self.assertRaisesRegex(ValueError, "exceeds"):
            several_fish_diagnostics.build_phase_recruitment_analysis(cohort, trials, {"left": ["C1", "S1"]}, (0, 15))
        with self.assertRaisesRegex(ValueError, "no 'bout_15s' window"):
            several_fish_diagnostics.build_phase_recruitment_analysis(
                cohort, trials, {"left": ["C1", "S1"]}, (0, 3.6), bout_window="bout_15s")
        cohort, trials = self.recruitment_fixture(recovery_s=15)
        trials = trials.assign(window=trials.window.replace({"bout": "bout_15s"}))
        result = several_fish_diagnostics.build_phase_recruitment_analysis(
            cohort, trials, {"left": ["C1", "S1"]}, (0, 15), bout_window="bout_15s")
        display = result["displays"]["left"]
        self.assertEqual(display["time_s"]["S1"][[0, -1]].tolist(), [0, 14.5])
        self.assertEqual(display["matrices"]["C1"].shape, (7, 30))
        summary = result["fish_summary"].set_index(["fish_id", "condition"])
        self.assertEqual(summary.loc[("second", "S1"), "n_shared_c1"], 3)

    def test_recruitment_bar_and_raster_style_options(self):
        cohort, trials = self.recruitment_fixture()
        conditions = {"left": ["C1", "S1"]}
        result = several_fish_diagnostics.build_phase_recruitment_analysis(cohort, trials, conditions, (0, 3.6))
        fig, axes = several_fish_figures.plot_phase_recruitment_counts(
            result, cohort["fish_ids"], conditions, figsize=(4, 4), show_fish_identity=False,
            fish_color="#9AAF9F", mean_color="#385F73", legend_fontsize=5, value_fontsize=5)
        from matplotlib.colors import to_hex
        self.assertEqual(to_hex(axes[0, 0].patches[1].get_facecolor()), "#385f73")
        self.assertEqual(axes[1, 0].patches[1].get_height(), 2.5)
        self.assertEqual(len(axes[0, 0].lines), 2)
        for line in axes[0, 0].lines:
            self.assertEqual(line.get_linestyle(), "None")
            self.assertEqual(to_hex(line.get_color()), "#9aaf9f")
        legend = fig.legends[0]
        self.assertEqual([t.get_text() for t in legend.get_texts()], ["Fish", "Mean"])
        self.assertEqual(legend.get_texts()[0].get_fontsize(), 5)
        fig, axes = several_fish_figures.plot_phase_recruitment_proportions(
            result, cohort["fish_ids"], conditions, show_legend=False, show_title=False)
        self.assertEqual(fig.legends, [])
        self.assertAlmostEqual(axes[1, 0].patches[1].get_height(), 0.65)
        default_fig, default_axes = several_fish_figures.plot_phase_recruitment_counts(
            result, cohort["fish_ids"], conditions)
        self.assertEqual([t.get_text() for t in default_fig.legends[0].get_texts()], ["first", "second"])
        self.assertEqual(default_axes[0, 0].lines[0].get_linestyle(), "-")
        fig, axes = several_fish_figures.plot_phase_recruitment_rasters(
            result, "left", figsize=(8, 4), show_row_labels=False, show_legend=False, panel_fontsize=6)
        self.assertEqual(len(axes[0].get_yticks()), 0)
        self.assertEqual(axes[0].get_ylabel(), "Neurons")
        self.assertEqual(fig.legends, [])
        self.assertEqual(axes[1].title.get_fontsize(), 6)
        fig, axes = several_fish_figures.plot_phase_recruitment_rasters(result, "left", figsize=(8, 4))
        self.assertGreater(len(axes[0].get_yticklabels()), 0)
        fig, axes = several_fish_figures.plot_phase_recruitment_decisions(
            result, "left", figsize=(8, 4), show_row_labels=False, show_title=False)
        self.assertEqual(len(axes[0].get_yticks()), 0)
        self.assertIsNone(fig._suptitle)

    def test_reorder_recruitment_rows_pre_then_reference_auc(self):
        cohort, trials = self.recruitment_fixture()
        result = several_fish_diagnostics.build_phase_recruitment_analysis(
            cohort, trials, {"left": ["C1", "S1"]}, (0, 3.6))
        display = result["displays"]["left"]
        metadata = display["row_metadata"]
        original_order = display["neuron_order"].copy()
        # Pool (group 0) = first neuron of each fish: rows 0 ("first", 3) and 2 ("second", 1).
        pre_rows = [i for i in original_order if metadata.group.iloc[i] == 0]
        self.assertEqual(pre_rows, [0, 2])
        auc = {("first", 8): 1.0, ("second", 2): 5.0, ("second", 3): np.nan,
               ("second", 4): 1.0, ("second", 5): 3.0, ("first", 3): 9.0, ("second", 1): -9.0}
        trials = trials.assign(auc_z_s=[auc[(f, n)] + (10.0 if r == 0 else 0.0)
                                        if c == "C1" else -100.0
                                        for f, n, c, r in zip(trials.fish_id, trials.neuron_id,
                                                              trials.condition, trials.repetition)])
        # Repetition 1 (index 0) is excluded by the display's repetitions 2-4.
        snapshot = trials.copy()
        reordered = several_fish_diagnostics.reorder_phase_recruitment_rows(
            result, trials, window="bout")
        new = reordered["displays"]["left"]
        # Pool first in original order; rest by descending C1 AUC, ties stable, NaN last.
        expected_rest = [3, 6, 1, 5, 4]  # second/2 (5), second/5 (3), first/8 (1), second/4 (1), NaN.
        self.assertEqual(new["neuron_order"].tolist(), pre_rows + expected_rest)
        self.assertEqual(new["displayed_row_metadata"].neuron_id.tolist(), [3, 1, 2, 5, 8, 4, 3])
        np.testing.assert_array_equal(new["displayed_row_metadata"].sort_value.to_numpy(),
                                      [9, -9, 5, 3, 1, 1, np.nan])
        self.assertEqual(new["row_segments"][0][1:], (0, 2))
        # Inputs are not mutated.
        np.testing.assert_array_equal(display["neuron_order"], original_order)
        self.assertNotIn("sort_value", display["displayed_row_metadata"])
        self.assertNotIn("row_segments", display)
        pd.testing.assert_frame_equal(trials, snapshot)
        same = several_fish_diagnostics.reorder_phase_recruitment_rows(result, trials, "groups", window="bout")
        np.testing.assert_array_equal(same["displays"]["left"]["neuron_order"], original_order)
        with self.assertRaisesRegex(ValueError, "mode"):
            several_fish_diagnostics.reorder_phase_recruitment_rows(result, trials, "bad", window="bout")
        with self.assertRaisesRegex(ValueError, "no 'bout_15s'"):
            several_fish_diagnostics.reorder_phase_recruitment_rows(result, trials, window="bout_15s")
        # Decisions and rasters follow the new order; only the pre/rest boundary exists.
        fig, axes = several_fish_figures.plot_phase_recruitment_decisions(reordered, "left", figsize=(8, 4))
        np.testing.assert_array_equal(axes[1].images[0].get_array(),
                                      display["decision_matrix"][new["neuron_order"]])
        self.assertEqual([line.get_ydata()[0] for line in axes[1].lines if line.get_linestyle() == ":"], [1.5])
        fig, axes = several_fish_figures.plot_phase_recruitment_rasters(reordered, "left", figsize=(8, 4))
        self.assertEqual([line.get_ydata()[0] for line in axes[0].lines if line.get_linestyle() == ":"], [1.5])
        self.assertEqual(len(axes[0].get_yticklabels()), 2)
        self.assertIn("bottom to top", fig._suptitle.get_text())
        bottom, top = axes[0].get_ylim()
        self.assertLess(bottom, top)
        # rows_top_down flips rows, strips and decisions together; labels stay aligned.
        fig, axes = several_fish_figures.plot_phase_recruitment_rasters(
            reordered, "left", figsize=(8, 4), rows_top_down=True)
        n_rows = len(new["neuron_order"])
        self.assertEqual(tuple(axes[0].get_ylim()), (n_rows - 0.5, -0.5))
        self.assertEqual(tuple(axes[1].get_ylim()), (n_rows - 0.5, -0.5))
        strip = axes[0].child_axes[0]
        self.assertGreater(strip.get_ylim()[0], strip.get_ylim()[1])
        self.assertIn("top to bottom", fig._suptitle.get_text())
        self.assertNotIn("bottom to top", fig._suptitle.get_text())
        self.assertEqual([line.get_ydata()[0] for line in axes[0].lines if line.get_linestyle() == ":"], [1.5])
        self.assertLess(axes[0].get_yticks()[0], axes[0].get_yticks()[1])  # pool label is row 0 side
        fig, axes = several_fish_figures.plot_phase_recruitment_decisions(
            reordered, "left", figsize=(8, 4), rows_top_down=True)
        self.assertGreater(axes[0].get_ylim()[0], axes[0].get_ylim()[1])
        np.testing.assert_array_equal(axes[1].images[0].get_array(),
                                      display["decision_matrix"][new["neuron_order"]])
        fig, axes = several_fish_figures.plot_phase_recruitment_decisions(reordered, "left", figsize=(8, 4))
        self.assertLess(axes[0].get_ylim()[0], axes[0].get_ylim()[1])
        # 'groups' order has no direction in its description: the flipped title states it.
        fig, _ = several_fish_figures.plot_phase_recruitment_rasters(
            same, "left", figsize=(8, 4), rows_top_down=True)
        self.assertIn("Rows, top to bottom: consistent preconditioning", fig._suptitle.get_text())
        fig, _ = several_fish_figures.plot_phase_recruitment_rasters(same, "left", figsize=(8, 4))
        self.assertIn("\nRows: consistent preconditioning", fig._suptitle.get_text())

    def test_recruitment_raster_hides_strips_legend_and_group_lines(self):
        cohort, trials = self.recruitment_fixture()
        result = several_fish_diagnostics.build_phase_recruitment_analysis(
            cohort, trials, {"left": ["C1", "S1"]}, (0, 3.6))
        fig, axes = several_fish_figures.plot_phase_recruitment_rasters(result, "left", figsize=(8, 4))
        self.assertEqual(len(axes[0].child_axes), 1)
        self.assertTrue(any(line.get_linestyle() == ":" for line in axes[0].lines))
        self.assertEqual(len(fig.legends), 1)
        fig, axes = several_fish_figures.plot_phase_recruitment_rasters(
            result, "left", figsize=(8, 4), show_legend=False,
            show_membership_strips=False, show_group_lines=False)
        for ax in axes:
            self.assertEqual(len(ax.child_axes), 0)
            self.assertEqual(len(ax.lines), 0)
        self.assertEqual(fig.legends, [])
        self.assertEqual(len(fig.axes), len(axes) + 1)  # Colorbar kept.
        fig, axes = several_fish_figures.plot_phase_recruitment_decisions(
            result, "left", figsize=(8, 4), show_group_lines=False)
        self.assertFalse(any(line.get_linestyle() == ":" for ax in axes for line in ax.lines))

    def control_comparison_fixture(self):
        # S1 stands in for C2: it has its own preconditioning before its bout.
        cohort, trials = self.recruitment_fixture()
        result = several_fish_diagnostics.build_phase_control_comparison(
            cohort, trials, {"left": ("C1", "S1")}, "bout", (0, 3.6), labels=("C1", "C2"))
        return cohort, trials, result

    def test_control_comparison_groups_rows_and_fractions(self):
        cohort, trials, result = self.control_comparison_fixture()
        display = result["displays"]["left"]
        rows = display["displayed_row_metadata"]
        self.assertEqual(rows.neuron_id.tolist(), [3, 1, 3, 5, 8, 2, 4])
        self.assertEqual(rows.fish_id.tolist(), ["first", "second", "second", "second", "first", "second", "second"])
        self.assertEqual(rows.group.tolist(), [0, 0, 2, 2, 3, 3, 3])
        self.assertEqual(display["membership_codes"]["C1"].tolist(), [1, 0, 1, 0, 1, 0, 1])
        self.assertEqual(display["membership_codes"]["S1"].tolist(), [1, 2, 1, 2, 1, 2, 1])
        np.testing.assert_array_equal(display["decision_matrix"][0], [1, 1, 1])
        self.assertEqual(set(display["matrices"]), {"preconditioning", "C1", "S1"})
        self.assertEqual(display["matrices"]["preconditioning"].shape[0], 7)
        self.assertEqual(result["categories"]["left"], ["C2 bout", "C2 pre", "C1 bout"])
        table = result["fish_summary"].set_index(["fish_id", "category"])
        self.assertEqual(table.loc[("first", "test_bout"), "n_count"], 2)
        self.assertEqual(table.loc[("second", "test_bout"), "n_count"], 5)
        self.assertEqual(table.loc[("first", "test_pre"), "fraction_of_test"], 0.5)
        self.assertAlmostEqual(table.loc[("second", "test_pre"), "fraction_of_test"], 0.2)
        self.assertAlmostEqual(table.loc[("second", "reference_bout"), "fraction_of_test"], 0.6)
        fig, axes = several_fish_figures.plot_phase_recruitment_rasters(result, "left", figsize=(8, 4))
        self.assertEqual([ax.get_title() for ax in axes], ["C2 own preconditioning", "C1 bout", "C2 bout"])
        self.assertIn("C2 bout only", [t.get_text() for t in fig.legends[0].get_texts()])
        fig, axes = several_fish_figures.plot_phase_recruitment_decisions(result, "left", figsize=(8, 4))
        self.assertEqual([t.get_text() for t in axes[1].get_xticklabels()], ["C2 pre", "C1 bout", "C2 bout"])
        np.testing.assert_array_equal(axes[1].images[0].get_array(), display["decision_matrix"][display["neuron_order"]])
        fig, axes = several_fish_figures.plot_phase_control_comparison_summary(
            result, cohort["fish_ids"], figsize=(4, 4), show_fish_identity=False)
        self.assertEqual(axes.shape, (2, 1))
        self.assertEqual(axes[0, 0].patches[0].get_height(), 3.5)
        np.testing.assert_allclose([p.get_height() for p in axes[1, 0].patches], [0.35, 0.55])
        self.assertEqual([t.get_text() for t in axes[1, 0].get_xticklabels()], ["C2 pre", "C1 bout"])

    def test_control_comparison_missing_and_reference_only(self):
        cohort, trials = self.recruitment_fixture()
        # A missing C2-bout decision removes the row and makes that fish's counts N/A.
        trials.loc[trials.fish_id.eq("second") & trials.neuron_id.eq(2) & trials.condition.eq("S1")
                   & trials.window.eq("bout") & trials.repetition.eq(2), "active_trial"] = np.nan
        # A C1-only neuron: first fish neuron 3 inactive in the C2 bout.
        trials.loc[trials.fish_id.eq("first") & trials.neuron_id.eq(3) & trials.condition.eq("S1")
                   & trials.window.eq("bout") & trials.repetition.gt(0), "active_trial"] = 0
        result = several_fish_diagnostics.build_phase_control_comparison(
            cohort, trials, {"left": ("C1", "S1")}, "bout", (0, 3.6), include_reference_only=True)
        rows = result["displays"]["left"]["displayed_row_metadata"]
        self.assertEqual((rows.fish_id + rows.neuron_id.astype(str)).tolist()[-1], "first3")
        self.assertEqual(rows.group.tolist()[-1], 5)
        self.assertNotIn(("second", 2), list(zip(rows.fish_id, rows.neuron_id)))
        table = result["fish_summary"].set_index(["fish_id", "category"])
        self.assertTrue(np.isnan(table.loc[("second", "test_bout"), "n_count"]))
        self.assertTrue(np.isnan(table.loc[("second", "reference_bout"), "fraction_of_test"]))
        self.assertEqual(table.loc[("first", "test_bout"), "n_count"], 1)
        self.assertEqual(table.loc[("first", "reference_bout"), "fraction_of_test"], 0)
        display = result["displays"]["left"]  # Codes follow row_metadata; order displays them.
        self.assertEqual(display["membership_codes"]["C1"][display["neuron_order"]][-1], 3)
        self.assertEqual(display["membership_codes"]["S1"][display["neuron_order"]][-1], 0)

    def shared_nopre_fixture(self):
        # S1 stands in for C2. Make first-fish neuron 8 C1-bout-active so both fish
        # have neurons active in both bouts but not in S1 preconditioning (group 2).
        cohort, trials = self.recruitment_fixture()
        mask = (trials.fish_id.eq("first") & trials.neuron_id.eq(8) & trials.condition.eq("C1")
                & trials.window.eq("bout"))
        trials.loc[mask, "active_trial"] = trials.loc[mask, "repetition"].gt(0).astype(float)
        # Window mean z: C1 = neuron ID, S1 = 1; repetition 1 (excluded) = 100.
        trials["mean_z"] = np.where(trials.condition.eq("C1"), trials.neuron_id.astype(float), 1.0)
        trials.loc[trials.repetition.eq(0), "mean_z"] = 100.0
        for fish in cohort["all_fish_data"].values():
            ids = np.asarray(fish["kept_neuron_indices"], dtype=float)
            bundle = fish["aligned"]["bout"]["zscore"]["trial_aligned_traces"]
            for sid, level in [(1, ids[:, None]), (2, np.ones((len(ids), 1)))]:
                traces = np.repeat(level[:, :, None], bundle[sid].shape[1], axis=1)
                traces = np.repeat(traces, 4, axis=2)
                traces[:, :, 0] = 100.0
                bundle[sid] = traces
        comparison = several_fish_diagnostics.build_phase_control_comparison(
            cohort, trials, {"left": ("C1", "S1")}, "bout", (0, 3.6), labels=("C1", "C2"))
        return cohort, trials, comparison

    def test_shared_nopre_response_selects_group_and_weights_fish(self):
        cohort, trials, comparison = self.shared_nopre_fixture()
        result = several_fish_diagnostics.build_phase_control_shared_nopre_response(
            cohort, trials, comparison, trace_window_s=(-5, 10))
        groups = comparison["neurons"].query("group == 2")
        self.assertEqual(sorted(zip(groups.fish_id, groups.neuron_id)), [("first", 8), ("second", 3), ("second", 5)])
        self.assertEqual(sorted(zip(result["neurons"].fish_id, result["neurons"].neuron_id)),
                         [("first", 8), ("second", 3), ("second", 5)])
        fish = result["fish_summary"].set_index("fish_id")
        self.assertEqual(fish.n_selected.tolist(), [1, 2])
        self.assertEqual(fish.n_neurons.tolist(), [2, 5])
        self.assertEqual(fish.reference_mean_z.tolist(), [8, 4])
        self.assertEqual(fish.test_mean_z.tolist(), [1, 1])
        self.assertEqual(fish.delta_mean_z.tolist(), [7, 3])
        cohort_row = result["cohort_summary"].iloc[0]
        self.assertEqual(cohort_row.reference_mean_z, 6)  # Equal fish, not pooled neurons (16/3).
        self.assertEqual(cohort_row.delta_mean_z, 5)
        self.assertAlmostEqual(cohort_row.reference_sem_z, 2)
        self.assertEqual(cohort_row.n_fish_defined, 2)
        self.assertEqual(cohort_row.n_selected_total, 3)
        traces = result["traces"]["left"]
        self.assertEqual(traces["time_s"][[0, -1]].tolist(), [-5, 9.5])
        np.testing.assert_allclose(traces["reference"], [[8] * 30, [4] * 30])
        np.testing.assert_allclose(traces["reference_mean"], 6)
        np.testing.assert_allclose(traces["reference_sem"], 2)
        np.testing.assert_allclose(traces["test_mean"], 1)
        self.assertAlmostEqual(traces["bout_duration_s"]["reference"], 3.6)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            several_fish_diagnostics.build_phase_control_shared_nopre_response(cohort, trials, comparison)
        broken = dict(comparison, neurons=comparison["neurons"].assign(group=-1))
        with self.assertRaisesRegex(ValueError, "group"):
            several_fish_diagnostics.build_phase_control_shared_nopre_response(
                cohort, trials, broken, trace_window_s=(-5, 10))

    def test_shared_nopre_response_missing_decisions_and_empty_fish(self):
        cohort, trials, _ = self.shared_nopre_fixture()
        # A missing C1 decision excludes the neuron (never counted as inactive).
        trials.loc[trials.fish_id.eq("second") & trials.neuron_id.eq(5) & trials.condition.eq("C1")
                   & trials.window.eq("bout") & trials.repetition.eq(2), "active_trial"] = np.nan
        # First fish: neuron 8 becomes C2-preconditioning-active, leaving no selected neuron.
        trials.loc[trials.fish_id.eq("first") & trials.neuron_id.eq(8) & trials.condition.eq("S1")
                   & trials.window.eq("preconditioning"), "active_trial"] = 1.0
        comparison = several_fish_diagnostics.build_phase_control_comparison(
            cohort, trials, {"left": ("C1", "S1")}, "bout", (0, 3.6), labels=("C1", "C2"))
        result = several_fish_diagnostics.build_phase_control_shared_nopre_response(
            cohort, trials, comparison, trace_window_s=(-5, 10))
        fish = result["fish_summary"].set_index("fish_id")
        self.assertEqual(fish.loc["second", "n_selected"], 1)
        self.assertEqual(fish.loc["second", "n_unclassified"], 1)
        self.assertEqual(fish.loc["second", "reference_mean_z"], 3)
        self.assertEqual(fish.loc["first", "n_selected"], 0)
        self.assertTrue(np.isnan(fish.loc["first", "reference_mean_z"]))
        self.assertTrue(np.isnan(result["traces"]["left"]["reference"][0]).all())
        cohort_row = result["cohort_summary"].iloc[0]
        self.assertEqual(cohort_row.reference_mean_z, 3)  # Missing fish is not zero.
        self.assertEqual(cohort_row.n_fish_defined, 1)
        self.assertTrue(np.isnan(cohort_row.reference_sem_z))
        fig, axes = several_fish_figures.plot_phase_control_shared_nopre_response(result, "left")
        self.assertIn("1/2 fish", axes[1].get_title())

    def test_shared_nopre_response_plot(self):
        from matplotlib.colors import to_hex
        cohort, trials, comparison = self.shared_nopre_fixture()
        result = several_fish_diagnostics.build_phase_control_shared_nopre_response(
            cohort, trials, comparison, trace_window_s=(-5, 10))
        fig, axes = several_fish_figures.plot_phase_control_shared_nopre_response(
            result, "left", figsize=(3, 4), test_color="#C2703D", legend_fontsize=5)
        np.testing.assert_allclose([p.get_height() for p in axes[1].patches], [6, 1])
        self.assertEqual(to_hex(axes[1].patches[0].get_facecolor()), "#385f73")
        dots = [line for line in axes[1].lines if line.get_marker() == "o"]
        self.assertEqual(len(dots), 2)
        for line in dots:
            self.assertEqual(line.get_linestyle(), "None")
            self.assertEqual(to_hex(line.get_color()), "#9aaf9f")
        self.assertIn("+5.00", axes[1].get_title())
        self.assertEqual([t.get_text() for t in axes[1].get_xticklabels()], ["C1 bout", "C2 bout"])
        self.assertEqual(sorted(t.get_text() for t in axes[1].texts), ["1.00", "6.00"])
        curves = {line.get_label(): line for line in axes[0].lines}
        np.testing.assert_allclose(curves["C1 bout"].get_ydata(), 6)
        self.assertEqual(to_hex(curves["C2 bout"].get_color()), "#c2703d")
        self.assertEqual(len(axes[0].collections), 2)  # SEM bands.
        self.assertEqual(fig.legends[0].get_texts()[0].get_fontsize(), 5)
        self.assertIn("3 neurons", fig._suptitle.get_text())
        fig, axes = several_fish_figures.plot_phase_control_shared_nopre_response(
            result, "left", show_sem=False, show_fish_lines=True, show_legend=False, show_title=False,
            show_values=False, show_difference=False)
        self.assertEqual(len(axes[0].collections), 0)
        self.assertTrue(all(line.get_linestyle() == "-" for line in axes[1].lines if line.get_marker() == "o"))
        self.assertEqual(fig.legends, [])
        self.assertIsNone(fig._suptitle)
        self.assertEqual(axes[1].get_title(), "")
        self.assertEqual(len(axes[1].texts), 0)

    def test_select_phase_panel_rows_restricts_matrices_and_means(self):
        cohort = self.synthetic_cohort()
        panels = several_fish_diagnostics.build_phase_stimulus_panel_displays(
            cohort, {"left": ["C1", "S1"]}, alignment="sequence", sort_mode="unsorted")
        panel = panels["zscore"]["left"]
        rows = pd.DataFrame(dict(fish_id=["second", "first", "second"], neuron_id=[4, 8, 1], group=[0, 0, 1]))
        subset = several_fish_diagnostics.select_phase_panel_rows(panel, rows)
        np.testing.assert_array_equal(subset["source_positions"], [5, 1, 2])
        np.testing.assert_array_equal(subset["matrices"][1], panel["matrices"][1][[5, 1, 2]])
        np.testing.assert_allclose(subset["fish_means"][1][:, 0], [2, 10])
        np.testing.assert_array_equal(subset["neuron_order"], [0, 1, 2])
        self.assertEqual(subset["row_metadata"].group.tolist(), [0, 0, 1])
        self.assertEqual(panel["matrices"][1].shape[0], 7)
        result = several_fish_figures.plot_all_fish_stimulus_figures(subset, {"left": [1, 2]},
            layouts=("combined",), mean_mode="pooled_neurons", show_fish_traces=False,
            z_limits=(-1, 4), show_legend=False)
        curve = next(line for line in result["axes"]["left_combined"][1, 0].lines
                     if line.get_label() == "Pooled-neuron mean")
        np.testing.assert_allclose(curve.get_ydata(), 22 / 3)
        self.assertEqual(result["figures"]["left_combined"].legends, [])
        result = several_fish_figures.plot_all_fish_stimulus_figures(subset, {"left": [1]},
            layouts=("raster",), legend_fontsize=5)
        self.assertEqual(result["figures"]["left_raster"].legends[0].get_texts()[0].get_fontsize(), 5)
        with self.assertRaisesRegex(ValueError, "identities"):
            several_fish_diagnostics.select_phase_panel_rows(panel, rows.assign(neuron_id=[4, 99, 1]))

    def test_recruitment_strength_inputs_match_summary_plot(self):
        cohort, trials = self.recruitment_fixture()
        trials["mean_z"] = np.where(trials.condition.eq("C1"), 3.0, 1.0)
        trials.loc[trials.repetition.eq(0), "mean_z"] = 100.0  # Excluded repetition.
        inputs = several_fish_diagnostics.build_phase_recruitment_strength_inputs(
            cohort, trials, {"left": ["C1", "S1"]}, ["C1", "C2"], "bout")
        table = inputs["analysis"]["tables"]["recruitment"].set_index(["fish_id", "condition"])
        self.assertEqual(table.loc[("first", "C1"), "active_fraction"], 0.5)
        self.assertEqual(table.loc[("second", "C1"), "active_fraction"], 0.6)
        self.assertEqual(table.loc[("second", "C2"), "mean_normalized_auc"], 1)
        shared = inputs["shared_auc"]["tables"]["shared_auc_per_fish"].set_index("fish_id")
        self.assertEqual(shared.n_shared.tolist(), [1, 3])
        self.assertEqual(shared.mean_delta_normalized_auc.tolist(), [2, 2])
        panels = several_fish_figures.plot_recruitment_strength_summary(
            inputs["analysis"], inputs["shared_auc"], ylabels=["Mean z", "Active", "C1 - stimulus"],
            legend_fontsize=5)
        fig, axes = panels["left"]
        mean = [line for line in axes[1].lines if line.get_label() == "Mean"][0]
        np.testing.assert_allclose(mean.get_ydata(), [0.55, 1.0])
        self.assertEqual(axes[2].get_ylabel(), "C1 - stimulus")
        self.assertEqual(fig.legends[0].get_texts()[0].get_fontsize(), 5)
        self.assertEqual([t.get_text() for t in axes[2].get_xticklabels()], ["C1", "C2"])
        trials.loc[trials.fish_id.eq("first") & trials.neuron_id.eq(8) & trials.condition.eq("S1")
                   & trials.window.eq("bout") & trials.repetition.eq(2), "active_trial"] = np.nan
        missing = several_fish_diagnostics.build_phase_recruitment_strength_inputs(
            cohort, trials, {"left": ["C1", "S1"]}, ["C1", "C2"], "bout")
        table = missing["analysis"]["tables"]["recruitment"].set_index(["fish_id", "condition"])
        self.assertTrue(np.isnan(table.loc[("first", "C2"), "active_fraction"]))
        shared = missing["shared_auc"]["tables"]["shared_auc_per_fish"].set_index("fish_id")
        self.assertTrue(np.isnan(shared.loc["first", "mean_delta_normalized_auc"]))
        self.assertEqual(fig._suptitle.get_text(), "Left stimuli")  # Default title unchanged.
        panels = several_fish_figures.plot_recruitment_strength_summary(
            inputs["analysis"], inputs["shared_auc"], title_suffix="trials 2, 3, 4 (>= 2/3)")
        self.assertEqual(panels["left"][0]._suptitle.get_text(), "Left stimuli\ntrials 2, 3, 4 (>= 2/3)")

    def test_all_four_repetitions_switch_through_helpers(self):
        # Fixture trials: repetition 1 is the opposite of repetitions 2-4; S1 neuron j=1
        # has repetition-4 bout 0 and preconditioning 1. With all four trials and >= 2/4,
        # that neuron's S1 preconditioning becomes active (2/4) though it is 1/3 in trials 2-4.
        cohort, trials = self.shared_nopre_fixture()[:2]
        trials["auc_z_s"] = trials["mean_z"]
        rule = dict(repetitions=(1, 2, 3, 4), min_active_reps=2)
        recruitment = several_fish_diagnostics.build_phase_recruitment_analysis(
            cohort, trials, {"left": ["C1", "S1"]}, (0, 3.6), **rule)
        self.assertEqual(recruitment["displays"]["left"]["repetitions"], [1, 2, 3, 4])
        pool = recruitment["preconditioning_pool"]
        self.assertEqual(set(pool.n_expected_trials), {4})
        self.assertEqual(set(pool.min_passing_trials), {3})  # ceil(0.75 * 4 presentations).
        summary = recruitment["fish_summary"].set_index(["fish_id", "condition"])
        self.assertEqual(summary.loc[("first", "S1"), "n_new_after_pre"], 0)  # 1 with trials 2-4.
        self.assertEqual(summary.loc[("second", "S1"), "n_new_after_pre"], 3)
        reordered = several_fish_diagnostics.reorder_phase_recruitment_rows(recruitment, trials, window="bout")
        shown = reordered["displays"]["left"]["displayed_row_metadata"].set_index(["fish_id", "neuron_id"])
        self.assertEqual(shown.loc[("second", 5), "sort_value"], (100 + 3 * 5) / 4)  # All four averaged.
        comparison = several_fish_diagnostics.build_phase_control_comparison(
            cohort, trials, {"left": ("C1", "S1")}, "bout", (0, 3.6), labels=("C1", "C2"), **rule)
        rows = comparison["displays"]["left"]["displayed_row_metadata"]
        self.assertEqual(list(zip(rows.fish_id, rows.neuron_id, rows.group)),
                         [("first", 3, 0), ("first", 8, 0), ("second", 1, 0), ("second", 2, 1),
                          ("second", 3, 2), ("second", 5, 2), ("second", 4, 3)])
        response = several_fish_diagnostics.build_phase_control_shared_nopre_response(
            cohort, trials, comparison, trace_window_s=(-5, 10))
        fish = response["fish_summary"].set_index("fish_id")
        self.assertEqual(fish.loc["second", "reference_mean_z"], ((100 + 9) / 4 + (100 + 15) / 4) / 2)
        self.assertEqual(fish.loc["second", "test_mean_z"], (100 + 3) / 4)
        self.assertEqual(fish.loc["first", "n_selected"], 0)
        self.assertEqual(response["repetitions"], [1, 2, 3, 4])
        strength = several_fish_diagnostics.build_phase_recruitment_strength_inputs(
            cohort, trials, {"left": ["C1", "S1"]}, ["C1", "C2"], "bout", **rule)
        self.assertEqual(strength["analysis"]["repetitions"], [1, 2, 3, 4])
        # The 07a z-score rows are built independently of Cell 04a with the analysis
        # repetitions. This fixture's raw trials have two repetitions, so (1, 2) is "all".
        plain = self.synthetic_cohort()
        panels = several_fish_diagnostics.build_phase_stimulus_panel_displays(
            plain, {"left": ["C1", "S1"]}, alignment="sequence", repetitions=(1, 2), sort_mode="unsorted")
        sequence = several_fish_diagnostics.build_phase_sequence_panel_displays(plain, panels)["zscore"]
        self.assertEqual(sequence["left"]["repetition_numbers"], {1: [1, 2], 2: [1, 2]})
        subset = several_fish_diagnostics.select_phase_panel_rows(
            sequence["left"], comparison["displays"]["left"]["displayed_row_metadata"])
        self.assertEqual(subset["matrices"][1].shape[0], 7)
        fig, _ = several_fish_figures.plot_phase_recruitment_rasters(comparison, "left", figsize=(8, 4))
        self.assertIn("repetitions 1, 2, 3, 4", fig._suptitle.get_text())

    def test_pre_bout_window_covers_control_and_conditioned_bouts(self):
        settings = dict(self.measurements, bout_windows={"bout": (0, 3.6), "pre_bout": (-2, 0)})
        result = several_fish_diagnostics.build_phase_population_analysis(self.synthetic_cohort(), settings)
        coverage = result["window_coverage"].query("window == 'pre_bout'")
        self.assertEqual(set(coverage.n_samples), {4})
        bout_start = {"C1": 0.0, "S1": 8.4}  # Seconds from sequence onset; C1 bout = sequence onset.
        for row in coverage.itertuples():
            self.assertGreaterEqual(row.first_sample_s, bout_start[row.condition] - 2)
            self.assertLess(row.last_sample_s, bout_start[row.condition])
        values = result["fish_summary"].query("window == 'pre_bout'").set_index(["fish_id", "condition"])
        self.assertEqual(values.loc[("first", "C1"), "mean_z"], 2)

    def test_shared_nopre_response_pre_bout_baseline(self):
        cohort, trials, comparison = self.shared_nopre_fixture()
        # Per-trial pre-bout mean z: C1 = 0.5, C2 (S1) = 2 + repetition (carry-over); rep 1 excluded.
        baseline = trials.loc[trials.window.eq("bout")].assign(window="pre_bout")
        baseline["mean_z"] = np.where(baseline.condition.eq("C1"), 0.5, 2.0 + baseline.repetition)
        baseline.loc[baseline.repetition.eq(0), "mean_z"] = 100.0
        trials = pd.concat([trials, baseline], ignore_index=True)
        for fish in cohort["all_fish_data"].values():
            bout = fish["aligned"]["bout"]["zscore"]
            before = (bout["time_s"][2] >= -2) & (bout["time_s"][2] < 0)
            bout["trial_aligned_traces"][2][:, before, 1:] = 3.0  # C2 pre-bout carry-over level.
        default = several_fish_diagnostics.build_phase_control_shared_nopre_response(
            cohort, trials, comparison, trace_window_s=(-5, 10))
        self.assertIsNone(default["baseline_window"])
        self.assertEqual(default["fish_summary"].set_index("fish_id").reference_mean_z.tolist(), [8, 4])
        result = several_fish_diagnostics.build_phase_control_shared_nopre_response(
            cohort, trials, comparison, trace_window_s=(-5, 10),
            baseline_window="pre_bout", baseline_window_s=(-2, 0))
        fish = result["fish_summary"].set_index("fish_id")
        self.assertEqual(fish.reference_mean_z.tolist(), [7.5, 3.5])  # ID - 0.5.
        self.assertEqual(fish.test_mean_z.tolist(), [-3, -3])  # 1 - mean(3, 4, 5).
        self.assertEqual(fish.delta_mean_z.tolist(), [10.5, 6.5])
        self.assertEqual(result["cohort_summary"].iloc[0].delta_mean_z, 8.5)  # Equal fish.
        self.assertEqual((result["baseline_window"], result["baseline_window_s"]), ("pre_bout", (-2.0, 0.0)))
        traces = result["traces"]["left"]
        after = traces["time_s"] >= 0
        np.testing.assert_allclose(traces["reference_mean"], 0)  # Constant traces minus their baseline.
        np.testing.assert_allclose(traces["test_mean"][after], -2)
        for bad in [dict(baseline_window="pre_bout"), dict(baseline_window_s=(-2, 0)),
                    dict(baseline_window="missing", baseline_window_s=(-2, 0))]:
            with self.subTest(options=bad), self.assertRaises(ValueError):
                several_fish_diagnostics.build_phase_control_shared_nopre_response(
                    cohort, trials, comparison, trace_window_s=(-5, 10), **bad)
        fig, axes = several_fish_figures.plot_phase_control_shared_nopre_response(default, "left")
        self.assertEqual(axes[1].get_ylabel(), "Mean z-score\n(0-3.6 s)")
        self.assertEqual(axes[0].get_ylabel(), "Mean z-score")
        self.assertNotIn("pre-bout", fig._suptitle.get_text())
        fig, axes = several_fish_figures.plot_phase_control_shared_nopre_response(result, "left")
        self.assertEqual(axes[1].get_ylabel(), "Mean z-score − pre-bout\n(0-3.6 s)")
        self.assertEqual(axes[0].get_ylabel(), "Mean z-score − pre-bout")
        self.assertIn("minus pre-bout mean (-2 to 0 s)", fig._suptitle.get_text())
        self.assertIn("Pre-bout -2 to 0 s", [t.get_text() for t in fig.legends[0].get_texts()])

    def prepare_reviewed(self):
        self.fixture.prepare_recording()
        loaded = self.fixture.load()
        out = self.fixture.root / "cache"
        zres = dict(z_traces=np.ones((320, 3)), baseline_mean=np.zeros(3),
                    baseline_std=np.ones(3), valid_onsets=np.array([24, 76, 184, 236]))
        rel = dict(kept_neuron_indices=np.array([0, 2]), otsu_threshold=0.4)
        sig = (None, np.zeros((320, 2)), None, None, None, None, np.zeros((320, 2)), None)
        data_loading.save_phase_analysis_outputs(loaded, zres, rel, sig,
            dict(baseline_s=10, significant=dict(n_bins=64)), out)
        args = dict(fish_id="L1_f01", experiment_name="experiment", main_path=self.fixture.root / "data",
            stimuli_main_path=self.fixture.root / "stimuli", phase_metadata_path=self.fixture.phase_path,
            stimulus_aliases=self.fixture.aliases, selected_blocks=["B1", "B2"], cache_root=out)
        return args, out

    def test_reviewed_loader_legacy_settings_and_missing_fish(self):
        args, out = self.prepare_reviewed()
        file = out / "significant_traces/L1_f01_significant_traces.npz"
        with np.load(file) as cache:
            legacy = {k: cache[k] for k in ["raster", "deltaF_center", "kept_neuron_indices"]}
        np.savez(file, **legacy)
        loaded = data_loading.load_reviewed_phase_experiment(**args)
        self.assertEqual(loaded["detector_settings"], dict(n_bins=64))
        settings = {k: v for k, v in args.items() if k not in ["fish_id", "cache_root"]}
        settings.update(fish_ids=["L1_f01", "missing"], cache_roots={"L1_f01": out}, display_pre_s=5, recovery_s=10)
        cohort = several_fish_loading.load_reviewed_phase_cohort(settings)
        self.assertEqual(cohort["fish_ids"], ["L1_f01"])
        self.assertEqual(cohort["readiness"].status.tolist(), ["ready", "missing"])
        np.save(out / "filtered_neurons_by_stimuli/L1_f01_kept_neuron_indices.npy", [2, 0])
        with self.assertRaisesRegex(ValueError, "mapping"):
            data_loading.load_reviewed_phase_experiment(**args)

    def test_reviewed_cache_can_move_between_data_roots(self):
        args, out = self.prepare_reviewed()
        config_file = out / "phase_alignment/L1_f01_analysis_configuration.json"
        config = json.loads(config_file.read_text())
        original_source = Path(config["paths"]["dfof_file"])
        relative = original_source.relative_to(args["main_path"])
        # Model provenance saved on the other Windows workstation.
        saved_source = str(PureWindowsPath("D:/Alejandro/Data/OneDrive - Universit\u00e9 de Lausanne/Lab/Data/2p") / relative)
        config["paths"]["dfof_file"] = saved_source
        config_file.write_text(json.dumps(config))
        cache_file = out / "significant_traces/L1_f01_significant_traces.npz"
        with np.load(cache_file) as cache:
            values = {key: cache[key] for key in cache.files}
        values["source_file"] = saved_source
        np.savez(cache_file, **values)
        moved = self.fixture.root / "other-computer" / "data"
        shutil.copytree(args["main_path"], moved)
        args["main_path"] = moved
        loaded = data_loading.load_reviewed_phase_experiment(**args)
        np.testing.assert_array_equal(loaded["kept_neuron_indices"], [0, 2])
        self.assertEqual(loaded["paths"]["dfof_file"], moved / relative)
        # Matching filenames alone cannot bypass source provenance or fish identity.
        values["source_file"] = str(PureWindowsPath(saved_source).with_name("other_dFoF_merged.npy"))
        np.savez(cache_file, **values)
        with self.assertRaisesRegex(ValueError, "source file differs"):
            data_loading.load_reviewed_phase_experiment(**args)
        config["paths"]["dfof_file"] = values["source_file"]
        config_file.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "source file differs"):
            data_loading.load_reviewed_phase_experiment(**args)

    def test_inconsistent_settings_and_timestamps_fail(self):
        args, out = self.prepare_reviewed()
        config = out / "phase_alignment/L1_f01_analysis_configuration.json"
        content = json.loads(config.read_text())
        content["configuration"]["significant"]["n_bins"] = 32
        config.write_text(json.dumps(content))
        with self.assertRaisesRegex(ValueError, "settings differ"):
            data_loading.load_reviewed_phase_experiment(**args)
        content["configuration"]["significant"]["n_bins"] = 64
        config.write_text(json.dumps(content))
        content["session_id"] = "wrong-session"
        config.write_text(json.dumps(content))
        with self.assertRaisesRegex(ValueError, "session"):
            data_loading.load_reviewed_phase_experiment(**args)
        content["session_id"] = "session"
        config.write_text(json.dumps(content))
        timing_file = out / "phase_alignment/L1_f01_trial_timing.csv"
        timing = pd.read_csv(timing_file)
        timing.loc[0, "onset_s"] += 1
        timing.to_csv(timing_file, index=False)
        with self.assertRaisesRegex(ValueError, "trial timing"):
            data_loading.load_reviewed_phase_experiment(**args)

    def test_incomplete_activity_keeps_denominator_and_reports_missing_fraction(self):
        cohort = self.synthetic_cohort()
        raster = cohort["all_fish_data"]["first"]["aligned"]["sequence"]["raster"]["trial_aligned_traces"][1]
        raster[0, :, 0] = np.nan
        result = several_fish_diagnostics.build_phase_population_analysis(cohort, self.measurements)
        row = result["fish_summary"].loc[lambda t: t.fish_id.eq("first") & t.condition.eq("C1") & t.window.eq("bout")].iloc[0]
        self.assertEqual(row.n_neurons, 2)
        self.assertEqual(row.n_valid_active, 1)
        self.assertTrue(np.isnan(row.active_fraction))

    def test_pooled_preconditioning_requires_twelve_of_fifteen_trials(self):
        cohort = self.synthetic_cohort()
        conditions = {"left": ["C1", "S1", "S2", "S3", "S4", "S5"]}
        rows = []
        for fid, fish in cohort["all_fish_data"].items():
            for j, neuron in enumerate(fish["kept_neuron_indices"]):
                for condition_index, condition in enumerate(conditions["left"][1:]):
                    for repetition in range(4):
                        passing = (condition_index * 3 + repetition - 1) < (11 if j == 0 else 12)
                        rows.append(dict(fish_id=fid, neuron_id=neuron, condition=condition,
                            window="preconditioning", repetition=repetition,
                            active_trial=float(passing if repetition else 1)))
        trials = pd.DataFrame(rows)
        pool = several_fish_diagnostics._phase_preconditioning_pool(
            cohort, trials, conditions, [2, 3, 4], 0.75)
        first = pool.loc[pool.fish_id.eq("first")]
        self.assertEqual(first.n_expected_trials.tolist(), [15, 15])
        self.assertEqual(first.min_passing_trials.tolist(), [12, 12])
        self.assertEqual(first.n_passing_trials.tolist(), [11, 12])
        self.assertEqual(first.active.tolist(), [0, 1])
        row = trials.index[(trials.fish_id == "first") & (trials.neuron_id == 8)
                           & (trials.repetition == 1)][0]
        trials.loc[row, "active_trial"] = np.nan
        pool = several_fish_diagnostics._phase_preconditioning_pool(cohort, trials, conditions, [2, 3, 4], 0.75)
        missing = pool.loc[pool.fish_id.eq("first") & pool.neuron_id.eq(8)].iloc[0]
        self.assertTrue(np.isnan(missing.active))
        self.assertEqual(missing.n_valid_trials, 14)

    def test_report_save_gate(self):
        root = self.fixture.root / "reports"
        off = several_fish_reporting.save_analysis_report_run({}, {"save_report": False}, root)
        self.assertIsNone(off)
        self.assertFalse(root.exists())
        saved = several_fish_reporting.save_analysis_report_run(dict(experiment_name="example"),
            dict(save_report=True, export_notebook=False), root)
        self.assertTrue(saved["run_dir"].exists())
        self.assertNotIn("figures", saved)


if __name__ == "__main__":
    unittest.main()

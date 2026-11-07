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
            pre_bout_s=2, active_fraction_threshold=0.1, min_epoch_s=1,
            min_active_reps=1, expected_reps=2)

    def synthetic_cohort(self):
        fish = {}
        for fid, ids, level, active in [("first", [3, 8], 2, 1), ("second", [1, 2, 3, 4, 5], 10, 0)]:
            z = np.full((320, len(ids)), level, dtype=float)
            raster = np.full_like(z, active)
            data = dict(kept_neuron_indices=np.array(ids), trials=self.fixture.trials,
                        fps_2p=2, stimuli_ids=[1, 2], stimuli_id_map={"C1_Bout": 1, "S1_Overlap": 2})
            data["aligned"] = {anchor: {kind: trial_alignment.build_event_aligned_traces(values,
                self.fixture.trials, 2, alignment=anchor) for kind, values in [("zscore", z), ("raster", raster)]}
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

    def test_measurements_memberships_and_absent_phase(self):
        result = several_fish_diagnostics.build_phase_population_analysis(self.synthetic_cohort(),
            self.measurements, {"S1-C1": ("S1", "C1")})
        fish = result["fish_summary"]
        pre_control = fish.loc[fish.condition.eq("C1") & fish.window.eq("preconditioning")]
        self.assertTrue(pre_control.auc_z_s.isna().all())
        self.assertTrue(pre_control.active_fraction.isna().all())
        short = fish.loc[fish.condition.eq("C1") & fish.window.eq("bout")].set_index("fish_id")
        self.assertEqual(short.loc["first", "auc_z_s"], 6)
        self.assertEqual(short.loc["first", "active_fraction"], 1)
        self.assertEqual(short.loc["second", "active_fraction"], 0)
        self.assertEqual(short.loc["first", "n_neurons"], 2)
        self.assertEqual(short.loc["second", "n_neurons"], 5)
        self.assertEqual(short.loc["first", "delta_auc_z_s"], 0)
        membership = result["phase_memberships"]
        self.assertEqual(membership.loc[membership.fish_id.eq("first") & membership.category.eq("both"), "fraction"].iloc[0], 1)
        self.assertEqual(membership.loc[membership.fish_id.eq("second") & membership.category.eq("neither"), "fraction"].iloc[0], 1)
        self.assertFalse(membership.condition.eq("C1").any())
        self.assertTrue(result["correlations"].spearman_rho.isna().all())

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

    def recruitment_fixture(self):
        cohort = self.synthetic_cohort()
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
        result = several_fish_diagnostics.build_phase_population_analysis(cohort, self.measurements, {})
        row = result["fish_summary"].loc[lambda t: t.fish_id.eq("first") & t.condition.eq("C1") & t.window.eq("bout")].iloc[0]
        self.assertEqual(row.n_neurons, 2)
        self.assertEqual(row.n_valid_active, 1)
        self.assertTrue(np.isnan(row.active_fraction))

    def test_numeric_delays_keep_actual_spacing(self):
        from src.plotting_diagnostics import plot_paired_fish_values
        table = pd.DataFrame(dict(fish_id=["one"] * 3, condition=["a", "b", "c"], value=[2, 3, 1]))
        fig, ax = plt.subplots()
        plot_paired_fish_values(table, "value", ["a", "b", "c"], ["one"], ax, "value", x_values=[0.3, 0.6, 8])
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [0.3, 0.6, 8])
        np.testing.assert_array_equal(ax.lines[1].get_ydata(), [2, 3, 1])

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

    def test_report_save_gate_and_optional_figures(self):
        root = self.fixture.root / "reports"
        off = several_fish_reporting.save_analysis_report_run({}, {"save_report": False}, root)
        self.assertIsNone(off)
        self.assertFalse(root.exists())
        fig, ax = plt.subplots()
        ax.plot([0, 1])
        saved = several_fish_reporting.save_analysis_report_run(dict(experiment_name="example"),
            dict(save_report=True, export_notebook=False), root, figures={"diagnostic": fig})
        self.assertTrue(saved["figures"]["diagnostic"].exists())


if __name__ == "__main__":
    unittest.main()

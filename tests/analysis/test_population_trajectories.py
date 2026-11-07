"""Known-output checks for shared PCA and unbiased population distances."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.population_trajectories import (build_population_trajectory_analysis,
    test_population_separation as run_separation_test, evaluate_population_contrasts,
    check_population_subsampling, build_joint_population_pca)
from src.stimuli_timeline import first_trajectory_difference


class PopulationTrajectoryTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        base = rng.normal(size=(6, 12, 1))
        a = np.repeat(base, 4, axis=2)
        b = a.copy()
        b[:, 6:, :] += 2
        self.fish = {fid: dict(kept_neuron_indices=np.array([4, 1, 3, 0]),
            stimuli_id_map={'A': 1, 'B': 2}, trial_aligned_traces_z_core={1: a.copy(), 2: b.copy()})
            for fid in ['f1', 'f2', 'f3']}
        self.sides = {'left': dict(stimuli=['A', 'B'], pairs=[('A', 'B')])}

    def build(self, **kwargs):
        return build_population_trajectory_analysis(self.fish, list(self.fish), self.sides,
            fps_2p=2, t_pre_s=0, program_onset_s=0,
            window_s=(0, 6), fit_window_s=(0, 6), **kwargs)

    def test_known_distances_and_shared_projection(self):
        result = self.build()
        first = result['metrics'].query('fish_id == "f1"')
        np.testing.assert_allclose(first.rms_distance, [0] * 6 + [2] * 6)
        np.testing.assert_allclose(first.cv_squared_distance, [0] * 6 + [4] * 6, atol=1e-14)
        np.testing.assert_allclose(first.pattern_correlation, 1)
        fit = result['fits']['left']['f1']
        np.testing.assert_array_equal(fit['neuron_ids'], [4, 1, 3, 0])
        np.testing.assert_allclose(fit['means']['A'], fit['trials']['A'].mean(axis=1), atol=1e-14)
        sig = run_separation_test(result, n_permutations=999, seed=1)
        self.assertFalse(sig.significant.iloc[:6].any())
        self.assertTrue(sig.significant.iloc[6:].all())
        pd.testing.assert_frame_equal(sig, run_separation_test(result, n_permutations=999, seed=1))

    def test_negative_noise_estimate_is_retained(self):
        for fish in self.fish.values():
            fish['trial_aligned_traces_z_core'][2] = (
                fish['trial_aligned_traces_z_core'][1] + np.array([1, -1, 1, -1]))
        result = self.build()
        np.testing.assert_allclose(result['metrics'].cv_squared_distance, -1 / 3, atol=1e-14)
        np.testing.assert_allclose(result['metrics'].rms_distance, 0, atol=1e-14)

    def test_pooled_count_balance_is_invariant_to_duplicate_neuron_features(self):
        result = self.build(pca_scope='pooled')
        original = result['pooled_fits']['left']
        self.assertEqual(len(original['neuron_metadata']), 12)
        self.assertEqual(result['fits']['left'], {})
        fish = self.fish['f1']
        fish['trial_aligned_traces_z_core'] = {sid: np.repeat(arr, 2, axis=0)
            for sid, arr in fish['trial_aligned_traces_z_core'].items()}
        fish['kept_neuron_indices'] = (fish['kept_neuron_indices'][:, None] * 2 + [0, 1]).ravel()
        duplicated = self.build(pca_scope='pooled')['pooled_fits']['left']
        a, b = [np.concatenate([fit['means'][label] for label in ['A', 'B']])
                for fit in [original, duplicated]]
        np.testing.assert_allclose(a @ a.T, b @ b.T, atol=1e-12)
        np.testing.assert_allclose(original['explained_variance_ratio'],
                                   duplicated['explained_variance_ratio'], atol=1e-12)
        evidence = run_separation_test(result, n_permutations=999, seed=1)
        self.assertTrue(evidence.significant.iloc[6:].all())

    def test_equal_count_resampling_matches_full_result_when_all_neurons_used(self):
        result = self.build(pca_scope='pooled')
        sampled = check_population_subsampling(result, n_draws=5, seed=3)
        np.testing.assert_allclose(sampled.all_neuron_distance, sampled.subsample_median, atol=1e-12)
        np.testing.assert_allclose(sampled.subsample_p05, sampled.subsample_p95, atol=1e-12)
        pd.testing.assert_frame_equal(sampled, check_population_subsampling(result, n_draws=5, seed=3))
        with self.assertRaises(ValueError):
            check_population_subsampling(result, neurons_per_fish=5)

    def test_contrast_discovery_validation_split_and_constant_identity(self):
        result = self.build(pca_scope='pooled')
        table = evaluate_population_contrasts(result, {'left': 'A'},
            {'left': {'identical': 'A', 'different': 'B'}}, window_s=(0, 6))
        np.testing.assert_allclose(table.loc[table.candidate.eq('identical'), 'pattern_correlation'], 1)
        discovery = table[table.split.eq('discovery')].set_index(['fish_id', 'candidate'])
        validation = table[table.split.eq('validation')].set_index(['fish_id', 'candidate'])
        np.testing.assert_allclose(discovery.pattern_correlation, validation.pattern_correlation)
        with self.assertRaises(ValueError):
            evaluate_population_contrasts(result, {'left': 'A'}, {'left': {'B': 'B'}},
                window_s=(0, 6), discovery_reps=(0, 1), validation_reps=(1, 2))

    def test_joint_pca_aligns_original_ids_and_identical_cross_side_paths(self):
        self.sides['right'] = dict(stimuli=['C', 'D'], pairs=[('C', 'D')])
        for fish in self.fish.values():
            fish['stimuli_id_map'].update(C=3, D=4)
            fish['trial_aligned_traces_z_core'].update({
                3: fish['trial_aligned_traces_z_core'][1].copy(),
                4: fish['trial_aligned_traces_z_core'][2].copy()})
        self.fish['f1']['trial_aligned_traces_z_core'][3][4, 0, 0] = np.nan
        result = self.build(pca_scope='pooled')
        joint = build_joint_population_pca(result)
        self.assertEqual(joint['pca'].n_samples_, 48)
        np.testing.assert_allclose(joint['means'][('left', 'A')], joint['means'][('right', 'C')], atol=1e-12)
        np.testing.assert_allclose(joint['means'][('left', 'B')], joint['means'][('right', 'D')], atol=1e-12)
        np.testing.assert_array_equal(joint['neuron_metadata'].query('fish_id == "f1"').neuron_id, [1, 3, 0])
        self.assertEqual(joint['joint_audit'].excluded_from_joint.sum(), 1)

    def test_control_distance_overlay_reports_known_first_evidence(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from src.plotting_population_trajectories import plot_control_distance_overlay
        result = self.build(pca_scope='pooled')
        evidence = run_separation_test(result, n_permutations=999, seed=1)
        plotted = plot_control_distance_overlay(result, {'left': 'A'}, {'left': ['A', 'B']},
            significance=evidence, time_window_s=(0, 5.5))
        self.assertEqual(plotted['axes'].size, 1)
        self.assertEqual(plotted['detection_table'].iloc[0].first_corrected_evidence_s, 3)
        plt.close(plotted['figure'])

    def test_joint_nonfinite_exclusion_and_repetition_validation(self):
        self.fish['f1']['trial_aligned_traces_z_core'][2][4, 0, 0] = np.nan
        result = self.build()
        np.testing.assert_array_equal(result['fits']['left']['f1']['neuron_ids'], [1, 3, 0])
        self.assertEqual(result['audit'].iloc[0].n_excluded_nonfinite, 1)
        self.fish['f1']['trial_aligned_traces_z_core'][1] = np.zeros((6, 12, 3))
        with self.assertRaisesRegex(ValueError, 'expected'):
            self.build()

    def test_asset_difference_uses_all_spatial_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a.csv', Path(directory) / 'b.csv'
            df = pd.DataFrame({'dot0_x': [1., 1., 1.], 'dot0_y': [0., 0., 0.],
                               'dot0_radius': [0.2, 0.2, 0.2]})
            df.to_csv(a, index=False)
            df.loc[2, 'dot0_radius'] = 0
            df.to_csv(b, index=False)
            self.assertEqual(first_trajectory_difference(a, b, framerate=60)['frame'], 2)
            self.assertAlmostEqual(first_trajectory_difference(a, b, framerate=60)['time_from_appearance_s'], 2 / 60)
            self.assertTrue(np.isnan(first_trajectory_difference(a, a, framerate=60)['time_from_appearance_s']))

    def test_plot_keeps_native_window_when_asset_difference_is_earlier(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from src.plotting_population_trajectories import (plot_population_separation,
                                                         plot_population_trajectories)
        result = self.build()
        plotted = plot_population_separation(result, 'left', stimulus_events={('A', 'B'): -8})
        np.testing.assert_allclose(plotted['axes'][0, 0].get_xlim(), (0, 5.5))
        plt.close(plotted['figure'])
        for dimensions in [(0, 1), (0, 1, 2)]:
            plotted = plot_population_trajectories(result, 'left', 'f1', dimensions=dimensions)
            self.assertIn('PC 1', plotted['axes'].get_xlabel())
            plt.close(plotted['figure'])

    def test_pooled_plots_preserve_basis_and_use_mean_only_panels(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from src.plotting_population_trajectories import (plot_pooled_population_trajectories,
                                                         plot_population_distance)
        result = self.build(pca_scope='pooled')
        components = result['pooled_fits']['left']['pca'].components_.copy()
        for dimensions in [(0, 1), (0, 1, 2)]:
            plotted = plot_pooled_population_trajectories(result, 'left',
                {'Both': ['A', 'B'], 'One': ['A']}, dimensions=dimensions, time_window_s=(0, 4))
            self.assertEqual(len(plotted['axes']), 2)
            np.testing.assert_allclose(plotted['axes'][0].get_xlim(), plotted['axes'][1].get_xlim())
            plt.close(plotted['figure'])
        plotted = plot_population_distance(result, 'left', focus_window_s=(0, 4), context_window_s=(0, 5.5))
        np.testing.assert_allclose(plotted['axes'][0, 0].get_xlim(), (0, 4))
        plt.close(plotted['figure'])
        np.testing.assert_array_equal(components, result['pooled_fits']['left']['pca'].components_)
        joint = build_joint_population_pca(result)
        plotted = plot_pooled_population_trajectories(result, 'left', {'': [('left', 'A'), ('left', 'B')]},
            pca_fit=joint, focus_window_s=None, legend_labels={('left', 'A'): 'A', ('left', 'B'): 'B'},
            line_styles={('left', 'B'): '--'}, time_window_s=(0, 5.5), title='Joint test basis')
        self.assertEqual(len(plotted['axes']), 1)
        self.assertEqual(len(plotted['axes'][0].lines), 2)
        self.assertEqual(plotted['axes'][0].lines[1].get_linestyle(), '--')
        self.assertIn('Joint test basis', plotted['figure']._suptitle.get_text())
        plt.close(plotted['figure'])


if __name__ == '__main__':
    unittest.main()

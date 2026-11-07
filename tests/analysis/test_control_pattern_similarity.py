"""Known same-time patterns, repeatability and missing-correlation checks."""
import unittest

import numpy as np
import pandas as pd

from src.population_trajectories import build_population_trajectory_analysis, build_control_pattern_similarity
from src.stimulus_similarity import compute_population_pattern_correlation


def delayed_population():
    p = np.array([1., -1., 1., -1.])
    q = np.array([1., 1., -1., -1.])
    r = np.array([1., -1., -1., 1.])
    control = np.column_stack([p, q, r, -p])
    stimulus = control[:, [0, 0, 1, 2]]
    fish = {fid: dict(kept_neuron_indices=np.array([2, 0, 3, 1]),
            stimuli_id_map={'Control': 1, 'Delayed': 2},
            trial_aligned_traces_z_core={1: np.repeat(control[:, :, None], 4, axis=2),
                                        2: np.repeat(stimulus[:, :, None], 4, axis=2)})
            for fid in ['fish1', 'fish2']}
    return fish


def prepare(fish):
    return build_population_trajectory_analysis(fish, list(fish),
        {'left': {'stimuli': ['Control', 'Delayed'], 'pairs': [('Control', 'Delayed')]}},
        fps_2p=2, t_pre_s=0, program_onset_s=0, window_s=(0, 2), fit_window_s=(0, 2),
        pca_scope='pooled')


class ControlPatternSimilarityTests(unittest.TestCase):
    def test_pearson_sign_gain_offset_and_constant_vectors(self):
        a = np.array([[-1., 2], [0., -1], [1., 0]])
        np.testing.assert_allclose(compute_population_pattern_correlation(a, 3 * a + 7), 1)
        np.testing.assert_allclose(compute_population_pattern_correlation(a, -a), -1)
        self.assertEqual(compute_population_pattern_correlation(a, a).shape, (2,))
        self.assertTrue(np.isnan(compute_population_pattern_correlation(a, np.ones_like(a))).all())
        with self.assertRaises(ValueError):
            compute_population_pattern_correlation(a, np.array([[np.nan], [1], [2]]))

    def test_known_same_time_curves_match_existing_metric(self):
        analysis = prepare(delayed_population())
        for estimator in ['trial_mean', 'split_half']:
            result = build_control_pattern_similarity(analysis, {'left': 'Control'},
                time_window_s=(0, 1.5), estimator=estimator)
            np.testing.assert_allclose(result['curve_summary'].mean_r, [1, 0, 0, 0], atol=1e-14)
            self.assertTrue(result['curve_summary'].n_fish.eq(2).all())
            if estimator == 'split_half':
                np.testing.assert_allclose(result['repeatability_summary'].mean_r, 1)
            else:
                self.assertTrue(result['repeatability_summary'].empty)
                expected = analysis['metrics'].groupby('time_s').pattern_correlation.mean()
                np.testing.assert_allclose(result['curve_summary'].mean_r, expected, atol=1e-14)

    def test_split_half_matches_explicit_independent_two_trial_means(self):
        fish = delayed_population()
        rng = np.random.default_rng(21)
        for f in fish.values():
            for arr in f['trial_aligned_traces_z_core'].values():
                arr += rng.normal(0, 0.3, arr.shape)
        analysis = prepare(fish)
        result = build_control_pattern_similarity(analysis, {'left': 'Control'}, time_window_s=(0, 1.5))
        arrays = analysis['responses']['left']['fish1']
        explicit = []
        for first, second in [([0, 1], [2, 3]), ([0, 2], [1, 3]), ([0, 3], [1, 2])]:
            for a, b in [(first, second), (second, first)]:
                stimulus = arrays['Delayed'][:, :, a].mean(axis=2)
                control = arrays['Control'][:, :, b].mean(axis=2)
                explicit.append([np.corrcoef(stimulus[:, t], control[:, t])[0, 1]
                                 for t in range(stimulus.shape[1])])
        actual = result['fish_time_table'].query('fish_id == "fish1"').correlation
        np.testing.assert_allclose(actual, np.mean(explicit, axis=0), atol=1e-14)
        pd.testing.assert_frame_equal(result['curve_summary'], build_control_pattern_similarity(
            analysis, {'left': 'Control'}, time_window_s=(0, 1.5))['curve_summary'])

    def test_constant_patterns_keep_nan_and_valid_fish_counts(self):
        fish = delayed_population()
        for f in fish.values():
            f['trial_aligned_traces_z_core'][1][:, 0, :] = 2
        fish['fish1']['trial_aligned_traces_z_core'][2][:, 3, :] = 3
        result = build_control_pattern_similarity(prepare(fish), {'left': 'Control'}, time_window_s=(0, 1.5))
        self.assertEqual(result['curve_summary'].iloc[-1].n_fish, 1)
        self.assertEqual(result['curve_summary'].iloc[0].n_fish, 0)
        self.assertTrue(np.isnan(result['curve_summary'].iloc[0].mean_r))
        with self.assertRaises(ValueError):
            build_control_pattern_similarity(prepare(fish), {'left': 'Control'}, time_window_s=(-1, 1.5))

    def test_plot_consumer_preserves_same_time_curve(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from src.plotting_population_trajectories import plot_control_similarity_curves
        result = build_control_pattern_similarity(prepare(delayed_population()), {'left': 'Control'},
                                                  time_window_s=(0, 1.5))
        plotted = plot_control_similarity_curves(result, {'left': ['Control', 'Delayed']})
        np.testing.assert_allclose(plotted['axes'][0, 0].lines[0].get_ydata(),
                                   result['curve_summary'].mean_r)
        plotted['figure'].canvas.draw()
        plt.close(plotted['figure'])


if __name__ == '__main__':
    unittest.main()

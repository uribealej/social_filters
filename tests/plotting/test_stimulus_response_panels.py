"""Known values, shared neuron identity, phases, and both new panel layouts."""
import copy
import unittest
from unittest import mock

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.several_fish_diagnostics import build_stimulus_panel_display
from src.several_fish_figures import build_all_fish_stimulus_figures, plot_all_fish_stimulus_figures
from src.stimuli_timeline import stimulus_phase_intervals


def panel_cohort():
    data = {}
    mapping = {'left': 7, 'right': 14}
    for fid, count, kept in [('fish_b', 4, [3, 0]), ('fish_a', 6, [4, 1, 0])]:
        data[fid] = {
            'kept_neuron_indices': np.array(kept), 'stimuli_id_map': mapping,
            'trial_aligned_traces_z_core': {
                sid: np.arange(count)[:, None, None] + np.arange(60)[None, :, None] / 60
                     + np.arange(4)[None, None, :] / 10 + sid
                for sid in mapping.values()},
            'stimuli_durations': {label: dict(static_before_sec=2., motion_sec=3.,
                static_after_sec=1., total_sec=6.) for label in mapping},
        }
    return data


class StimulusResponsePanelTests(unittest.TestCase):
    def setUp(self):
        self.data = panel_cohort()
        self.fish_ids = list(self.data)
        self.timing = dict(fps_2p=2., t_pre_s=5.)
        self.settings = dict(combine_mode='concat', trace_type='raster',
                             raster_sort_mode='unsorted', left_control=7, right_control=14)

    def tearDown(self):
        plt.close('all')

    def prepare(self, **kwargs):
        return build_stimulus_panel_display(self.data, self.fish_ids, [14, 7],
            self.data['fish_b'], self.timing, self.settings, **kwargs)

    def test_trial_mean_values_fish_weights_and_shared_order(self):
        settings = self.settings.copy()
        result = self.prepare(neuron_order=np.array([4, 0, 3, 2, 1]))
        np.testing.assert_array_equal(result['neuron_order'], [4, 0, 3, 2, 1])
        self.assertEqual(result['row_metadata'].source_neuron_index.tolist(), [3, 0, 4, 1, 0])
        self.assertEqual(result['displayed_row_metadata'].source_neuron_index.tolist(), [0, 3, 1, 4, 0])
        self.assertEqual(result['stimulus_ids'], [14, 7])
        np.testing.assert_allclose(result['time_s'][7], np.arange(60) / 2 - 5)
        for sid in [14, 7]:
            expected = np.concatenate([self.data[fid]['trial_aligned_traces_z_core'][sid][
                self.data[fid]['kept_neuron_indices']].mean(axis=2) for fid in self.fish_ids])
            np.testing.assert_allclose(result['matrices'][sid], expected)
            expected_fish = np.stack([expected[:2].mean(axis=0), expected[2:].mean(axis=0)])
            np.testing.assert_allclose(result['fish_means'][sid], expected_fish)
            self.assertNotAlmostEqual(expected_fish.mean(axis=0)[0], expected.mean(axis=0)[0])
        self.assertEqual(self.settings, settings)

    def test_all_layouts_plot_same_values_and_common_limits(self):
        result = build_all_fish_stimulus_figures(self.data, self.fish_ids, [14, 7],
            self.data['fish_b'], self.timing, self.settings, {'left': [7], 'right': [14]},
            neuron_order=np.array([4, 0, 3, 2, 1]), z_limits=(-1, 25))
        display = result['display']
        self.assertEqual(len(result['figures']), 6)
        limits = []
        for side, sid in [('left', 7), ('right', 14)]:
            for layout in ['raster', 'mean', 'combined']:
                axes = result['axes'][f'{side}_{layout}']
                self.assertEqual(axes.shape, (2 if layout == 'combined' else 1, 1))
                if layout != 'mean':
                    image = axes[0, 0].images[0]
                    np.testing.assert_allclose(image.get_array(), display['matrices'][sid][display['neuron_order']])
                    self.assertEqual(image.get_clim(), (-1, 25))
                if layout != 'raster':
                    ax = axes[-1, 0]
                    curves = {line.get_label(): line for line in ax.lines}
                    np.testing.assert_allclose(curves['Equal-fish mean'].get_ydata(),
                                               display['fish_means'][sid].mean(axis=0))
                    for index, fid in enumerate(self.fish_ids):
                        np.testing.assert_allclose(curves[fid].get_ydata(), display['fish_means'][sid][index])
                    limits.append(ax.get_ylim())
                for ax in axes.flat:
                    phase_starts = [float(line.get_xdata()[0]) for line in ax.lines
                                    if line.get_linestyle() in (':', '--')]
                    self.assertEqual(phase_starts, [0, 6])
        self.assertTrue(all(limit == limits[0] for limit in limits))

    def test_phase_and_row_contract_failures_are_explicit(self):
        self.assertEqual(stimulus_phase_intervals(dict(static_before_sec=2, motion_sec=3,
            static_after_sec=1, total_sec=6)), [('static', 0., 2.), ('active', 2., 5.), ('static', 5., 6.)])
        for duration in [dict(static_before_sec=2, motion_sec=3, total_sec=4),
                         dict(static_before_sec=-1, motion_sec=3, total_sec=2)]:
            with self.assertRaises(ValueError):
                stimulus_phase_intervals(duration)
        with self.assertRaisesRegex(ValueError, 'every pooled neuron'):
            self.prepare(neuron_order=np.array([0, 0, 1, 2, 3]))
        bad = copy.deepcopy(self.data)
        bad['fish_a']['trial_aligned_traces_z_core'][7] = np.zeros((6, 58, 4))
        with self.assertRaisesRegex(ValueError, 'shape'):
            build_stimulus_panel_display(bad, self.fish_ids, [7], bad['fish_b'],
                                        self.timing, self.settings)

    def test_appearance_redraw_never_prepares_or_sorts_and_uses_compact_labels(self):
        prepared = self.prepare()
        original = prepared['matrices'][7].copy()
        style = {key: 6 for key in ('font.size', 'axes.labelsize', 'axes.titlesize',
                 'figure.titlesize', 'xtick.labelsize', 'ytick.labelsize')}
        with mock.patch('src.several_fish_figures.build_stimulus_panel_display',
                        side_effect=AssertionError('Redraw must not prepare/sort')), \
                mock.patch('src.neuron_ordering.compute_single_sort_order',
                           side_effect=AssertionError('Redraw must not sort')), plt.rc_context(style):
            for cmap in ('Greys', 'magma'):
                result = plot_all_fish_stimulus_figures(prepared, {'left': [7]},
                    raster_figsize=(150 / 25.4, 45 / 25.4),
                    mean_figsize=(150 / 25.4, 42.5 / 25.4),
                    combined_figsize=(150 / 25.4, 65 / 25.4), cmap=cmap)
                self.assertIs(result['display'], prepared)
                for key, figure in result['figures'].items():
                    self.assertEqual(len(figure.legends), 0)
                    self.assertEqual(figure._supxlabel.get_text(), 'Time (s)')
                    self.assertEqual(figure._supxlabel.get_fontsize(), 6)
                    for ax in result['axes'][key].flat:
                        self.assertEqual(ax.get_xlabel(), '')
                        self.assertNotIn('Mean of', ax.get_title())
                        if ax.get_title():
                            self.assertEqual(ax.get_title(), 'left')
                plt.close('all')
        np.testing.assert_array_equal(prepared['matrices'][7], original)


if __name__ == '__main__':
    unittest.main()

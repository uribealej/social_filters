"""Known values, shared neuron identity, phases, and both new panel layouts."""
import copy
import unittest
from unittest import mock

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.several_fish_diagnostics import build_stimulus_panel_display
from src import several_fish_diagnostics as diagnostics
from src.several_fish_figures import (
    build_all_fish_stimulus_figures, plot_all_fish_stimulus_figures,
    plot_prepared_left_right_active_overlap_diagnostics,
)
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
                    self.assertEqual(len(figure.legends), 1)
                    self.assertEqual([text.get_text() for text in figure.legends[0].get_texts()],
                                     ['Static', 'Motion / flicker'])
                    self.assertEqual(figure._supxlabel.get_text(), 'Time (s)')
                    self.assertEqual(figure._supxlabel.get_fontsize(), 6)
                    for ax in result['axes'][key].flat:
                        self.assertEqual(ax.get_xlabel(), '')
                        self.assertNotIn('Mean of', ax.get_title())
                        if ax.get_title():
                            self.assertEqual(ax.get_title(), 'left')
                plt.close('all')
        np.testing.assert_array_equal(prepared['matrices'][7], original)

    def prepare_cell06_orders(self):
        """Known, different side patterns on permuted retained source rows."""
        for fid, fish in self.data.items():
            left = np.array([0, 1]) if fid == 'fish_b' else np.array([1, 0, 1])
            fish['trial_aligned_traces_raster'] = {
                7: np.broadcast_to(left[:, None, None], (len(left), 60, 4)).copy(),
                14: np.broadcast_to((1-left)[:, None, None], (len(left), 60, 4)).copy(),
            }
        with mock.patch.object(diagnostics, 'build_decision_then_mean_sort_order',
                wraps=diagnostics.build_decision_then_mean_sort_order) as sorter:
            source = diagnostics.build_left_right_active_overlap_diagnostics(
                self.data, self.fish_ids, self.data['fish_b'], self.timing,
                {'left': [7], 'right': [14]}, motion_onset_s=2, tau_s=0.5)
        self.assertEqual(sorter.call_count, 2)
        np.testing.assert_array_equal(source['left']['diagnostic_neuron_order'], [1, 2, 4, 0, 3])
        np.testing.assert_array_equal(source['right']['diagnostic_neuron_order'], [0, 3, 1, 2, 4])
        return source

    def test_cell06_orders_match_each_side_without_reclassifying_or_sorting(self):
        source = self.prepare_cell06_orders()
        prepared = self.prepare(neuron_order=np.arange(5))
        original = prepared['matrices'][7].copy()
        with mock.patch.object(diagnostics, 'build_decision_then_mean_sort_order',
                side_effect=AssertionError('Redraw must reuse the cached order')), \
             mock.patch.object(diagnostics, 'build_active_neuron_matrices_all_fish',
                side_effect=AssertionError('Redraw must reuse the activity decisions')), \
             mock.patch('src.neuron_ordering.compute_single_sort_order',
                side_effect=AssertionError('Redraw must not compute another order')):
            for cmap in ('Greys', 'magma'):
                result = plot_all_fish_stimulus_figures(prepared,
                    {'left': [7], 'right': [14]}, side_order_sources=source, cmap=cmap)
                for side, sid in [('left', 7), ('right', 14)]:
                    order = source[side]['diagnostic_neuron_order']
                    display = result['side_displays'][side]
                    self.assertIs(display['neuron_order'], order)
                    self.assertEqual(display['displayed_row_metadata'].global_neuron_id.tolist(), order.tolist())
                    for layout in ('raster', 'combined'):
                        image = result['axes'][f'{side}_{layout}'][0, 0].images[0]
                        np.testing.assert_allclose(image.get_array(), prepared['matrices'][sid][order])
                    curves = {line.get_label(): line for line in result['axes'][f'{side}_mean'][0, 0].lines}
                    np.testing.assert_allclose(curves['Equal-fish mean'].get_ydata(),
                        prepared['fish_means'][sid].mean(axis=0))
                plotted = plot_prepared_left_right_active_overlap_diagnostics(
                    source, self.data['fish_b'], self.timing, show_overlap=False,
                    show_significant_raster=True)
                for side in ('left', 'right'):
                    self.assertIs(plotted[side]['diagnostic_neuron_order'], source[side]['diagnostic_neuron_order'])
                    self.assertNotIn('diagnostic_figure', source[side])
                plt.close('all')
        np.testing.assert_array_equal(prepared['matrices'][7], original)
        np.testing.assert_array_equal(prepared['neuron_order'], np.arange(5))

    def test_cell06_reuse_rejects_changed_source_identities_and_bad_orders(self):
        source = self.prepare_cell06_orders()
        prepared = self.prepare(neuron_order=np.arange(5))
        for bad in ('source', 'rows', 'permutation'):
            changed = copy.deepcopy(source)
            if bad == 'source':
                changed['left']['source_row_metadata'].loc[0, 'source_neuron_index'] = 100
            elif bad == 'rows':
                rows = changed['left']['active_trace_diagnostic']['row_metadata']
                changed['left']['active_trace_diagnostic']['row_metadata'] = rows.iloc[::-1]
            else:
                changed['left']['diagnostic_neuron_order'] = np.zeros(5, dtype=int)
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, 'identities|every pooled neuron'):
                plot_all_fish_stimulus_figures(prepared, {'left': [7]}, side_order_sources=changed)

    def test_significant_trial_fractions_keep_retained_rows_and_pooled_weights(self):
        # Rasters already contain retained rows, whose source IDs exceed their
        # length. Reapplying source indices here would fail or misalign neurons.
        for fish in self.data.values():
            count = len(fish['kept_neuron_indices'])
            fish['trial_aligned_traces_raster'] = {
                sid: np.broadcast_to(np.arange(4)[None, None, :] <
                    np.arange(1, count + 1)[:, None, None], (count, 60, 4)).copy()
                for sid in [7, 14]}
        prepared = self.prepare(trace_type='raster', neuron_order=np.array([4, 0, 3, 2, 1]))
        expected = np.array([.25, .5, .25, .5, .75])
        np.testing.assert_allclose(prepared['matrices'][7], np.broadcast_to(expected[:, None], (5, 60)))
        np.testing.assert_allclose(prepared['fish_means'][7][:, 0], [.375, .5])
        np.testing.assert_allclose(prepared['pooled_means'][7], .45)
        self.assertEqual(prepared['trace_type'], 'raster')
        result = plot_all_fish_stimulus_figures(prepared, {'left': [7]},
            layouts=('combined',), mean_mode='pooled_neurons', show_fish_traces=False,
            z_limits=(0, 1))
        self.assertEqual(list(result['figures']), ['left_combined'])
        axes = result['axes']['left_combined']
        image = axes[0, 0].images[0]
        np.testing.assert_allclose(image.get_array()[:, 0], expected[[4, 0, 3, 2, 1]])
        self.assertEqual(image.get_clim(), (0, 1))
        self.assertIn('trial fraction', image.colorbar.ax.get_ylabel())
        curves = {line.get_label(): line for line in axes[1, 0].lines}
        np.testing.assert_allclose(curves['Pooled-neuron mean'].get_ydata(), .45)
        self.assertNotIn('fish_a', curves)
        self.assertIn('fraction', axes[1, 0].get_ylabel())
        raster = plot_all_fish_stimulus_figures(prepared, {'left': [7]}, layouts=('raster',))
        self.assertEqual(raster['axes']['left_raster'].shape, (1, 1))

    def test_significant_cell06_orders_and_shape_validation(self):
        source = self.prepare_cell06_orders()
        prepared = self.prepare(trace_type='raster')
        result = plot_all_fish_stimulus_figures(prepared,
            {'left': [7], 'right': [14]}, side_order_sources=source, layouts=('raster',))
        for side, sid in [('left', 7), ('right', 14)]:
            np.testing.assert_array_equal(result['axes'][f'{side}_raster'][0, 0].images[0].get_array(),
                prepared['matrices'][sid][source[side]['diagnostic_neuron_order']])
        self.data['fish_a']['trial_aligned_traces_raster'][7] = np.zeros((6, 60, 4))
        with self.assertRaisesRegex(ValueError, 'retained-neuron order'):
            self.prepare(trace_type='raster')
        with self.assertRaisesRegex(ValueError, 'trace_type'):
            self.prepare(trace_type='unknown')


if __name__ == '__main__':
    unittest.main()

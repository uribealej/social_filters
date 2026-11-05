"""Exercise the actual Exp 7 panel cell with deterministic aligned traces."""
import json
from pathlib import Path
import unittest
from unittest import mock

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src import plotting_common, several_fish_figures
from src.multifish_matrices import build_matrix_all_fish


NOTEBOOK = (Path(__file__).resolve().parents[2]
            / 'notebooks/calcium/exp_07_trans_stat/01_all_fish_raster.ipynb')


class Exp07RasterControlsTests(unittest.TestCase):
    def test_setup_refreshes_sorter_in_an_existing_kernel(self):
        """Reproduce cached function imports, then rerun the actual setup cell."""
        from src import neuron_ordering, plotting_all_fish

        notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        source = next(''.join(c['source']) for c in notebook['cells']
                      if c['cell_type'] == 'code' and '# Cell 01 - Setup' in ''.join(c['source']))
        stale = mock.Mock(side_effect=ValueError("Unknown sort_mode 'rastermap'"))
        with mock.patch.object(neuron_ordering, 'compute_single_sort_order', stale), \
                mock.patch.object(plotting_all_fish, 'compute_single_sort_order', stale):
            exec(compile(source, str(NOTEBOOK), 'exec'), {})
            self.assertIs(plotting_all_fish.compute_single_sort_order,
                          neuron_ordering.compute_single_sort_order)
            with mock.patch.object(neuron_ordering, '_rastermap_sort_order',
                                   return_value=np.array([2, 1, 0])) as sorter:
                order = plotting_all_fish.compute_single_sort_order(np.eye(3), 'rastermap')
                np.testing.assert_array_equal(order, [2, 1, 0])
                sorter.assert_called_once()
            stale.assert_not_called()

    def test_side_selection_preserves_values_neurons_mean_and_global_settings(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        source = next(''.join(c['source']) for c in notebook['cells']
                      if c['cell_type'] == 'code' and 'RASTER_SIDE =' in ''.join(c['source']))
        config = next(''.join(c['source']) for c in notebook['cells']
                      if c['cell_type'] == 'code' and 'EXPECTED_STIMULUS_MAP =' in ''.join(c['source']))
        scope = {}
        exec(config, scope)
        settings = scope['USER_SETTINGS']
        mapping = scope['EXPECTED_STIMULUS_MAP']
        rng = np.random.default_rng(71)
        data = {fid: {
            'trial_aligned_traces_z_core': {sid: rng.normal(size=(5, 60, 4)) for sid in mapping.values()},
            'kept_neuron_indices': np.array([3, 0, 4, 1]),
            'stimuli_id_map': mapping,
            'stimuli_durations': {label: dict(static_before_sec=8., motion_sec=4., total_sec=20.)
                                  for label in mapping},
        } for fid in ('fish_b', 'fish_a')}
        colors, styles = plotting_common.build_stimulus_style_maps(stimuli_names=list(mapping))
        namespace = dict(plt=plt, several_figures=several_fish_figures, settings=settings,
                         stim_order=settings['stim_order'], all_fish_data=data,
                         fish_ids=list(data), reference_fish=data['fish_b'], timing=settings['timing'],
                         stimuli_colors=colors, stimuli_linestyles=styles,
                         PUBLICATION_STYLE={}, MM=1/25.4)
        for side, ids in [('both', settings['stim_order']), ('left', [7, 8, 10, 12, 3, 5, 1]),
                          ('right', [14, 9, 11, 13, 4, 6, 2])]:
            with self.subTest(side=side), mock.patch.object(plt, 'show'):
                exec(source.replace("RASTER_SIDE = 'both'", f'RASTER_SIDE = {side!r}'), namespace)
                result = namespace['raster_result']
                expected = build_matrix_all_fish(data, ids, list(data), 'mean', 'zscore')
                np.testing.assert_array_equal(result['flat_matrix_all_fish'], expected)
                np.testing.assert_array_equal(np.sort(result['neuron_order']), np.arange(8))
                np.testing.assert_array_equal(result['image'].get_array(), expected[result['neuron_order']])
                np.testing.assert_allclose(result['figure'].axes[1].lines[0].get_ydata(), expected.mean(axis=0))
                self.assertEqual(namespace['raster_stim_order'], ids)
                self.assertEqual(settings['raster_sort_mode'], 'left_right_index')
                self.assertEqual(settings['stim_order'], [14, 9, 11, 13, 4, 6, 2, 7, 8, 10, 12, 3, 5, 1])
                plt.close('all')
        with self.assertRaisesRegex(ValueError, 'RASTER_SIDE'):
            exec(source.replace("RASTER_SIDE = 'both'", "RASTER_SIDE = 'invalid'"), namespace)


if __name__ == '__main__':
    unittest.main()

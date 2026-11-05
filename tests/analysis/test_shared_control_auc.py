"""Known-output shared-neuron intersections and motion-plus-decay AUC."""
import unittest

import numpy as np
import pandas as pd

from src.active_neuron_analysis import build_pooled_active_trace_diagnostic
from src.response_metrics import compute_motion_decay_auc
from src.several_fish_diagnostics import build_shared_control_auc_analysis


def shared_cohort():
    ids = [1, 2, 3, 4, 11, 12, 13, 14]
    labels = ['CL', 'AL', 'BL', 'EL', 'CR', 'AR', 'BR', 'ER']
    mapping = dict(zip(labels, ids))
    data, active = {}, {}
    for fid, count, kept, difference in [('fish_b', 4, [3, 0], 1.), ('fish_a', 5, [4, 1, 0], 3.)]:
        pattern = np.array([[1, 1, 0, 0], [1, 0, 1, 0]]) if fid == 'fish_b' else np.array([
            [1, 1, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0]])
        decisions = np.concatenate([pattern, pattern], axis=1)
        active[fid] = pd.DataFrame(decisions, columns=ids)
        z, raster = {}, {}
        for index, sid in enumerate(ids):
            scale = 1 if index < 4 else 2
            level = (np.arange(count) + 5.) * scale
            if index % 4 == 1:
                level -= difference * scale
            elif index % 4 == 2:
                level += 2 * scale
            z[sid] = np.broadcast_to(level[:, None, None], (count, 12, 4)).copy()
            z[sid][:, :3, :] = 1000  # Exclude baseline/static from the measurement.
            raster[sid] = np.broadcast_to(decisions[:, index, None, None], (len(kept), 12, 4)).copy()
        data[fid] = dict(kept_neuron_indices=np.array(kept), stimuli_id_map=mapping,
            trial_aligned_traces_z_core=z, trial_aligned_traces_raster=raster,
            stimuli_durations={label: dict(static_before_sec=2., motion_sec=2., total_sec=4.)
                                for label in labels})
    fish_ids = list(data)
    overlap = {side: {'active_trace_diagnostic': build_pooled_active_trace_diagnostic(
        data, active, selected, fish_ids=fish_ids)}
        for side, selected in [('left', ids[:4]), ('right', ids[4:])]}
    return data, fish_ids, overlap


class SharedControlAucTests(unittest.TestCase):
    def prepare(self, data=None, overlap=None):
        original, fish_ids, results = shared_cohort()
        return build_shared_control_auc_analysis(data or original, fish_ids, overlap or results,
            {'left': 1, 'right': 11}, {'fps_2p': 1., 't_pre_s': 1.}, decay_s=5.)

    def test_window_excludes_static_and_includes_five_second_decay(self):
        trace = np.full((2, 12, 4), 2.)
        trace[:, :3, :] = 1000
        duration = dict(static_before_sec=2., motion_sec=2., total_sec=4.)
        result = compute_motion_decay_auc(trace, duration, 1., 1., decay_s=5.)
        np.testing.assert_allclose(result['raw_auc'], [12, 12])
        np.testing.assert_allclose(result['normalized_auc'], [2, 2])
        self.assertEqual(result['window']['start_s'], 2)
        self.assertEqual(result['window']['requested_end_s'], 9)
        self.assertEqual(result['window']['last_sample_s'], 8)
        self.assertFalse(result['window']['window_clipped'])
        clipped = compute_motion_decay_auc(trace, duration, 1., 1., decay_s=20.)
        self.assertTrue(clipped['window']['window_clipped'])
        self.assertEqual(clipped['window']['observed_end_s'], 11.)
        trace[0, 3, :] = np.nan
        missing = compute_motion_decay_auc(trace, duration, 1., 1.)
        self.assertTrue(np.isnan(missing['raw_auc'][0]))
        self.assertEqual(missing['n_valid_trials'][0], 0)
        with self.assertRaises(ValueError):
            compute_motion_decay_auc(trace, duration, 1., 1., decay_s=-1)

    def test_shared_pairs_source_identity_sign_and_equal_fish_weight(self):
        result = self.prepare()
        rows = result['tables']['shared_neuron_auc']
        chosen = rows.loc[(rows.side == 'left') & (rows.condition == 'AL')]
        self.assertEqual(chosen.source_neuron_index.tolist(), [3, 4, 1])
        np.testing.assert_allclose(chosen.delta_normalized_auc, [1., 3., 3.])
        np.testing.assert_allclose(chosen.delta_raw_auc, [6., 18., 18.])
        summary = result['tables']['shared_auc_summary'].set_index('condition')
        self.assertEqual(summary.loc['AL', 'mean_delta_normalized_auc'], 2.)
        self.assertEqual(summary.loc['AR', 'mean_delta_normalized_auc'], 4.)
        self.assertEqual(summary.loc['BL', 'mean_delta_normalized_auc'], -2.)
        self.assertNotEqual(summary.loc['AL', 'mean_delta_normalized_auc'], chosen.delta_normalized_auc.mean())
        self.assertEqual(summary.loc['AL', 'n_shared_total'], 3)
        self.assertEqual(summary.loc['AL', 'n_fish'], 2)
        self.assertEqual(summary.loc['EL', 'n_fish'], 0)
        self.assertTrue(np.isnan(summary.loc['EL', 'mean_delta_normalized_auc']))
        self.assertEqual(result['tables']['shared_auc_windows'].requested_end_s.unique().tolist(), [9.])

    def test_missing_paired_values_retain_shared_count_but_not_false_zero(self):
        data, _, overlap = shared_cohort()
        data['fish_a']['trial_aligned_traces_z_core'][2][4, 3, :] = np.nan
        result = self.prepare(data, overlap)
        fish = result['tables']['shared_auc_per_fish'].set_index(['fish_id', 'condition'])
        self.assertEqual(fish.loc[('fish_a', 'AL'), 'n_shared'], 2)
        self.assertEqual(fish.loc[('fish_a', 'AL'), 'n_finite_pairs'], 1)
        self.assertEqual(fish.loc[('fish_a', 'AL'), 'mean_delta_normalized_auc'], 3)

    def test_rejects_misaligned_decisions_and_shapes(self):
        data, _, overlap = shared_cohort()
        overlap['left']['active_trace_diagnostic']['row_metadata'] = (
            overlap['left']['active_trace_diagnostic']['row_metadata'].iloc[::-1])
        with self.assertRaisesRegex(ValueError, 'decision rows'):
            self.prepare(data, overlap)
        data, _, overlap = shared_cohort()
        overlap['left']['active_trace_diagnostic']['decision_matrix'][0, 0] = 2
        with self.assertRaisesRegex(ValueError, 'binary decision'):
            self.prepare(data, overlap)
        data, _, overlap = shared_cohort()
        data['fish_a']['trial_aligned_traces_z_core'][2] = np.zeros((5, 10, 4))
        with self.assertRaisesRegex(ValueError, 'axes differ'):
            self.prepare(data, overlap)


if __name__ == '__main__':
    unittest.main()

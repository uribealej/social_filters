"""Joint clustering, paired identities and unchanged pooled response values."""

import copy
import unittest

import numpy as np
import pandas as pd

from src.neuron_ordering import build_response_profile_sort_order
from src.several_fish_diagnostics import (
    build_pooled_response_profiles, build_response_breadth_profiles, build_response_similarity_comparison,
)
from tests.analysis.test_recruitment_containment import prepared_analysis


class PooledResponseProfileTests(unittest.TestCase):
    def test_pearson_groups_positive_affine_profiles_and_preserves_undefined_rows(self):
        values = pd.DataFrame([[0, 1, 2], [2, 1, 0], [10, 12, 14],
                               [3, 3, 3], [np.nan, 1, 2]], columns=['A', 'B', 'C'])
        before = values.copy(deep=True)
        order = build_response_profile_sort_order(values, list(values.columns), sort_mode='pearson')
        self.assertEqual(set(order), set(range(5)))
        self.assertEqual(abs(np.flatnonzero(order == 0)[0] - np.flatnonzero(order == 2)[0]), 1)
        np.testing.assert_array_equal(order[-2:], [3, 4])
        pd.testing.assert_frame_equal(values, before)
        np.testing.assert_array_equal(build_response_profile_sort_order(values.iloc[:0], list(values.columns), 'pearson'), [])
        np.testing.assert_array_equal(build_response_profile_sort_order(values.iloc[[3, 0, 4]], list(values.columns), 'pearson'), [1, 0, 2])
        np.testing.assert_array_equal(build_response_profile_sort_order(values.iloc[[3, 4]], list(values.columns), 'pearson'), [0, 1])

    def test_both_modes_share_one_identity_order_and_retain_signed_values(self):
        analysis = prepared_analysis()
        profiles = build_response_breadth_profiles(analysis, build_response_similarity_comparison(analysis))
        before = copy.deepcopy(profiles)
        for mode in ('pearson', 'euclidean', 'preferred'):
            result = build_pooled_response_profiles(profiles, sort_mode=mode)
            ids = result['row_metadata'].global_neuron_id.tolist()
            self.assertEqual(len(ids), 5)
            self.assertEqual(len(set(ids)), 5)
            self.assertEqual(result['profile_color_limit'], profiles['profile_color_limit'])
            for side in ('left', 'right'):
                expected = pd.concat([profiles['profiles'][fid][side]['all_kept']['responses']
                                      for fid in profiles['fish_ids']]).loc[ids]
                pd.testing.assert_frame_equal(result['responses'][side], expected)
            changed_control = copy.deepcopy(profiles)
            for fid in profiles['fish_ids']:
                for side in ('left', 'right'):
                    changed_control['profiles'][fid][side]['all_kept']['responses'].iloc[:, 0] *= 10000
            changed = build_pooled_response_profiles(changed_control, sort_mode=mode)
            self.assertEqual(changed['row_metadata'].global_neuron_id.tolist(), ids)
        for fid in profiles['fish_ids']:
            pd.testing.assert_frame_equal(profiles['profiles'][fid]['left']['all_kept']['responses'],
                                          before['profiles'][fid]['left']['all_kept']['responses'])

    def test_mismatched_side_identities_raise(self):
        analysis = prepared_analysis()
        profiles = build_response_breadth_profiles(analysis, build_response_similarity_comparison(analysis))
        fid = profiles['fish_ids'][0]
        profiles['profiles'][fid]['right']['all_kept']['row_metadata'].loc[0, 'source_neuron_id'] += 100
        with self.assertRaisesRegex(ValueError, 'same neuron identities'):
            build_pooled_response_profiles(profiles)


if __name__ == '__main__':
    unittest.main()

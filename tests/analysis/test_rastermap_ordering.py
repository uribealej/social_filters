"""Rastermap's optional-dependency and pooled-row identity contracts."""
import sys
import types
import unittest
from unittest import mock

import numpy as np

from src.neuron_ordering import compute_single_sort_order


class RastermapOrderingTests(unittest.TestCase):
    def test_maps_valid_rows_back_and_preserves_input(self):
        values = np.array([[0., 1., 2., 0., np.nan],
                           [1., 1., 0., 3., 1.],
                           [2., 1., 1., 2., 2.],
                           [0., 1., 3., 1., 3.]])
        original = values.copy()
        model = mock.Mock()
        model.fit.return_value.isort = np.array([2, 0, 1])
        constructor = mock.Mock(return_value=model)
        package = types.SimpleNamespace(Rastermap=constructor)
        with mock.patch.dict(sys.modules, rastermap=package):
            order = compute_single_sort_order(values, 'rastermap', random_state=17)
        np.testing.assert_array_equal(order, [3, 0, 2, 1, 4])
        np.testing.assert_array_equal(values, original)
        np.testing.assert_array_equal(model.fit.call_args.args[0], values[:, [0, 2, 3]].T)
        self.assertEqual(constructor.call_args.kwargs['random_state'], 17)
        self.assertEqual(constructor.call_args.kwargs['n_PCs'], 2)
        self.assertFalse(model.fit.call_args.kwargs['compute_X_embedding'])

    def test_tiny_or_invalid_populations_keep_every_row(self):
        with mock.patch.dict(sys.modules, rastermap=mock.Mock()) as modules:
            for data, expected in [
                (np.zeros((4, 3)), [0, 1, 2]),
                (np.array([[0., 1., 1.], [0., 2., np.inf]]), [1, 0, 2]),
                (np.array([[0., 1.], [1., 0.]]), [0, 1]),
                (np.empty((4, 0)), []),
            ]:
                np.testing.assert_array_equal(
                    compute_single_sort_order(data, 'rastermap'), expected)
            modules['rastermap'].Rastermap.assert_not_called()

    def test_missing_package_has_actionable_error_and_other_modes_work(self):
        with mock.patch.dict(sys.modules, rastermap=None):
            with self.assertRaisesRegex(ImportError, '%pip install'):
                compute_single_sort_order(np.eye(4), 'rastermap')
            self.assertIsNone(compute_single_sort_order(np.eye(4), 'unsorted'))
            self.assertEqual(len(compute_single_sort_order(np.eye(4), 'corravg')), 4)

    def test_rejects_incomplete_or_duplicate_library_order(self):
        for bad_order in ([0, 0, 2, 3], [0, 1], [0., 1., 2., 3.]):
            package = mock.Mock()
            package.Rastermap.return_value.fit.return_value.isort = np.array(bad_order)
            with mock.patch.dict(sys.modules, rastermap=package):
                with self.assertRaisesRegex(ValueError, 'exactly once'):
                    compute_single_sort_order(np.eye(4), 'rastermap')


if __name__ == '__main__':
    unittest.main()

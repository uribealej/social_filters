"""Display-only diagnostics preserve filtered-neuron and time identities."""
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import plotting_single_fish as plotting


class SignificantTraceReviewTests(unittest.TestCase):
    def setUp(self):
        self.ids = np.array([11, 42, 99])
        self.raw = np.arange(80)[:, None] * np.array([[1, 3, 2]]) / 1000
        self.centered = self.raw - 0.01
        self.raster = (self.raw > 0.1).astype(int)
        self.addCleanup(plt.close, "all")

    def test_shared_raster_sort_and_display_only_zoom(self):
        fig, axes, order = plotting.plot_significant_trace_comparison(
            self.raw, self.centered, self.raster, 2, self.ids,
            sort_by_peak=True, time_window_s=(10, 20), figsize=(8, 6))
        np.testing.assert_array_equal(order, [42, 99, 11])
        np.testing.assert_array_equal(axes[0].images[0].get_array(), self.raw[:, [1, 2, 0]].T)
        np.testing.assert_array_equal(axes[2].images[0].get_array(), self.raster[:, [1, 2, 0]].T)
        self.assertEqual(axes[0].get_xlim(), (10, 20))
        self.assertEqual(axes[2].images[0].get_clim(), (0, 1))
        self.assertEqual(self.raw.shape, (80, 3))

    def test_trace_uses_original_id_and_all_event_frames(self):
        fig, axes, selected = plotting.plot_significant_trace_examples(
            self.raw, self.centered, self.raster, 2, self.ids,
            neuron_ids=[42], time_window_s=(10, 20), figsize=(8, 3))
        np.testing.assert_array_equal(selected, [42])
        ax = axes[0, 0]
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), self.raw[:, 1])
        np.testing.assert_array_equal(ax.collections[0].get_offsets()[:, 0], np.flatnonzero(self.raster[:, 1]) / 2)
        self.assertIn("42", ax.get_title())
        with self.assertRaisesRegex(ValueError, "original merged-neuron IDs"):
            plotting.plot_significant_trace_examples(self.raw, self.centered, self.raster, 2, self.ids, neuron_ids=[1])

    def test_random_examples_reproducible_and_zero_event_neurons_retained(self):
        args = (self.raw, self.centered, self.raster, 2, self.ids)
        _, _, first = plotting.plot_significant_trace_examples(*args,
            n_examples=2, high_activity_fraction=0.34, seed=5, figsize=(8, 3))
        _, _, second = plotting.plot_significant_trace_examples(*args,
            n_examples=2, high_activity_fraction=0.34, seed=5, figsize=(8, 3))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(set(first), {42, 99})
        _, axes, ids = plotting.plot_significant_trace_comparison(*args, figsize=(8, 6))
        np.testing.assert_array_equal(ids, self.ids)
        self.assertEqual(axes[2].images[0].get_array().shape[0], 3)
        self.assertFalse(self.raster[:, 0].any())

    def test_invalid_shapes_and_time_limits_fail(self):
        with self.assertRaisesRegex(ValueError, "share"):
            plotting.plot_significant_trace_comparison(self.raw, self.centered[:, :2], self.raster, 2, self.ids)
        with self.assertRaisesRegex(ValueError, "time window"):
            plotting.plot_significant_trace_examples(self.raw, self.centered, self.raster, 2, self.ids, time_window_s=(-1, 20))


if __name__ == "__main__":
    unittest.main()

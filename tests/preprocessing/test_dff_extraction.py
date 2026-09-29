"""Known-output checks for the S10 dFoF extraction boundary."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src import dff_extraction as dff


class DffExtractionTests(unittest.TestCase):
    """Protect baseline math, ROI ordering, and the batch notebook gateway."""

    def test_percentile_baseline_rejects_unstable_roi(self):
        traces = np.array([
            [10, 1, 5],
            [12, 1, 5],
            [14, 19, 5],
            [16, 19, 5],
            [18, 19, 5],
            [20, 1, 5],
        ], dtype=float)

        baseline = dff.compute_percentile_baseline(
            traces, fps=1, tau=1, percentile=0, instability_ratio=0.1,
            min_window_s=1, window_tau_multiplier=1,
        )

        np.testing.assert_array_equal(
            baseline[:, 0], [10, 10, 12, 14, 16, 18]
        )
        self.assertTrue(np.isnan(baseline[:, 1]).all())
        np.testing.assert_array_equal(baseline[:, 2], [5] * 6)

    def test_percentile_baseline_smooths_local_minimum(self):
        trace = np.array([[10], [12], [14], [16], [18], [20]], dtype=float)

        baseline = dff.compute_percentile_baseline(
            trace, fps=1, tau=1, percentile=0, instability_ratio=0.1,
            min_window_s=3, window_tau_multiplier=1,
        )

        np.testing.assert_allclose(
            baseline[:, 0], [10, 10, 10, 32 / 3, 12, 40 / 3]
        )

    def test_dim_and_activity_filters_preserve_original_roi_order(self):
        traces = np.array([
            [10, 10, 10, 10, 10, 0],
            [10, 10, 10, 10, 10, 0],
            [10, 10, 10, 10, 10, 0],
            [14, 10, 10, 10, 10, 0],
            [10, 10, 10, 10, 10, 0],
            [10, 10, 10, 10, 10, 0],
        ], dtype=float)
        bright, mask = dff.filter_dim_rois(traces)
        np.testing.assert_array_equal(mask, [True] * 5 + [False])
        self.assertEqual((6, 5), bright.shape)

        dfof = np.column_stack((
            [0, 0, 0, 0.4, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0.1, 0, 0, 0, 0],
        ))
        with contextlib.redirect_stdout(io.StringIO()):
            active, indices = dff.filter_inactive_rois_by_std_or_z(
                dfof, np.array([2, 4, 6]), min_std=0.05
            )
        np.testing.assert_array_equal(indices, [2])
        np.testing.assert_array_equal(active[:, 0], dfof[:, 0])

    def test_one_plane_gateway_returns_known_dfof_and_suite2p_index(self):
        with tempfile.TemporaryDirectory() as folder:
            plane = Path(folder) / "plane0"
            plane.mkdir()
            # Suite2p F.npy is ROI x frames; the public output is frames x ROI.
            fluorescence = np.array([
                [10, 10, 10, 14, 10, 10],  # active cell
                [10, 10, 10, 10, 10, 10],  # stable inactive cell
                [1, 1, 19, 19, 19, 1],    # unstable cell
                [10, 11, 10, 10, 10, 10], # below activity threshold
                [10, 10, 10, 10, 10, 10], # stable inactive cell
                [0, 0, 0, 0, 0, 0],       # dim cell
                [100, 100, 100, 100, 100, 100], # non-cell
            ], dtype=float)
            np.save(plane / "F.npy", fluorescence)
            np.save(plane / "iscell.npy", np.array([
                [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [0, 0]
            ]))

            with contextlib.redirect_stdout(io.StringIO()):
                result, indices = dff.process_suite2p_fluorescence(
                    f_path=plane, fps=1, tau=1, percentile=0,
                    instability_ratio=0.1, min_window_s=1,
                    window_tau_multiplier=1, min_std=0.05,
                )

        self.assertEqual((6, 1), result.shape)
        np.testing.assert_array_equal(indices, [0])
        np.testing.assert_allclose(result[:, 0], [0, 0, 0, 0.4, 0, 0])


if __name__ == "__main__":
    unittest.main()

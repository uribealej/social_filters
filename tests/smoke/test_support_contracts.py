"""Self-tests for shared fixtures and regression assertion helpers."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MATPLOTLIB_CONFIG_ROOT = REPOSITORY_ROOT / ".test_cache" / "matplotlib"
MATPLOTLIB_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MATPLOTLIB_CONFIG_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tests.support.assertions import (
    assert_array_allclose_contract,
    assert_array_equal_contract,
    assert_figure_contract,
    assert_frame_contract,
)
from tests.support.fixtures import (
    continuous_traces,
    response_raster,
    response_table,
    stimulus_log_table,
    stimulus_timing,
    trial_aligned_traces,
)


class SupportContractTests(unittest.TestCase):
    """Verify that the shared S00 testing foundation is usable."""

    def test_numpy_blas_runtime_environment(self) -> None:
        """Require the configured BLAS runtime to complete a tiny dot product."""

        code = (
            "import numpy as np; "
            "result = np.dot(np.eye(3), np.eye(3)); "
            "np.testing.assert_array_equal(result, np.eye(3))"
        )
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            self.fail(
                "NumPy BLAS runtime did not complete a 3x3 dot product within "
                "10 seconds; check the environment's BLAS/OpenMP installation"
            )
        detail = result.stderr.strip() or result.stdout.strip()
        self.assertEqual(0, result.returncode, detail)

    def test_fixture_dimension_conventions(self) -> None:
        """Keep fixture axes explicit and stable for later characterization."""

        self.assertEqual((12, 3), continuous_traces().shape)
        self.assertEqual((3, 6, 2), trial_aligned_traces().shape)
        self.assertEqual((12, 3), response_raster().shape)
        self.assertEqual(["stimulus", "onset_frame", "block"], list(stimulus_log_table().columns))
        self.assertEqual(
            ["fish_id", "neuron_id", "stimulus", "response"],
            list(response_table().columns),
        )
        self.assertEqual({"stim_a", "stim_b"}, set(stimulus_timing()))

    def test_array_assertion_helpers(self) -> None:
        """Exercise exact and tolerance-based array contract helpers."""

        expected = continuous_traces()
        assert_array_equal_contract(
            self,
            expected.copy(),
            expected,
            shape=(12, 3),
            dtype=np.float64,
        )
        assert_array_allclose_contract(
            self,
            expected + 1e-10,
            expected,
            shape=(12, 3),
            rtol=1e-7,
            atol=1e-9,
        )

    def test_dataframe_assertion_helper(self) -> None:
        """Exercise DataFrame schema, order, and value checks."""

        expected = response_table()
        assert_frame_contract(
            self,
            expected.copy(),
            expected,
            columns=["fish_id", "neuron_id", "stimulus", "response"],
        )

    def test_figure_assertion_helper(self) -> None:
        """Exercise structural figure checks without risking a native crash."""

        figure, axis = plt.subplots()
        axis.plot([0, 1], [0, 1])
        try:
            assert_figure_contract(self, figure, expected_axes=1, render=False)
        finally:
            plt.close(figure)

    def test_matplotlib_render_environment(self) -> None:
        """Probe native Matplotlib rendering without crashing the test runner."""

        environment = os.environ.copy()
        environment["MPLCONFIGDIR"] = str(MATPLOTLIB_CONFIG_ROOT)
        environment["PYTHONFAULTHANDLER"] = "1"
        code = (
            "import matplotlib; matplotlib.use('Agg'); "
            "import matplotlib.pyplot as plt; "
            "figure, axis = plt.subplots(); "
            "axis.plot([0, 1], [0, 1]); "
            "figure.canvas.draw(); plt.close(figure)"
        )
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=REPOSITORY_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )
        except subprocess.TimeoutExpired:
            self.fail(
                "Active environment did not complete a minimal Matplotlib "
                "render within 15 seconds; run the BLAS runtime probe and "
                "check the configured Matplotlib backend"
            )
        if result.returncode == 0:
            return

        detail = result.stderr.strip() or result.stdout.strip()
        if "Windows fatal exception" in detail and "0xc06d007f" in detail:
            self.fail(
                "Active environment cannot render Matplotlib figures: "
                "native Windows exception 0xc06d007f"
            )
        self.fail(f"Unexpected Matplotlib render failure:\n{detail}")


if __name__ == "__main__":
    unittest.main()

"""Characterize the duplicated V1 and V2 significant-trace stages."""

from __future__ import annotations

import inspect
import unittest
import warnings
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np

from src import significant_traces as v1
from src import significant_traces_v2 as v2
from src import significant_trace_detection as canonical
from tests.support.assertions import assert_array_allclose_contract
from tests.support.fixtures import significant_trace_cases


SHARED_PUBLIC_STAGES = {
    "compute_global_sigma",
    "compute_noise_model_romano_fast_modular",
    "compute_significant_odds",
    "create_grid",
    "estimate_and_center_noise_model",
    "estimate_kde_peak",
    "extract_transition_points",
    "gaussfit_neg",
    "generate_synthetic_noise",
    "histogram2d",
    "normalize_dff",
    "plot_dff_and_raster",
    "rasterize_with_odds",
}


class SignificantTraceVersionCharacterizationTests(unittest.TestCase):
    """Expose the shared contracts and intentional V1/V2 differences."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = significant_trace_cases()

    def tearDown(self) -> None:
        plt.close("all")

    def test_all_duplicated_public_stages_are_in_the_comparison(self) -> None:
        """Keep this characterization synchronized with both public modules."""

        public_v1 = {
            name
            for name, value in vars(v1).items()
            if inspect.isfunction(value)
            and value.__module__ == v1.__name__
            and not name.startswith("_")
        }
        public_v2 = {
            name
            for name, value in vars(v2).items()
            if inspect.isfunction(value)
            and value.__module__ == v2.__name__
            and not name.startswith("_")
        }
        self.assertEqual(SHARED_PUBLIC_STAGES, public_v1 & public_v2)

    def test_finite_noise_fit_centering_normalization_and_transitions(self) -> None:
        """Shared early stages agree for ordinary finite, varying traces."""

        traces = self.cases["finite_with_transients"]
        fit_v1 = v1.gaussfit_neg(traces[:, 0])
        fit_v2 = v2.gaussfit_neg(traces[:, 0])
        np.testing.assert_allclose(fit_v1[:3], fit_v2[:3], equal_nan=True)
        self.assertEqual(fit_v1[3], fit_v2[3])

        centered_v1 = v1.estimate_and_center_noise_model(traces, verbose=False)
        centered_v2 = v2.estimate_and_center_noise_model(traces, verbose=False)
        for actual, expected in zip(centered_v1[:4], centered_v2[:4]):
            np.testing.assert_allclose(actual, expected, equal_nan=True)
        self.assertEqual(centered_v1[4], centered_v2[4])

        norm_v1 = v1.normalize_dff(centered_v1[0], centered_v1[1])
        norm_v2 = v2.normalize_dff(centered_v2[0], centered_v2[1])
        assert_array_allclose_contract(self, norm_v1, norm_v2, shape=traces.shape)
        np.testing.assert_array_equal(
            v1.extract_transition_points(norm_v1),
            v2.extract_transition_points(norm_v2),
        )

    def test_v2_handles_constant_nan_and_low_sample_noise_fits(self) -> None:
        """V2 has finite fallbacks where V1 raises from SciPy's KDE."""

        edge_traces = {
            "constant": self.cases["constant"][:, 0],
            "nan": np.array([np.nan, -0.1, 0.0, 0.1, np.nan]),
            "low_sample": self.cases["low_sample"][:, 0],
        }
        expected_v1_errors = {
            "constant": np.linalg.LinAlgError,
            "nan": ValueError,
            "low_sample": ValueError,
        }
        expected_v2_fallback = {
            "constant": True,
            "nan": False,
            "low_sample": True,
        }

        for name, trace in edge_traces.items():
            with self.subTest(case=name):
                with self.assertRaises(expected_v1_errors[name]):
                    v1.gaussfit_neg(trace)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    sigma, mu, amplitude, used_fallback = v2.gaussfit_neg(trace)
                self.assertEqual([], caught)
                self.assertTrue(np.isfinite(sigma) and sigma > 0)
                self.assertTrue(np.isfinite(mu))
                self.assertEqual(expected_v2_fallback[name], used_fallback)
                self.assertEqual(used_fallback, bool(np.isnan(amplitude)))

    def test_v2_normalization_avoids_v1_nonfinite_division(self) -> None:
        """Invalid sigma values warn and divide in V1 but become NaN in V2."""

        centered = np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
        sigma = np.array([0.0, np.nan, 2.0])
        with warnings.catch_warnings(record=True) as caught_v1:
            warnings.simplefilter("always")
            normalized_v1 = v1.normalize_dff(centered, sigma)
        with warnings.catch_warnings(record=True) as caught_v2:
            warnings.simplefilter("always")
            normalized_v2 = v2.normalize_dff(centered, sigma)

        self.assertGreaterEqual(len(caught_v1), 1)
        self.assertEqual([], caught_v2)
        self.assertTrue(np.any(~np.isfinite(normalized_v1[:, :2])))
        self.assertTrue(np.isnan(normalized_v2[:, :2]).all())
        np.testing.assert_allclose(normalized_v1[:, 2], normalized_v2[:, 2])

    def test_kde_grid_histogram_and_smoothing_scale_contracts(self) -> None:
        """Common finite results agree; V2 defines safer low-count behavior."""

        norm = self.cases["finite_with_transients"][:, :2]
        points = v1.extract_transition_points(norm)
        peak_v1 = v1.estimate_kde_peak(points)
        peak_v2 = v2.estimate_kde_peak(points)
        np.testing.assert_allclose(peak_v1, peak_v2)

        small_points = np.array([[0.0, 0.0], [1.0, 1.0]])
        peak_small_v1 = v1.estimate_kde_peak(small_points)
        self.assertTrue(np.isfinite(peak_small_v1).all())
        with self.assertRaisesRegex(ValueError, "Not enough valid transition points"):
            v2.estimate_kde_peak(small_points)

        np.random.seed(23)
        synthetic_v1_a = v1.generate_synthetic_noise(points, *peak_v1)
        np.random.seed(23)
        synthetic_v1_b = v1.generate_synthetic_noise(points, *peak_v1)
        synthetic_v2_a = v2.generate_synthetic_noise(
            points, *peak_v2, rng=np.random.default_rng(23)
        )
        synthetic_v2_b = v2.generate_synthetic_noise(
            points, *peak_v2, rng=np.random.default_rng(23)
        )
        np.testing.assert_array_equal(synthetic_v1_a, synthetic_v1_b)
        np.testing.assert_array_equal(synthetic_v2_a, synthetic_v2_b)
        self.assertEqual(synthetic_v1_a.shape, synthetic_v2_a.shape)
        self.assertFalse(np.array_equal(synthetic_v1_a, synthetic_v2_a))

        grid_v1 = v1.create_grid(points, synthetic_v1_a, 24)
        grid_v2 = v2.create_grid(points, synthetic_v1_a, 24)
        for actual, expected in zip(grid_v1, grid_v2):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(
            v1.histogram2d(points, 24, grid_v1[0]),
            v2.histogram2d(points, 24, grid_v2[0]),
        )

        sigma_v1 = v1.compute_global_sigma(points, grid_v1[1], grid_v1[2], 5)
        sigma_v2 = v2.compute_global_sigma(points, grid_v2[1], grid_v2[2], 5)
        self.assertAlmostEqual(sigma_v1, sigma_v2)
        with self.assertRaisesRegex(ValueError, "n_neighbors"):
            v1.compute_global_sigma(small_points, grid_v1[1], grid_v1[2], 100)
        self.assertTrue(
            np.isfinite(
                v2.compute_global_sigma(small_points, grid_v2[1], grid_v2[2], 100)
            )
        )

    def test_significance_and_rasterization_defaults_and_differences(self) -> None:
        """Odds defaults match; V2 explicitly offers stricter corrected behavior."""

        grid = np.linspace(-2.0, 2.0, 5)
        xev, yev = np.meshgrid(grid, grid)
        points = np.array([[-1.0, -0.5], [0.5, 1.0]])
        density_data = np.array(
            [
                [0.0, 0.1, 0.2, 0.0, 0.0],
                [0.0, 0.2, 0.3, 0.1, 0.0],
                [0.0, 0.1, 0.4, 0.2, 0.0],
                [0.0, 0.0, 0.1, 0.2, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        density_noise = 0.04 * density_data
        odds_v1 = v1.compute_significant_odds(
            points, density_data, density_noise, xev, yev
        )
        odds_v2 = v2.compute_significant_odds(
            points, density_data, density_noise, xev, yev
        )
        np.testing.assert_array_equal(odds_v1, odds_v2)
        strict_odds_v2 = v2.compute_significant_odds(
            points,
            density_data,
            density_noise,
            xev,
            yev,
            exclude_empty_bins=True,
        )
        self.assertTrue(odds_v1[density_data == 0].all())
        self.assertFalse(strict_odds_v2[density_data == 0].any())

        map_of_odds = np.ones((5, 5), dtype=bool)
        map_of_odds[2, 2] = False
        map_of_odds[0, 3] = False
        norm_data = np.tile(np.array([-1.5, -0.5, 0.0, 0.5, 1.0, 0.5, 0.0]), (2, 1)).T
        raster_v1, joint_v1 = v1.rasterize_with_odds(
            norm_data, map_of_odds, xev, yev, fps=2.0, tauDecay=6.0
        )
        raster_v2, joint_v2 = v2.rasterize_with_odds(
            norm_data, map_of_odds, xev, yev, fps=2.0, tauDecay=6.0
        )
        self.assertEqual(norm_data.shape, raster_v1.shape)
        self.assertEqual(norm_data.shape, raster_v2.shape)
        self.assertEqual(np.dtype(int), raster_v1.dtype)
        self.assertEqual(np.dtype(int), raster_v2.dtype)
        self.assertFalse(np.array_equal(joint_v1, joint_v2))
        self.assertTrue(joint_v1[2, 2])
        self.assertFalse(joint_v2[2, 2])
        strict_raster, strict_joint = v2.rasterize_with_odds(
            norm_data,
            map_of_odds,
            xev,
            yev,
            fps=2.0,
            tauDecay=6.0,
            joint_map_mode="strict",
        )
        self.assertEqual(norm_data.shape, strict_raster.shape)
        self.assertFalse(strict_joint[2, 2])

    def test_full_pipeline_contract_randomness_and_degenerate_inputs(self) -> None:
        """V2 preserves the tuple while adding reproducibility and diagnostics."""

        traces = self.cases["finite_with_transients"]
        detector_kwargs = {
            "n_bins": 24,
            "k_neighbors": 5,
            "confCutOff": 95,
            "fps": 2.0,
            "tauDecay": 6.0,
        }
        np.random.seed(41)
        with mock.patch.object(canonical, "print", create=True):
            result_v1_a = v1.compute_noise_model_romano_fast_modular(
                traces, **detector_kwargs
            )
        np.random.seed(41)
        with mock.patch.object(canonical, "print", create=True):
            result_v1_b = v1.compute_noise_model_romano_fast_modular(
                traces, **detector_kwargs
            )
        with mock.patch.object(canonical, "print", create=True):
            result_v2_a = v2.compute_noise_model_romano_fast_modular(
                traces,
                random_state=41,
                return_diagnostics=True,
                **detector_kwargs,
            )
            result_v2_b = v2.compute_noise_model_romano_fast_modular(
                traces,
                random_state=41,
                return_diagnostics=True,
                **detector_kwargs,
            )

        self.assertEqual(8, len(result_v1_a))
        self.assertEqual(9, len(result_v2_a))
        expected_shapes = [
            (24, 24),
            traces.shape,
            (24, 24),
            (24, 24),
            (24, 24),
            (24, 24),
            traces.shape,
            (24, 24),
        ]
        expected_dtypes = [bool, float, float, float, float, float, int, bool]
        for index, (shape, dtype) in enumerate(zip(expected_shapes, expected_dtypes)):
            self.assertEqual(shape, result_v1_a[index].shape)
            self.assertEqual(shape, result_v2_a[index].shape)
            self.assertEqual(np.dtype(dtype), result_v1_a[index].dtype)
            self.assertEqual(np.dtype(dtype), result_v2_a[index].dtype)
            np.testing.assert_array_equal(result_v1_a[index], result_v1_b[index])
            np.testing.assert_array_equal(result_v2_a[index], result_v2_b[index])
        np.testing.assert_allclose(result_v1_a[1], result_v2_a[1])
        self.assertEqual(
            detector_kwargs | {"random_state": 41, "smoothing_mode": "v1", "joint_map_mode": "v1", "exclude_empty_bins": False, "mode": "current"},
            result_v2_a[8]["params"],
        )
        self.assertEqual(traces.shape[1], result_v2_a[8]["event_counts"].shape[0])

        constant = self.cases["constant"]
        with self.assertRaises(np.linalg.LinAlgError):
            v1.compute_noise_model_romano_fast_modular(
                constant, n_bins=8, k_neighbors=3
            )
        with mock.patch.object(canonical, "print", create=True):
            constant_v2 = v2.compute_noise_model_romano_fast_modular(
                constant,
                n_bins=8,
                k_neighbors=3,
                random_state=41,
                return_diagnostics=True,
            )
        self.assertEqual(9, len(constant_v2))
        self.assertTrue(constant_v2[8]["degenerate_transition_cloud"])
        self.assertFalse(constant_v2[6].any())

        with_nan = self.cases["with_nan"]
        with self.assertRaises(ValueError):
            v1.compute_noise_model_romano_fast_modular(
                with_nan, n_bins=16, k_neighbors=3
            )
        with warnings.catch_warnings(record=True) as nan_warnings:
            warnings.simplefilter("always")
            with mock.patch.object(canonical, "print", create=True):
                nan_v2 = v2.compute_noise_model_romano_fast_modular(
                    with_nan,
                    n_bins=16,
                    k_neighbors=3,
                    random_state=41,
                    return_diagnostics=True,
                )
        self.assertEqual(with_nan.shape, nan_v2[6].shape)
        self.assertEqual(1, nan_v2[8]["n_invalid_traces"])
        self.assertFalse(nan_v2[6][:, 3].any())
        self.assertEqual(2, len(nan_warnings))
        self.assertTrue(
            all(issubclass(item.category, RuntimeWarning) for item in nan_warnings)
        )
        self.assertTrue(
            all("Degrees of freedom" in str(item.message) for item in nan_warnings)
        )

    def test_legacy_mixed_detection_cleanup_and_plot_chain(self) -> None:
        """Keep the pre-S05 mixed workflow reproducible for compatibility."""

        traces = self.cases["finite_with_transients"]
        with mock.patch.object(canonical, "print", create=True):
            result = v2.compute_noise_model_romano_fast_modular(
                traces,
                n_bins=24,
                k_neighbors=5,
                random_state=71,
            )
        cleanup = v1.clean_binary_raster_columns(result[6])
        self.assertEqual(result[6].shape[0], cleanup["raster_clean"].shape[0])
        self.assertEqual(result[6].shape[1], cleanup["good_mask"].shape[0])

        with mock.patch.object(plt, "show"):
            sort_idx = v1.plot_dff_and_raster(result[1], result[6], fps=2.0)
        np.testing.assert_array_equal(np.sort(sort_idx), np.arange(traces.shape[1]))
        figure = plt.gcf()
        self.assertEqual(4, len(figure.axes))
        figure.canvas.draw()


if __name__ == "__main__":
    unittest.main()

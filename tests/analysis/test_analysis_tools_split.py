"""Known-output and facade tests for the S06 analysis-tools split."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import analysis_tools
from src import data_loading
from src import motion_metrics
from src import multifish_analysis
from src import plotting
from src import reliability
from src import response_classification
from src import response_metrics
from src import response_normalization
from src import response_selectivity
from src import trial_alignment


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class AnalysisToolsSplitTests(unittest.TestCase):
    """Protect owner boundaries and representative numerical contracts."""

    def test_analysis_tools_is_a_thin_compatibility_facade(self) -> None:
        """Require public names to resolve directly to their narrow owners."""

        expected_owners = {
            "build_trial_aligned_traces": trial_alignment,
            "compute_response_window_frames": trial_alignment,
            "compute_trial_mean_response_metrics": response_metrics,
            "compute_motion_delta_integrals": motion_metrics,
            "filter_neurons_by_trial_reliability": reliability,
            "build_active_neuron_matrix_from_trial_raster": response_classification,
            "compute_stimulus_selectivity_metrics": response_selectivity,
            "zscore_dfof_from_prestim_baseline": response_normalization,
            "plot_accepted_rejected_rasters": plotting,
            "plot_venn_3stim": plotting,
        }
        for name, owner in expected_owners.items():
            with self.subTest(name=name):
                self.assertIs(getattr(analysis_tools, name), getattr(owner, name))

        source = (REPOSITORY_ROOT / "src" / "analysis_tools.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        self.assertEqual([], functions)
        self.assertLess(len(source.splitlines()), 100)

    def test_alignment_selection_and_response_window_known_outputs(self) -> None:
        """Preserve stimulus order, aligned axes, and clipped frame semantics."""

        selection = trial_alignment.resolve_selected_stimuli(
            ["stim_b", 1],
            {"stim_a": 1, "stim_b": 2},
            available_stimuli=[1, 2],
        )
        self.assertEqual([2, 1], selection["stimulus_ids"])
        self.assertEqual(["stim_b", "stim_a"], selection["stimulus_labels"])

        window = trial_alignment.compute_response_window_frames(
            n_time=20,
            fps_2p=2.0,
            t_pre_s=2.0,
            motion_onset_s=1.0,
            motion_duration_s=2.0,
            tau_s=1.0,
        )
        np.testing.assert_array_equal(window["frame_indices"], np.arange(6, 14))
        self.assertEqual((6, 14, 8), (window["start_frame"], window["stop_frame"], window["n_frames"]))

        dfof = np.arange(24, dtype=float).reshape(12, 2)
        stimulus_60 = np.zeros(360, dtype=int)
        stimulus_60[90:150] = 1
        stimulus_60[240:300] = 1
        aligned = trial_alignment.build_trial_aligned_traces(
            dfof,
            stimulus_60,
            fps_2p=2.0,
            t_pre_s=1.0,
            t_post_s=2.0,
            stimuli_id_map={"stim_a": 1},
            verbose=False,
        )
        self.assertEqual([1], aligned["stimuli_ids"])
        self.assertEqual(["stim_a"], aligned["stimuli_names"])
        expected = np.stack([dfof[1:7].T, dfof[6:12].T], axis=2)
        np.testing.assert_array_equal(aligned["trial_aligned_traces"][1], expected)

    def test_response_metrics_known_outputs(self) -> None:
        """Preserve mean traces, peaks, averages, and AUC dimension order."""

        trial = np.empty((2, 4, 2), dtype=float)
        trial[0, :, :] = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        trial[1, :, :] = np.array([[4.0, 4.0], [3.0, 3.0], [2.0, 2.0], [1.0, 1.0]])
        result = response_metrics.compute_trial_mean_response_metrics(
            {7: trial},
            fps_2p=1.0,
            t_pre_s=1.0,
            t_post_s=2.0,
        )
        np.testing.assert_array_equal(result["mean_traces"][7], trial[:, :, 0])
        np.testing.assert_allclose(result["peaks"][7], [2.0, 3.0])
        np.testing.assert_allclose(result["averages"][7], [1.5, 2.5])
        np.testing.assert_allclose(result["aucs"][7], [1.5, 2.5])

        trial_auc = response_metrics.compute_trial_auc_by_neuron(
            trial,
            frame_indices=[1, 2],
            fps_2p=1.0,
        )
        np.testing.assert_allclose(trial_auc, [[1.5, 1.5], [2.5, 2.5]])
        np.testing.assert_allclose(
            response_metrics.compute_zscore_response_auc(
                trial,
                frame_indices=[1, 2],
                fps_2p=1.0,
            ),
            [1.5, 2.5],
        )

        self.assertIs(
            data_loading.build_trial_aligned_traces,
            trial_alignment.build_trial_aligned_traces,
        )
        downstream = multifish_analysis.build_zscore_response_matrix_for_fish(
            trial_aligned_traces_z_core={7: trial},
            selected_stimuli=[7],
            stimuli_durations={"stim_a": {"motion_sec": 2.0}},
            stimuli_id_map={"stim_a": 7},
            fps_2p=1.0,
            t_pre_s=0.0,
            motion_onset_s=0.0,
            tau_s=0.0,
        )
        pd.testing.assert_frame_equal(
            downstream,
            pd.DataFrame(
                {"stim_a": [0.5, 3.5]},
                index=pd.Index([0, 1], name="neuron_id"),
            ),
        )

    def test_motion_metrics_known_outputs(self) -> None:
        """Preserve tidy schema and motion-minus-fixed calculations."""

        trace = np.array([0.0, 1.0, 2.0, 5.0, 7.0]).reshape(1, 5, 1)
        kwargs = {
            "trial_aligned_traces": {1: trace},
            "fps_2p": 1.0,
            "t_pre_s": 1.0,
            "pre_motion_fixed_s": 2.0,
            "stimuli_id_map": {"LeB1": 1},
            "fish_id": "fish_01",
        }
        integrals = motion_metrics.compute_motion_delta_integrals(**kwargs)
        peaks = motion_metrics.compute_motion_delta_peaks(**kwargs)
        self.assertEqual(
            [
                "fish_id",
                "trial_id",
                "neuron_id",
                "stim_id",
                "stimulus",
                "segment",
                "side",
                "fixed_before_motion_integral",
                "motion_integral",
                "delta_integral",
            ],
            list(integrals.columns),
        )
        self.assertEqual(("LeB1", "B1", "left"), tuple(integrals.loc[0, ["stimulus", "segment", "side"]]))
        np.testing.assert_allclose(
            integrals.loc[0, ["fixed_before_motion_integral", "motion_integral", "delta_integral"]].to_numpy(dtype=float),
            [5.0, 6.0, 1.0],
        )
        np.testing.assert_allclose(
            peaks.loc[0, ["fixed_before_motion_peak", "motion_peak", "delta_peak"]].to_numpy(dtype=float),
            [5.0, 7.0, 2.0],
        )

    def test_reliability_filter_known_output_without_side_effects(self) -> None:
        """Preserve reliability values and selected neuron order."""

        trial = np.empty((3, 4, 3), dtype=float)
        trial[0] = np.column_stack([[0, 1, 2, 3]] * 3)
        trial[1] = np.column_stack(([0, 1, 2, 3], [3, 2, 1, 0], [0, 1, 0, 1]))
        trial[2] = np.column_stack(([0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 1, 1]))
        result = reliability.filter_neurons_by_trial_reliability(
            dfof=np.arange(18, dtype=float).reshape(6, 3),
            trial_aligned_traces={1: trial},
            stimuli_ids=[1],
            fps_2p=2.0,
            plots_path=None,
            custom_threshold=0.9,
            make_plots=False,
            save_indices=False,
        )
        np.testing.assert_allclose(
            result["max_stimuli_correlation"],
            [1.0, -1.0 / 3.0, -1.0 / 3.0],
            atol=1e-12,
        )
        np.testing.assert_array_equal(result["kept_neuron_indices"], [0])

    def test_classification_and_selectivity_known_outputs(self) -> None:
        """Preserve active decisions, pair indices, and selectivity metrics."""

        raster = np.zeros((2, 5, 3), dtype=int)
        raster[0, 0:2, 0:2] = 1
        raster[1, [0, 2], :] = 1
        active, counts = response_classification.build_active_neuron_matrix_from_trial_raster(
            {1: raster},
            stimuli_durations={"stim_a": {"motion_sec": 5.0}},
            stimuli_id_map={"stim_a": 1},
            fps_2p=1.0,
            t_pre_s=0.0,
            motion_onset_s=0.0,
            active_fraction_threshold=0.4,
            min_epoch_s=2.0,
            min_active_reps=2,
            expected_reps=3,
            tau_s=0.0,
            return_counts=True,
        )
        pd.testing.assert_frame_equal(active, pd.DataFrame({1: [1, 0]}, index=pd.Index([0, 1], name="neuron_id")))
        pd.testing.assert_frame_equal(counts, pd.DataFrame({1: [2, 0]}, index=pd.Index([0, 1], name="neuron_id")))

        pair_index = response_selectivity.compute_response_pair_index(
            pd.DataFrame({"left": [3.0, 1.0], "right": [1.0, 3.0]}),
            "left",
            "right",
        )
        np.testing.assert_allclose(pair_index, [0.5, -0.5])
        metrics = response_selectivity.compute_stimulus_selectivity_metrics(
            [4.0, 1.0, 0.0],
            ["a", "b", "c"],
        )
        self.assertEqual("a", metrics["preferred_stimulus"])
        self.assertAlmostEqual(7.0 / 9.0, metrics["selectivity_index"])
        self.assertAlmostEqual(13.0 / 17.0, metrics["lifetime_sparseness"])

    def test_baseline_normalization_known_output(self) -> None:
        """Preserve baseline selection, orientation, and z-score values."""

        dfof = np.column_stack(
            [
                np.arange(6, dtype=float),
                np.arange(2, 14, 2, dtype=float),
            ]
        )
        result = response_normalization.zscore_dfof_from_prestim_baseline(
            dfof,
            onsets_by_id={1: np.array([3])},
            fps_2p=1.0,
            baseline_s=2.0,
            verbose=False,
        )
        np.testing.assert_allclose(result["baseline_mean"], [1.5, 5.0])
        np.testing.assert_allclose(result["baseline_std"], [0.5, 1.0])
        np.testing.assert_allclose(result["z_traces"][0], [-3.0, -3.0])
        np.testing.assert_array_equal(result["valid_onsets"], [3])

    def test_moved_reliability_diagnostics_render(self) -> None:
        """Keep the reliability figure family renderable in its plotting owner."""

        dfof = np.arange(24, dtype=float).reshape(6, 4)
        with mock.patch.object(plt, "show"):
            figures = plotting.plot_reliability_diagnostics(
                dfof=dfof,
                nanfiltered_max=np.array([0.95, 0.7, 0.2, -0.1]),
                otsu_threshold=0.7,
                kept_mask=np.array([True, True, False, False]),
                kept_neuron_indices=np.array([0, 1]),
                fps_2p=2.0,
                hist_bins=4,
            )
        self.assertEqual(
            {"histogram", "threshold_curve", "rasters"},
            set(figures),
        )
        for figure, _axes in figures.values():
            figure.canvas.draw()
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()

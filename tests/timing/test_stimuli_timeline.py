"""Characterize stimulus-duration normalization and its loader consumers."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import src.data_loading as data_loading
import src.stimuli_timeline as stimuli_timeline


class StimulusDurationNormalizationTests(unittest.TestCase):
    """Protect the established downstream timing dictionary contract."""

    def test_missing_keys_and_output_structure(self) -> None:
        """Default absent duration fields without dropping caller metadata."""

        source = {
            "missing": {"note": "preserve"},
            "motion_alias": {
                "total_sec": 4.5,
                "static_before_sec": 1.25,
                "motion_end_frame": 270,
            },
        }

        result = stimuli_timeline.transform_stimuli_duration(source)

        self.assertEqual(["missing", "motion_alias"], list(result))
        self.assertEqual(
            {
                "note": "preserve",
                "motion_sec": 0,
                "static_after_sec": 0,
            },
            result["missing"],
        )
        self.assertEqual(3.25, result["motion_alias"]["motion_sec"])
        self.assertEqual(270, result["motion_alias"]["motion_end_frame"])
        self.assertEqual(270, result["motion_alias"]["end_frame"])

    def test_total_frames_aliases_and_rounding(self) -> None:
        """Prefer total_frames for aliases and retain three-decimal rounding."""

        source = {
            "stimulus": {
                "total_sec": 3.3334,
                "static_before_sec": 1.1111,
                "total_frames": 201,
                "motion_end_frame": 199,
                "custom": {"unit": "frames"},
            }
        }

        result = stimuli_timeline.transform_stimuli_duration(source)

        self.assertEqual(2.222, result["stimulus"]["motion_sec"])
        self.assertEqual(0, result["stimulus"]["static_after_sec"])
        self.assertEqual(201, result["stimulus"]["motion_end_frame"])
        self.assertEqual(201, result["stimulus"]["end_frame"])
        self.assertEqual({"unit": "frames"}, result["stimulus"]["custom"])

    def test_input_is_not_mutated(self) -> None:
        """Return new per-stimulus dictionaries without changing the input."""

        source = {
            "stimulus": {
                "total_sec": 2.0,
                "static_before_sec": 0.5,
                "total_frames": 120,
                "metadata": {"side": "left"},
            }
        }
        before = copy.deepcopy(source)

        result = stimuli_timeline.transform_stimuli_duration(source)

        self.assertEqual(before, source)
        self.assertIsNot(result, source)
        self.assertIsNot(result["stimulus"], source["stimulus"])

    def test_data_loading_entrypoint_delegates_to_timing_owner(self) -> None:
        """Keep the old import path as a compatibility wrapper only."""

        source = {"stimulus": {"total_sec": 1.0}}
        expected = {"delegated": {"motion_sec": 1.0}}
        with patch.object(
            stimuli_timeline,
            "transform_stimuli_duration",
            return_value=expected,
        ) as canonical:
            result = data_loading.transform_stimuli_duration(source)

        self.assertIs(expected, result)
        canonical.assert_called_once_with(source)

    def test_compatibility_entrypoint_matches_canonical_results(self) -> None:
        """Preserve the old data_loading result for representative timing data."""

        source = {
            "stimulus": {
                "total_sec": 5.0,
                "static_before_sec": 1.3333,
                "motion_end_frame": 300,
            }
        }

        self.assertEqual(
            stimuli_timeline.transform_stimuli_duration(source),
            data_loading.transform_stimuli_duration(source),
        )


class TimingLoaderConsumerTests(unittest.TestCase):
    """Exercise the loader and first alignment consumer around timing data."""

    def test_load_2p_experiment_returns_canonical_timing_contract(self) -> None:
        """Require the loader to expose normalized timing from the owner module."""

        raw_timing = {
            "total_sec": 3.3334,
            "static_before_sec": 1.1111,
            "motion_end_frame": 200,
        }
        trace = np.zeros(240, dtype=int)
        table = pd.DataFrame({"stimulus": ["stim_a"]})

        def fake_glob(path, pattern):
            if pattern == "*trajectory.*":
                return [Path("stim_a_trajectory.csv")]
            if pattern == "*block_log.csv":
                return [Path("trial_block_log.csv")]
            return []

        with (
            patch("builtins.print"),
            patch.object(Path, "glob", autospec=True, side_effect=fake_glob),
            patch.object(data_loading, "_resolve_merged_map_file", return_value=None),
            patch.object(
                data_loading,
                "_resolve_merged_dfof_file",
                return_value=Path("synthetic_dfof.npy"),
            ),
            patch.object(
                data_loading.np,
                "load",
                return_value=np.arange(16).reshape(8, 2),
            ),
            patch.object(
                stimuli_timeline,
                "get_motion_timing_simple",
                return_value=raw_timing,
            ),
            patch.object(
                stimuli_timeline,
                "make_stimulus_traces_2",
                return_value=(pd.DataFrame(), trace, table, {"stim_a": 1}),
            ) as make_traces,
        ):
            result = data_loading.load_2p_experiment(
                fish_id="fish_01",
                experiment_name="exp",
                main_path=Path("data"),
                stimuli_main_path=Path("stimuli"),
                fps_2p=2.0,
                selected_blocks=["B1", "B2"],
            )

        expected_timing = {
            "stim_a": {
                **raw_timing,
                "motion_sec": 2.222,
                "static_after_sec": 0,
                "end_frame": 200,
            }
        }
        self.assertEqual(expected_timing, result["stimuli_durations"])
        self.assertEqual(expected_timing, make_traces.call_args.args[1])

    def test_first_alignment_consumer_preserves_timing_contract(self) -> None:
        """Keep normalized timing unchanged through load-and-align orchestration."""

        timepoints = 8
        traces = np.arange(timepoints * 2, dtype=float).reshape(timepoints, 2)
        stimulus_trace = np.zeros(240, dtype=int)
        stimulus_trace[60:120] = 1
        timings = {
            "stim_a": {
                "total_sec": 2.0,
                "static_before_sec": 0.5,
                "motion_sec": 1.5,
                "static_after_sec": 0,
                "motion_end_frame": 120,
                "end_frame": 120,
            }
        }
        loader_result = {
            "dfof": traces,
            "raster": (traces > 5).astype(int),
            "deltaF_center": traces - traces.mean(axis=0),
            "z_traces": traces / 10.0,
            "stimuli_trace_60": stimulus_trace,
            "stimuli_id_map": {"stim_a": 1},
            "stimuli_durations": timings,
            "dFoF_merged_map": None,
            "kept_neuron_indices": None,
            "frames_per_block": 4,
            "duration_2p_block_sec": 2.0,
            "paths": {},
            "plane_ids": [],
            "stat_per_plane": {},
        }

        with patch.object(data_loading, "load_2p_experiment", return_value=loader_result):
            result = data_loading.load_and_align_2p_experiment(
                fish_id="fish_01",
                experiment_name="exp",
                main_path=Path("unused"),
                stimuli_main_path=Path("unused"),
                fps_2p=2.0,
                selected_blocks=["B1", "B2"],
                t_pre_s=0.5,
                t_post_s=1.0,
                verbose=False,
            )

        self.assertIs(timings, result["stimuli_durations"])
        self.assertEqual([1], result["stimuli_ids"])
        self.assertEqual((2, 3, 1), result["trial_aligned_traces"][1].shape)


if __name__ == "__main__":
    unittest.main()

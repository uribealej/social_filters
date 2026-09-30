"""S10 fixture checks for experiment lookup, optional files, and bundle assembly."""

import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import data_loading


class ExperimentLoaderTests(unittest.TestCase):
    """Exercise the public loader against a small canonical experiment tree."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        root = Path(self.folder.name)
        self.main = root / "data"
        self.stimuli_main = root / "stimuli"
        self.fish = "f01_exp"
        self.prefix = "f01_exp"
        self.analysis = self.main / self.fish / "03_analysis" / "functional"
        self.dfof_dir = self.analysis / "suite2P" / "merged_dFoF"
        self.plots_dir = self.analysis / "plots"
        self.metadata_dir = self.main / self.fish / "01_raw" / "2p" / "metadata"
        self.stimuli_dir = self.stimuli_main / "exp" / "stimuli"
        for path in (self.dfof_dir, self.metadata_dir, self.stimuli_dir):
            path.mkdir(parents=True)
        (self.stimuli_dir / "stim_a_trajectory.csv").touch()
        (self.metadata_dir / "old_block_log.csv").touch()
        latest_log = self.metadata_dir / "new_block_log.csv"
        latest_log.touch()
        os.utime(self.metadata_dir / "old_block_log.csv", (100, 100))
        os.utime(latest_log, (200, 200))
        self.latest_log = latest_log
        self.dfof = np.arange(16, dtype=float).reshape(8, 2)
        self.trace = np.zeros(240, dtype=int)
        self.trace[60:120] = 1

    def load(self):
        raw_timing = {
            "total_sec": 2.0,
            "static_before_sec": 0.5,
            "motion_end_frame": 120,
        }
        with (
            patch.object(data_loading.st, "get_motion_timing_simple", return_value=raw_timing),
            patch.object(
                data_loading.st,
                "make_stimulus_traces_2",
                return_value=(pd.DataFrame(), self.trace, pd.DataFrame(), {"stim_a": 1}),
            ) as make_traces,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = data_loading.load_2p_experiment(
                "f01", "exp", self.main, self.stimuli_main,
                fps_2p=2.0, selected_blocks=["B1", "B2"],
            )
        self.assertEqual(self.latest_log, make_traces.call_args.args[0])
        self.assertEqual(["B1", "B2"], make_traces.call_args.args[2])
        self.assertEqual(2.0, make_traces.call_args.args[3])
        return result

    def save_plane(self):
        plane_dir = self.analysis / "suite2P" / "plane0"
        plane_dir.mkdir(parents=True)
        np.save(plane_dir / "ops.npy", {"meanImg": np.array([[1, 2]])})
        np.save(plane_dir / "stat.npy", np.array([{"xpix": [0]}], dtype=object))

    def test_canonical_files_caches_and_first_alignment_consumer(self):
        canonical = self.dfof_dir / f"{self.prefix}_dFoF_merged.npy"
        np.save(canonical, self.dfof)
        np.save(self.dfof_dir / "other_dFoF_merged.npy", np.full((8, 2), 99))
        map_file = self.dfof_dir / f"{self.prefix}_dFoF_merged_map.csv"
        pd.DataFrame({"plane": ["plane0", "plane0"]}).to_csv(map_file, index=False)
        pd.DataFrame({"wrong": [0]}).to_csv(
            self.dfof_dir / f"{self.prefix}_dFoF_merged_map_old.csv", index=False
        )
        self.save_plane()

        sig_dir = self.plots_dir / "significant_traces"
        kept_dir = self.plots_dir / "filtered_neurons_by_stimuli"
        z_dir = self.plots_dir / "z_core"
        for path in (sig_dir, kept_dir, z_dir):
            path.mkdir(parents=True)
        np.savez(
            sig_dir / f"{self.prefix}_significant_traces.npz",
            raster=(self.dfof > 5).astype(int), deltaF_center=self.dfof - 7.5,
        )
        np.save(kept_dir / f"{self.prefix}_kept_neuron_indices.npy", np.array([0, 1]))
        np.save(self.dfof_dir / f"{self.prefix}_dFoF_merged_filtered_roi_indices.npy", [3, 7])
        np.savez(z_dir / f"{self.prefix}_zcore.npz", z_traces=self.dfof / 10)

        result = self.load()
        self.assertEqual(
            {
                "dfof", "fps_2p", "frames_per_block", "duration_2p_block_sec",
                "stimuli_durations", "adjusted_log", "stimuli_trace_60", "stimuli_table",
                "stimuli_id_map", "paths", "dFoF_merged_map", "z_traces", "raster",
                "deltaF_center", "kept_neuron_indices", "filtered_roi_indices",
                "plane_ids", "mean_imgs", "stat_per_plane",
            },
            set(result),
        )
        self.assertEqual(
            {
                "fish", "prefix", "stimuli_path", "metadata_dir", "dfof_dir",
                "plots_path", "experiment_log_path", "dfof_file", "merged_map_file",
                "planes_dir",
            },
            set(result["paths"]),
        )
        np.testing.assert_array_equal(self.dfof, result["dfof"])
        np.testing.assert_array_equal([3, 7], result["filtered_roi_indices"])
        self.assertEqual(canonical, result["paths"]["dfof_file"])
        self.assertEqual(map_file, result["paths"]["merged_map_file"])
        self.assertEqual(["plane0"], result["plane_ids"])
        np.testing.assert_array_equal([[1, 2]], result["mean_imgs"]["plane0"])
        self.assertEqual((8, 2), result["z_traces"].shape)
        self.assertEqual(4, result["frames_per_block"])

        with (
            patch.object(data_loading.st, "get_motion_timing_simple", return_value={
                "total_sec": 2.0, "static_before_sec": 0.5, "motion_end_frame": 120,
            }),
            patch.object(data_loading.st, "make_stimulus_traces_2", return_value=(
                pd.DataFrame(), self.trace, pd.DataFrame(), {"stim_a": 1},
            )),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            aligned = data_loading.load_and_align_2p_experiment(
                "f01", "exp", self.main, self.stimuli_main, 2.0,
                ["B1", "B2"], 0.5, 1.0, verbose=False,
            )
        self.assertEqual((2, 3, 1), aligned["trial_aligned_traces"][1].shape)
        self.assertEqual(["plane0"], aligned["meta"]["plane_ids"])

    def test_missing_optional_files_and_compatibility_dfof(self):
        fallback = self.dfof_dir / "old_dFoF_merged.npy"
        np.save(fallback, self.dfof)
        np.save(self.dfof_dir / "old_dFoF_merged_filtered_roi_indices.npy", [99])
        result = self.load()
        self.assertEqual(fallback, result["paths"]["dfof_file"])
        self.assertIsNone(result["paths"]["merged_map_file"])
        self.assertEqual([], result["plane_ids"])
        self.assertEqual({}, result["mean_imgs"])
        for key in ("dFoF_merged_map", "raster", "deltaF_center", "z_traces",
                    "kept_neuron_indices", "filtered_roi_indices"):
            self.assertIsNone(result[key], key)

    def test_compatibility_map_and_invalid_map_contract(self):
        np.save(self.dfof_dir / f"{self.prefix}_dFoF_merged.npy", self.dfof)
        map_file = self.dfof_dir / f"{self.prefix}_dFoF_merged_map_old.csv"
        pd.DataFrame({"plane": ["plane0", "plane0"]}).to_csv(map_file, index=False)
        self.save_plane()
        self.assertEqual(map_file, self.load()["paths"]["merged_map_file"])

        pd.DataFrame({"roi": [0, 1]}).to_csv(map_file, index=False)
        with self.assertRaisesRegex(ValueError, "missing required column.*plane"):
            self.load()

    def test_stale_canonical_map_uses_matching_timestamped_map(self):
        np.save(self.dfof_dir / f"{self.prefix}_dFoF_merged.npy", self.dfof)
        canonical = self.dfof_dir / f"{self.prefix}_dFoF_merged_map.csv"
        fallback = self.dfof_dir / f"{self.prefix}_dFoF_merged_map_20251028.csv"
        pd.DataFrame({"plane": ["plane0"]}).to_csv(canonical, index=False)
        pd.DataFrame({"plane": ["plane0", "plane0"]}).to_csv(fallback, index=False)
        self.save_plane()
        result = self.load()
        self.assertEqual(fallback, result["paths"]["merged_map_file"])
        self.assertEqual(2, len(result["dFoF_merged_map"]))

    def test_no_map_matching_merged_neurons_fails(self):
        np.save(self.dfof_dir / f"{self.prefix}_dFoF_merged.npy", self.dfof)
        canonical = self.dfof_dir / f"{self.prefix}_dFoF_merged_map.csv"
        pd.DataFrame({"plane": ["plane0"]}).to_csv(canonical, index=False)
        with self.assertRaisesRegex(ValueError, "matches the merged dFoF's 2 neuron columns"):
            self.load()


if __name__ == "__main__":
    unittest.main()

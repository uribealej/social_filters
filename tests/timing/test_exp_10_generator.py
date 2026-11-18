"""Known-output, geometry, blank-control, and downstream checks for Exp 10."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.stimuli.generation import Trajectory_exp_10_generator as generator
from src.stimuli_timeline import validate_phase_metadata, build_phase_trial_table
from src.stimulus_visualization import load_stimulus_trajectories, plot_stimulus_phase_timing, plot_stimulus_geometry, summarize_stimuli

ROOT = Path(__file__).resolve().parents[2]


class Experiment10Tests(unittest.TestCase):
    def setUp(self):
        self.experiments = [generator.load_experiment(ROOT / "configs/stimuli" / name)
                            for name in generator.DEFAULT_CONFIGS]

    def test_shared_angles_speed_rotation_and_size_units(self):
        angles = 40 + np.arange(9) * 0.497 / 1.2 * 0.6 * 180 / np.pi
        for experiment in self.experiments:
            config, base = experiment["config"], experiment["base"]
            sign = -1 if config["hemifield"] == "left" else 1
            np.testing.assert_allclose(base.angle_deg, sign * angles, atol=1e-6)
            self.assertEqual(45, config["rotation_angle_deg"])
            tables = generator.position_tables(config, base)
            for radius, group in tables.groupby("position_radius_mm"):
                np.testing.assert_allclose(group.angle_deg, base.angle_deg, atol=1e-12)
                np.testing.assert_allclose(np.hypot(group.x, group.y), radius / 10, atol=0.0012)
                theta = np.deg2rad(sign * angles)
                rotation = np.deg2rad(45)
                expected = np.column_stack((radius / 10 * (np.sin(theta)*np.cos(rotation)-np.cos(theta)*np.sin(rotation)),
                                            radius / 10 * (np.sin(theta)*np.sin(rotation)+np.cos(theta)*np.cos(rotation))))
                np.testing.assert_allclose(group[["x", "y"]], expected, atol=0.0015)
                self.assertAlmostEqual(config["dot_diameters_mm"][f"{radius:g}"] / 20, group.dot_radius_cm.iloc[0])
            self.assertAlmostEqual(0.1425, tables.loc[tables.position_radius_mm == 12].dot_radius_cm.iloc[0])

    def test_all_frames_bouts_pulses_and_matched_controls(self):
        for experiment in self.experiments:
            config = experiment["config"]
            tables = generator.position_tables(config, experiment["base"])
            reference = None
            specs = generator.condition_specs(config)
            self.assertEqual(7, len(specs))
            for spec in specs:
                frame, record = generator.build_stimulus(config, tables, spec)
                self.assertEqual((870, 546, 870), (len(frame), record["test_window_onset_frame"], record["test_window_offset_frame"]))
                self.assertEqual(14.5, record["csv_duration_sec"])
                self.assertEqual(1, record["delay_sec"])
                radii = frame.filter(regex="_radius$").to_numpy()
                self.assertFalse(radii[486:546].any())
                if spec["pre_radius_mm"] is None:
                    self.assertFalse(radii[:546].any())
                    self.assertEqual(8.1, record["initial_blank_sec"])
                    self.assertEqual(0, record["preconditioning_sec"])
                else:
                    expected = np.arange(486) % 36 < 18
                    np.testing.assert_array_equal(radii[:486] > 0, np.repeat(expected[:, None], 9, axis=1))
                    self.assertEqual(14, record["preconditioning_pulses"])
                    self.assertTrue(radii[485].all())
                    expected_radius = config["dot_diameters_mm"][f"{spec['pre_radius_mm']:g}"] / 20
                    np.testing.assert_allclose(radii[:486][radii[:486] > 0], expected_radius)
                if spec["bout_present"]:
                    bout = frame.iloc[546:][["dot0_x", "dot0_y", "dot0_radius"]].reset_index(drop=True)
                    if reference is None:
                        reference = bout
                    pd.testing.assert_frame_equal(reference, bout)
                    np.testing.assert_array_equal(bout.dot0_radius, np.tile(np.r_[np.full(18, .1425), np.zeros(18)], 9))
                    self.assertFalse(radii[546:, 1:].any())
                else:
                    self.assertFalse(radii[546:].any())
                    self.assertIsNone(record["bout_onset_frame"])
                    self.assertIsNone(record["bout_offset_frame"])
                    self.assertEqual(0, record["bout_sec"])
                    self.assertEqual(5.4, record["observation_sec"])

    def test_written_package_validation_plots_and_trial_alignment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "stimuli"
            timings = generator.save_experiments(self.experiments, root, previews=False)
            definitions = root / "definitions"
            self.assertEqual(14, len(list(definitions.glob("*_trajectory.csv"))))
            self.assertAlmostEqual(203, timings.csv_duration_sec.sum())
            self.assertAlmostEqual(553, timings.duration_with_pauses_sec.sum())
            saved = pd.read_csv(definitions / "parameters/stimulus_time_summary.csv")
            phases = validate_phase_metadata(saved, definitions)
            self.assertTrue(phases.loc[~phases.bout_present, "bout_onset_frame"].isna().all())
            samples = load_stimulus_trajectories(definitions)
            summary = summarize_stimuli(samples)
            self.assertTrue((summary.n_unique_angles == 9).all())
            for builder in (plot_stimulus_phase_timing, plot_stimulus_geometry):
                figure = builder(samples, saved)
                self.assertTrue(figure.axes)
                plt.close(figure)
            for key in ("LeC1_Bout", "LeC2_Pre18Only"):
                subset = phases.loc[[key]]
                log = pd.DataFrame({"event": ["B1_start", "B1_prestim1_pause", f"B1_stim1_{key}",
                                               "B1_poststim1_pause", "B1_end"],
                                    "timestamp": [0, 0, 12.5, 27, 39.5]})
                trials = build_phase_trial_table(log, [key], subset, {key: key}, ["B1"], 100, 2)
                self.assertAlmostEqual(21.6, trials.test_window_onset_s.iloc[0])
                self.assertAlmostEqual(20.6, trials.pre_end_s.iloc[0])
                self.assertEqual(key == "LeC1_Bout", bool(trials.bout_present.iloc[0]))
                if key.endswith("Only"):
                    self.assertTrue(pd.isna(trials.bout_onset_s.iloc[0]))
            invalid = saved.copy()
            invalid.loc[0, "initial_blank_sec"] = 7
            with self.assertRaises(ValueError):
                validate_phase_metadata(invalid, definitions)
            omitted = saved.index[~saved.bout_present][0]
            invalid = saved.copy()
            invalid.loc[omitted, "bout_onset_frame"] = 546
            with self.assertRaises(ValueError):
                validate_phase_metadata(invalid, definitions)
            # A falsely visible omitted-bout control must not pass downstream validation.
            control_path = definitions / saved.loc[omitted, "trajectory_file"]
            frame = pd.read_csv(control_path)
            frame.loc[546, "dot0_radius"] = .2
            frame.to_csv(control_path, index=False)
            with self.assertRaises(ValueError):
                validate_phase_metadata(saved, definitions)
            for name in ("README.md", "definitions/STIMULUS_DESCRIPTIONS.md", "definitions/parameters/resolved_settings.json"):
                self.assertTrue((root / name).is_file())

    def test_invalid_configs_and_archive_preservation(self):
        raw = json.loads(self.experiments[0]["path"].read_text(encoding="utf-8"))
        cases = ({"n_positions": 10}, {"static_period_sec": 8}, {"speed_cm_sec": 0},
                 {"bout_off_ms": 200}, {"blank_pause_sec": -1}, {"rotation_angle_deg": float("nan")},
                 {"dot_diameters_mm": {"18": 4}})
        with tempfile.TemporaryDirectory() as temp:
            config_path = Path(temp) / "config.json"
            for overrides in cases:
                modified = copy.deepcopy(raw)
                modified["_experiment"].update(overrides)
                config_path.write_text(json.dumps(modified), encoding="utf-8")
                with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                    generator.load_experiment(config_path)
            output = Path(temp) / "stimuli"
            output.mkdir()
            (output / "original.txt").write_text("Preserve me", encoding="utf-8")
            backup = generator.archive_outputs(output)
            self.assertEqual("Preserve me", (backup / "original.txt").read_text(encoding="utf-8"))
            self.assertTrue((output / "original.txt").exists())


if __name__ == "__main__":
    unittest.main()

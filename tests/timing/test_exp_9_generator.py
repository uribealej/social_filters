"""Known-output and first-consumer checks for RF-memory stimulus assets."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.stimuli.generation import Trajectory_exp_9_generator as generator
from scripts.stimuli.generation.Trajectory_unified_generator import build_base_position_table
from src.stimuli_timeline import get_motion_timing_simple
from src.stimulus_visualization import load_stimulus_trajectories, plot_stimulus_phase_timing


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = [ROOT / "configs" / "stimuli" / name for name in generator.DEFAULT_CONFIGS]


class Experiment9Tests(unittest.TestCase):
    def setUp(self):
        self.experiments = [generator.load_experiment(path) for path in CONFIGS]

    def test_default_geometry_and_all_bouts_match(self):
        for experiment in self.experiments:
            config, base = experiment["config"], experiment["base"]
            original, _ = build_base_position_table(config)
            pd.testing.assert_frame_equal(base.drop(columns="position_index"), original)
            self.assertEqual(list(range(15)), base.position_index.tolist())
            reference = None
            for spec in generator.condition_specs(config):
                trajectory, record = generator.build_stimulus(config, base, spec)
                bout = trajectory.iloc[record["bout_onset_frame"]:][["dot0_x", "dot0_y", "dot0_radius"]].reset_index(drop=True)
                self.assertEqual(216, len(bout))
                self.assertEqual(3.6, record["bout_sec"])
                if reference is None:
                    reference = bout
                pd.testing.assert_frame_equal(reference, bout)
                expected_xy = np.repeat(base.loc[base.position_index.isin(range(7, 13)), ["x", "y"]].to_numpy(), 36, axis=0)
                np.testing.assert_array_equal(expected_xy, bout[["dot0_x", "dot0_y"]])
                np.testing.assert_array_equal(np.tile(np.r_[np.full(18, .2), np.zeros(18)], 6), bout.dot0_radius)
                other_radii = [c for c in trajectory if c.endswith("_radius") and c != "dot0_radius"]
                self.assertFalse(trajectory.iloc[record["bout_onset_frame"]:][other_radii].to_numpy().any())

    def test_preconditioning_is_synchronous_and_delay_has_no_extra_off_interval(self):
        experiment = self.experiments[0]
        config, base = experiment["config"], experiment["base"]
        for spec in generator.condition_specs(config)[1:]:
            trajectory, record = generator.build_stimulus(config, base, spec)
            self.assertEqual(486, record["preconditioning_offset_frame"])
            self.assertEqual(14, record["preconditioning_pulses"])
            self.assertEqual(8.1, record["preconditioning_sec"])
            radii = trajectory.filter(regex="_radius$").to_numpy()
            np.testing.assert_array_equal(radii[:486], np.repeat(radii[:486, :1], radii.shape[1], axis=1))
            self.assertTrue((radii[468:486] == .2).all())
            self.assertTrue((radii[486:record["bout_onset_frame"]] == 0).all())
            self.assertEqual(round(spec["delay_requested_sec"] * 60), record["bout_onset_frame"] - 486)
            self.assertEqual(.2, radii[record["bout_onset_frame"], 0])
            positions = config["distant_positions"] if spec["preconditioning"] == "distant" else config["bout_positions"]
            self.assertEqual(len(positions), record["n_dots"])
            for dot, position in enumerate(positions):
                np.testing.assert_array_equal(trajectory.loc[0, [f"dot{dot}_x", f"dot{dot}_y"]], base.loc[position, ["x", "y"]])

    def test_rounding_short_trains_ties_and_custom_cadence(self):
        self.assertEqual((14, 486), generator.preconditioning_frames(8, 18, 18, 60))
        self.assertEqual((2, 54), generator.preconditioning_frames(.6, 18, 18, 60))
        self.assertEqual((1, 18), generator.preconditioning_frames(.01, 18, 18, 60))
        experiment = self.experiments[1]
        config = {**experiment["config"], "framerate": 50, "flicker_on_ms": 200,
                  "flicker_off_ms": 400, "preconditioning_duration_sec": 2,
                  "bout_on_ms": 200, "bout_off_ms": 400}
        _, record = generator.build_stimulus(config, experiment["base"], generator.condition_specs(config)[1])
        self.assertEqual(100, record["preconditioning_offset_frame"])
        self.assertEqual(4, record["preconditioning_pulses"])
        self.assertEqual(15, record["delay_frames"])
        self.assertEqual(180, record["bout_offset_frame"] - record["bout_onset_frame"])

    def test_invalid_scientific_settings_fail(self):
        cases = ({"bout_positions": [7, 15]}, {"distant_positions": [7]},
                 {"bout_positions": [7, 7]}, {"framerate": 0}, {"static_period_sec": 8},
                 {"preconditioning_duration_sec": float("nan")}, {"overlap_delays_sec": [-1]},
                 {"angle_ranges": [[30, 160]]})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            for overrides in cases:
                raw = copy.deepcopy(self.experiments[0]["raw"])
                raw["_experiment"].update(overrides)
                path.write_text(json.dumps(raw), encoding="utf-8")
                with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                    generator.load_experiment(path)

    def test_written_assets_reports_and_existing_consumers(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            for relative in (*generator.OBSOLETE_REPORTS,
                             *[f"parameters/configs/{path.name}" for path in CONFIGS]):
                obsolete = output / relative
                obsolete.parent.mkdir(parents=True, exist_ok=True)
                obsolete.write_text("Superseded generated output", encoding="utf-8")
            (output / "stimulus_timing_log.csv").write_text("measured playback", encoding="utf-8")
            (output / "notes.md").write_text("User notes", encoding="utf-8")
            records = generator.save_experiments(self.experiments, output)
            self.assertEqual(14, len(records))
            self.assertAlmostEqual(178, sum(r["csv_duration_sec"] for r in records))
            self.assertAlmostEqual(528, sum(r["duration_with_pauses_sec"] for r in records))
            for record in records:
                timing = get_motion_timing_simple(output / record["trajectory_file"])
                self.assertEqual(record["n_frames"], timing["total_frames"])
                self.assertAlmostEqual(record["csv_duration_sec"], timing["total_sec"])
            report = (output / "experiment_9_report.md").read_text(encoding="utf-8")
            for record in records:
                self.assertIn(record["stimulus_key"], report)
            self.assertIn("(experiment_9_timing.xlsx)", report)
            self.assertLess(len(report), 3500)
            self.assertNotIn("Purpose:", report)
            self.assertNotIn("Prediction", report)
            for relative in generator.OBSOLETE_REPORTS:
                self.assertFalse((output / relative).exists(), relative)
            self.assertFalse((output / "parameters/configs").exists())
            self.assertEqual("measured playback", (output / "stimulus_timing_log.csv").read_text(encoding="utf-8"))
            self.assertEqual("User notes", (output / "notes.md").read_text(encoding="utf-8"))
            resolved = json.loads((output / "parameters/resolved_settings.json").read_text(encoding="utf-8"))
            for experiment in self.experiments:
                side = experiment["config"]["hemifield"]
                self.assertEqual(experiment["config"]["bout_positions"], resolved[side]["bout_positions"])
            with zipfile.ZipFile(output / "experiment_9_timing.xlsx") as excel:
                ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                workbook = ET.fromstring(excel.read("xl/workbook.xml"))
                self.assertEqual(["Timing", "Frames"], [s.get("name") for s in workbook.findall("s:sheets/s:sheet", ns)])
                sheet = ET.fromstring(excel.read("xl/worksheets/sheet1.xml"))
                for cell, expected in (("B7", 14), ("C7", 178), ("D7", 350), ("E7", 528)):
                    value = sheet.find(f".//s:c[@r='{cell}']/s:v", ns)
                    self.assertIsNotNone(value, f"Missing cached Excel total {cell}")
                    self.assertAlmostEqual(expected, float(value.text))
            timings = pd.read_csv(output / "parameters/stimulus_time_summary.csv")
            samples = load_stimulus_trajectories(output)
            figure = plot_stimulus_phase_timing(samples, timings)
            self.assertEqual(14, len(figure.axes[0].get_yticklabels()))
            plt.close(figure)
            with self.assertRaisesRegex(ValueError, "same stimuli"):
                plot_stimulus_phase_timing(samples, timings.iloc[:-1])
            with self.assertRaisesRegex(ValueError, "other trajectories"):
                generator.save_experiments(self.experiments[:1], output)


if __name__ == "__main__":
    unittest.main()

"""Known-output contracts for explicit phase timing and block-local alignment."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import data_loading, stimuli_timeline, trial_alignment


class PhaseAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.assets = self.root / "stimuli" / "experiment" / "stimuli"
        self.assets.mkdir(parents=True)
        records = []
        for name, pre, delay in [("C1_Bout", 0, 0), ("S1_Overlap", 486, 18)]:
            bout = pre + delay
            count = bout + 216
            visible = np.zeros(count)
            if pre:
                visible[:pre] = (np.arange(pre) % 36 < 18) * 0.2
            visible[bout:] = (np.arange(216) % 36 < 18) * 0.2
            pd.DataFrame({"dot0_radius": visible}).to_csv(self.assets / f"{name}_trajectory.csv", index=False)
            records.append(dict(stimulus_key=name, trajectory_file=f"{name}_trajectory.csv",
                framerate=60, n_frames=count, preconditioning_offset_frame=pre,
                bout_onset_frame=bout, bout_offset_frame=count,
                preconditioning_sec=pre / 60, delay_sec=delay / 60,
                bout_sec=3.6, csv_duration_sec=count / 60))
        self.phases_raw = pd.DataFrame(records)
        self.phase_path = self.root / "phases.csv"
        self.phases_raw.to_csv(self.phase_path, index=False)
        self.phases = stimuli_timeline.validate_phase_metadata(self.phases_raw, self.assets)
        self.aliases = {"C1": "C1_Bout", "S1": "S1_Overlap"}
        events = []
        self.sequence = []
        for bi, block in enumerate(["B1", "B2"]):
            events.append((block + "_start", 0.0005))
            for j, (name, start) in enumerate([("C1", 12.1), ("S1", 38.2)]):
                i = bi * 2 + j  # Global trial indices, as in the real recordings.
                duration = 3.6 if name == "C1" else 12.0
                events.extend([(f"{block}_prestim{i}_pause", start - 10.1),
                               (f"{block}_stim{i}_{name}", start),
                               (f"{block}_poststim{i}_pause", start + duration + 0.02)])
                self.sequence.append(name)
            events.append((block + "_end", 65))
        self.log = pd.DataFrame(events, columns=["event", "timestamp"])
        self.trials = self.build()

    def build(self, **kwargs):
        args = dict(block_log=self.log, trial_sequence=self.sequence,
                    phases=self.phases, stimulus_aliases=self.aliases,
                    selected_blocks=["B1", "B2"], frames_per_block=160,
                    fps_2p=2, baseline_s=10)
        args.update(kwargs)
        return stimuli_timeline.build_phase_trial_table(**args)

    def test_recorded_times_offsets_and_control(self):
        t = self.trials
        np.testing.assert_allclose(t.onset_s, [12.1, 38.2, 92.1, 118.2])
        np.testing.assert_allclose(t.bout_onset_s - t.onset_s, [0, 8.4, 0, 8.4])
        np.testing.assert_allclose(t.pre_end_s - t.onset_s, [0, 8.1, 0, 8.1])
        np.testing.assert_allclose(t.logged_end_s - t.end_s, 0.02)

    def test_extraction_shape_order_and_actual_sample_times(self):
        dfof = np.tile(np.arange(320)[:, None], (1, 3))
        result = trial_alignment.build_event_aligned_traces(dfof, self.trials, 2)
        control = result["trial_aligned_traces"][1]
        self.assertEqual(control.shape, (3, 38, 2))
        np.testing.assert_array_equal(control[0, :, 0], np.arange(14, 52))
        np.testing.assert_array_equal(control[0, :, 1], np.arange(174, 212))
        np.testing.assert_allclose(result["actual_time_s"][1][0], [-5.1, -5.1])
        np.testing.assert_allclose(result["alignment_error_s"][1], [-0.1, -0.1])
        bout = trial_alignment.build_event_aligned_traces(dfof, self.trials, 2, alignment="bout")
        self.assertEqual(bout["trial_aligned_traces"][2].shape, control.shape)
        self.assertEqual(bout["onsets_by_id"][2][0], 93)

    def test_different_delays_preserve_real_times(self):
        longer = self.phases.copy()
        longer.loc["S1_Overlap", "bout_onset_frame"] = 606
        longer.loc["S1_Overlap", "csv_duration_sec"] = 13.7
        log = self.log.copy()
        log.loc[log.event.str.contains("poststim[13]_"), "timestamp"] += 1.7
        trials = self.build(block_log=log, phases=longer)
        np.testing.assert_allclose(trials.loc[trials.stimulus_id == 2, "bout_onset_s"]
                                   - trials.loc[trials.stimulus_id == 2, "onset_s"], 10.1)

    def test_unknown_names_and_missing_trials_fail(self):
        log = self.log.copy()
        log.loc[log.event == "B1_stim0_C1", "event"] = "B1_stim0_UNKNOWN"
        with self.assertRaisesRegex(ValueError, "Unknown logged stimulus"):
            self.build(block_log=log)
        log = self.log.loc[~self.log.event.str.contains("B2_stim3_")]
        with self.assertRaisesRegex(ValueError, "missing or duplicate"):
            self.build(block_log=log, trial_sequence=self.sequence[:-1])

    def test_sequence_and_baseline_failures(self):
        with self.assertRaisesRegex(ValueError, "Trial sequence"):
            self.build(trial_sequence=list(reversed(self.sequence)))
        with self.assertRaisesRegex(ValueError, "baseline crosses"):
            self.build(baseline_s=20)
        with self.assertRaisesRegex(ValueError, "one-to-one"):
            self.build(stimulus_aliases={"C1": "C1_Bout", "S1": "C1_Bout"})

    def test_boundary_crossing_and_nonfinite_settings_fail(self):
        for options in [dict(t_pre_s=20), dict(recovery_s=80)]:
            with self.assertRaisesRegex(ValueError, "block boundary"):
                trial_alignment.build_event_aligned_traces(np.zeros((320, 3)), self.trials, 2, **options)
        with self.assertRaises(ValueError):
            trial_alignment.build_event_aligned_traces(np.zeros((320, 3)), self.trials, np.nan)

    def test_bad_assets_fail(self):
        frame = self.phases_raw.copy()
        frame.loc[1, "delay_sec"] = 0.6
        with self.assertRaisesRegex(ValueError, "durations disagree"):
            stimuli_timeline.validate_phase_metadata(frame, self.assets)
        asset = self.assets / "S1_Overlap_trajectory.csv"
        csv = pd.read_csv(asset)
        csv.loc[490, "dot0_radius"] = 0.2
        csv.to_csv(asset, index=False)
        with self.assertRaisesRegex(ValueError, "Visibility"):
            stimuli_timeline.validate_phase_metadata(self.phases_raw, self.assets)

    def prepare_recording(self):
        fish = self.root / "data" / "L1_f01_experiment"
        metadata_dir = fish / "01_raw/2p/metadata"
        metadata_dir.mkdir(parents=True)
        meta = dict(fish_ID="L1_f01", experiment_name="experiment", framerate=2,
                    n_rep_stim=2, n_volumes=160, n_trials_per_block=2)
        pd.DataFrame(meta.items(), columns=["parameter", "value"]).to_csv(metadata_dir / "session_metadata.csv", index=False)
        self.log.to_csv(metadata_dir / "session_block_log.csv", index=False)
        global_log = self.log.copy()
        global_log.loc[global_log.event.str.startswith("B2_"), "timestamp"] += 100
        global_log.to_csv(metadata_dir / "session_experiment_log.csv", index=False)
        pd.DataFrame({"stimulus": self.sequence}).to_csv(metadata_dir / "session_trial_sequence.csv", index=False)
        out = fish / "03_analysis/functional/suite2P/merged_dFoF"
        out.mkdir(parents=True)
        np.save(out / "L1_f01_dFoF_merged.npy", np.ones((320, 3)))
        return metadata_dir, out

    def load(self, **kwargs):
        return data_loading.load_phase_aligned_experiment("L1_f01", "experiment", self.root / "data",
            self.root / "stimuli", self.phase_path, self.aliases, ["B1", "B2"], **kwargs)

    def test_loader_session_selection_and_frame_count(self):
        folder, out = self.prepare_recording()
        result = self.load()
        self.assertEqual(len(result["trials"]), 4)
        self.log.to_csv(folder / "other_block_log.csv", index=False)
        with self.assertRaisesRegex(ValueError, "Select one recording session"):
            self.load()
        self.assertEqual(self.load(session_id="session")["session_id"], "session")
        np.save(out / "L1_f01_dFoF_merged.npy", np.ones((316, 3)))
        with self.assertRaisesRegex(ValueError, "Incompatible frame count"):
            self.load(session_id="session")

    def test_writer_and_first_cache_consumer_preserve_column_mapping(self):
        self.prepare_recording()
        loaded = self.load()
        zres = dict(z_traces=np.zeros((320, 3)), baseline_mean=np.zeros(3),
                    baseline_std=np.ones(3), valid_onsets=np.array([24, 76, 184, 236]))
        rel = dict(kept_neuron_indices=np.array([0, 2]), otsu_threshold=0.4)
        sig = (None, np.ones((320, 2)), None, None, None, None, np.zeros((320, 2)), None)
        out = self.root / "outputs"
        record = data_loading.save_phase_analysis_outputs(loaded, zres, rel, sig, {}, out)
        self.assertTrue(record.exists())
        caches = data_loading._load_optional_caches(dict(prefix="L1_f01", plots_path=out, dfof_dir=out))
        np.testing.assert_array_equal(caches["kept_neuron_indices"], [0, 2])
        self.assertEqual(caches["raster"].shape, (320, 2))
        with self.assertRaisesRegex(ValueError, "Cache shapes"):
            data_loading.save_phase_analysis_outputs(loaded, zres,
                dict(kept_neuron_indices=np.array([0]), otsu_threshold=0.4), sig, {}, self.root / "bad")
        self.assertFalse((self.root / "bad").exists())

    def test_review_save_requires_current_settings_and_neuron_ids(self):
        import json
        self.prepare_recording()
        loaded = self.load()
        zres = dict(z_traces=np.zeros((320, 3)), baseline_mean=np.zeros(3),
                    baseline_std=np.ones(3), valid_onsets=np.array([24, 76, 184, 236]))
        rel = dict(kept_neuron_indices=np.array([0, 2]), otsu_threshold=0.4)
        sig = (None, np.ones((320, 2)), None, None, None, None, np.zeros((320, 2)), None)
        run = dict(fish=loaded["paths"]["fish"], session_id=loaded["session_id"],
                   source_file=str(loaded["paths"]["dfof_file"]), fps=2,
                   settings=dict(n_bins=64, confCutOff=90), neuron_indices=[0, 2])
        config = dict(significant=dict(run["settings"]))
        out = self.root / "reviewed"
        data_loading.save_phase_analysis_outputs(loaded, zres, rel, sig, config, out, significant_run=run)
        cache = np.load(out / "significant_traces/L1_f01_significant_traces.npz")
        self.assertEqual(json.loads(str(cache["settings_json"])), run["settings"])
        np.testing.assert_array_equal(cache["kept_neuron_indices"], [0, 2])
        self.assertEqual(float(cache["fps"]), 2)
        for changed in (dict(significant=dict(n_bins=32, confCutOff=90)), config):
            selected = rel if changed is not config else dict(kept_neuron_indices=np.array([2, 0]), otsu_threshold=0.4)
            with self.assertRaisesRegex(ValueError, "rerun detection"):
                data_loading.save_phase_analysis_outputs(loaded, zres, selected, sig,
                    changed, self.root / "stale", significant_run=run)
        self.assertFalse((self.root / "stale").exists())

    def test_legacy_alignment_contract_unchanged(self):
        dfof = np.arange(200).reshape(100, 2)
        stimulus = np.zeros(3000, dtype=int)
        stimulus[600:720] = 1
        result = trial_alignment.build_trial_aligned_traces(dfof, stimulus, 2,
            t_pre_s=1, t_post_s=2, verbose=False)
        np.testing.assert_array_equal(result["onsets_by_id"][1], [20])
        np.testing.assert_array_equal(result["trial_aligned_traces"][1][:, :, 0], dfof[18:24].T)

    def test_phase_plots_preserve_condition_and_neuron_order(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src import plotting_single_fish as plotting

        data = np.column_stack([np.arange(320), np.arange(320) * 2, np.arange(320) * 3])
        aligned = trial_alignment.build_event_aligned_traces(data, self.trials, 2)
        fig, lines = plotting.plot_phase_aligned_responses(aligned,
            ["C1_Bout", "S1_Overlap"], [1, 2], [2, 0, 1],
            neuron_indices=[0, 2], trace_count=1, figsize=(8, 6), trace_figsize=(8, 6))
        self.addCleanup(plt.close, fig)
        self.addCleanup(plt.close, lines)
        expected = aligned["trial_aligned_traces"][1].mean(axis=2)[[2, 0]]
        np.testing.assert_array_equal(fig.axes[0].images[0].get_array(), expected)
        self.assertEqual(len(fig.axes[0].patches), 1)  # C1: only the bout phase.
        self.assertEqual(len(fig.axes[1].patches), 3)
        self.assertEqual(fig.axes[0].get_xlim(), fig.axes[1].get_xlim())
        self.assertEqual([tick.get_text() for tick in lines.axes[0].get_yticklabels()], ["2"])
        binary, _ = plotting.plot_phase_aligned_responses(aligned,
            ["C1_Bout", "S1_Overlap"], [1, 2], [2, 0, 1], binary=True, trace_count=0, figsize=(8, 6))
        self.addCleanup(plt.close, binary)
        self.assertEqual(binary.axes[0].images[0].get_clim(), (0, 1))

    def test_population_mean_uses_exact_raster_rows_with_equal_neuron_weights(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src import plotting_single_fish as plotting

        aligned = trial_alignment.build_event_aligned_traces(np.zeros((320, 3)), self.trials, 2)
        arr = aligned["trial_aligned_traces"][1]
        arr[0, 0, :] = [2, np.nan]  # Per-neuron trial mean = 2.
        arr[1, 0, :] = 1000        # Must not contribute to the filtered mean.
        arr[2, 0, :] = [4, 8]      # Per-neuron trial mean = 6.
        for selection, expected in [([0, 2], 4.0), (None, 336.0)]:
            fig, lines = plotting.plot_phase_aligned_responses(aligned,
                ["C1_Bout", "S1_Overlap"], [1, 2], [2, 0, 1],
                neuron_indices=selection, trace_mode="population", trace_count=0,
                figsize=(8, 6), trace_figsize=(8, 6))
            self.addCleanup(plt.close, fig)
            self.addCleanup(plt.close, lines)
            population_line = next(line for line in lines.axes[0].lines if line.get_label() == "Population mean")
            self.assertEqual(population_line.get_ydata()[0], expected)
            self.assertEqual(lines.axes[0].get_ylabel(), "Population mean z-score")
            self.assertEqual(lines.axes[0].get_ylim(), lines.axes[1].get_ylim())
            self.assertEqual(len(lines.axes[0].patches), 1)
            count = 3 if selection is None else 2
            self.assertEqual(fig.axes[0].images[0].get_array().shape[0], count)
            self.assertIn(f"all {count} raster neurons", lines._suptitle.get_text())


if __name__ == "__main__":
    unittest.main()

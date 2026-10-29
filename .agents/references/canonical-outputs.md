# Canonical Outputs

Purpose

Record the authoritative writer stages, file names, and folders that downstream workflows expect.

Use this file when

- A task changes output names or output folders.
- A loader or notebook cannot find a file it expects.
- You need to know which stage owns a cache or derived artifact.

## Calcium preprocessing outputs
- Per-plane writer stage: `DeltaFF_batch_pipeline.ipynb`, using `process_suite2p_fluorescence` for extraction.
- Canonical per-plane files written by `DeltaFF_batch_pipeline.ipynb`:
  - `{prefix}_dFoF.npy`
  - `{prefix}_filtered_roi_indices.npy`
  - `{prefix}_dFoF_outputs.npz`
  - `metadata.json`
- The historical standalone runner in `src/dff_extraction.py` was retired in S11. The batch notebook is the sole supported per-plane writer; its prefixed file names and metadata contract are unchanged.
- Experiment-level merge writer stage: `DeltaFF_batch_pipeline.ipynb`.
- Canonical merged folder:
  - `03_analysis/functional/suite2P/merged_dFoF`
- Canonical merged files:
  - `{prefix}_dFoF_merged.npy`
  - `{prefix}_dFoF_merged_filtered_roi_indices.npy`
  - `{prefix}_dFoF_merged_map.csv`
- Loader-side authority for discovering merged outputs and compatibility variants:
  - `src/data_loading.py`
- The loader requires a merged map to have one row per merged dFoF neuron. It
  prefers the canonical map when consistent, otherwise selects the newest
  matching timestamped map from the writer's locked-CSV fallback. It raises
  when maps exist but none match; a missing map remains optional.
- Preprocessing notebook helpers that search for experiments, planes, or per-plane files are wrapper logic, not canonical output contracts.

## Calcium analysis caches and plots
- Exp 9 significant-trace review is non-writing until its final `SAVE_OUTPUTS` cell is enabled. Its filtered canonical significant NPZ includes `raster`, `deltaF_center`, `kept_neuron_indices`, `fps`, `source_file`, and `settings_json`. Columns follow the original merged-neuron IDs in the canonical retained-index file; zero-event neurons are not removed. The writer validates the successful detector run's settings/session/source/IDs before saving. Four optional review PNGs (original/centered/raster comparison, peak-sorted comparison, single-neuron trace, high-activity examples) are saved beside that NPZ. Display order, example IDs and review settings are recorded in the phase-analysis configuration. This differs intentionally from the separate `significant_traces_all` reference notebook, whose outputs include every processed input column and are not loaded as filtered canonical caches.
- Canonical plots root:
  - `03_analysis/functional/plots`
- Common cache folders and files:
  - `z_core/{prefix}_zcore.npz`
  - `significant_traces/{prefix}_significant_traces.npz`
  - `filtered_neurons_by_stimuli/{prefix}_kept_neuron_indices.npy`
  - `merged_dFoF/` for merge-side plot exports and sort-order CSVs written during validation notebooks
- `src/data_loading.py` is the loader-side authority for how these outputs are discovered and reused.

## Several-fish reviewed-run reports
- Exp 9 population analysis reuses `save_analysis_report_run`, with `analysis_path` set to `Lab/Results/Paper/2p_derived` and run label `population_analysis`. The usual `<experiment>/reports/<timestamp>_<label>/` contains readiness, per-trial/per-neuron/per-fish metrics, native sample coverage, phase memberships, correlations, paired differences and original-neuron display mappings in `tables/`. Optional `figures` are exported into `figures/` as PNGs. Settings and run metadata retain included fish, sessions and per-fish detector settings. The final save gate defaults off; individual-fish canonical caches are read-only.
- `src/several_fish_reporting.py` writes optional reports when `REPORT_SETTINGS['save_report']` is enabled. Active several-fish notebooks call it directly; `src.reusable_several_fish` remains a supported historical compatibility facade under the [S12 decision](src-compatibility-decision.md).
- Report folder: `<analysis_path>/<experiment_name>/reports/<YYYY-MM-DD_HHMM>_<run_label>/`.
- Each saved run contains `settings.json`, `report_settings.json`, `run_metadata.json`, `comments.md`, and `tables/`. Nonempty supplied tables become `<slugified_table_name>.csv` files under `tables/`.
- When notebook export is requested, nbconvert writes `<report_name>.html` and/or `<report_name>.pdf` in the run folder. A failed PDF export may create an HTML fallback. The returned export status and paths report what succeeded.

## Stimulus authoring outputs
- Experiment 9 also writes `experiment_9_timing.xlsx`, linked from `experiment_9_report.md`, with per-condition timings and formula totals on `Timing` plus frame boundaries/cadence on `Frames`. The Python generator calls `exp_9_timing_workbook.mjs` using Node.js and `@oai/artifact-tool` (the bundled Codex runtime is supported). Its cached Excel totals are checked against the generated records before export.
- Experiment 9 uses `Trajectory_exp_9_generator.py` and left/right configs under `configs/stimuli/`. The default root is `<2p_derived>/Exp_9_RF_memory_inhibition/stimuli/`. Its compact output set is dot-prefixed trajectories, `parameters/base_position_table_left.csv` and `base_position_table_right.csv` (zero-based position_index plus legacy one-based position_id), `parameters/stimulus_time_summary.csv`, `parameters/resolved_settings.json`, one short `experiment_9_report.md` stimulus specification, and the Excel workbook. Source configs remain in the repository; the resolved JSON stores all effective settings. The generator removes only named superseded reports/CSV exports/config copies; it preserves playback logs and unrelated files. Initial static display is disabled. External before/after pauses are not embedded in CSVs. `preconditioning_offset_frame`, `bout_onset_frame`, and `bout_offset_frame` are explicit exclusive phase boundaries; generic movement timing is not test-bout onset. A partial-side rerun into a folder containing other trajectories is rejected rather than silently producing incomplete reports.
- Stimulus asset writers live in `scripts/stimuli/generation/`; their JSON configuration inputs live in `configs/stimuli/`.
- Other established stimulus writers retain these canonical generated assets:
  - `*_trajectory.csv`
  - `parameters/experiment_parameters.csv`
  - `parameters/total_time_sec.csv`
  - package or mapping JSON files kept alongside the generating scripts
- Playback runs can write:
  - `stimulus_timing_log.csv`
- Inspection figures from `notebooks/stimuli/plots_stimuli_batch.ipynb` are notebook outputs, not canonical writer-stage contracts.
- The generic `src/stimulus_visualization.py::save_stimulus_report` helper remains available for earlier workflows, writing `stimuli_standard.png`, `stimuli_polar.png`, and `stimuli_summary.csv`. The CSV contains `stimulus_name`, `unique_angles_deg` (JSON lists), `n_unique_angles`, and `n_dots`; visible spatial directions are rounded to 0.1 degree, whereas dot count includes all recorded identities.
- The Experiment 9 batch notebook saves only `stimuli_polar.png` and `stimuli_timing.png` at 300 dpi. Angle/dot-size traces and angle summaries are displayed in the notebook without creating redundant files. Timing phases use supplied metadata via `plot_stimulus_phase_timing`.

## Timing ownership
- Exp 9 individual-fish analysis preserves canonical `plots/z_core/{prefix}_zcore.npz`, `plots/filtered_neurons_by_stimuli/{prefix}_kept_neuron_indices.npy`, and `plots/significant_traces/{prefix}_significant_traces.npz`. The significant cache additionally records original `kept_neuron_indices` for its column mapping. `src.data_loading.save_phase_analysis_outputs` adds `plots/phase_alignment/{prefix}_trial_timing.csv` and `{prefix}_analysis_configuration.json` containing the matched session, input paths, aliases, metadata and settings. Validation may redirect this output tree without writing to the source data.
- `src/stimuli_timeline.py` is the authority for turning trajectory CSVs and block logs into downstream timing semantics.
- Downstream notebooks should consume these timing semantics, not redefine movement onset or duration logic locally.
- Stimulus generator helpers may compute packaging metadata such as total duration, but that metadata does not supersede `src/stimuli_timeline.py` as downstream timing authority.

## Writer-stage rules
- Change file naming or folder layout only when the task explicitly requires a migration.
- When a writer-stage contract changes, verify the first downstream consumer immediately after the writer, not just the writer itself.
- For calcium analysis, the first downstream consumer is often `src/data_loading.py` or a notebook that loads the written cache.
- Only writer-stage filenames and folders listed in this file are authoritative downstream contracts. Notebook or script convenience helpers are not canonical outputs unless a later slice promotes them.

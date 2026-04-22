# Canonical Outputs

Purpose

Record the authoritative writer stages, file names, and folders that downstream workflows expect.

Use this file when

- A task changes output names or output folders.
- A loader or notebook cannot find a file it expects.
- You need to know which stage owns a cache or derived artifact.

## Calcium preprocessing outputs
- Per-plane writer stage: preprocessing notebooks using `process_suite2p_fluorescence`.
- Typical per-plane files:
  - `dFoF.npy`
  - `roi_filtered.npy`
  - `meta.json`
- Experiment-level merge writer stage: `DeltaFF_batch_pipeline.ipynb`.
- Canonical merged folder:
  - `03_analysis/functional/suite2P/merged_dFoF`
- Canonical merged files:
  - `{prefix}_dFoF_merged.npy`
  - `{prefix}_dFoF_merged_filtered_roi_indices.npy`
  - `{prefix}_dFoF_merged_map.csv`
- Loader-side authority for discovering merged outputs and compatibility variants:
  - `src/data_loading.py`
- Preprocessing notebook helpers that search for experiments, planes, or per-plane files are not canonical outputs by themselves; they are wrapper logic and backlog candidates listed in `duplicate-helper-inventory.md`.

## Calcium analysis caches and plots
- Canonical plots root:
  - `03_analysis/functional/plots`
- Common cache folders and files:
  - `z_core/{prefix}_zcore.npz`
  - `significant_traces/{prefix}_significant_traces.npz`
  - `filtered_neurons_by_stimuli/{prefix}_kept_neuron_indices.npy`
  - `merged_dFoF/` for merge-side plot exports and sort-order CSVs written during validation notebooks
- `src/data_loading.py` is the loader-side authority for how these outputs are discovered and reused.

## Stimulus authoring outputs
- Stimulus asset writers live in `scripts/stimuli/`.
- Canonical generated assets include:
  - `*_trajectory.csv`
  - `parameters/experiment_parameters.csv`
  - `parameters/total_time_sec.csv`
  - package or mapping JSON files kept alongside the generating scripts
- Playback runs can write:
  - `stimulus_timing_log.csv`
- Inspection figures from `plots_stimuli.ipynb` are notebook outputs, not canonical writer-stage contracts.

## Timing ownership
- `src/stimuli_timeline.py` is the authority for turning trajectory CSVs and block logs into downstream timing semantics.
- Downstream notebooks should consume these timing semantics, not redefine movement onset or duration logic locally.
- Stimulus generator helpers may compute packaging metadata such as total duration, but that metadata does not supersede `src/stimuli_timeline.py` as downstream timing authority.

## Writer-stage rules
- Change file naming or folder layout only when the task explicitly requires a migration.
- When a writer-stage contract changes, verify the first downstream consumer immediately after the writer, not just the writer itself.
- For calcium analysis, the first downstream consumer is often `src/data_loading.py` or a notebook that loads the written cache.
- Only writer-stage filenames and folders listed in this file are authoritative downstream contracts. Notebook or script convenience helpers are not canonical outputs unless a later slice promotes them.

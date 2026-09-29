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
- The standalone `__main__` block in `src/dff_extraction.py` writes unprefixed example files. Those names are not the batch notebook's per-plane output contract; review that standalone path during S10 of `src-refactor-roadmap.md`.
- Experiment-level merge writer stage: `DeltaFF_batch_pipeline.ipynb`.
- Canonical merged folder:
  - `03_analysis/functional/suite2P/merged_dFoF`
- Canonical merged files:
  - `{prefix}_dFoF_merged.npy`
  - `{prefix}_dFoF_merged_filtered_roi_indices.npy`
  - `{prefix}_dFoF_merged_map.csv`
- Loader-side authority for discovering merged outputs and compatibility variants:
  - `src/data_loading.py`
- Preprocessing notebook helpers that search for experiments, planes, or per-plane files are wrapper logic, not canonical output contracts.

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
- Stimulus asset writers live in `scripts/stimuli/generation/`; their JSON configuration inputs live in `configs/stimuli/`.
- Canonical generated assets include:
  - `*_trajectory.csv`
  - `parameters/experiment_parameters.csv`
  - `parameters/total_time_sec.csv`
  - package or mapping JSON files kept alongside the generating scripts
- Playback runs can write:
  - `stimulus_timing_log.csv`
- Inspection figures from `notebooks/stimuli/plots_stimuli_batch.ipynb` are notebook outputs, not canonical writer-stage contracts.
- The batch inspection report is saved by `src/stimulus_visualization.py` under the configured derived root, `<experiment>/stimuli/`: `stimuli_standard.png`, `stimuli_polar.png` (300 dpi), and `stimuli_summary.csv`. Reruns replace these report files. The CSV contains `stimulus_name`, `unique_angles_deg` (JSON lists), `n_unique_angles`, and `n_dots`; its visible spatial directions are rounded to 0.1 degree, whereas dot count includes all recorded identities. These reports do not alter generated trajectory or timing contracts.

## Timing ownership
- `src/stimuli_timeline.py` is the authority for turning trajectory CSVs and block logs into downstream timing semantics.
- Downstream notebooks should consume these timing semantics, not redefine movement onset or duration logic locally.
- Stimulus generator helpers may compute packaging metadata such as total duration, but that metadata does not supersede `src/stimuli_timeline.py` as downstream timing authority.

## Writer-stage rules
- Change file naming or folder layout only when the task explicitly requires a migration.
- When a writer-stage contract changes, verify the first downstream consumer immediately after the writer, not just the writer itself.
- For calcium analysis, the first downstream consumer is often `src/data_loading.py` or a notebook that loads the written cache.
- Only writer-stage filenames and folders listed in this file are authoritative downstream contracts. Notebook or script convenience helpers are not canonical outputs unless a later slice promotes them.

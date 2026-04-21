# Calcium Analysis Stage Map

Purpose

Provide the ordered flow used by the calcium-analysis notebooks that load cached experiment artifacts, align traces to stimuli, classify responses, and generate figures.

Use this file when

- You need to place a bug in loading, alignment, filtering, classification, or plotting.
- You need to know which analysis-stage cache is authoritative.
- You need the smallest notebook rerun surface after editing an analysis owner.

## End-to-end stages
1. Select fish, experiment, paths, stimulus blocks, and style dictionaries in the target notebook.
2. Load the experiment bundle with `load_2p_experiment`, including dFoF, paths, stimulus durations, logs, traces, and optional cached outputs.
3. Build trial-aligned traces for continuous dFoF and any cached binary or normalized matrices.
4. Compute derived outputs such as reliability filters, z-scored traces, significant rasters, response classes, or left-right metrics.
5. Plot single-fish outputs such as sorted chunks, mean traces, grouped rasters, and anatomy-linked views.
6. Build all-fish flattened matrices and comparative plots in the several-fish notebooks.
7. Save or reuse caches in `03_analysis/functional/plots/...` as the notebook workflow requires.

## Key outputs by stage
- Stage 2 returns `stimuli_trace_60`, `stimuli_table`, `stimuli_id_map`, `paths`, and optional cache arrays.
- Stage 4 writes reusable caches such as `{prefix}_zcore.npz`, `{prefix}_significant_traces.npz`, and `{prefix}_kept_neuron_indices.npy`.
- Stages 5 to 6 write plot PNGs and notebook-level summaries, often under `03_analysis/functional/plots/`.

## Concept ownership
- `src/data_loading.py` owns experiment-bundle assembly and cache lookup.
- `src/stimuli_timeline.py` owns timing extraction and stimulus-trace construction used by loaders and plots.
- `src/analysis_tools.py` owns trial alignment, filtering, classification, and derived metrics.
- `src/significant_traces.py` owns the noise-model and rasterization pipeline.
- `src/plotting.py` owns reusable raster, chunk, and mean-trace figure construction.
- Analysis notebooks own configuration, orchestration, and interpretation around those shared helpers.

## Navigation notes
- Read `current-state.md` before trusting older notebook-local helpers or legacy imports.
- Read `canonical-outputs.md` for cache names and downstream file contracts.
- Read `symbol-index.md` if you need the public callable surface first.

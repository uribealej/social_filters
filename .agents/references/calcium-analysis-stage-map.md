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
- `src/analysis_tools.py` is the compatibility facade for narrow S06 owners: `trial_alignment`, `response_metrics`, `motion_metrics`, `reliability`, `response_classification`, `response_selectivity`, and `response_normalization`.
- `src/multifish_matrices.py` owns pure multi-fish matrix and response-matrix construction from already trial-aligned per-fish traces. Other S07 scientific families live in `src/static_flicker_analysis.py`, `src/bout_flicker_analysis.py`, `src/stimulus_specificity.py`, `src/stimulus_similarity.py`, and `src/active_neuron_analysis.py`; `src/multifish_analysis.py` is their compatibility facade.
- `src/significant_trace_detection.py` owns current and explicit legacy detector modes plus raster cleanup and diagnostic plotting; the versioned modules are compatibility facades only.
- The `src/plotting_*` modules own figure construction by family; `src/plotting.py` is their compatibility facade. `src/neuron_ordering.py` and `src/stimuli_timeline.py` own reusable preparation used by those figures.
- The shared single-fish notebook imports `trial_alignment`, `response_normalization`, `response_metrics`, `reliability`, `plotting_common`, `plotting_single_fish`, and `plotting_all_fish` directly. Its loader and significant-trace imports remain on their established public owners.
- The shared several-fish notebook and Exp 1, 5, and 8 all-fish raster notebooks import their scientific, plotting, cohort-loading, diagnostics, figure, and reporting owners directly. Their configured fish/stimulus order, response-row alignment, report folders, and output stages remain unchanged.
- Analysis notebooks own configuration, orchestration, and interpretation around those shared helpers.

## Navigation notes
- Consult `current-state.md` for a known loader caveat or suspected notebook migration drift.
- Consult `canonical-outputs.md` when cache names or downstream file contracts matter.
- Consult `symbol-index.md` when locating a helper or checking its stable public import path.

# Symbol Index

Purpose

List the stable callable surface that notebooks and scripts already use so future changes start from the real owner modules instead of copying helpers into notebooks.

Use this file when

- A task mentions a helper name but not its owner.
- You need to decide whether a notebook cell should call an existing `src/` function.
- You need the smallest entrypoint into shared repo logic.

## Trusted implementation patterns
- Import shared helpers from `src.*` in notebooks and scripts.
- Keep dFoF and raster arrays in the owner module conventions already used by the repo, usually `(T, N)` until a plotting helper intentionally transposes.
- Treat notebook-local copies of shared helpers as legacy unless the repo clearly moved authority back into the notebook.

## Public surface by module

### `src/dff_extraction.py`
- `process_suite2p_fluorescence` - end-to-end per-plane Suite2p to filtered dFoF extraction and retained ROI indices.

### `src/auxtrigger_extraction.py`
- `extract_aux_trigger_frames` - parse ScanImage TIFF metadata to recover aux-trigger events by frame.

### `src/data_loading.py`
- `load_2p_experiment` - high-level experiment bundle loader that assembles dFoF, cache lookups, paths, stimulus traces, and plane metadata.

### `src/stimuli_timeline.py`
- `get_motion_timing_simple` - derive timing from trajectory CSVs using x, y, and radius changes.
- `make_stimulus_traces_2` - convert experiment logs plus stimulus durations into the numeric stimulus trace and table used downstream.
- `extract_stimulus_chunks` - extract aligned chunks for sorted raster-style plots.

### `src/analysis_tools.py`
- `build_trial_aligned_traces` - build trial windows keyed by stimulus id.
- `filter_neurons_by_trial_reliability` - keep neurons based on per-stimulus trial-to-trial reliability.
- `classify_responses_from_raster` - classify evoked and tardive responses from binary rasters and onset tables.
- `compute_left_right_index` - compute left-right preference metrics from mean traces.
- `build_neuron_order_groupwise_onset` - derive onset-based neuron ordering across response groups.
- `zscore_dfof_from_prestim_baseline` - z-score dFoF using pre-stimulus baselines.

### `src/significant_traces.py`
- `compute_noise_model_romano_fast_modular` - build centered dFoF, significance maps, and event rasters from the Romano-style noise model.

### `src/plotting.py`
- `plot_sorted_chunks_single_mode` - build and plot one sorted stimulus-chunk raster.
- `plot_stimulus_means` - plot per-stimulus mean traces with style dictionaries and optional save behavior.
- `plot_allfish_flat_raster` - render flattened multi-fish matrices with stimulus movement markers.

## Ownership notes
- If a notebook calls one of the symbols above, start in the owning module before editing the notebook.
- If a task changes any public helper signature or return contract, update this file and the relevant stage map in the same change.

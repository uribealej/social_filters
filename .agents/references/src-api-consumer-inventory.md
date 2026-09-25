# `src` API and Consumer Inventory

Purpose

Record the active notebook/script consumers of `src` and the compatibility contracts that later refactor slices must preserve.

Use this file when

- Moving or renaming a public helper.
- Deciding whether a compatibility facade is required.
- Checking whether a function is active, internal, documented-but-uncalled, or a retirement candidate.
- Selecting the first downstream consumer for validation.

This inventory was established on 2026-09-25 during Step 2 of `src-refactor-roadmap.md`. The exact machine-readable contract is `tests/contracts/src_api_consumers.json`, and `tests/contracts/test_src_api_consumers.py` fails when an active consumer or referenced symbol changes without review.

## Classification language

- **External-active:** referenced by an active notebook or script. Preserve its import path and observable contract until consumers migrate.
- **Internal-active:** called by another `src` module or by another helper in its owner module. It may move internally, but callers must migrate in the same slice.
- **Active facade candidate:** external-active but implemented in a large mixed-responsibility module scheduled to become a compatibility facade.
- **Documented, currently uncalled:** listed in `symbol-index.md` but not called by an active notebook/script. It is not dead merely because it is currently uncalled.
- **Deprecation review candidate:** duplicated, accidentally re-exported, or unreferenced code that may be retired only through Step 11 evidence.
- **Private implementation:** underscore-prefixed helper or a helper deliberately excluded from the stable notebook API.

## Audit coverage

- 12 active notebooks were scanned cell-by-cell after IPython syntax transformation.
- 4 active Python scripts under `scripts/` were scanned with the Python AST.
- 11 notebooks currently import or call `src`.
- `notebooks/calcium/preprocessing/2P_Experiment_FileOps.ipynb` currently has no `src` import.
- None of the 4 stimulus generation/playback scripts currently imports `src`.
- Raw notebook cell index 2 in `notebooks/calcium/exp_08_accumulation_ev/02_bout_flicker_position_onset.ipynb` remains the one registered syntax failure. Imports and uses from its other cells are included.
- Files under `archive/` and `.agents/history/` were excluded.

## Consumer overview

| Consumer | Active `src` owners | Compatibility role |
| --- | --- | --- |
| Exp 1 all-fish raster | `plotting`, `reusable_several_fish` | Shared cohort loading, raster/mean figures, diagnostics, report saving |
| Exp 1 bout/flicker position | `multifish_analysis`, `plotting`, `reusable_several_fish` | Bout-position analysis and Cell 06-style figure |
| Exp 1 static/flicker recruitment | `analysis_tools`, `multifish_analysis`, `plotting`, `reusable_several_fish` | Recruitment calculation, validation, statistics, and figures |
| Exp 5 all-fish raster | `analysis_tools`, `multifish_analysis`, `plotting`, `reusable_several_fish` | Response matrices, active decisions, pooled diagnostics, standard reports |
| Exp 5 LME decomposition | `analysis_tools`, `data_loading`, `lme_feature_decomposition`, `multifish_analysis`, `plotting` | Loader/alignment, response matrices, model fitting, and figures |
| Exp 8 all-fish raster | `analysis_tools`, `multifish_analysis`, `plotting`, `reusable_several_fish` | Response selection, response matrices, diagnostics, standard reports |
| Exp 8 bout/flicker position | `multifish_analysis`, `plotting`, `reusable_several_fish` | Same public bout-position surface as Exp 1 |
| dFoF batch preprocessing | `dff_extraction` | Per-plane Suite2p-to-dFoF extraction entrypoint |
| Shared single-fish analysis | `analysis_tools`, `data_loading`, `plotting`, `significant_traces`, `significant_traces_v2` | Load, align, filter, detect significant traces, and plot |
| Shared several-fish analysis | `analysis_tools`, `data_loading`, `multifish_analysis`, `plotting`, `reusable_several_fish` | Reusable pooled matrices, selection, overlap, diagnostics, and figures |
| Stimulus batch plots | `stimulus_visualization` | Load trajectories, summarize, plot, and save reports |

The exact per-consumer function list lives in the machine contract rather than being duplicated here.

## External-active surface by owner

These symbols are directly referenced by active notebooks.

### `src.analysis_tools`

- `build_trial_aligned_traces`
- `compute_motion_delta_integrals`
- `compute_motion_delta_peaks`
- `compute_response_window_frames`
- `compute_trial_mean_response_metrics`
- `filter_neurons_by_trial_reliability`
- `resolve_selected_stimuli`
- `validate_static_flicker_recruitment_result`
- `zscore_dfof_from_prestim_baseline`

Classification: active facade candidates. Preserve `src.analysis_tools.<name>` while Step 6 separates their implementations.

### `src.data_loading`

- `load_2p_experiment`
- `load_and_align_2p_experiment`

Classification: stable public gateways. Their internals may be split in Step 10, but their input/output contracts remain public.

### `src.dff_extraction`

- `process_suite2p_fluorescence`

Classification: stable preprocessing gateway.

### `src.lme_feature_decomposition`

- `build_lme_response_table`
- `fit_lme_models`
- `summarize_lme_model_results`
- `validate_lme_response_table`

Classification: stable public scientific workflow.

### `src.multifish_analysis`

- `build_active_neuron_matrices_all_fish`
- `build_bout_flicker_position_analysis`
- `build_matrix_all_fish`
- `build_pooled_active_trace_diagnostic`
- `build_static_flicker_recruitment_analysis`
- `build_stimulus_vector_similarity`
- `build_zscore_response_matrices_all_fish`
- `build_zscore_response_matrix_for_fish`
- `compute_static_flicker_fish_level_statistics`

Classification: active facade candidates. Preserve `src.multifish_analysis.<name>` while Step 7 separates scientific domains.

### `src.plotting`

- `add_stimuli_markers`
- `build_stimulus_style_maps`
- `list_stimulus_names`
- `plot_active_stimuli_histogram`
- `plot_active_trace_decision_diagnostic`
- `plot_allfish_flat_raster`
- `plot_bout_flicker_position_cell06_style`
- `plot_lme_model_outputs`
- `plot_motion_delta_distribution`
- `plot_pooled_static_flicker_category_proportions`
- `plot_preferred_stimulus_distribution`
- `plot_recruitment_amplification`
- `plot_shared_static_flicker_auc_summary`
- `plot_similarity_by_distance`
- `plot_similarity_heatmaps`
- `plot_sorted_chunks_single_mode`
- `plot_static_flicker_classification_raster`
- `plot_stimulus_means`
- `plot_stimulus_specificity_selectivity_index`
- `plot_stimulus_specificity_sparseness`
- `raster_with_stimuli`

Classification: active facade candidates. Preserve the current names while Step 8 moves figure families.

### `src.reusable_several_fish`

- `build_all_fish_raster_figure`
- `build_filtered_trial_aligned_traces_for_fish`
- `build_fish_keep_masks`
- `build_high_sparseness_raster_data`
- `build_overlap_diagnostic_data`
- `build_plot_all_fish_mean_zscore_traces`
- `build_pooled_mean_trace_by_stimulus`
- `build_response_window_validation`
- `build_selected_neuron_summary`
- `load_and_preflight_fish_raster_inputs`
- `plot_left_right_active_overlap_diagnostics`
- `plot_lifetime_sparseness_analysis`
- `plot_motion_active_neuron_counts`
- `resolve_stimulus_set`
- `save_analysis_report_run`

Classification: active facade candidates. Step 9 may separate workflow, reporting, data preparation, and plotting, but notebooks need compatibility during migration.

### Significant traces

- `src.significant_traces.clean_binary_raster_columns`
- `src.significant_traces.plot_dff_and_raster`
- `src.significant_traces_v2.compute_noise_model_romano_fast_modular`

Classification: active mixed-version contract. The shared single-fish notebook intentionally combines V1 cleanup/plotting with V2 detection. Preserve this until Steps 4-5 characterize and consolidate both implementations.

### `src.stimulus_visualization`

- `load_stimulus_trajectories`
- `plot_stimulus_positions`
- `plot_stimulus_traces`
- `save_stimulus_report`
- `summarize_stimuli`

Classification: stable public stimulus-inspection workflow.

## Important internal-active surface

The following are not necessarily direct notebook calls, but active `src` code depends on them:

- Timing: `get_angles_from_positions`, `get_motion_timing_simple`, `make_stimulus_traces_2`, and `extract_stimulus_chunks`.
- Analysis primitives: `find_file_with_suffix`, `compute_response_pair_index`, `build_response_index_keep_mask`, `compute_zscore_response_auc`, `compute_trial_auc_by_neuron`, `compute_static_flicker_trial_metrics`, `compute_stimulus_selectivity_metrics`, `classify_stimulus_specificity_neuron`, and `build_active_neuron_matrix_from_trial_raster`.
- Multifish composition: `combine_reps_one_stim`, `build_matrix_for_fish`, `build_neuron_stimulus_summary_table`, `add_selectivity_metrics_to_summary_table`, `resolve_segment_labels`, `compute_active_neuron_jaccard_overlap`, and `build_active_neuron_overlap_matrices_all_fish`.
- Several-fish reporting: `export_notebook_report` and `resolve_response_control_columns`.
- V1 comparison path: `significant_traces_v2.compare_significant_trace_versions` imports the V1 `compute_noise_model_romano_fast_modular` implementation.

Internal-active helpers can move only with their immediate callers and focused tests.

## Documented but currently uncalled public helpers

These are listed in `symbol-index.md` but have no active notebook/script call and no confirmed active public workflow call at the time of this audit:

- `analysis_tools.classify_responses_from_raster`
- `analysis_tools.compute_left_right_index`
- `analysis_tools.build_neuron_order_groupwise_onset`
- `multifish_analysis.classify_stimulus_specificity_summary_table`
- `multifish_analysis.build_stimulus_specificity_neuron_order`
- `multifish_analysis.build_segment_selectivity_permutation_summary`
- `plotting.plot_stimulus_specificity_summary`
- `plotting.plot_static_flicker_category_proportions`
- `stimuli_timeline.transform_stimuli_duration` as a direct caller surface; its duplicate in `data_loading` is currently used instead and is scheduled for Step 3 consolidation.
- `auxtrigger_extraction.extract_aux_trigger_frames`; this remains an intentional documented preprocessing API despite having no active repository caller.

This classification does not authorize deletion. Step 11 requires stronger evidence and explicit compatibility review.

## Deprecation and boundary review candidates

- `data_loading.transform_stimuli_duration` duplicates the documented timing owner and is scheduled for Step 3 compatibility delegation.
- `multifish_analysis.compute_stimulus_selectivity_metrics` is an accidental re-export created by importing the function from `analysis_tools`; `reusable_several_fish.py` currently accesses it through `mfa`. That hidden coupling must be corrected deliberately, with tests, rather than disappearing during an import cleanup.
- `analysis_tools.inspect_obj`, `analysis_tools.plot_venn_3stim`, and other public-looking but undocumented helpers remain Step 11 review candidates.
- V1/V2 significant-trace names are not deprecation candidates until Step 4 establishes their behavioral differences.
- No `__all__` declarations were added in Step 2 because the current modules contain accidental and transitional exports. Public export lists should be introduced only after the facade boundaries are agreed.

## High-value compatibility contracts

### Array conventions

- Continuous dFoF and raster arrays normally use `(time, neurons)`.
- Trial-aligned arrays use `(neurons, time, trials)`.
- Pooled flat matrices use `(pooled neurons, concatenated stimulus time)`.
- Neuron order must remain aligned across traces, rasters, response matrices, decisions, metadata, and figures.

### `build_trial_aligned_traces`

Inputs are `(T, N)` continuous traces, a 60 Hz stimulus-ID trace, imaging FPS, and pre/post windows. It returns a dictionary containing:

- `cell_traces`: `(N, T)`;
- `stimuli_trace`: imaging-rate stimulus IDs;
- ordered `stimuli_ids` and `stimuli_names`;
- `trial_aligned_traces`: `stimulus_id -> (N, window, trials)`;
- `onsets_by_id`, `pre_frames`, `post_frames`, and `win_length`.

### Experiment loaders

`load_2p_experiment` returns a dictionary whose established keys include:

- base data: `dfof`, `fps_2p`, `frames_per_block`, `duration_2p_block_sec`;
- timing/log data: `stimuli_durations`, `adjusted_log`, `stimuli_trace_60`, `stimuli_table`, `stimuli_id_map`;
- locations: `paths`;
- optional caches/metadata: `dFoF_merged_map`, `z_traces`, `raster`, `deltaF_center`, `kept_neuron_indices`, `filtered_roi_indices`, `plane_ids`, `mean_imgs`, and `stat_per_plane`.

`load_and_align_2p_experiment` requires dFoF, raster, centered dFoF, and z-score caches. It returns the four aligned trace families plus stimulus identity, onsets, raw arrays, kept indices, durations, map data, and a nested `meta` dictionary. Several-fish helpers rely on these exact keys.

### Preprocessing gateway

`process_suite2p_fluorescence` reads Suite2p plane inputs and returns:

- dFoF with shape `(T, retained neurons)`;
- retained ROI indices relative to the full Suite2p ROI list.

Its scientific thresholds remain explicit parameters. Output filenames are owned by the calling preprocessing notebook, not this function.

### Multifish matrices

- `build_matrix_all_fish` returns `(sum selected neurons, sum stimulus-block time)` and must preserve fish order, neuron order, and configured stimulus order.
- `build_zscore_response_matrices_all_fish` returns `response_matrices`, `pooled_response_matrix`, `row_metadata`, `selected_stimulus_ids`, and `selected_stimulus_labels`. `row_metadata` columns are `fish_id`, `neuron_id`, `source_neuron_id`, and `global_neuron_id`.
- `build_active_neuron_matrices_all_fish` returns `fish_id -> neuron-by-stimulus active matrix`; with `return_counts=True`, it returns `(active_matrices, active_counts)`.
- `build_pooled_active_trace_diagnostic` returns `trace_matrix`, `decision_matrix`, `row_metadata`, `stim_order`, `stim_labels`, block width/time/repetition metadata, and `combine_mode`.

### Several-fish workflow gateways

`load_and_preflight_fish_raster_inputs` returns settings, timing, fish/stimulus order, resolved paths, `all_fish_data`, reference-fish identity/data, aligned z-score traces, the stimulus map, durations, and names. It raises if any configured fish or stimulus is unavailable rather than silently producing a partial cohort.

`build_all_fish_raster_figure` returns a dictionary containing the flat matrix, optional left-right index, neuron order, figure, axes, and image.

`save_analysis_report_run` is intentionally stateful. When enabled it creates a timestamped report folder containing settings JSON, report-settings JSON, run metadata JSON, comments Markdown, optional CSV tables, and an optional notebook export. It returns `run_id`, `run_dir`, saved table paths, and export status; when saving is disabled it returns `None`.

### Significant-trace detector

The active V2 detector accepts `(T, N)` dFoF and returns an eight-item tuple:

1. significance map;
2. centered dFoF;
3. data density;
4. noise density;
5. x grid;
6. y grid;
7. `(T, N)` binary raster;
8. joint constrained significance map.

With `return_diagnostics=True`, a ninth diagnostics dictionary is appended. Tuple order is a compatibility contract until an additive named-result API is introduced.

### Stimulus visualization

- `load_stimulus_trajectories` returns a tidy table with `stimulus_name`, `dot_id`, `time_s`, `angle_deg`, `dot_size`, and `visible`.
- `summarize_stimuli` returns `stimulus_name`, `unique_angles_deg`, `n_unique_angles`, and `n_dots` in stable column order.
- Plot helpers return Matplotlib figures without saving.
- `save_stimulus_report` overwrites `stimuli_standard.png`, `stimuli_polar.png`, and `stimuli_summary.csv` under the supplied output directory and returns that directory.

### Plotting contracts

Active plotting helpers must preserve configured stimulus order, neuron row alignment, return object structure, and existing optional save behavior. Moving a plotting family requires figure rendering and screenshot inspection. The current environment's Matplotlib native render failure remains documented in `tests/README.md` and must be repaired or replaced by a working validation environment before Step 8 can close.

## Change protocol

When an active consumer changes:

1. Update `tests/contracts/src_api_consumers.json` only after reviewing the change.
2. Update this reference if classification, owner, shape, keys, side effects, or output paths changed.
3. Update `symbol-index.md` when the stable public surface changed.
4. Validate the owner and the first downstream consumer.
5. Preserve an old import through a facade until the planned migration slice closes.

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
- **Active compatibility facade:** external-active import path whose implementation has moved to a narrower owner. Preserve the import until consumers migrate.
- **Retained compatibility facade:** historical public import path with no active notebook/script caller; keep its documented exports until the compatibility decision is recorded.
- **Active facade candidate:** external-active mixed-responsibility module whose implementation is scheduled to move.
- **Documented, currently uncalled:** listed in `symbol-index.md` but not called by an active notebook/script. It is not dead merely because it is currently uncalled.
- **Deprecation review candidate:** duplicated, accidentally re-exported, or unreferenced code requiring per-symbol evidence and a documented compatibility decision before retirement.
- **Private implementation:** underscore-prefixed helper or a helper deliberately excluded from the stable notebook API.

## Audit coverage

- 12 active notebooks were scanned cell-by-cell after IPython syntax transformation.
- 4 active Python scripts under `scripts/` were scanned with the Python AST.
- 11 notebooks currently import or call `src`.
- `notebooks/calcium/preprocessing/2P_Experiment_FileOps.ipynb` currently has no `src` import.
- None of the 4 stimulus generation/playback scripts currently imports `src`.
- All active notebook code cells now compile after S12 step 3 corrected the pre-existing syntax error in raw cell 2 of `notebooks/calcium/exp_08_accumulation_ev/02_bout_flicker_position_onset.ipynb`.
- Files under `archive/` and `.agents/history/` were excluded.

S12 step 4 checked the remaining preprocessing and stimulus consumers against
their owner and output stages. The dFoF batch notebook already calls
`src.dff_extraction.process_suite2p_fluorescence`; its import guidance now
points only to this repository. The file-ops notebook has no reusable `src`
call. The stimulus inspection notebook calls `src.stimulus_visualization`
directly. The three trajectory generators and playback script intentionally
remain experiment-specific asset wrappers without `src` imports; downstream
timing interpretation belongs to `src.stimuli_timeline`. No consumer import or
machine-contract symbol changed in this audit.

## Consumer overview

| Consumer | Active `src` imports | Compatibility role |
| --- | --- | --- |
| Exp 1 all-fish raster | `plotting_common`, `several_fish_figures`, `several_fish_loading`, `several_fish_reporting` | Shared cohort loading, raster/mean figures, diagnostics, report saving |
| Exp 1 bout/flicker position | `bout_flicker_analysis`, `plotting_bout_position`, `several_fish_loading` | Bout-position analysis and Cell 06-style figure |
| Exp 1 static/flicker recruitment | `plotting_static_flicker`, `response_metrics`, `several_fish_loading`, `several_fish_reporting`, `static_flicker_analysis` | Recruitment calculation, validation, statistics, and figures |
| Exp 5 all-fish raster | `active_neuron_analysis`, `multifish_matrices`, `plotting_common`, `plotting_diagnostics`, `several_fish_figures`, `several_fish_loading`, `several_fish_reporting`, `trial_alignment` | Response matrices, active decisions, pooled diagnostics, standard reports |
| Exp 5 LME decomposition | `data_loading`, `lme_feature_decomposition`, `multifish_matrices`, `plotting_lme`, `trial_alignment` | Loader/alignment, response matrices, model fitting, and figures |
| Exp 8 all-fish raster | `multifish_matrices`, `plotting_common`, `plotting_diagnostics`, `several_fish_figures`, `several_fish_loading`, `several_fish_reporting`, `trial_alignment` | Response selection, response matrices, diagnostics, standard reports |
| Exp 8 bout/flicker position | `bout_flicker_analysis`, `plotting_bout_position`, `several_fish_loading` | Same public bout-position surface as Exp 1 |
| dFoF batch preprocessing | `dff_extraction` | Per-plane Suite2p-to-dFoF extraction entrypoint |
| Shared single-fish analysis | `data_loading`, `trial_alignment`, `response_normalization`, `response_metrics`, `reliability`, `plotting_common`, `plotting_single_fish`, `plotting_all_fish`, `significant_trace_detection` | Load, align, filter, detect significant traces, and plot through the owning modules |
| Shared several-fish analysis | `active_neuron_analysis`, `data_loading`, `motion_metrics`, `multifish_matrices`, `plotting_all_fish`, `plotting_common`, `plotting_diagnostics`, `plotting_specificity`, `several_fish_diagnostics`, `several_fish_reporting`, `several_fish_selection`, `stimulus_similarity` | Reusable pooled matrices, selection, overlap, diagnostics, and figures |
| Stimulus batch plots | `stimulus_visualization` | Load trajectories, summarize, plot, and save reports |

The exact per-consumer function list lives in the machine contract rather than being duplicated here.

## External-active surface by import module

Canonical owner sections list symbols directly referenced by active notebooks. The facade sections record compatibility paths retained after consumer migration; they have no active notebook or script import.

### `src.analysis_tools`

Classification: retained compatibility facade. S06 moved implementations to responsibility-specific owners while preserving historical `src.analysis_tools.<name>` imports and signatures. No active notebook or script imports this module after S12 step 3. Its supported export set awaits the step 5 compatibility decision; see `symbol-index.md` for the documented surface.

### Single-fish analysis owners

- `src.trial_alignment.build_trial_aligned_traces`, `resolve_selected_stimuli`
- `src.response_normalization.zscore_dfof_from_prestim_baseline`
- `src.response_metrics.compute_trial_mean_response_metrics`
- `src.reliability.filter_neurons_by_trial_reliability`

Classification: canonical owner imports used by the shared single-fish notebook after its S12 migration. The function arguments, return values, and notebook stage order are unchanged.

### `src.data_loading`

- `load_2p_experiment`
- `load_and_align_2p_experiment`

Classification: stable public gateways. S10 separated loader internals into private stages; their input/output contracts remain public.

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

Classification: retained compatibility facade. S07 moved implementations to scientific-domain owners. No active notebook or script imports this module after S12 step 3. Its supported export set awaits the step 5 compatibility decision.

### Several-fish scientific owners

- `src.motion_metrics` - motion-minus-fixed peak and integral calculations in the shared several-fish notebook.
- `src.multifish_matrices` - pooled matrices and per-fish or pooled z-score response matrices.
- `src.active_neuron_analysis` - active-neuron matrices and pooled trace diagnostics.
- `src.stimulus_similarity` - selected stimulus-vector similarity.
- `src.trial_alignment` - response selection and response-window frames in the cohort notebooks.

Classification: canonical owner imports used by the shared several-fish and all-fish raster notebooks after their S12 migration. The response-row, fish, and stimulus ordering contracts above remain in force.

### Specialized analysis owners

- `src.bout_flicker_analysis.build_bout_flicker_position_analysis` - pooled bout-position matching and ordered responses for Exp 1 and Exp 8.
- `src.static_flicker_analysis.build_static_flicker_recruitment_analysis`, `compute_static_flicker_fish_level_statistics` - Exp 1 recruitment summaries and fish-level inference.
- `src.response_metrics.validate_static_flicker_recruitment_result` - per-fish result validation.
- `src.multifish_matrices.build_zscore_response_matrices_all_fish` and `src.trial_alignment` - Exp 5 LME input preparation.

Classification: canonical owner imports used by the specialized notebooks after S12 step 3. Exp 8 bout/flicker cell 2's pre-existing syntax error and duplicate load were corrected without changing its block or timing settings; those settings still need a data-backed comparison with the maintained Exp 8 raster workflow.

### `src.plotting`

Classification: retained compatibility facade. S08 moved implementations to figure-family owners. No active notebook or script imports this module after S12 step 3. Its supported export set awaits the step 5 compatibility decision.

### Single-fish plotting owners

- `src.plotting_common.list_stimulus_names`, `build_stimulus_style_maps`
- `src.plotting_single_fish.raster_with_stimuli`, `add_stimuli_markers`, `plot_sorted_chunks_single_mode`
- `src.plotting_all_fish.plot_stimulus_means`

Classification: canonical owner imports used by the shared single-fish notebook after its S12 migration. The plotted data, figure arguments, and optional save behavior are unchanged.

### Several-fish plotting owners

- `src.plotting_common` - stimulus style maps.
- `src.plotting_all_fish` - flat pooled rasters and stimulus means.
- `src.plotting_diagnostics` - active-trace decisions and motion-delta figures.
- `src.plotting_specificity` - similarity and stimulus-specificity figures.

Classification: canonical owner imports used by the shared several-fish and all-fish raster notebooks after their S12 migration. Figure arguments, row ordering, and optional save behavior are unchanged.

### Specialized plotting owners

- `src.plotting_bout_position.plot_bout_flicker_position_cell06_style` - Exp 1 and Exp 8 bout-position figures.
- `src.plotting_static_flicker` - Exp 1 category, shared AUC, and recruitment figures.
- `src.plotting_lme.plot_lme_model_outputs` - Exp 5 model figures.

Classification: canonical owner imports used by the specialized notebooks after S12 step 3. Figure call arguments and configured save behavior are unchanged.

### `src.reusable_several_fish`

Classification: retained import-only compatibility facade after S09. No active notebook or script imports this module after S12 step 3. Its supported export set awaits the step 5 compatibility decision.

The migrated notebooks now import `src.several_fish_loading`, `src.several_fish_selection`, `src.several_fish_diagnostics`, `src.several_fish_figures`, and `src.several_fish_reporting` as their calls require. The canonical file, table, and figure contracts below are unchanged.

S09 step 1: `save_analysis_report_run` and `export_notebook_report` now belong to `src.several_fish_reporting`; `src.reusable_several_fish` re-exports both for active notebook callers. The report folder and return contracts below are unchanged.

S09 step 2: `load_and_preflight_fish_raster_inputs` now belongs to `src.several_fish_loading`; `src.reusable_several_fish` re-exports it for active notebook callers. The returned bundle, path choices, fish/stimulus order, and all-or-error contract below are unchanged.

S09 step 3: pure stimulus/control selection, response summary and keep-mask preparation, trace filtering, and active-matrix row/column selection now belong to `src.several_fish_selection`. `src.reusable_several_fish` re-exports the notebook-facing names. Pooled response-row ordering and summary return keys are unchanged.

S09 step 4: response-window tables, high-sparseness raster inputs, mean traces, and overlap-diagnostic preparation now belong to `src.several_fish_diagnostics`. `src.reusable_several_fish` re-exports the notebook-facing names and former diagnostic subhelpers. Plotting and report dependencies are absent from the diagnostic owner; prepared arrays and row order are unchanged.

S09 step 5: pooled raster inputs, lifetime-sparseness summaries, and motion active-count tables now also belong to `src.several_fish_diagnostics`. The five high-level figure functions belong to `src.several_fish_figures`; `src.reusable_several_fish` re-exports them. The historical public signatures, returned structures, figure data, and ordering are unchanged. No active notebook import was migrated.

### Significant traces

- `src.significant_trace_detection.clean_binary_raster_columns`
- `src.significant_trace_detection.compute_noise_model_romano_fast_modular`
- `src.significant_trace_detection.plot_dff_and_raster`

Classification: stable canonical surface. The shared single-fish notebook uses the current-mode detector, cleanup, and plotting helpers from one owner. Historical `src.significant_traces` and `src.significant_traces_v2` paths remain compatibility facades but have no active notebook/script consumer.

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
- Analysis primitives retain their `analysis_tools` facade paths but are implemented by `analysis_io`, `response_selectivity`, `response_metrics`, and `response_classification`. Internal owner-to-owner dependencies now use the narrow modules rather than the facade.
- Multifish composition: `combine_reps_one_stim`, `build_matrix_for_fish`, `build_neuron_stimulus_summary_table`, `add_selectivity_metrics_to_summary_table`, `resolve_segment_labels`, `compute_active_neuron_jaccard_overlap`, and `build_active_neuron_overlap_matrices_all_fish`.
- Several-fish reporting: `export_notebook_report` and `resolve_response_control_columns`.
- Significant-trace comparison: `significant_trace_detection.compare_significant_trace_versions` calls the canonical detector in explicit legacy and current modes; the V2 facade delegates to that comparison helper.

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
- `stimuli_timeline.transform_stimuli_duration` as a direct notebook/script caller surface; it is now the canonical implementation used internally by `data_loading`.
- `auxtrigger_extraction.extract_aux_trigger_frames`; this remains an intentional documented preprocessing API despite having no active repository caller.

This classification does not authorize deletion. S11 retained these documented
helpers after its evidence review; any later change requires a new
compatibility decision. See the [S11 completion record](../history/src-refactor-s11.md).

## Deprecation and boundary review candidates

- `data_loading.transform_stimuli_duration` is a compatibility wrapper that delegates to the canonical timing owner; it contains no normalization logic.
- `multifish_analysis.compute_stimulus_selectivity_metrics` is a compatibility re-export from `response_selectivity`; S09 moved the several-fish calculation to `several_fish_diagnostics.py`, which imports `response_selectivity` directly. The `mfa` alias remains on `reusable_several_fish.py` for historical compatibility; review its export boundary during S12.
- `analysis_tools.inspect_obj`, `analysis_tools.plot_venn_3stim`, and other public-looking helpers were retained in S11 because explicit exports or characterization contracts leave external use uncertain. Review facade exposure during S12.
- `src.significant_traces` and `src.significant_traces_v2` are thin compatibility facades after S05. Their removal remains a later compatibility/deprecation decision; neither contains scientific array logic.
- `analysis_tools.py` now has an explicit `__all__`; other transitional facades still expose historical imports. Review their export boundaries during S12 rather than treating every imported name as permanent API.

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

### Several-fish selection

`build_selected_neuron_summary` returns `neuron_summary_table`, `neuron_summary_all_stimuli_table`, `left_right_index`, `plot_neuron_keep_mask`, `plot_neuron_keep_indices`, `resolved_preference_controls`, `selected_auc_response_matrix`, and `plot_filter_table`. The index, keep mask, selected responses, and filter table follow pooled response-row order; the summary-table owner sorts its rows by `global_neuron_id`. `build_fish_keep_masks` splits the pooled mask by fish without reordering rows; trace and active-matrix filtering then apply each fish's rows in their original order.

### Significant-trace detector

The canonical current-mode detector accepts `(T, N)` dFoF and returns an eight-item tuple:

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

Active plotting helpers must preserve configured stimulus order, neuron row alignment, return object structure, and existing optional save behavior. Moving a plotting family requires figure rendering and screenshot inspection. S08 is complete; `tests/README.md` documents the historical Matplotlib rendering failure and the current render probe.

## Change protocol

When an active consumer changes:

1. Update `tests/contracts/src_api_consumers.json` only after reviewing the change.
2. Update this reference if classification, owner, shape, keys, side effects, or output paths changed.
3. Update `symbol-index.md` when the stable public surface changed.
4. Validate the owner and the first downstream consumer.
5. Preserve an old import through a facade until the planned migration slice closes.

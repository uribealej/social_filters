# Symbol Index

Purpose

List the stable callable surface that notebooks and scripts already use so future changes start from the real owner modules instead of copying helpers into notebooks.

Use this file when

- A task mentions a helper name but not its owner.
- You need to decide whether a notebook cell should call an existing `src/` function.
- You need the smallest entrypoint into shared repo logic.

The active notebook/script evidence and compatibility classification for these symbols is recorded in `src-api-consumer-inventory.md`. The machine-readable consumer contract is `tests/contracts/src_api_consumers.json`.

## Trusted implementation patterns
- Import shared helpers from `src.*` in notebooks and scripts.
- Keep dFoF and raster arrays in the owner module conventions already used by the repo, usually `(T, N)` until a plotting helper intentionally transposes.
- Treat notebook-local copies of shared helpers as legacy unless the repo clearly moved authority back into the notebook.

## Public surface by module

### `src/dff_extraction.py`
- `process_suite2p_fluorescence` - end-to-end per-plane Suite2p to filtered dFoF extraction and retained ROI indices.

This is the documented notebook-facing gateway. The module's unprefixed
loading, baseline, and ROI-filter helpers remain callable for compatibility,
but have no active external repository consumer. S11 retained them because the
extraction gateway and S10 tests use them; review their public names during S12.

### `src/auxtrigger_extraction.py`
- `extract_aux_trigger_frames` - parse ScanImage TIFF metadata to recover aux-trigger events by frame.

`ScanImageTiffReader` is required when this function runs and is imported at
call time. The module itself must import without the TIFF reader installed;
missing readers raise an actionable `ImportError` from the function.

### `src/data_loading.py`
- `load_2p_experiment` - high-level experiment bundle loader that assembles dFoF, cache lookups, paths, stimulus traces, and plane metadata; verifies merged-map rows against dFoF neurons and accepts a matching timestamped fallback.
- `load_and_align_2p_experiment` - load one fish and build aligned dFoF, raster, normalized, and z-score traces for several-fish notebooks.

Compatibility entrypoint:
- `transform_stimuli_duration` - preserves the historical data-loading import path while delegating all normalization logic to `src.stimuli_timeline.transform_stimuli_duration`.

Not public here yet:
- preprocessing notebook helpers such as `ensure_dir`, `find_experiments`, `plane_dir` / `plane_dirs`, and `experiment_prefix` are still duplicated outside `src/` and should not be treated as stable callable API.
- per-plane lookup helpers such as `load_dfof_for_plane` and `load_filtered_indices_for_plane` are still notebook-local duplication backlog until a later extraction slice moves them into an owner module.

### `src/stimuli_timeline.py`
- `get_motion_timing_simple` - derive timing from trajectory CSVs using plain or dot-prefixed x, y, and radius changes.
- `transform_stimuli_duration` - canonical owner for normalizing extracted timing dictionaries into the downstream-facing timing contract.
- `get_angles_from_positions` - reverse rotated trajectory x/y coordinates into angle values for shared stimulus interpretation.
- `make_stimulus_traces_2` - convert experiment logs plus stimulus durations into the numeric stimulus trace and table used downstream.
- `extract_stimulus_chunks` - extract aligned chunks for sorted raster-style plots.

Boundary note:
- duplicated generator-side helpers such as `generate_circular_trajectory` are still backlog items under `scripts/stimuli/` and are not yet part of the public `src/` timing surface.

### `src/stimulus_visualization.py`
- `load_stimulus_trajectories` - load plain or multi-dot trajectory CSVs once into angle, dot-size, and visibility samples; uses the shared angle converter.
- `summarize_stimuli` - one row per stimulus with unique visible angles rounded to 0.1 degree in `[-180, 180)`, angle count, and recorded dot count.
- `plot_stimulus_traces` - per-stimulus angle/dot-size traces; unwrap angles only for plotting and retain off frames.
- `plot_stimulus_positions` - per-stimulus equal-marker polar panels using the summary's exact angle lists.
- `save_stimulus_report` - save two 300-dpi PNGs and a CSV with complete angle lists under the supplied output directory.

### `src/analysis_tools.py`
Supported compatibility facade: its explicit `__all__` callable surface remains available while new code imports the narrow owners listed below. See the [S12 compatibility decision](src-compatibility-decision.md).

- `build_trial_aligned_traces` - build trial windows keyed by stimulus id.
- `compute_trial_mean_response_metrics` - build per-stimulus trial-mean traces plus peak, AUC, and average response metrics.
- `resolve_selected_stimuli` - normalize ordered stimulus selections from names or IDs.
- `compute_response_window_frames` - compute clipped aligned-trace response-window frame indices.
- `compute_zscore_response_auc` - compute per-neuron mean z-score AUC across repetitions for one stimulus.
- `compute_trial_auc_by_neuron` - compute per-neuron/per-trial AUC values for one stimulus response window.
- `compute_static_flicker_trial_metrics` - build per-trial and per-position static--flicker AUC, activity, category, and raster-display data for one fish.
- `validate_static_flicker_recruitment_result` - smoke-check one fish's static--flicker windows, categories, and raster-display rows.
- `compute_response_pair_index` - compute a paired stimulus preference index from a neuron-by-stimulus response matrix.
- `build_response_index_keep_mask` - convert a paired response index into a reusable neuron keep mask.
- `compute_stimulus_selectivity_metrics` - compute raw-response preference, raw selectivity index, and non-negative selectivity metrics.
- `classify_stimulus_specificity_neuron` - classify one neuron from sparseness, breadth, and positive response strength.
- `filter_neurons_by_trial_reliability` - keep neurons based on per-stimulus trial-to-trial reliability.
- `classify_responses_from_raster` - classify evoked and tardive responses from binary rasters and onset tables.
- `compute_left_right_index` - compute left-right preference metrics from mean traces.
- `compute_motion_delta_integrals` - build tidy per-neuron/per-trial motion-minus-fixed integral metrics for selected stimuli.
- `compute_motion_delta_peaks` - build tidy per-neuron/per-trial motion-minus-fixed peak metrics for selected stimuli.
- `build_neuron_order_groupwise_onset` - derive onset-based neuron ordering across response groups.
- `build_active_neuron_matrix_from_trial_raster` - build a neuron-by-stimulus active-decision matrix from trial-aligned rasters.
- `find_file_with_suffix` - resolve a matching analysis file from a directory.
- `inspect_obj` - inspect a loaded object's type, shape, and available keys for interactive analysis.
- `plot_accepted_rejected_rasters` - show reliability-accepted and rejected neural traces.
- `plot_venn_3stim` - show three-stimulus response-set overlap.
- `zscore_dfof_from_prestim_baseline` - z-score dFoF using pre-stimulus baselines.

Canonical implementation owners after S06:

| Owner module | Responsibility |
| --- | --- |
| `src.trial_alignment` | Trial alignment, stimulus selection, timing lookup, and response-window frames |
| `src.response_metrics` | Trial means, AUCs, static--flicker trial metrics, and validation |
| `src.motion_metrics` | Motion-minus-fixed integral and peak metrics |
| `src.reliability` | Trial-to-trial reliability calculation, selection, and optional index saving |
| `src.response_classification` | Active-neuron matrices, raster response classes, left/right indices, and onset ordering |
| `src.response_selectivity` | Paired response indices and stimulus-selectivity metrics/classes |
| `src.response_normalization` | Pre-stimulus baseline z-scoring |
| `src.analysis_io` | File discovery and interactive object inspection |
| `src.plotting_reliability` | Reliability raster diagnostics and three-stimulus Venn figures; `src.plotting` remains the compatibility import |

### `src/trial_alignment.py`
- `build_trial_aligned_traces` - build trial windows keyed by stimulus id; used directly by the shared single-fish notebook.
- `resolve_selected_stimuli` - normalize ordered stimulus selections from names or IDs.
- `compute_response_window_frames` - compute clipped aligned-trace response-window frame indices.

### `src/response_normalization.py`
- `zscore_dfof_from_prestim_baseline` - z-score dFoF using pre-stimulus baselines.

### `src/response_metrics.py`
- `compute_trial_mean_response_metrics` - build per-stimulus trial-mean traces and response metrics.
- `validate_static_flicker_recruitment_result` - check one fish's static--flicker windows, categories, and raster rows.

### `src/reliability.py`
- `filter_neurons_by_trial_reliability` - select reliable neurons and optionally save their indices.

### `src/multifish_analysis.py`
Supported compatibility facade: S07 moved scientific implementations into `src.multifish_matrices`, `src.static_flicker_analysis`, `src.bout_flicker_analysis`, `src.stimulus_specificity`, `src.stimulus_similarity`, and `src.active_neuron_analysis`. Documented public functions and the characterized selectivity re-export remain available; active notebooks use the owners. See the [S12 compatibility decision](src-compatibility-decision.md).

- `build_bout_flicker_position_analysis` - build pooled bout-referenced flicker-position comparison data.
- `compute_static_flicker_fish_level_statistics` - compute fish-level static--flicker tests, confidence intervals, and Holm-adjusted p-values.
- `combine_reps_one_stim` - combine one stimulus' trial-aligned repetitions by concatenating time or averaging repeats.
- `build_matrix_for_fish` - concatenate selected stimulus blocks into a per-fish matrix with optional kept-neuron indexing.
- `build_matrix_all_fish` - stack per-fish matrices for dFoF, raster, or z-score trial-aligned traces.
- `build_zscore_response_matrix_for_fish` - build one fish's neuron-by-stimulus z-score AUC response matrix.
- `build_zscore_response_matrices_all_fish` - build per-fish and pooled z-score AUC response matrices plus row metadata.
- `build_neuron_stimulus_summary_table` - join response matrices, active decisions, and neuron identity metadata.
- `add_selectivity_metrics_to_summary_table` - add preference, sparseness, raw selectivity index, selectivity, and active-breadth metrics.
- `classify_stimulus_specificity_summary_table` - add adaptive neuron classes using configurable thresholds.
- `build_stimulus_specificity_neuron_order` - sort pooled rows by preferred stimulus, sparseness, and response strength.
- `build_stimulus_vector_similarity` - compute Pearson/cosine stimulus-vector similarity matrices and pairwise distance summaries.
- `resolve_segment_labels` - resolve short segment labels against selected stimulus labels.
- `build_segment_selectivity_permutation_summary` - compute pooled per-neuron segment-selectivity permutation summaries.
- `build_active_neuron_matrices_all_fish` - build one binary neuron-by-stimulus active matrix per fish.
- `compute_active_neuron_jaccard_overlap` - compute pairwise Jaccard overlap between active-neuron condition sets.
- `build_active_neuron_overlap_matrices_all_fish` - build pooled and mean-per-fish left/right active-neuron overlap matrices.
- `build_static_flicker_recruitment_analysis` - combine per-fish, per-position static--flicker metrics into category, shared-ΔAUC, and recruitment/amplification summaries.
- `build_pooled_active_trace_diagnostic` - pool binary trial-aligned traces and active decisions across fish for strictness diagnostics.
- `compute_stimulus_selectivity_metrics` - characterized compatibility re-export from `src.response_selectivity`.

### `src/motion_metrics.py`
- `compute_motion_delta_integrals` - build per-trial motion-minus-fixed integral metrics.
- `compute_motion_delta_peaks` - build per-trial motion-minus-fixed peak metrics.

### `src/multifish_matrices.py`
- `build_matrix_all_fish` - stack per-fish matrices in fish and stimulus order.
- `build_zscore_response_matrix_for_fish` - build one fish's neuron-by-stimulus response matrix.
- `build_zscore_response_matrices_all_fish` - build per-fish and pooled response matrices with row metadata.

### `src/active_neuron_analysis.py`
- `build_active_neuron_matrices_all_fish` - build per-fish active-neuron decisions.
- `build_pooled_active_trace_diagnostic` - align pooled binary traces with active decisions and row metadata.

### `src/stimulus_similarity.py`
- `build_stimulus_vector_similarity` - compute selected stimulus-vector similarity and pairwise summaries.

### `src/bout_flicker_analysis.py`
- `build_bout_flicker_position_analysis` - derive trajectory-matched bout/flicker position responses and neuron ordering.

### `src/static_flicker_analysis.py`
- `build_static_flicker_recruitment_analysis` - combine per-fish static--flicker responses into recruitment summaries.
- `compute_static_flicker_fish_level_statistics` - calculate fish-level comparisons and adjusted p-values.

### `src/several_fish_reporting.py`
- `save_analysis_report_run` - save a timestamped several-fish run folder with settings, comments, metadata, tables, and optional notebook export.
- `export_notebook_report` - export a saved notebook to a report folder through nbconvert.

### `src/several_fish_loading.py`
- `load_and_preflight_fish_raster_inputs` - load and validate an ordered several-fish raster cohort and return its notebook bundle.

### `src/several_fish_selection.py`
- `resolve_stimulus_set` - resolve ordered stimulus IDs or names against a reference fish.
- `resolve_response_control_columns` - resolve left/right controls to response-matrix columns.
- `build_selected_neuron_summary` - prepare row-aligned summary tables, preference index, keep mask, and selected responses.
- `build_fish_keep_masks` - split a pooled neuron keep mask into per-fish response-row masks.
- `build_filtered_trial_aligned_traces_for_fish` - select one fish's filtered neuron-by-time-by-trial traces.
- `subset_active_matrices` - subset per-fish active matrices to ordered stimulus columns.
- `filter_active_matrices_by_keep_mask` - apply a pooled keep mask to per-fish active-matrix rows.

### `src/several_fish_diagnostics.py`
- `build_all_fish_raster_inputs` - prepare pooled raster matrix, optional left-right index, and stable neuron ordering.
- `build_lifetime_sparseness_summary` - prepare selected response matrix, per-neuron selectivity metrics, and sparseness summary.
- `build_motion_active_counts` - prepare ordered per-fish active-neuron counts.
- `build_response_window_validation` - prepare ordered response-window rows across fish and stimuli.
- `build_high_sparseness_raster_data` - prepare a sorted high-sparseness raster matrix and row order.
- `build_pooled_mean_trace_by_stimulus` - prepare per-fish mean traces for each selected stimulus.
- `build_diagnostic_trace_source_data` - align z-score diagnostic traces to active-matrix rows.
- `apply_diagnostic_keep_mask` - filter pooled diagnostic trace, decision, and metadata rows.
- `build_decision_then_mean_sort_order` - order diagnostic rows by active decisions and mean trace.
- `build_metric_sort_order` - order diagnostic rows by a per-neuron summary metric.
- `build_overlap_diagnostic_data` - prepare overlap matrices and pooled trace diagnostics.

### `src/several_fish_figures.py`
- `build_all_fish_raster_figure` - orchestrate the pooled raster figure with configured labels and sorting.
- `build_plot_all_fish_mean_zscore_traces` - render mean z-score traces for an ordered stimulus set.
- `plot_left_right_active_overlap_diagnostics` - render left/right active-neuron overlap and raster diagnostics.
- `plot_motion_active_neuron_counts` - render ordered per-stimulus active counts.
- `plot_lifetime_sparseness_analysis` - render lifetime sparseness and high-sparseness raster figures.

### `src/reusable_several_fish.py`
Supported compatibility facade for the documented historical several-fish callables below; new code imports their owners directly. See the [S12 compatibility decision](src-compatibility-decision.md).

- `resolve_stimulus_set` - historical notebook import; implementation in `src/several_fish_selection.py`.
- `save_analysis_report_run` - historical notebook import; implementation in `src/several_fish_reporting.py`.
- `export_notebook_report` - historical notebook import; implementation in `src/several_fish_reporting.py`.
- `build_response_window_validation` - historical notebook import; implementation in `src/several_fish_diagnostics.py`.
- `resolve_response_control_columns` - historical import; implementation in `src/several_fish_selection.py`.
- `build_selected_neuron_summary` - historical notebook import; implementation in `src/several_fish_selection.py`.
- `build_high_sparseness_raster_data` - historical notebook import; implementation in `src/several_fish_diagnostics.py`.
- `build_pooled_mean_trace_by_stimulus` - historical notebook import; implementation in `src/several_fish_diagnostics.py`.
- `build_fish_keep_masks` - historical notebook import; implementation in `src/several_fish_selection.py`.
- `build_filtered_trial_aligned_traces_for_fish` - historical notebook import; implementation in `src/several_fish_selection.py`.
- `subset_active_matrices` - historical import; implementation in `src/several_fish_selection.py`.
- `filter_active_matrices_by_keep_mask` - historical import; implementation in `src/several_fish_selection.py`.
- `build_overlap_diagnostic_data` - historical notebook import; implementation in `src/several_fish_diagnostics.py`.
- `build_diagnostic_trace_source_data` - historical import; implementation in `src/several_fish_diagnostics.py`.
- `apply_diagnostic_keep_mask` - historical import; implementation in `src/several_fish_diagnostics.py`.
- `build_decision_then_mean_sort_order` - historical import; implementation in `src/several_fish_diagnostics.py`.
- `build_metric_sort_order` - historical import; implementation in `src/several_fish_diagnostics.py`.
- `load_and_preflight_fish_raster_inputs` - historical notebook import; implementation in `src/several_fish_loading.py`.
- `build_all_fish_raster_figure` - historical notebook import; implementation in `src/several_fish_figures.py`.
- `build_plot_all_fish_mean_zscore_traces` - historical notebook import; implementation in `src/several_fish_figures.py`.
- `plot_left_right_active_overlap_diagnostics` - historical notebook import; implementation in `src/several_fish_figures.py`.
- `plot_motion_active_neuron_counts` - historical notebook import; implementation in `src/several_fish_figures.py`.
- `plot_lifetime_sparseness_analysis` - historical notebook import; implementation in `src/several_fish_figures.py`.

### `src/lme_feature_decomposition.py`
- `build_lme_response_table` - convert per-fish neuron-by-stimulus response matrices plus editable stimulus metadata into a long LME response table.
- `validate_lme_response_table` - validate fish/neuron/stimulus row contracts, metadata labels, duplicates, missing responses, and response ranges while printing compact summaries.
- `fit_lme_models` - fit an editable dict/list of statsmodels mixed-effects model specs while reporting failures and continuing remaining fits.
- `summarize_lme_model_results` - extract fixed effects, random-effect variances, fit statistics, and model metadata into tidy result tables.

### `src/significant_trace_detection.py`
- `compute_noise_model_romano_fast_modular` - canonical Romano-style detector; current behavior is the default, while `mode="legacy"` reproduces V1 behavior through the same implementation.
- `clean_binary_raster_columns` - remove non-finite or zero-variance raster columns before correlation-based sorting.
- `plot_dff_and_raster` - plot centered dFoF and a binary raster with one shared neuron order.
- `compare_significant_trace_versions` - diagnostic comparison of canonical legacy and current modes.

Compatibility facades:
- `src.significant_traces` preserves historical V1 signatures and delegates differing stages to canonical `mode="legacy"`.
- `src.significant_traces_v2` preserves historical V2 signatures and delegates to canonical current mode.
- New analysis must import `src.significant_trace_detection`; the shared single-fish notebook has migrated to this owner.
- Both mode-specific paths remain supported under the [S12 compatibility decision](src-compatibility-decision.md).

### `src/plotting.py`
Supported compatibility facade: S08 moved figure construction into the `src.plotting_*` modules. Neuron ordering and timing preparation live in `src.neuron_ordering` and `src.stimuli_timeline`. Documented public figure functions remain available; active notebooks use the owners. See the [S12 compatibility decision](src-compatibility-decision.md).

- `add_stimuli_markers` - add stimulus timing markers to an existing axis.
- `list_stimulus_names` - discover stimulus names from `*_trajectory.*` files by stripping `_trajectory`.
- `build_stimulus_style_maps` - build reusable stimulus color and linestyle dictionaries from discovered stimulus names.
- `plot_similarity_heatmaps` - plot Pearson and cosine stimulus-vector similarity heatmaps.
- `plot_similarity_by_distance` - plot pairwise stimulus-vector similarity as a function of selected-order distance.
- `plot_motion_delta_distribution` - plot motion-minus-fixed metric distributions grouped by selected stimulus labels.
- `plot_active_trace_decision_diagnostic` - plot pooled binary trace diagnostics beside strict active-neuron decisions, with an optional active-count trace panel.
- `plot_stimulus_specificity_sparseness` - plot lifetime sparseness against maximum response strength.
- `plot_stimulus_specificity_selectivity_index` - plot raw selectivity index against maximum response strength.
- `plot_active_stimuli_histogram` - plot the distribution of active-stimulus counts.
- `plot_preferred_stimulus_distribution` - plot preferred-stimulus counts in selected stimulus order.
- `plot_stimulus_specificity_summary` - render the three stimulus-specificity summary plots together.
- `plot_lme_model_outputs` - render fixed-effect, AIC/BIC, observed-vs-fitted, and response-distribution plots for LME feature-decomposition results.
- `plot_sorted_chunks_single_mode` - build and plot one sorted stimulus-chunk raster.
- `plot_stimulus_means` - plot per-stimulus mean traces with style dictionaries and optional save behavior.
- `plot_allfish_flat_raster` - render flattened multi-fish matrices with stimulus movement markers and optional mean trace panels.
- `plot_static_flicker_classification_raster` - render independent left and right category-ordered static--flicker significant-raster figures.
- `plot_static_flicker_category_proportions` - render per-position, per-fish stacked category proportions.
- `plot_pooled_static_flicker_category_proportions` - render descriptive category proportions after pooling neurons across fish.
- `plot_shared_static_flicker_auc_summary` - render separate left/right shared-neuron AUC comparisons with per-stimulus distributions of per-fish mean ΔAUC.
- `plot_recruitment_amplification` - render per-position, per-fish recruitment and amplification components.
- `plot_bout_flicker_position_cell06_style` - render the active Cell 06-style bout/flicker position comparison figure.
- `raster_with_stimuli` - render a single-fish raster with stimulus timing annotations.

### `src/plotting_common.py`
- `list_stimulus_names` - discover stimulus names from trajectory files.
- `build_stimulus_style_maps` - build stimulus color and linestyle dictionaries.

### `src/plotting_single_fish.py`
- `add_stimuli_markers` - add stimulus timing markers to an existing axis.
- `raster_with_stimuli` - render a single-fish raster with stimulus timing annotations.
- `plot_sorted_chunks_single_mode` - build and plot one sorted stimulus-chunk raster.

### `src/plotting_all_fish.py`
- `plot_stimulus_means` - plot per-stimulus mean traces with optional save behavior.
- `plot_allfish_flat_raster` - render pooled flattened matrices with optional mean panels.

### `src/plotting_diagnostics.py`
- `plot_active_trace_decision_diagnostic` - plot pooled trace and strict active-decision diagnostics.
- `plot_motion_delta_distribution` - plot motion-minus-fixed metric distributions.

### `src/plotting_specificity.py`
- `plot_similarity_heatmaps` - plot Pearson and cosine stimulus-vector similarity heatmaps.
- `plot_similarity_by_distance` - plot selected-order distance summaries.
- `plot_stimulus_specificity_sparseness` - plot lifetime sparseness and response strength.
- `plot_stimulus_specificity_selectivity_index` - plot selectivity index and response strength.
- `plot_active_stimuli_histogram` - plot active-stimulus counts.
- `plot_preferred_stimulus_distribution` - plot preferred-stimulus counts in selected order.

### `src/plotting_bout_position.py`
- `plot_bout_flicker_position_cell06_style` - render the matched bout/flicker Cell 06-style figure.

### `src/plotting_static_flicker.py`
- `plot_static_flicker_classification_raster` - render independent left/right category-ordered rasters.
- `plot_pooled_static_flicker_category_proportions` - render pooled category proportions.
- `plot_shared_static_flicker_auc_summary` - render shared-neuron AUC comparisons.
- `plot_recruitment_amplification` - render recruitment and amplification components.

### `src/plotting_lme.py`
- `plot_lme_model_outputs` - render model fit and response-distribution figures.

## Ownership notes
- If a notebook calls one of the symbols above, start in its implementation owner before editing the notebook; inspect a facade only when its compatibility surface matters.
- If a public helper signature or return contract changes, update this file and the consumer inventory. Update a stage map or output document when its stage flow or file contract changes.

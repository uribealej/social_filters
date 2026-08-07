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
- Treat helpers listed in `duplicate-helper-inventory.md` as backlog items or notebook/script-local wrappers, not as part of the stable public surface until they are extracted into an owner module.

## Public surface by module

### `src/dff_extraction.py`
- `process_suite2p_fluorescence` - end-to-end per-plane Suite2p to filtered dFoF extraction and retained ROI indices.

### `src/auxtrigger_extraction.py`
- `extract_aux_trigger_frames` - parse ScanImage TIFF metadata to recover aux-trigger events by frame.

### `src/data_loading.py`
- `load_2p_experiment` - high-level experiment bundle loader that assembles dFoF, cache lookups, paths, stimulus traces, and plane metadata.
- `load_and_align_2p_experiment` - load one fish and build aligned dFoF, raster, normalized, and z-score traces for several-fish notebooks.

Not public here yet:
- preprocessing notebook helpers such as `ensure_dir`, `find_experiments`, `plane_dir` / `plane_dirs`, and `experiment_prefix` are still duplicated outside `src/` and should not be treated as stable callable API.
- per-plane lookup helpers such as `load_dfof_for_plane` and `load_filtered_indices_for_plane` are still notebook-local duplication backlog until a later extraction slice moves them into an owner module.

### `src/stimuli_timeline.py`
- `get_motion_timing_simple` - derive timing from trajectory CSVs using x, y, and radius changes.
- `transform_stimuli_duration` - normalize extracted timing dictionaries into the downstream-facing timing contract.
- `get_angles_from_positions` - reverse rotated trajectory x/y coordinates into angle values for shared stimulus interpretation.
- `make_stimulus_traces_2` - convert experiment logs plus stimulus durations into the numeric stimulus trace and table used downstream.
- `extract_stimulus_chunks` - extract aligned chunks for sorted raster-style plots.

Boundary note:
- duplicated generator-side helpers such as `generate_circular_trajectory` are still backlog items under `scripts/stimuli/` and are not yet part of the public `src/` timing surface.

### `src/analysis_tools.py`
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
- `zscore_dfof_from_prestim_baseline` - z-score dFoF using pre-stimulus baselines.

### `src/multifish_analysis.py`
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

### `src/reusable_several_fish.py`
- `resolve_stimulus_set` - resolve editable notebook stimulus sets against a reference fish.
- `save_analysis_report_run` - save a timestamped several-fish run folder with settings, comments, metadata, tables, and optional notebook export.
- `export_notebook_report` - export a saved notebook to a report folder through nbconvert.
- `build_response_window_validation` - build compact response-window validation tables for selected fish and stimuli.
- `resolve_response_control_columns` - resolve left/right control IDs or names to response-matrix columns.
- `build_selected_neuron_summary` - build selected/all-stimulus summary tables plus response-index filter outputs.
- `build_high_sparseness_raster_data` - prepare high lifetime-sparseness raster matrices and row order.
- `build_pooled_mean_trace_by_stimulus` - build per-fish mean traces for pooled time-course plots.
- `build_fish_keep_masks` - split a pooled neuron keep mask into per-fish masks in response-row order.
- `build_filtered_trial_aligned_traces_for_fish` - subset one fish's selected trial-aligned traces by preprocessing and optional pooled-filter rows.
- `build_overlap_diagnostic_data` - prepare active-neuron overlap matrices and pooled trace diagnostics for reusable notebooks.
- `load_and_preflight_fish_raster_inputs` - load and validate a configured several-fish raster cohort.
- `build_all_fish_raster_figure` - build an all-fish raster with configurable labels and sorting.
- `build_plot_all_fish_mean_zscore_traces` - render mean z-score traces for an ordered stimulus set.
- `plot_left_right_active_overlap_diagnostics` - render left/right active-neuron overlap and raster diagnostics.
- `plot_motion_active_neuron_counts` - render ordered per-stimulus active counts with one point per fish and no fish legend.
- `plot_lifetime_sparseness_analysis` - render lifetime sparseness and a high-sparseness raster with an optional analysis label.

### `src/lme_feature_decomposition.py`
- `build_lme_response_table` - convert per-fish neuron-by-stimulus response matrices plus editable stimulus metadata into a long LME response table.
- `validate_lme_response_table` - validate fish/neuron/stimulus row contracts, metadata labels, duplicates, missing responses, and response ranges while printing compact summaries.
- `fit_lme_models` - fit an editable dict/list of statsmodels mixed-effects model specs while reporting failures and continuing remaining fits.
- `summarize_lme_model_results` - extract fixed effects, random-effect variances, fit statistics, and model metadata into tidy result tables.

### `src/significant_traces.py`
- `compute_noise_model_romano_fast_modular` - build centered dFoF, significance maps, and event rasters from the Romano-style noise model.
- `clean_binary_raster_columns` - remove non-finite or zero-variance raster columns before correlation-based sorting.

### `src/plotting.py`
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
- `plot_shared_static_flicker_auc_summary` - render separate left/right shared-neuron AUC comparisons with per-stimulus distributions of per-fish mean ΔAUC.
- `plot_recruitment_amplification` - render per-position, per-fish recruitment and amplification components.

## Ownership notes
- If a notebook calls one of the symbols above, start in the owning module before editing the notebook.
- If a task changes any public helper signature or return contract, update this file and the relevant stage map in the same change.
- For repeated helper triage that does not change code yet, use `duplicate-helper-inventory.md` to distinguish public `src/` owners from extract-later notebook or script helpers.

# Recent Changes: Calcium Analysis

Use this log for experiment loading, alignment, filtering, classification, plotting, and multi-fish analysis slices.

## Entry template
- Date and label:
- Slice goal:
- Passes completed in this session:
- What changed:
- What remains broken:
- Remaining in-slice work:
- Next likely breakpoint:
- Rerun implications:

## 2026-04-21 - Initial guidance system seed
- Slice goal: create the routed guidance docs for calcium-analysis ownership and handoff.
- Passes completed in this session: repo inspection, router creation, stage-map creation, symbol indexing, and current-state documentation.
- What changed: added the calcium-analysis router and reference docs that point agents to `src/` owners before large notebooks.
- What remains broken: known mixed-state issues are documented in `current-state.md`; no analysis code changed here.
- Remaining in-slice work: append future entries when analysis helpers, loader contracts, or notebook handoffs change.
- Next likely breakpoint: first loader, alignment, or plotting task that changes public behavior.
- Rerun implications: none for runtime behavior; docs only.

## 2026-04-22 - Slice 1 merged-output loader hardening
- Slice goal: make merged dFoF loading less brittle around merged-file discovery and missing or compatibility map CSVs.
- Passes completed in this session: routed workflow read, owner-module update in `src/data_loading.py`, canonical and compatibility-path validation, and doc refresh.
- What changed: `src/data_loading.py` now resolves canonical merged filenames by `{prefix}` first, falls back to compatibility merged files when needed, and keeps base experiment loading working when the merged map CSV is missing while skipping map-dependent plane metadata.
- What remains broken: experiments that truly require plane metadata still need a valid merged map CSV with a `plane` column; rebuilding or normalizing merge outputs remains upstream work.
- Remaining in-slice work: none after validation.
- Next likely breakpoint: slice 2 if merge-output generation or extraction-side contracts need refactoring.
- Rerun implications: rerun the smallest `load_2p_experiment` smoke checks on one canonical dataset and one compatibility or missing-map case after future loader changes.

## 2026-05-05 - Multi-fish matrix helper extraction slice
- Slice goal: extract pure all-fish matrix construction helpers from one representative several-fish notebook into `src/`.
- Passes completed in this session: routed workflow read, duplicate-helper inspection, new `src/multifish_analysis.py` module, one notebook import replacement, synthetic behavior validation, and symbol index update.
- What changed: `combine_reps_one_stim`, `build_matrix_for_fish`, and `build_matrix_all_fish` are now public helpers in `src/multifish_analysis.py`; `Exp3_rocking_2_several_fish.ipynb` imports them instead of defining local copies.
- What remains broken: other several-fish notebooks still contain local copies; older `is_raster` variants remain in `Exp1_flickers_several_fish.ipynb` and `Exp2_rocking_1_several_fish.ipynb`.
- Remaining in-slice work: none for the representative notebook after validation.
- Next likely breakpoint: migrate the remaining several-fish notebooks after confirming whether their older `is_raster` helper variants should be preserved or retired.
- Rerun implications: run synthetic matrix-helper checks and, when data are available, rerun the helper/import cell plus first `build_matrix_all_fish` call in the migrated notebook.

## 2026-05-05 - Slice 1 several-fish notebook migration completion
- Slice goal: finish migrating the several-fish calcium-analysis notebooks to shared multi-fish matrix helpers.
- Passes completed in this session: routed workflow read, status inspection across Exp1 through Exp5, notebook import migration, `is_raster` call migration, JSON/search validation, and synthetic helper validation.
- What changed: `Exp1_flickers_several_fish.ipynb`, `Exp2_rocking_1_several_fish.ipynb`, `Exp4_distances_several_fish.ipynb`, and `Exp5_map_positions_several.ipynb` now import the three helpers from `src.multifish_analysis`; remaining `build_matrix_all_fish` calls in Exp1 through Exp5 now use `trace_type`.
- What remains broken: no known Slice 1 helper duplication remains in the target several-fish notebooks.
- Remaining in-slice work: none after validation.
- Next likely breakpoint: later slices should avoid changing dFoF extraction, timing/alignment semantics, z-score logic, significant-trace computation, or plotting behavior unless explicitly scoped.
- Rerun implications: full notebook reruns were not needed; when data are available, rerun each migrated import cell and first downstream `build_matrix_all_fish` cell.

## 2026-05-13 - Exp5 active-neuron overlap helper
- Slice goal: add Step 2 active-neuron Jaccard overlap for Exp5 left/right B conditions.
- Passes completed in this session: routed workflow read, owner-module update, notebook orchestration cell update, doc refresh, and synthetic overlap validation.
- What changed: `src/multifish_analysis.py` now computes active-neuron Jaccard matrices and returns both pooled and mean-per-fish left/right overlap summaries for several-fish notebooks.
- What remains broken: no known active-overlap helper issues remain after synthetic validation.
- Remaining in-slice work: rerun the new Exp5 notebook Step 2 cell with the full local dataset when available.
- Next likely breakpoint: any later change to the active-neuron definition should stay in `build_active_neuron_matrices_all_fish` or its `analysis_tools` owner.
- Rerun implications: rerun the Exp5 import/setup cells, the active-matrix build cell, and the new overlap heatmap cell.

## 2026-05-15 - Exp5 active-decision strictness diagnostic
- Slice goal: compare pooled binary significant-response traces with strict active-neuron decisions.
- Passes completed in this session: owner-module diagnostic builder, plotting helper, notebook orchestration cell, docs, and synthetic validation.
- What changed: `src.multifish_analysis` now builds pooled trace/decision diagnostic matrices; `src.plotting` now renders a shared-row trace heatmap plus active-decision strip.
- What remains broken: no known diagnostic helper issues after synthetic validation.
- Remaining in-slice work: rerun the Exp5 diagnostic cell with the full dataset and chosen stimulus list.
- Next likely breakpoint: future strictness changes should adjust the active-neuron thresholds passed to `build_active_neuron_matrices_all_fish`, not the diagnostic plot.
- Rerun implications: rerun imports, active-matrix build, and the diagnostic visualization cell.

## 2026-05-15 - Exp5 stimulus specificity slice 1
- Slice goal: add selected-stimuli configuration and reusable response-window frame validation for the neuron-level stimulus specificity analysis.
- Passes completed in this session: routed workflow read, owner-module helper addition, Exp5 notebook config/validation cell, public symbol index update, and synthetic helper validation.
- What changed: `src.analysis_tools` now resolves selected stimuli from names or IDs and computes clipped response-window frame indices; `Exp5_map_positions_several.ipynb` defines the initial `Sparseness_Left` stimulus subset and validates response windows across loaded fish.
- What remains broken: full notebook validation still requires the local Exp5 data paths and execution of cells through the new slice-1 cell.
- Remaining in-slice work: none for slice 1 after helper smoke checks.
- Next likely breakpoint: slice 2 should build per-fish z-score response matrices from `trial_aligned_traces_z_core` using the selected stimulus IDs and these response-window frames.
- Rerun implications: rerun the Exp5 setup/import cell, load-and-align cell, then the new Cell 03b validation cell.

## 2026-05-15 - Exp5 stimulus specificity slice 2
- Slice goal: build per-fish and pooled neuron-by-selected-stimulus z-score AUC response matrices.
- Passes completed in this session: owner-module AUC primitive, multi-fish matrix builders, Exp5 notebook orchestration cell, public symbol index update, and synthetic matrix validation.
- What changed: `src.analysis_tools.compute_zscore_response_auc` computes one mean AUC response per neuron; `src.multifish_analysis` now builds filtered per-fish response matrices and pooled response matrices with row metadata; `Exp5_map_positions_several.ipynb` builds `response_matrices_by_fish`, `pooled_response_matrix`, and `response_row_metadata`.
- What remains broken: full notebook validation still requires executing through the new Cell 03c on the local Exp5 dataset.
- Remaining in-slice work: none for slice 2 after synthetic validation.
- Next likely breakpoint: slice 3 should join `pooled_response_matrix` with `active_matrices` and promote row metadata into the neuron summary table.
- Rerun implications: rerun Exp5 setup/imports, load-and-align, Cell 03b, then Cell 03c.

## 2026-05-15 - Exp5 stimulus specificity slice 3
- Slice goal: join z-score response matrices with binary active-neuron decisions and identity metadata.
- Passes completed in this session: owner-module summary-table builder, Exp5 notebook orchestration cell, public symbol index update, and synthetic join validation.
- What changed: `src.multifish_analysis.build_neuron_stimulus_summary_table` now creates one row per filtered neuron with fish/global identity, `analysis_label`, selected-stimuli metadata, and per-stimulus `response_*`/`active_*` columns; the Exp5 notebook now builds `stimulus_specificity_active_matrices` and `stimulus_specificity_summary_table`.
- What remains broken: full notebook validation still requires executing through the new Cell 03d on the local Exp5 dataset.
- Remaining in-slice work: none for slice 3 after synthetic validation.
- Next likely breakpoint: slice 4 should add selectivity metrics to `stimulus_specificity_summary_table`.
- Rerun implications: rerun Exp5 setup/imports, load-and-align, Cells 03b through 03d.

## 2026-05-15 - Exp5 stimulus specificity slice 4
- Slice goal: add selectivity metrics to the neuron-level stimulus specificity summary table.
- Passes completed in this session: owner-module selectivity primitive, summary-table metric helper, Exp5 notebook orchestration cell, public symbol index update, and synthetic metric validation including all-negative responses.
- What changed: `src.analysis_tools.compute_stimulus_selectivity_metrics` computes raw preferred/max/mean response and non-negative simple selectivity/lifetime sparseness; `src.multifish_analysis.add_selectivity_metrics_to_summary_table` adds those fields plus active-stimulus count and response breadth; the Exp5 notebook now builds `stimulus_specificity_metric_table` and updates `stimulus_specificity_summary_table` to include metrics.
- What remains broken: full notebook validation still requires executing through the new Cell 03e on the local Exp5 dataset.
- Remaining in-slice work: none for slice 4 after synthetic validation.
- Next likely breakpoint: slice 5 should add configurable neuron classification thresholds using the metric-enriched summary table.
- Rerun implications: rerun Exp5 setup/imports, load-and-align, Cells 03b through 03e.

## 2026-05-15 - Exp5 stimulus specificity slice 5
- Slice goal: add configurable neuron classification on top of the metric-enriched stimulus specificity table.
- Passes completed in this session: owner-module single-neuron classifier, summary-table classification helper, Exp5 notebook threshold config and classification cell, public symbol index update, and synthetic classification validation.
- What changed: `src.analysis_tools.classify_stimulus_specificity_neuron` implements the planned class order; `src.multifish_analysis.classify_stimulus_specificity_summary_table` derives strong/weak positive-response thresholds from the current analysis table and adds `neuron_class`; the Exp5 notebook now defines `classification_thresholds` and builds `stimulus_specificity_classified_table`.
- What remains broken: full notebook validation still requires executing through the new Cell 03f on the local Exp5 dataset.
- Remaining in-slice work: none for slice 5 after synthetic validation.
- Next likely breakpoint: slice 6 should add summary plots for sparseness vs response strength, active-stimuli counts, and preferred-stimulus distribution.
- Rerun implications: rerun Exp5 setup/imports, load-and-align, Cells 03b through 03f.

## 2026-05-15 - Exp5 stimulus specificity slice 6
- Slice goal: add summary plots for the classified stimulus specificity table.
- Passes completed in this session: reusable plotting helpers, Exp5 notebook plot cell, public symbol index update, and synthetic noninteractive plotting validation.
- What changed: `src.plotting` now includes the three summary plots from the plan plus `plot_stimulus_specificity_summary`; the Exp5 notebook now has Cell 03g to plot sparseness vs max response, active-stimuli histogram, and preferred-stimulus distribution.
- What remains broken: full notebook validation still requires executing through the new Cell 03g on the local Exp5 dataset.
- Remaining in-slice work: none for slice 6 after synthetic validation.
- Next likely breakpoint: slice 7 should re-sort the Cell 18 raster by preferred stimulus, lifetime sparseness, and max response.
- Rerun implications: rerun Exp5 setup/imports, load-and-align, Cells 03b through 03g.

## 2026-05-15 - Exp5 stimulus specificity slice 7
- Slice goal: re-sort the Cell 18-style diagnostic raster by preferred stimulus, lifetime sparseness, and max response.
- Passes completed in this session: pooled row-order helper, Exp5 notebook re-sorted diagnostic cell, public symbol index update, and synthetic sort-order validation.
- What changed: `src.multifish_analysis.build_stimulus_specificity_neuron_order` builds diagnostic row indices from the classified summary table; the Exp5 notebook now has Cell 03h to rebuild the selected-stimulus diagnostic and pass the specificity order into `plot_active_trace_decision_diagnostic`.
- What remains broken: full notebook validation still requires executing through the new Cell 03h on the local Exp5 dataset.
- Remaining in-slice work: none for the planned stimulus-specificity slices after validation.
- Next likely breakpoint: future refinement should focus on full-data reruns, saved outputs, or export/report formatting if needed.
- Rerun implications: rerun Exp5 setup/imports, load-and-align, Cells 03b through 03h.

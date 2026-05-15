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

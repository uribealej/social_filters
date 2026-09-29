# Calcium Analysis Router

Purpose

Route work related to experiment loading, stimulus alignment, response filtering and classification, significance detection, single-fish plotting, and multi-fish summary notebooks.

Use this file when

- The task targets `src/data_loading.py`, a narrow analysis or plotting owner, one of their compatibility facades, or `src/significant_trace_detection.py`.
- The task targets the single-fish or several-fish notebooks under `notebooks/calcium/`.
- The task mentions stimulus alignment, trial windows, reliability filtering, response types, left-right metrics, or figure behavior.

## Task routing table

Choose one row. A dash means no extra reference is needed before opening the owner.

| Task pattern | Reference if needed | Open next |
| --- | --- | --- |
| Experiment loading, merged-map failure, plane metadata, `paths` dict behavior | `current-state.md` for the known merged-map caveat | `src/data_loading.py` |
| Stimulus alignment, peri-stimulus windows, onset extraction, shared timing inside analysis | — | `src/stimuli_timeline.py`, then its direct consumer if the handoff is involved |
| Trial alignment, stimulus selection, response windows | — | `src/trial_alignment.py`; inspect the `src.analysis_tools` facade only for import compatibility |
| Trial response/AUC metrics or static--flicker trial metrics | — | `src/response_metrics.py` |
| Reliability filtering or its diagnostic figure | — | `src/reliability.py` for filtering; `src/plotting_reliability.py` for the figure |
| Motion-minus-fixed metrics | — | `src/motion_metrics.py` |
| Response classification, active-neuron decisions, left-right or onset ordering | — | `src/response_classification.py` |
| Selectivity metrics or response-pair filtering | — | `src/response_selectivity.py` |
| Pre-stimulus baseline z-scoring | — | `src/response_normalization.py` |
| Significant trace detection, noise model, raster generation | — | `src/significant_trace_detection.py` |
| Sorted chunk plots, all-fish flat rasters, mean traces, raster styling, saved PNG behavior | — | Relevant `src/plotting_*` owner; use `src/plotting.py` to locate it if unclear |
| Notebook helper duplication, notebook-local scientific logic, or notebook-to-module extraction | `refactor-rules.md` | Owning `src/` module before opening the notebook region |
| Cache names, filtered neuron index naming, saved figure locations | `canonical-outputs.md` | `src/data_loading.py` or the relevant writer |
| Several-fish report folder, JSON/CSV serialization, notebook export | `canonical-outputs.md` for saved names and layout | `src/several_fish_reporting.py`; check `src/reusable_several_fish.py` for the historical notebook import |
| Several-fish cohort loading, data-root lookup, or stimulus preflight | `src-api-consumer-inventory.md` for the returned bundle | `src/several_fish_loading.py`; check `src/reusable_several_fish.py` for the historical notebook import |
| Several-fish stimulus/control selection, response summaries, pooled keep masks, or filtered trace/active rows | `src-api-consumer-inventory.md` for the pooled-row contract | `src/several_fish_selection.py`; check `src/reusable_several_fish.py` for historical notebook imports |
| Several-fish response-window tables, high-sparseness raster inputs, mean traces, or active-overlap diagnostics | `src-api-consumer-inventory.md` for pooled-row ordering | `src/several_fish_diagnostics.py`; check `src/reusable_several_fish.py` for historical notebook imports |
| Several-fish pooled raster, sparseness, mean-trace, active-overlap, or active-count figure orchestration | `src-api-consumer-inventory.md` for figure return contracts | `src/several_fish_figures.py`; check `src/reusable_several_fish.py` for historical notebook imports |
| Analysis stage order or smallest rerun chain | `calcium-analysis-stage-map.md` | Relevant owner and first consumer |
| Helper name without a known owner, or stable public import path | `symbol-index.md` | Owner identified there |
| Historical facade import or removal question | `src-compatibility-decision.md` | Canonical owner named in the decision |

## Ownership guidance
- `src/data_loading.py` owns experiment bundle assembly and output lookup semantics.
- Narrow S06 modules own alignment, metrics, reliability, classification, selectivity, and normalization; `src.analysis_tools` preserves the established import surface as a facade.
- `src/multifish_matrices.py` owns pure multi-fish matrix construction from already trial-aligned per-fish traces; `src/multifish_analysis.py` preserves historical imports for this and other S07 scientific owners.
- `src/significant_trace_detection.py` owns the Romano-style noise-model and rasterization pipeline; the versioned significant-trace modules are compatibility facades.
- The `src/plotting_*` modules own reusable figure families; `src/plotting.py` preserves historical imports. Inspect its imports to locate the exact owner.
- `src/several_fish_reporting.py` owns several-fish report serialization and notebook export; `src/reusable_several_fish.py` keeps historical imports available under the S12 compatibility decision.
- `src/several_fish_loading.py` owns several-fish cohort loading and stimulus preflight; its existing notebook import remains available through `src/reusable_several_fish.py`.
- `src/several_fish_selection.py` owns pure several-fish stimulus selection, response summaries, keep masks, and row filtering; `src/reusable_several_fish.py` preserves historical notebook imports.
- `src/several_fish_diagnostics.py` owns pure several-fish diagnostic and figure-input preparation, including raster inputs, sparseness summaries, and active-count tables.
- `src/several_fish_figures.py` owns high-level several-fish figure orchestration; `src/plotting_*` modules own the reusable figure primitives. `src/reusable_several_fish.py` is an import-only compatibility facade.
- Active analysis notebooks and reusable owners use narrow scientific, plotting, and several-fish workflow modules directly. `analysis_tools`, `multifish_analysis`, `plotting`, and `reusable_several_fish` remain supported historical compatibility facades; see `src-compatibility-decision.md` for their boundaries.
- Analysis notebooks own experiment selection, scientific narration, style dictionaries, and orchestration across cached outputs.

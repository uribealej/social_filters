# Calcium Analysis Router

Purpose

Route work related to experiment loading, stimulus alignment, response filtering and classification, significance detection, single-fish plotting, and multi-fish summary notebooks.

Use this file when

- The task targets `src/data_loading.py`, a narrow analysis or plotting owner, one of their compatibility facades, or `src/significant_trace_detection.py`.
- The task targets the single-fish or several-fish notebooks under `notebooks/calcium/`.
- The task mentions stimulus alignment, trial windows, reliability filtering, response types, left-right metrics, or figure behavior.

## Task routing table

Choose one row. A dash means no extra reference is needed before opening the owner.
For everyday analysis or a new `src/` responsibility, dependency, or public contract decision, read
[the organization policy](../references/src-organization-policy.md) and then
use the matching row below to open the owner.

| Task pattern | Reference if needed | Open next |
| --- | --- | --- |
| Experiment loading, merged-map failure, plane metadata, `paths` dict behavior | `current-state.md` for current map-selection behavior | `src/data_loading.py` |
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
| Several-fish report folder, JSON/CSV serialization, notebook export | `canonical-outputs.md` for saved names and layout | `src/several_fish_reporting.py` |
| Several-fish cohort loading, data-root lookup, or stimulus preflight | `src-api-consumer-inventory.md` for the returned bundle | `src/several_fish_loading.py` |
| Directional active-neuron containment or empty-source overlap behavior | `src-api-consumer-inventory.md` for fraction semantics | `src/active_neuron_analysis.py` |
| Mirrored recruitment, observed-window activity, control/all-rocking containment summaries | `src-api-consumer-inventory.md` for neuron identity and equal-fish contracts | `src/several_fish_diagnostics.py`, then `src/several_fish_figures.py` |
| Mirrored all-kept/shared-rocking response-pattern similarity | `src-api-consumer-inventory.md` for conditional-population and valid-count semantics | `src/several_fish_diagnostics.py`, then `src/several_fish_figures.py`; `src/stimulus_similarity.py` owns Pearson/cosine calculations |
| Family response breadth, exact memberships, or signed per-neuron response profiles | `src-api-consumer-inventory.md` for kept-neuron denominators and profile order | `src/stimulus_specificity.py` for memberships; `src/neuron_ordering.py` for display order; `src/several_fish_diagnostics.py` and `src/several_fish_figures.py` for orchestration |
| Within-family lifetime sparseness from prepared response profiles | `src-api-consumer-inventory.md` for positive-response and missing-profile semantics | `src/response_selectivity.py` owns the metric; `src/several_fish_diagnostics.py` and `src/several_fish_figures.py` own mirrored population summaries |
| Several-fish stimulus/control selection, response summaries, pooled keep masks, or filtered trace/active rows | `src-api-consumer-inventory.md` for the pooled-row contract | `src/several_fish_selection.py` |
| Several-fish response-window tables, high-sparseness raster inputs, mean traces, or active-overlap diagnostics | `src-api-consumer-inventory.md` for pooled-row ordering | `src/several_fish_diagnostics.py` |
| Several-fish pooled raster, sparseness, mean-trace, active-overlap, or active-count figure orchestration | `src-api-consumer-inventory.md` for figure return contracts | `src/several_fish_figures.py` |
| Analysis stage order or smallest rerun chain | `calcium-analysis-stage-map.md` | Relevant owner and first consumer |
| Helper name without a known owner, or stable public import path | `symbol-index.md` | Owner identified there |
| Historical facade import or removal question | `src-compatibility-decision.md` | Canonical owner named in the decision |

## Ownership guidance
- Pooled left/right response-profile heatmaps: `build_pooled_response_profiles` in `src/several_fish_diagnostics.py`; `build_response_profile_sort_order` in `src/neuron_ordering.py` owns preferred/Pearson/Euclidean ordering; `plot_pooled_response_profiles` in `src/several_fish_figures.py` owns the two figures.
- Family-responsive breadth/combination denominators: `src/stimulus_specificity.py::build_family_responsive_membership_fractions`; prepared tables are assembled by `build_response_breadth_profiles`, and selected by the figure helper's `denominator` option.
- Compact independent summary panels are owned by `src/several_fish_figures.py` (`separate=True`, `individual_style="anonymous_points"`); anonymous categorical dots and CDF dots belong to `plotting_diagnostics.py` and `plotting_specificity.py`. Exp 7 notebook 02 owns their dimensions, fonts and export selection.
- Publication canvas sizes and font settings stay beside notebook plots; `src/plotting_common.py::save_publication_figure` exports fixed-size PDF/SVG/PNG. Exp 7 notebooks 01/02 use compact A4-compatible blocks; profile subsets are selected by `plot_response_profile_panels(..., fish_ids=...)` in `src/several_fish_figures.py`.
- `src/data_loading.py` owns experiment bundle assembly and output lookup semantics.
- Narrow modules own alignment, metrics, reliability, classification, selectivity, and normalization; `src.analysis_tools` preserves the established import surface as a facade.
- `src/multifish_matrices.py` owns pure multi-fish matrix construction from already trial-aligned per-fish traces; `src/multifish_analysis.py` preserves historical imports for this and other scientific owners.
- `src/significant_trace_detection.py` owns the Romano-style noise-model and rasterization pipeline; the versioned significant-trace modules are compatibility facades.
- The `src/plotting_*` modules own reusable figure families; `src/plotting.py` preserves historical imports. Inspect its imports to locate the exact owner.
- `src/several_fish_reporting.py` owns several-fish report serialization and notebook export.
- `src/several_fish_loading.py` owns several-fish cohort loading and stimulus preflight.
- `src/several_fish_selection.py` owns pure several-fish stimulus selection, response summaries, keep masks, and row filtering.
- `src/several_fish_diagnostics.py` owns pure several-fish diagnostic and figure-input preparation, including raster inputs, sparseness summaries, and active-count tables.
- `src/several_fish_figures.py` owns high-level several-fish figure orchestration; `src/plotting_*` modules own the reusable figure primitives. `src/reusable_several_fish.py` is an import-only compatibility facade.
- Active analysis notebooks and reusable owners use narrow scientific, plotting, and several-fish workflow modules directly. `analysis_tools`, `multifish_analysis`, `plotting`, and `reusable_several_fish` remain supported historical compatibility facades; see `src-compatibility-decision.md` for their boundaries.
- Analysis notebooks own experiment selection, scientific narration, style dictionaries, and orchestration across cached outputs.

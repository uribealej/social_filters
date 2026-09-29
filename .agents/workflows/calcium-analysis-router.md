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
| Analysis stage order or smallest rerun chain | `calcium-analysis-stage-map.md` | Relevant owner and first consumer |
| Helper name without a known owner, or stable public import path | `symbol-index.md` | Owner identified there |

## Ownership guidance
- `src/data_loading.py` owns experiment bundle assembly and output lookup semantics.
- Narrow S06 modules own alignment, metrics, reliability, classification, selectivity, and normalization; `src.analysis_tools` preserves the established import surface as a facade.
- `src/multifish_matrices.py` owns pure multi-fish matrix construction from already trial-aligned per-fish traces; `src/multifish_analysis.py` preserves historical imports for this and other S07 scientific owners.
- `src/significant_trace_detection.py` owns the Romano-style noise-model and rasterization pipeline; the versioned significant-trace modules are compatibility facades.
- The `src/plotting_*` modules own reusable figure families; `src/plotting.py` preserves historical imports. Inspect its imports to locate the exact owner.
- Analysis notebooks own experiment selection, scientific narration, style dictionaries, and orchestration across cached outputs.

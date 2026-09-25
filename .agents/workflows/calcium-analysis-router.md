# Calcium Analysis Router

Purpose

Route work related to experiment loading, stimulus alignment, response filtering and classification, significance detection, single-fish plotting, and multi-fish summary notebooks.

Use this file when

- The task targets `src/data_loading.py`, `src/analysis_tools.py`, `src/plotting.py`, or `src/significant_trace_detection.py`.
- The task targets the single-fish or several-fish notebooks under `notebooks/calcium/`.
- The task mentions stimulus alignment, trial windows, reliability filtering, response types, left-right metrics, or figure behavior.

## Read order
1. `../references/calcium-analysis-stage-map.md`
2. `../references/current-state.md` for legacy notebook drift or known loader caveats
3. `../references/symbol-index.md`
4. `../references/canonical-outputs.md` for cache or file-contract questions
5. `../references/refactor-rules.md`
6. `../references/recent-changes.md` only when resuming incomplete work

## Task routing table

| Task pattern | Read first | Open next |
| --- | --- | --- |
| Experiment loading, missing merged-map crash, cache lookup, plane metadata loading, `paths` dict behavior | `current-state.md` | `src/data_loading.py` |
| Stimulus alignment, peri-stimulus windows, onset extraction, shared timing inside analysis | `symbol-index.md` | `src/stimuli_timeline.py`, then `src/analysis_tools.py` or `src/data_loading.py` |
| Trial alignment, stimulus selection, response windows | `symbol-index.md` | `src/trial_alignment.py`, then the `src.analysis_tools` facade if compatibility context is needed |
| Trial response/AUC metrics or static--flicker trial metrics | `symbol-index.md` | `src/response_metrics.py` |
| Reliability filtering | `symbol-index.md` | `src/reliability.py`, then `src/plotting.py` for diagnostics |
| Motion-minus-fixed metrics | `symbol-index.md` | `src/motion_metrics.py` |
| Response classification, active-neuron decisions, left-right or onset ordering | `symbol-index.md` | `src/response_classification.py` |
| Selectivity metrics or response-pair filtering | `symbol-index.md` | `src/response_selectivity.py` |
| Pre-stimulus baseline z-scoring | `symbol-index.md` | `src/response_normalization.py` |
| Significant trace detection, noise model, raster generation | `symbol-index.md` | `src/significant_trace_detection.py` |
| Sorted chunk plots, all-fish flat rasters, mean traces, raster styling, saved PNG behavior | `symbol-index.md` | `src/plotting.py` |
| Notebook helper duplication, notebook-local scientific logic, or notebook-to-module extraction | `refactor-rules.md` | Owning `src/` module before opening the notebook region |
| Plot cache semantics, z-score cache naming, filtered neuron index naming, saved figure locations | `canonical-outputs.md` | `src/data_loading.py` or `src/plotting.py` |

## Ownership guidance
- `src/data_loading.py` owns experiment bundle assembly and output lookup semantics.
- Narrow S06 modules own alignment, metrics, reliability, classification, selectivity, and normalization; `src.analysis_tools` preserves the established import surface as a facade.
- `src/multifish_analysis.py` owns pure matrix construction for several-fish notebooks after per-fish trial alignment is already available.
- `src/significant_trace_detection.py` owns the Romano-style noise-model and rasterization pipeline; the versioned significant-trace modules are compatibility facades.
- `src/plotting.py` owns reusable figure construction, chunk layout, and all-fish raster helpers.
- Analysis notebooks own experiment selection, scientific narration, style dictionaries, and orchestration across cached outputs.
- Smallest practical validation surface: rerun the smallest affected notebook cell chain on one fish or one stimulus block, then verify the first downstream notebook call or saved cache that depends on the edited owner.
- Record only incomplete work in `../references/recent-changes.md`; completed history is not part of normal routing.

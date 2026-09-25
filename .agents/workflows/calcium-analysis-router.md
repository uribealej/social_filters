# Calcium Analysis Router

Purpose

Route work related to experiment loading, stimulus alignment, response filtering and classification, significance detection, single-fish plotting, and multi-fish summary notebooks.

Use this file when

- The task targets `src/data_loading.py`, `src/analysis_tools.py`, `src/plotting.py`, or `src/significant_traces.py`.
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
| Reliability filtering, trial alignment, response classification, left-right or onset-based ordering | `symbol-index.md` | `src/analysis_tools.py` |
| Significant trace detection, noise model, raster generation | `symbol-index.md` | `src/significant_traces.py` |
| Sorted chunk plots, all-fish flat rasters, mean traces, raster styling, saved PNG behavior | `symbol-index.md` | `src/plotting.py` |
| Notebook helper duplication, notebook-local scientific logic, or notebook-to-module extraction | `refactor-rules.md` | Owning `src/` module before opening the notebook region |
| Plot cache semantics, z-score cache naming, filtered neuron index naming, saved figure locations | `canonical-outputs.md` | `src/data_loading.py` or `src/plotting.py` |

## Ownership guidance
- `src/data_loading.py` owns experiment bundle assembly and output lookup semantics.
- `src/analysis_tools.py` owns shared alignment, filtering, classification, and derived-metric logic.
- `src/multifish_analysis.py` owns pure matrix construction for several-fish notebooks after per-fish trial alignment is already available.
- `src/significant_traces.py` owns the Romano-style noise-model and rasterization pipeline.
- `src/plotting.py` owns reusable figure construction, chunk layout, and all-fish raster helpers.
- Analysis notebooks own experiment selection, scientific narration, style dictionaries, and orchestration across cached outputs.
- Smallest practical validation surface: rerun the smallest affected notebook cell chain on one fish or one stimulus block, then verify the first downstream notebook call or saved cache that depends on the edited owner.
- Record only incomplete work in `../references/recent-changes.md`; completed history is not part of normal routing.

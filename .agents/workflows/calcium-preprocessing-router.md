# Calcium Preprocessing Router

Purpose

Route work related to dFoF extraction, baseline or threshold sweeps, experiment-level merge artifacts, and preprocessing-side utility notebooks.

Use this file when

- The task targets `src/dff_extraction.py` or `src/auxtrigger_extraction.py`.
- The task targets `DeltaFF_batch_pipeline.ipynb`, `Baseline_Evaluation_single_plane.ipynb`, `STD_Zcore_threshold_analyis.ipynb`, or `2P_Experiment_FileOps.ipynb`.
- The task changes per-plane dFoF outputs, merged dFoF outputs, or preprocessing parameter sweeps.

## Read order
1. `../references/calcium-preprocessing-stage-map.md`
2. `../references/canonical-outputs.md` for output naming, folder layout, or writer-stage questions
3. `../references/symbol-index.md` for callable owners in `src/`
4. `../references/refactor-rules.md` before adding logic to notebooks
5. `../references/current-state.md` if the task touches utility notebooks or legacy experiment trees
6. `../references/recent-changes-calcium-preprocessing.md` for handoff context

## Task routing table

| Task pattern | Read first | Open next |
| --- | --- | --- |
| Per-plane dFoF extraction, ROI filtering, baseline math, `min_std`, `percentile`, `tau`, or Suite2p file handling | `symbol-index.md` | `src/dff_extraction.py` |
| Aux trigger extraction from ScanImage TIFF metadata | `symbol-index.md` | `src/auxtrigger_extraction.py` |
| Merged dFoF file naming, map CSV behavior, merge output folders, downstream lookup compatibility | `canonical-outputs.md` | `DeltaFF_batch_pipeline.ipynb`, then first consumer such as `src/data_loading.py` |
| Baseline sweep or threshold sweep notebook behavior | `calcium-preprocessing-stage-map.md` | Target sweep notebook, then extract reusable logic to `src/dff_extraction.py` if the logic is repeating |
| Experiment tree copy, rename, or migration utility work | `current-state.md` | `2P_Experiment_FileOps.ipynb` |

## Ownership guidance
- `src/dff_extraction.py` owns reusable fluorescence loading, filtering, baseline estimation, and dFoF extraction math.
- `src/auxtrigger_extraction.py` owns reusable TIFF metadata extraction, even if preprocessing notebooks call it rarely.
- Preprocessing notebooks own batch orchestration, sweep setup, summary tables, and ad hoc migration utilities.
- `2P_Experiment_FileOps.ipynb` is a file-ops surface, not the authority for scientific extraction semantics.
- Smallest practical validation surface: rerun the smallest affected notebook stage on one experiment or one plane, then verify the written output names and shapes expected by the first downstream consumer.
- Handoff log: `../references/recent-changes-calcium-preprocessing.md`.

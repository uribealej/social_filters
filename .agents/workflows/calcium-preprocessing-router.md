# Calcium Preprocessing Router

Purpose

Route work related to dFoF extraction, baseline or threshold sweeps, experiment-level merge artifacts, and preprocessing-side utility notebooks.

Use this file when

- The task targets `src/dff_extraction.py` or `src/auxtrigger_extraction.py`.
- The task targets preprocessing notebooks under `notebooks/calcium/preprocessing/`.
- The task changes per-plane dFoF outputs, merged dFoF outputs, or preprocessing parameter sweeps.

## Task routing table

Choose one row. A dash means no extra reference is needed before opening the owner.
For everyday preprocessing analysis or a new `src/` responsibility, dependency, or public contract decision, read
[the organization policy](../references/src-organization-policy.md) and then
use the matching row below to open the owner.

| Task pattern | Reference if needed | Open next |
| --- | --- | --- |
| Per-plane dFoF extraction, ROI filtering, baseline math, `min_std`, `percentile`, `tau`, or Suite2p file handling | — | `src/dff_extraction.py` |
| Aux trigger extraction from ScanImage TIFF metadata | — | `src/auxtrigger_extraction.py` |
| Merged dFoF file naming, map CSV behavior, merge output folders, downstream lookup compatibility | `canonical-outputs.md` | `DeltaFF_batch_pipeline.ipynb`, then first consumer such as `src/data_loading.py` |
| Baseline or threshold parameter exploration | `calcium-preprocessing-stage-map.md` if stage placement is unclear | `src/dff_extraction.py` for reusable math; target notebook for settings and reporting |
| Repeated preprocessing setup helpers such as experiment discovery, plane lookup, prefix derivation, or per-plane file lookup | `refactor-rules.md` if ownership is unclear | Likely owner, such as `src/data_loading.py` or `src/dff_extraction.py`, before the notebook region |
| Experiment tree copy, rename, or migration utility work | `current-state.md` for legacy layout caveats | `2P_Experiment_FileOps.ipynb` |
| Preprocessing stage order or smallest rerun chain | `calcium-preprocessing-stage-map.md` | Relevant writer and first consumer |
| Helper name without a known owner, or stable public import path | `symbol-index.md` | Owner identified there |

## Ownership guidance
- `src/dff_extraction.py` owns reusable fluorescence loading, filtering, baseline estimation, and dFoF extraction math.
- `src/auxtrigger_extraction.py` owns reusable TIFF metadata extraction, even if preprocessing notebooks call it rarely.
- Aux-trigger extraction loads `ScanImageTiffReader` when called; other preprocessing modules remain importable without that dependency.
- `src/data_loading.py` owns reusable merged-output and per-plane output discovery semantics consumed downstream.
- Preprocessing notebooks own batch orchestration, sweep setup, summary tables, and ad hoc migration utilities.
- `2P_Experiment_FileOps.ipynb` is a file-ops surface, not the authority for scientific extraction semantics.
- Diagnostic sweep plots such as `plot_raster_gray` stay notebook-local until a separately scoped change promotes them into shared plotting API.
- For extraction or writer behavior changes, rerun the affected stage on one experiment or plane when data are available; check written names and shapes with the first consumer when the output contract changes.

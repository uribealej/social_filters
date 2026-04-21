# Calcium Preprocessing Stage Map

Purpose

Provide the ordered preprocessing flow for extracting dFoF data and writing experiment-level artifacts before downstream analysis notebooks consume them.

Use this file when

- You need to know which stage writes a preprocessing output.
- You are deciding whether a bug belongs in extraction math, merge logic, or utility notebooks.
- You need the smallest rerun surface after a preprocessing edit.

## End-to-end stages
1. Discover candidate experiments and planes in preprocessing notebooks.
2. Load Suite2p fluorescence and cell masks for a plane.
3. Filter non-cells, dim ROIs, unstable baselines, and inactive ROIs, then compute dFoF.
4. Write per-plane outputs such as dFoF arrays, kept ROI indices, and metadata.
5. Merge plane outputs into experiment-level files under `03_analysis/functional/suite2P/merged_dFoF`.
6. Validate merged shapes, map CSVs, and optional merge-side plot exports.
7. Run utility file-copy or tree-reorganization notebooks only when the task is about storage layout rather than scientific extraction semantics.

## Key outputs by stage
- Stage 3 to 4 writes per-plane dFoF arrays and filtered ROI indices.
- Stage 5 writes `{prefix}_dFoF_merged.npy`, `{prefix}_dFoF_merged_filtered_roi_indices.npy`, and `{prefix}_dFoF_merged_map.csv`.
- Sweep notebooks write summary CSVs, diagnostics, and figures for parameter comparison.
- Utility notebooks copy or reorganize raw analysis folders but should not redefine extraction math.

## Concept ownership
- `src/dff_extraction.py` owns the extraction pipeline and per-plane filtering semantics.
- `src/auxtrigger_extraction.py` owns reusable aux-trigger parsing from TIFF metadata.
- `DeltaFF_batch_pipeline.ipynb` owns batch orchestration and experiment-level merge writing.
- `Baseline_Evaluation_single_plane.ipynb` and `STD_Zcore_threshold_analyis.ipynb` own sweep orchestration and summary reporting.
- `2P_Experiment_FileOps.ipynb` owns migration and copy utilities only.

## Navigation notes
- Read `canonical-outputs.md` for authoritative file names and folders.
- Read `symbol-index.md` before editing a callable helper in `src/`.
- Read `current-state.md` when a task depends on legacy experiment layout or utility-notebook behavior.

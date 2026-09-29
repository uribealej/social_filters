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
4. Write per-plane dFoF arrays, kept ROI indices, the NPZ bundle, and metadata from the batch notebook.
5. Merge plane outputs into experiment-level files under `03_analysis/functional/suite2P/merged_dFoF`.
6. Validate merged shapes, map CSVs, and optional merge-side plot exports.
7. Run utility file-copy or tree-reorganization notebooks only when the task is about storage layout rather than scientific extraction semantics.

## Key outputs by stage
- Stage 4 writes `{prefix}_dFoF.npy`, `{prefix}_filtered_roi_indices.npy`, `{prefix}_dFoF_outputs.npz`, and `metadata.json` per plane.
- Stage 5 writes `{prefix}_dFoF_merged.npy`, `{prefix}_dFoF_merged_filtered_roi_indices.npy`, and `{prefix}_dFoF_merged_map.csv`.
- Utility notebooks copy or reorganize raw analysis folders but should not redefine extraction math.

## Concept ownership
- `src/dff_extraction.py` owns the extraction pipeline and per-plane filtering semantics.
- `src/auxtrigger_extraction.py` owns reusable aux-trigger parsing from TIFF metadata.
- `DeltaFF_batch_pipeline.ipynb` owns batch orchestration, per-plane output writing, and experiment-level merge writing.
- `2P_Experiment_FileOps.ipynb` owns migration and copy utilities only.

## Navigation notes
- Consult `canonical-outputs.md` when a file name, folder, or writer contract matters.
- Consult `symbol-index.md` when locating a helper or checking its stable public import path.
- Consult `current-state.md` when a task depends on legacy experiment layout or utility-notebook behavior.

# Duplicate Helper Inventory

Scope: Slice 12 only from `.agents/references/refactor-plan.md`. This is a planning backlog for repeated helper logic across preprocessing notebooks in `scripts/calcium_analysis/` and stimulus inspection notebooks/scripts in `scripts/stimuli/`. No code was refactored.

## Extract Now

| Helper(s) | Files | Repeated logic | Recommended owner layer | Why |
| --- | --- | --- | --- | --- |
| `ensure_dir`, `find_experiments`, `plane_dir` / `plane_dirs`, `experiment_prefix` | `scripts/calcium_analysis/Baseline_Evaluation_single_plane.ipynb`, `scripts/calcium_analysis/STD_Zcore_threshold_analyis.ipynb`, `scripts/calcium_analysis/DeltaFF_batch_pipeline.ipynb` | Preprocessing setup for discovering experiments and planes and deriving canonical experiment prefixes | Small shared preprocessing helper in `src/`, near `src/dff_extraction.py` or `src/data_loading.py` | This is repeated path and file-contract logic that should not drift across notebooks. |
| `load_dfof_for_plane`, `load_filtered_indices_for_plane` | `scripts/calcium_analysis/DeltaFF_batch_pipeline.ipynb` | Repeated per-plane output lookup using canonical and fallback filenames | `src/data_loading.py` | Output lookup semantics belong at the owner layer rather than in notebook cells. |
| `generate_circular_trajectory` | `scripts/stimuli/trayectory_stimuli.py`, `scripts/stimuli/Trayectory_flicker.py` | Shared circular arc sampling, rotation, timing, and frame construction for generated trajectories | Shared helper in `scripts/stimuli/` | This is duplicated generator-side logic across stimulus wrappers. |
| `get_angles_from_positions` | `scripts/stimuli/trayectory_stimuli.py`, `scripts/stimuli/plots_stimuli.ipynb` | Reverse-rotation geometry used to interpret positions as angles | Shared helper in `scripts/stimuli/` | The same geometry convention is duplicated verbatim between generation and inspection. |

## Keep Notebook-Local

| Helper(s) | Files | Repeated logic | Recommended owner layer | Why |
| --- | --- | --- | --- | --- |
| `plot_raster_gray` | `scripts/calcium_analysis/Baseline_Evaluation_single_plane.ipynb`, `scripts/calcium_analysis/STD_Zcore_threshold_analyis.ipynb` | Quick grayscale diagnostic raster plotting for sweep/reporting cells | Notebook-local | This is sweep/reporting visualization, not clearly a stable shared plotting API. |
| `safe_to_csv`, `safe_savefig` | `scripts/calcium_analysis/DeltaFF_batch_pipeline.ipynb` | Convenience wrappers that save alternate files when targets are locked | Notebook-local unless reused elsewhere | Useful orchestration glue, but not scientific or contract authority. |
| `plot_angle_and_size`, `plot_right_left_on_circle` | `scripts/stimuli/plots_stimuli.ipynb` | Inspection and exploratory plotting for generated trajectories | Notebook-local | The stimulus inspection notebook should stay focused on visualization rather than becoming a shared timing or plotting owner. |
| `calculate_experiment_duration`, `calculate_experiment_duration_by_type` | `scripts/stimuli/Trayectory_rocking_stimuli.py`, `scripts/stimuli/Trayectory_flicker.py` | Per-script total-duration summaries for parameter/package metadata | Script-local in `scripts/stimuli/` | Similar purpose, but the config semantics differ enough that this is still wrapper-specific logic. |

## Legacy Only

| Helper(s) | Files | Repeated logic | Recommended owner layer | Why |
| --- | --- | --- | --- | --- |
| Duplicate copies of `ensure_dir`, `find_experiments`, `plane_dirs`, `experiment_prefix` within one notebook | `scripts/calcium_analysis/DeltaFF_batch_pipeline.ipynb` | Intra-notebook copy-paste variants of preprocessing setup helpers | Future shared `src` helper only | These duplicate sections are useful as backlog evidence, but should not become authoritative implementations. |
| Older preprocessing setup variants in sweep notebooks | `scripts/calcium_analysis/Baseline_Evaluation_single_plane.ipynb`, `scripts/calcium_analysis/STD_Zcore_threshold_analyis.ipynb` | Earlier notebook-local setup helpers overlapping with pipeline helpers | Shared `src` owner if extracted later | They show duplication, but are notebook drift rather than a source of truth. |


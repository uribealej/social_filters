# Duplicate Helper Inventory

Scope: Slice 12 only from `.agents/references/refactor-plan.md`. This is a planning backlog for repeated helper logic across preprocessing notebooks in `notebooks/calcium/preprocessing/`, stimulus inspection notebooks in `notebooks/stimuli/`, and generators in `scripts/stimuli/generation/`. No code was refactored.

## Extract Now

| Helper(s) | Files | Repeated logic | Recommended owner layer | Why |
| --- | --- | --- | --- | --- |
| `ensure_dir`, `find_experiments`, `plane_dir` / `plane_dirs`, `experiment_prefix` | Historical sweep notebooks plus `notebooks/calcium/preprocessing/DeltaFF_batch_pipeline.ipynb` | Preprocessing setup for discovering experiments and planes and deriving canonical experiment prefixes | Small shared preprocessing helper in `src/`, near `src/dff_extraction.py` or `src/data_loading.py` | This is repeated path and file-contract logic that should not drift across notebooks. |
| `load_dfof_for_plane`, `load_filtered_indices_for_plane` | `notebooks/calcium/preprocessing/DeltaFF_batch_pipeline.ipynb` | Repeated per-plane output lookup using canonical and fallback filenames | `src/data_loading.py` | Output lookup semantics belong at the owner layer rather than in notebook cells. |
| `generate_circular_trajectory` | Historical `trayectory_stimuli.py` plus `scripts/stimuli/generation/Trayectory_flicker.py` | Shared circular arc sampling, rotation, timing, and frame construction for generated trajectories | Shared helper in `scripts/stimuli/generation/` | This is duplicated generator-side logic across stimulus wrappers. |
| `get_angles_from_positions` | Historical generator/inspection copies; current owner is `src/stimulus_visualization.py` | Reverse-rotation geometry used to interpret positions as angles | `src/stimulus_visualization.py` | Preserve the established geometry convention. |

## Keep Notebook-Local

| Helper(s) | Files | Repeated logic | Recommended owner layer | Why |
| --- | --- | --- | --- | --- |
| `plot_raster_gray` | Historical baseline and threshold sweep notebooks | Quick grayscale diagnostic raster plotting for sweep/reporting cells | Notebook-local | This is sweep/reporting visualization, not clearly a stable shared plotting API. |
| `safe_to_csv`, `safe_savefig` | `notebooks/calcium/preprocessing/DeltaFF_batch_pipeline.ipynb` | Convenience wrappers that save alternate files when targets are locked | Notebook-local unless reused elsewhere | Useful orchestration glue, but not scientific or contract authority. |
| `plot_angle_and_size`, `plot_right_left_on_circle` | Historical inspection notebook copies | Inspection and exploratory plotting for generated trajectories | Notebook-local | The stimulus inspection notebook should stay focused on visualization rather than becoming a shared timing or plotting owner. |
| `calculate_experiment_duration`, `calculate_experiment_duration_by_type` | `scripts/stimuli/generation/Trayectory_rocking_stimuli.py`, `scripts/stimuli/generation/Trayectory_flicker.py` | Per-script total-duration summaries for parameter/package metadata | Script-local in `scripts/stimuli/generation/` | Similar purpose, but the config semantics differ enough that this is still wrapper-specific logic. |

## Legacy Only

| Helper(s) | Files | Repeated logic | Recommended owner layer | Why |
| --- | --- | --- | --- | --- |
| Duplicate copies of `ensure_dir`, `find_experiments`, `plane_dirs`, `experiment_prefix` within one notebook | `notebooks/calcium/preprocessing/DeltaFF_batch_pipeline.ipynb` | Intra-notebook copy-paste variants of preprocessing setup helpers | Future shared `src` helper only | These duplicate sections are useful as backlog evidence, but should not become authoritative implementations. |
| Older preprocessing setup variants in sweep notebooks | Historical baseline and threshold sweep notebooks | Earlier notebook-local setup helpers overlapping with pipeline helpers | Shared `src` owner if extracted later | They show duplication, but are notebook drift rather than a source of truth. |

"""Load calcium-imaging experiments and assemble downstream analysis inputs."""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

import src.stimuli_timeline as st
from src.analysis_tools import build_trial_aligned_traces, find_file_with_suffix


def transform_stimuli_duration(stimuli_durations: Dict[str, dict]) -> Dict[str, dict]:
    """Delegate to the shared timing owner while preserving this import path."""

    return st.transform_stimuli_duration(stimuli_durations)


def _pick_latest_file(candidates):
    """
    Helper: given a list of Path objects, return the most recent one.
    Assumes the list is non-empty.
    """
    if len(candidates) == 1:
        return candidates[0]

    print("Multiple files found, using the most recent:")
    for path in candidates:
        print(" -", path.name)

    candidates = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _resolve_merged_dfof_file(dfof_dir: Path, prefix: str) -> Path:
    """
    Resolve the merged dFoF file using the canonical writer name first,
    then fall back to compatibility variants in the canonical merged folder.
    """
    preferred_path = dfof_dir / f"{prefix}_dFoF_merged.npy"
    if preferred_path.exists():
        print(f"Using canonical dFoF file: {preferred_path}")
        return preferred_path

    candidates = [
        path for path in dfof_dir.glob("*.npy")
        if "dfof_merged" in path.name.lower()
        and "filtered_roi_indices" not in path.name.lower()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No dFoF merged file found in {dfof_dir} "
            f"(looked for '{preferred_path.name}' or compatibility '*dFoF_merged*.npy')."
        )

    resolved = _pick_latest_file(candidates)
    print(f"Using compatibility dFoF file: {resolved}")
    return resolved


def _resolve_merged_map_file(dfof_dir: Path, prefix: str):
    """
    Resolve the merged map CSV using the canonical writer name first,
    then fall back to compatibility variants in the canonical merged folder.
    """
    preferred_path = dfof_dir / f"{prefix}_dFoF_merged_map.csv"
    if preferred_path.exists():
        print(f"Using canonical merged map file: {preferred_path}")
        return preferred_path

    candidates = list(dfof_dir.glob(f"{prefix}_dFoF_merged_map*.csv"))
    if not candidates:
        print(
            f"Merged map CSV not found in {dfof_dir} "
            f"(looked for '{preferred_path.name}' or compatibility variants). "
            "Continuing without map-dependent plane metadata."
        )
        return None

    resolved = _pick_latest_file(candidates)
    print(f"Using compatibility merged map file: {resolved}")
    return resolved


def _load_merged_map(map_file: Path):
    """
    Load the merged map CSV and validate the minimum columns needed
    to derive plane metadata.
    """
    try:
        dfof_merged_map = pd.read_csv(map_file)
    except Exception as exc:
        raise ValueError(
            f"Could not read merged map CSV '{map_file}'. "
            "Plane metadata cannot be built without a readable merged map."
        ) from exc

    missing_columns = {"plane"}.difference(dfof_merged_map.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Merged map CSV '{map_file}' is missing required column(s): {missing}. "
            "Plane metadata cannot be built without them."
        )

    print("dFoF_merged_map:", dfof_merged_map.shape)
    return dfof_merged_map


def _experiment_paths(
    fish_id: str, experiment_name: str, main_path: Path, stimuli_main_path: Path
) -> Dict[str, Any]:
    """Build the established experiment and output locations."""
    fish = f"{fish_id}_{experiment_name}"
    prefix = "_".join(fish.split("_")[:2])
    analysis_dir = main_path / fish / "03_analysis" / "functional"
    return {
        "fish": fish,
        "prefix": prefix,
        "stimuli_path": stimuli_main_path / experiment_name / "stimuli",
        "metadata_dir": main_path / fish / "01_raw" / "2p" / "metadata",
        "dfof_dir": analysis_dir / "suite2P" / "merged_dFoF",
        "plots_path": analysis_dir / "plots",
        "planes_dir": analysis_dir / "suite2P",
    }


def _load_optional_caches(paths: Dict[str, Any]) -> Dict[str, Any]:
    """Load available analysis caches, retaining None for missing or unreadable files."""
    prefix = paths["prefix"]
    plots_path = paths["plots_path"]
    dfof_dir = paths["dfof_dir"]
    raster = None
    deltaF_center = None
    kept_neuron_indices = None
    filtered_roi_indices = None

    sig_dir = plots_path / "significant_traces"
    sig_file = sig_dir / f"{prefix}_significant_traces.npz"

    if sig_file.exists():
        try:
            with np.load(sig_file) as data:
                raster = data["raster"] if "raster" in data.files else None
                deltaF_center = data["deltaF_center"] if "deltaF_center" in data.files else None
            print(f"Loaded significant traces from {sig_file}")
        except Exception as exc:
            print(f"Could not load {sig_file}: {exc}")
    else:
        print(f"Significant traces file not found (skipping): {sig_file}")

    kept_dir = plots_path / "filtered_neurons_by_stimuli"
    kept_file = kept_dir / f"{prefix}_kept_neuron_indices.npy"

    if kept_file.exists():
        try:
            kept_neuron_indices = np.load(kept_file)
            print(f"Loaded kept_neuron_indices from {kept_file}")
        except Exception as exc:
            print(f"Could not load {kept_file}: {exc}")
    else:
        print(f"kept_neuron_indices file not found (skipping): {kept_file}")

    filtered_roi_file = dfof_dir / f"{prefix}_dFoF_merged_filtered_roi_indices.npy"

    if filtered_roi_file.exists():
        try:
            filtered_roi_indices = np.load(filtered_roi_file)
            print(f"Loaded filtered_roi_indices from {filtered_roi_file}")
        except Exception as exc:
            print(f"Could not load {filtered_roi_file}: {exc}")
    else:
        print(f"filtered_roi_indices file not found (skipping): {filtered_roi_file}")

    z_traces = None
    zcore_dir = plots_path / "z_core"
    zcore_file = zcore_dir / f"{prefix}_zcore.npz"

    if zcore_file.exists():
        try:
            with np.load(zcore_file) as data:
                z_traces = data["z_traces"] if "z_traces" in data.files else None
            print(f"Loaded z_traces from {zcore_file}")
        except Exception as exc:
            print(f"Could not load {zcore_file}: {exc}")
    else:
        print(f"z_core file not found (skipping): {zcore_file}")

    return {
        "raster": raster,
        "deltaF_center": deltaF_center,
        "kept_neuron_indices": kept_neuron_indices,
        "filtered_roi_indices": filtered_roi_indices,
        "z_traces": z_traces,
    }


def _load_plane_metadata(paths: Dict[str, Any]) -> Dict[str, Any]:
    """Load optional merged map and its Suite2p plane images and ROI records."""
    dfof_dir = paths["dfof_dir"]
    prefix = paths["prefix"]
    planes_dir = paths["planes_dir"]
    merged_map_file = _resolve_merged_map_file(dfof_dir, prefix)
    dfof_merged_map = _load_merged_map(merged_map_file) if merged_map_file else None

    plane_ids = sorted(dfof_merged_map["plane"].unique()) if dfof_merged_map is not None else []
    mean_imgs = {}
    stat_per_plane = {}

    for plane_id in plane_ids:
        plane_path = planes_dir / plane_id
        ops_file = find_file_with_suffix(plane_path, "ops.npy")
        stat_file = find_file_with_suffix(plane_path, "stat.npy")

        ops = np.load(ops_file, allow_pickle=True).item()
        stat = np.load(stat_file, allow_pickle=True)

        mean_imgs[plane_id] = ops["meanImg"]
        stat_per_plane[plane_id] = stat

    if plane_ids:
        print("Loaded planes:", list(mean_imgs.keys()))
    else:
        print("No merged map available; skipping Suite2P plane metadata loading.")

    return {
        "merged_map_file": merged_map_file,
        "dFoF_merged_map": dfof_merged_map,
        "plane_ids": plane_ids,
        "mean_imgs": mean_imgs,
        "stat_per_plane": stat_per_plane,
    }


def _load_merged_dfof(paths: Dict[str, Any], selected_blocks, fps_2p: float) -> Dict[str, Any]:
    """Load frames x neurons dFoF and derive the per-block frame and time counts."""
    dfof_file = _resolve_merged_dfof_file(paths["dfof_dir"], paths["prefix"])
    dfof = np.load(dfof_file)

    if dfof.ndim < 2:
        raise ValueError(
            f"dFoF array has shape {dfof.shape}, expected at least 2D "
            f"(frames x neurons). Did you load an index file by mistake?"
        )

    n_frames = dfof.shape[0]
    n_blocks = len(selected_blocks)
    if n_frames % n_blocks != 0:
        raise ValueError(f"{n_frames=} not divisible by {n_blocks=} (check selected_blocks or dFoF).")

    frames_per_block = n_frames // n_blocks
    duration_2p_block_sec = frames_per_block / fps_2p

    return {
        "dfof_file": dfof_file,
        "dfof": dfof,
        "frames_per_block": frames_per_block,
        "duration_2p_block_sec": duration_2p_block_sec,
    }


def _load_stimulus_timing(
    paths: Dict[str, Any], selected_blocks, duration_2p_block_sec: float
) -> Dict[str, Any]:
    """Find stimulus trajectories and the block log, then build 60 Hz timing."""
    stimuli_path = paths["stimuli_path"]
    metadata_dir = paths["metadata_dir"]
    stimuli_durations = {}
    for stim_file in stimuli_path.glob("*trajectory.*"):
        filename = stim_file.stem
        stim_name = filename.replace("_trajectory", "")

        stimuli_durations[stim_name] = st.get_motion_timing_simple(
            stim_file,
            framerate=60,
            include_xy=True,
            include_radius=True,
        )

    stimuli_durations = st.transform_stimuli_duration(stimuli_durations)

    if not stimuli_durations:
        raise FileNotFoundError(
            f"No stimulus trajectory files found in {stimuli_path} "
            "with pattern '*trajectory.*'. Check the stimuli_path and file names."
        )

    block_logs = list(metadata_dir.glob("*block_log.csv"))
    if not block_logs:
        raise FileNotFoundError(f"No '*block_log.csv' file found in {metadata_dir}")

    experiment_log_path = _pick_latest_file(block_logs)
    print("Using block log:", experiment_log_path)

    adjusted_log, stimuli_trace_60, stimuli_table, stimuli_id_map = st.make_stimulus_traces_2(
        experiment_log_path,
        stimuli_durations,
        selected_blocks,
        duration_2p_block_sec,
    )

    return {
        "experiment_log_path": experiment_log_path,
        "stimuli_durations": stimuli_durations,
        "adjusted_log": adjusted_log,
        "stimuli_trace_60": stimuli_trace_60,
        "stimuli_table": stimuli_table,
        "stimuli_id_map": stimuli_id_map,
    }


def _assemble_experiment_bundle(
    paths: Dict[str, Any],
    caches: Dict[str, Any],
    planes: Dict[str, Any],
    imaging: Dict[str, Any],
    timing: Dict[str, Any],
    fps_2p: float,
) -> Dict[str, Any]:
    """Preserve the public bundle keys, array order, and path dictionary contract."""
    public_paths = {
        "fish": paths["fish"],
        "prefix": paths["prefix"],
        "stimuli_path": paths["stimuli_path"],
        "metadata_dir": paths["metadata_dir"],
        "dfof_dir": paths["dfof_dir"],
        "plots_path": paths["plots_path"],
        "experiment_log_path": timing["experiment_log_path"],
        "dfof_file": imaging["dfof_file"],
        "merged_map_file": planes["merged_map_file"],
        "planes_dir": paths["planes_dir"],
    }

    return {
        "dfof": imaging["dfof"],
        "fps_2p": fps_2p,
        "frames_per_block": imaging["frames_per_block"],
        "duration_2p_block_sec": imaging["duration_2p_block_sec"],
        "stimuli_durations": timing["stimuli_durations"],
        "adjusted_log": timing["adjusted_log"],
        "stimuli_trace_60": timing["stimuli_trace_60"],
        "stimuli_table": timing["stimuli_table"],
        "stimuli_id_map": timing["stimuli_id_map"],
        "paths": public_paths,
        "dFoF_merged_map": planes["dFoF_merged_map"],
        "z_traces": caches["z_traces"],
        "raster": caches["raster"],
        "deltaF_center": caches["deltaF_center"],
        "kept_neuron_indices": caches["kept_neuron_indices"],
        "filtered_roi_indices": caches["filtered_roi_indices"],
        "plane_ids": planes["plane_ids"],
        "mean_imgs": planes["mean_imgs"],
        "stat_per_plane": planes["stat_per_plane"],
    }


def load_2p_experiment(
    fish_id: str,
    experiment_name: str,
    main_path: Path,
    stimuli_main_path: Path,
    fps_2p: float = 2.0,
    selected_blocks=None,
) -> Dict[str, Any]:
    """Load one 2P experiment and return its imaging, timing, and cache bundle.

    ``dfof`` is frames x neurons, ``stimuli_trace_60`` is sampled at 60 Hz,
    and ``duration_2p_block_sec`` is in seconds. Optional caches and the
    merged map are ``None`` when absent; their public keys remain present.
    """
    if selected_blocks is None:
        selected_blocks = [f"B{n}" for n in range(1, 3)]

    paths = _experiment_paths(fish_id, experiment_name, main_path, stimuli_main_path)
    caches = _load_optional_caches(paths)
    planes = _load_plane_metadata(paths)
    imaging = _load_merged_dfof(paths, selected_blocks, fps_2p)
    timing = _load_stimulus_timing(paths, selected_blocks, imaging["duration_2p_block_sec"])
    return _assemble_experiment_bundle(paths, caches, planes, imaging, timing, fps_2p)


def load_and_align_2p_experiment(
    fish_id: str,
    experiment_name: str,
    main_path: Path,
    stimuli_main_path: Path,
    fps_2p: float,
    selected_blocks,
    t_pre_s: float,
    t_post_s: float,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Load one fish experiment and build aligned dFoF, raster, normalized, and z-score traces.

    This is the reusable owner-layer version of the several-fish notebook
    loading cell. It preserves the notebook data shape expected by
    ``src.multifish_analysis`` helpers.
    """
    results = load_2p_experiment(
        fish_id=fish_id,
        experiment_name=experiment_name,
        main_path=main_path,
        stimuli_main_path=stimuli_main_path,
        fps_2p=fps_2p,
        selected_blocks=selected_blocks,
    )

    required_arrays = ["dfof", "raster", "deltaF_center", "z_traces"]
    missing_arrays = [name for name in required_arrays if results.get(name) is None]
    if missing_arrays:
        raise ValueError(
            "Cannot build aligned experiment bundle; missing cached array(s): "
            + ", ".join(missing_arrays)
        )

    dfof_center = np.asarray(results["deltaF_center"], dtype=float)
    min_val = np.nanmin(dfof_center)
    max_val = np.nanmax(dfof_center)
    value_range = max_val - min_val + 1e-12
    dfof_center_norm = (dfof_center - min_val) / value_range

    align_kwargs = {
        "stimuli_trace_60": results["stimuli_trace_60"],
        "fps_2p": fps_2p,
        "t_pre_s": t_pre_s,
        "t_post_s": t_post_s,
        "stimuli_id_map": results["stimuli_id_map"],
        "verbose": verbose,
    }
    aligned_dfof = build_trial_aligned_traces(dfof=results["dfof"], **align_kwargs)
    aligned_raster = build_trial_aligned_traces(dfof=results["raster"], **align_kwargs)
    aligned_norm = build_trial_aligned_traces(dfof=dfof_center_norm, **align_kwargs)
    aligned_zscore = build_trial_aligned_traces(dfof=results["z_traces"], **align_kwargs)

    return {
        "trial_aligned_traces": aligned_dfof["trial_aligned_traces"],
        "stimuli_ids": aligned_dfof["stimuli_ids"],
        "stimuli_names": aligned_dfof["stimuli_names"],
        "win_length": aligned_dfof["win_length"],
        "stimuli_trace": aligned_dfof["stimuli_trace"],
        "onsets_by_id": aligned_dfof["onsets_by_id"],
        "trial_aligned_traces_raster": aligned_raster["trial_aligned_traces"],
        "trial_aligned_traces_norm": aligned_norm["trial_aligned_traces"],
        "trial_aligned_traces_z_core": aligned_zscore["trial_aligned_traces"],
        "dfof": results["dfof"],
        "raster": results["raster"],
        "dfof_center": results["deltaF_center"],
        "dFoF_merged_map": results["dFoF_merged_map"],
        "stimuli_durations": results["stimuli_durations"],
        "kept_neuron_indices": results["kept_neuron_indices"],
        "stimuli_id_map": results["stimuli_id_map"],
        "meta": {
            "frames_per_block": results["frames_per_block"],
            "duration_2p_block_sec": results["duration_2p_block_sec"],
            "paths": results["paths"],
            "plane_ids": results["plane_ids"],
            "stat_per_plane": results["stat_per_plane"],
        },
    }

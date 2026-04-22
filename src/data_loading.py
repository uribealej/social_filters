from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

import src.stimuli_timeline as st
from src.analysis_tools import find_file_with_suffix


def transform_stimuli_duration(stimuli_durations: Dict[str, dict]) -> Dict[str, dict]:
    """
    Normalize per-stimulus timing dictionaries:
      - add 'motion_sec' (total_sec - static_before_sec)
      - set 'static_after_sec' = 0
      - ensure 'motion_end_frame' and 'end_frame' use 'total_frames' if present
    """
    out = {}
    for k, v in stimuli_durations.items():
        total_sec = v.get("total_sec", 0)
        static_before = v.get("static_before_sec", 0)
        total_frames = v.get("total_frames", v.get("motion_end_frame"))

        new_v = v.copy()
        new_v["motion_sec"] = round(total_sec - static_before, 3)
        new_v["static_after_sec"] = 0

        if total_frames is not None:
            new_v["motion_end_frame"] = total_frames
            new_v["end_frame"] = total_frames

        out[k] = new_v
    return out


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


def load_2p_experiment(
    fish_id: str,
    experiment_name: str,
    main_path: Path,
    stimuli_main_path: Path,
    fps_2p: float = 2.0,
    selected_blocks=None,
) -> Dict[str, Any]:
    """
    High-level loader for a 2P experiment.

    It will:
      - build paths from fish_id + experiment_name
      - load the merged dFoF file
      - load & transform stimuli timing info
      - find the correct block_log CSV
      - call st.make_stimulus_traces_2

    Returns a dict with:
      - 'dfof':            np.ndarray (frames, neurons)
      - 'fps_2p':          float
      - 'frames_per_block':int
      - 'duration_2p_block_sec': float
      - 'stimuli_durations': dict
      - 'adjusted_log':    DataFrame (from make_stimulus_traces_2)
      - 'stimuli_trace_60':np.ndarray
      - 'stimuli_table':   DataFrame
      - 'stimuli_id_map':  dict
      - 'paths':           dict with various Path objects and names
    """
    if selected_blocks is None:
        selected_blocks = [f"B{n}" for n in range(1, 3)]

    fish = f"{fish_id}_{experiment_name}"
    prefix = "_".join(fish.split("_")[:2])

    stimuli_path = stimuli_main_path / experiment_name / "stimuli"
    metadata_dir = main_path / fish / "01_raw" / "2p" / "metadata"
    dfof_dir = main_path / fish / "03_analysis" / "functional" / "suite2P" / "merged_dFoF"
    plots_path = main_path / fish / "03_analysis" / "functional" / "plots"
    planes_dir = main_path / fish / "03_analysis" / "functional" / "suite2P"

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

    dfof_file = _resolve_merged_dfof_file(dfof_dir, prefix)
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

    stimuli_durations = transform_stimuli_duration(stimuli_durations)

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

    paths = {
        "fish": fish,
        "prefix": prefix,
        "stimuli_path": stimuli_path,
        "metadata_dir": metadata_dir,
        "dfof_dir": dfof_dir,
        "plots_path": plots_path,
        "experiment_log_path": experiment_log_path,
        "dfof_file": dfof_file,
        "merged_map_file": merged_map_file,
        "planes_dir": planes_dir,
    }

    return {
        "dfof": dfof,
        "fps_2p": fps_2p,
        "frames_per_block": frames_per_block,
        "duration_2p_block_sec": duration_2p_block_sec,
        "stimuli_durations": stimuli_durations,
        "adjusted_log": adjusted_log,
        "stimuli_trace_60": stimuli_trace_60,
        "stimuli_table": stimuli_table,
        "stimuli_id_map": stimuli_id_map,
        "paths": paths,
        "dFoF_merged_map": dfof_merged_map,
        "z_traces": z_traces,
        "raster": raster,
        "deltaF_center": deltaF_center,
        "kept_neuron_indices": kept_neuron_indices,
        "filtered_roi_indices": filtered_roi_indices,
        "plane_ids": plane_ids,
        "mean_imgs": mean_imgs,
        "stat_per_plane": stat_per_plane,
    }

from pathlib import Path
from typing import Dict, Any

import numpy as np
import src.stimuli_timeline as st


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

    print("⚠️ Multiple files found, using the most recent:")
    for p in candidates:
        print(" -", p.name)

    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


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

    # ------------------------------------------------------------------ Paths
    fish = f"{fish_id}_{experiment_name}"

    stimuli_path = stimuli_main_path / experiment_name / "stimuli"
    metadata_dir = main_path / fish / "01_raw" / "2p" / "metadata"
    dfof_dir = main_path / fish / "03_analysis" / "functional" / "suite2P" / "merged_dFoF"
    plots_path = main_path / fish / "03_analysis" / "functional" / "plots"
    prefix = "_".join(fish.split("_")[:2])

    # ------------------------------------------------------------------ Optional: precomputed traces & indices
    raster = None
    deltaF_center = None
    kept_neuron_indices = None
    filtered_roi_indices = None

    # 1) significant_traces.npz   (raster, deltaF_center)
    sig_dir = plots_path / "significant_traces"
    sig_file = sig_dir / f"{prefix}_significant_traces.npz"

    if sig_file.exists():
        try:
            with np.load(sig_file) as data:
                raster = data["raster"] if "raster" in data.files else None
                deltaF_center = (
                    data["deltaF_center"] if "deltaF_center" in data.files else None
                )
            print(f"✅ Loaded significant traces from {sig_file}")
        except Exception as e:
            print(f"⚠️ Could not load {sig_file}: {e}")
    else:
        print(f"ℹ️ Significant traces file not found (skipping): {sig_file}")

    # 2) L433_f02_kept_neuron_indices.npy
    kept_dir = plots_path / "filtered_neurons_by_stimuli"
    kept_file = kept_dir / f"{prefix}_kept_neuron_indices.npy"

    if kept_file.exists():
        try:
            kept_neuron_indices = np.load(kept_file)
            print(f"✅ Loaded kept_neuron_indices from {kept_file}")
        except Exception as e:
            print(f"⚠️ Could not load {kept_file}: {e}")
    else:
        print(f"ℹ️ kept_neuron_indices file not found (skipping): {kept_file}")

    # 3) L433_f02_dFoF_merged_filtered_roi_indices.npy
    filtered_roi_file = dfof_dir / f"{prefix}_dFoF_merged_filtered_roi_indices.npy"

    if filtered_roi_file.exists():
        try:
            filtered_roi_indices = np.load(filtered_roi_file)
            print(f"✅ Loaded filtered_roi_indices from {filtered_roi_file}")
        except Exception as e:
            print(f"⚠️ Could not load {filtered_roi_file}: {e}")
    else:
        print(f"ℹ️ filtered_roi_indices file not found (skipping): {filtered_roi_file}")



    # ------------------------------------------------------------------ dFoF
    # 1) Prefer the exact merged file: <fish_id>_dFoF_merged.npy
    preferred_name = f"{fish_id}_dFoF_merged.npy"
    preferred_path = dfof_dir / preferred_name

    if preferred_path.exists():
        dfof_file = preferred_path
        print("✅ Using preferred dFoF file:", dfof_file)
    else:
        # 2) Fallback: search for *dfof_merged*.npy, but EXCLUDE filtered_roi_indices
        candidates = [
            p for p in dfof_dir.glob("*.npy")
            if "dfof_merged" in p.name.lower()
               and "filtered_roi_indices" not in p.name.lower()
        ]

        if not candidates:
            raise FileNotFoundError(
                f"No dFoF merged file found in {dfof_dir} "
                f"(looked for {preferred_name!r} or *dFoF_merged*.npy)."
            )

        dfof_file = _pick_latest_file(candidates)
        print("✅ Using dFoF file (fallback search):", dfof_file)

    dfof = np.load(dfof_file)

    # basic sanity check: expect at least 2D array
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
    # ------------------------------------------------------------------ Stimuli durations
    stimuli_durations = {}
    for stim_file in stimuli_path.glob("*trajectory.*"):
        filename = stim_file.stem  # e.g. 'FL2_trajectory'
        stim_name = filename.replace("_trajectory", "")

        # Use timing function depending on stimulus type
        if any(s in filename for s in ("B", "RR", "RL")):
            stimuli_durations[stim_name] = st.get_stimulus_timing(stim_file)
        else:
            stimuli_durations[stim_name] = st.get_radius_timing(stim_file)

    stimuli_durations = transform_stimuli_duration(stimuli_durations)

    if not stimuli_durations:
        raise FileNotFoundError(
            f"No stimulus trajectory files found in {stimuli_path} "
            f"with pattern '*trajectory.*'. "
            f"Check the stimuli_path and file names."
        )
    # ------------------------------------------------------------------ Block log (metadata)
    block_logs = list(metadata_dir.glob("*block_log.csv"))
    if not block_logs:
        raise FileNotFoundError(f"No '*block_log.csv' file found in {metadata_dir}")

    experiment_log_path = _pick_latest_file(block_logs)
    print("✅ Using block log:", experiment_log_path)

    # ------------------------------------------------------------------ Stimulus traces (your new function)
    adjusted_log, stimuli_trace_60, stimuli_table, stimuli_id_map = st.make_stimulus_traces_2(
        experiment_log_path,
        stimuli_durations,
        selected_blocks,
        duration_2p_block_sec,
    )

    # ------------------------------------------------------------------ Pack results
    paths = {
        "fish": fish,
        "prefix": prefix,
        "stimuli_path": stimuli_path,
        "metadata_dir": metadata_dir,
        "dfof_dir": dfof_dir,
        "plots_path": plots_path,
        "experiment_log_path": experiment_log_path,
        "dfof_file": dfof_file,
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
        # Optional extras (may be None if files are missing):
        "raster": raster,
        "deltaF_center": deltaF_center,
        "kept_neuron_indices": kept_neuron_indices,
        "filtered_roi_indices": filtered_roi_indices,
    }

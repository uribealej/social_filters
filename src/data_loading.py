"""Load calcium-imaging experiments and assemble downstream analysis inputs."""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

import src.stimuli_timeline as st
from src.analysis_io import find_file_with_suffix
from src.trial_alignment import build_trial_aligned_traces


def load_phase_aligned_experiment(fish_id, experiment_name, main_path,
                                  stimuli_main_path, phase_metadata_path,
                                  stimulus_aliases, selected_blocks,
                                  session_id=None, baseline_s=10.0, expected_fps_2p=2.0):
    """Load one complete, explicitly phased recording without legacy inference.

    Select a matching ``*_block_log.csv``/metadata/sequence session. Ambiguous
    sessions fail unless ``session_id`` is supplied (full filename prefix).
    Require B1..Bn concatenation in acquisition order, matching metadata volume
    count and the frames-by-neurons dFoF. Does not load or trust older caches.
    Returns paths, metadata, phases, trials, dFoF, rate, condition order and an
    audit table. This opt-in entry point leaves ``load_2p_experiment`` unchanged.
    """
    paths = _experiment_paths(fish_id, experiment_name, Path(main_path), Path(stimuli_main_path))
    logs = sorted(paths["metadata_dir"].glob("*_block_log.csv"))
    if session_id is not None:
        logs = [p for p in logs if p.name == f"{session_id}_block_log.csv"]
    if len(logs) != 1:
        raise ValueError(f"Select one recording session in {paths['metadata_dir']}; candidates: {[p.name for p in logs]}")
    log_path = logs[0]
    session = log_path.name.removesuffix("_block_log.csv")
    metadata_path = log_path.with_name(f"{session}_metadata.csv")
    sequence_path = log_path.with_name(f"{session}_trial_sequence.csv")
    experiment_log_path = log_path.with_name(f"{session}_experiment_log.csv")
    meta_rows = pd.read_csv(metadata_path, keep_default_na=False)
    if meta_rows.parameter.duplicated().any():
        raise ValueError("Duplicate recording metadata parameters.")
    metadata = dict(zip(meta_rows.parameter, meta_rows.value))
    if metadata.get("fish_ID") != fish_id or metadata.get("experiment_name") != experiment_name:
        raise ValueError("Recording metadata identity does not match selected fish/experiment.")
    fps = float(metadata["framerate"])
    if fps != expected_fps_2p:
        raise ValueError(f"Expected {expected_fps_2p:g}-Hz imaging, got {fps}.")
    blocks = list(selected_blocks)
    repetitions = int(metadata["n_rep_stim"])
    if blocks != [f"B{i}" for i in range(1, repetitions + 1)]:
        raise ValueError("Select all stimulus blocks in acquisition order; exclude resting B0.")
    frames = int(metadata["n_volumes"])
    if int(metadata["n_trials_per_block"]) != len(stimulus_aliases):
        raise ValueError("Metadata trial count disagrees with configured conditions.")
    imaging = _load_merged_dfof(paths, blocks, fps)
    if imaging["dfof"].ndim != 2 or imaging["frames_per_block"] != frames:
        raise ValueError(f"Incompatible frame count: {imaging['dfof'].shape}; expected {len(blocks)} x {frames} samples.")
    phases = st.validate_phase_metadata(pd.read_csv(phase_metadata_path), paths["stimuli_path"])
    block_log = pd.read_csv(log_path)
    trials = st.build_phase_trial_table(block_log,
        pd.read_csv(sequence_path).stimulus.tolist(), phases, stimulus_aliases,
        blocks, frames, fps, baseline_s)
    # Check the independently saved global clock against block-local clocks.
    global_log = pd.read_csv(experiment_log_path).set_index("event").timestamp
    local = block_log.set_index("event").timestamp
    for block in blocks:
        names = [e for e in local.index if e.startswith(block + "_")]
        if not set(names) <= set(global_log.index):
            raise ValueError(f"Experiment log is missing {block} events.")
        discrepancy = ((global_log.loc[names] - global_log[block + "_start"])
                       - (local.loc[names] - local[block + "_start"]))
        if not np.isfinite(discrepancy).all() or discrepancy.abs().max() >= 1 / fps:
            raise ValueError(f"Global and block logs disagree by an imaging frame in {block}.")
    audit = trials.groupby("block", sort=False).agg(
        trials=("stimulus_id", "size"), conditions=("stimulus_id", "nunique"),
        max_duration_error_s=("duration_error_s", lambda x: float(x.abs().max())),
        min_pre_pause_s=("pre_pause_s", "min"), min_post_pause_s=("post_pause_s", "min"))
    paths.update(experiment_log_path=log_path, recording_experiment_log_path=experiment_log_path,
                 metadata_file=metadata_path, sequence_file=sequence_path,
                 phase_metadata_file=Path(phase_metadata_path), dfof_file=imaging["dfof_file"])
    return dict(paths=paths, metadata=metadata, session_id=session,
                selected_blocks=blocks, stimulus_aliases=dict(stimulus_aliases),
                baseline_s=float(baseline_s), phases=phases, trials=trials,
                dfof=imaging["dfof"], fps_2p=fps, frames_per_block=frames,
                stimuli_ids=list(range(1, len(stimulus_aliases) + 1)),
                stimuli_id_map={name: i + 1 for i, name in enumerate(stimulus_aliases.values())},
                audit=audit)


def load_reviewed_phase_experiment(fish_id, experiment_name, main_path, stimuli_main_path,
                                   phase_metadata_path, stimulus_aliases, selected_blocks,
                                   cache_root=None):
    """Read reviewed phase caches and verify original neuron/session identities.

    Does not recompute normalization, reliability or significant activity and
    never writes files. Older phase caches may obtain detector settings/rate
    from their saved configuration and recording metadata. A mismatch raises.
    """
    import json

    paths = _experiment_paths(fish_id, experiment_name, Path(main_path), Path(stimuli_main_path))
    root = Path(cache_root) if cache_root is not None else paths["plots_path"]
    prefix = paths["prefix"]
    files = dict(configuration=root / "phase_alignment" / f"{prefix}_analysis_configuration.json",
        trial_timing=root / "phase_alignment" / f"{prefix}_trial_timing.csv",
        zscore=root / "z_core" / f"{prefix}_zcore.npz",
        kept=root / "filtered_neurons_by_stimuli" / f"{prefix}_kept_neuron_indices.npy",
        significant=root / "significant_traces" / f"{prefix}_significant_traces.npz")
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing reviewed outputs: " + "; ".join(missing))
    config = json.loads(files["configuration"].read_text(encoding="utf-8"))
    if config["selected_blocks"] != list(selected_blocks) or config["stimulus_aliases"] != dict(stimulus_aliases):
        raise ValueError("Saved blocks or stimulus aliases differ from cohort configuration.")
    loaded = load_phase_aligned_experiment(fish_id, experiment_name, main_path,
        stimuli_main_path, phase_metadata_path, stimulus_aliases, selected_blocks,
        session_id=config["session_id"], baseline_s=config["configuration"]["baseline_s"])
    if config["paths"]["fish"] != loaded["paths"]["fish"]:
        raise ValueError("Saved fish identity differs from the recording.")
    saved_trials = pd.read_csv(files["trial_timing"])
    expected_trials = loaded["trials"]
    for col in ("block", "trial_index", "stimulus_key", "stimulus_id"):
        if saved_trials[col].tolist() != expected_trials[col].tolist():
            raise ValueError(f"Saved trial identity/order differs: {col}.")
    for col in ("onset_s", "pre_end_s", "bout_onset_s", "end_s", "logged_end_s"):
        if not np.allclose(saved_trials[col], expected_trials[col], atol=1e-7, rtol=0):
            raise ValueError(f"Saved trial timing differs from current logs/metadata: {col}.")
    kept = np.load(files["kept"], allow_pickle=False)
    if (kept.ndim != 1 or not np.issubdtype(kept.dtype, np.integer) or not len(kept)
            or len(np.unique(kept)) != len(kept) or (kept < 0).any()
            or (kept >= loaded["dfof"].shape[1]).any()
            or not np.array_equal(kept, config["kept_neuron_indices"])):
        raise ValueError("Invalid or inconsistent saved retained-neuron mapping.")
    with np.load(files["zscore"], allow_pickle=False) as z:
        zscore = z["z_traces"]
    settings = config["configuration"].get("significant")
    if not isinstance(settings, dict) or not settings:
        raise ValueError("Saved detector settings are missing.")
    with np.load(files["significant"], allow_pickle=False) as z:
        raster = z["raster"]
        if not np.array_equal(z["kept_neuron_indices"], kept):
            raise ValueError("Significant-cache neuron IDs differ from retained-index file.")
        if z["deltaF_center"].shape != raster.shape:
            raise ValueError("Centered traces and raster have incompatible shapes.")
        if "settings_json" in z and json.loads(str(z["settings_json"])) != settings:
            raise ValueError("Significant-cache detector settings differ from provenance.")
        if "fps" in z and float(z["fps"]) != loaded["fps_2p"]:
            raise ValueError("Saved imaging rates differ.")
        if "source_file" in z and Path(str(z["source_file"])).resolve() != Path(loaded["paths"]["dfof_file"]).resolve():
            raise ValueError("Saved significant-cache source file differs.")
    if zscore.shape != loaded["dfof"].shape or raster.shape != (len(zscore), len(kept)):
        raise ValueError("Saved trace/raster shapes differ from source data and retained IDs.")
    if not np.isin(raster[np.isfinite(raster)], [0, 1]).all():
        raise ValueError("Saved significant raster is not binary.")
    loaded.update(z_traces=zscore, raster=raster, kept_neuron_indices=kept,
                  detector_settings=settings, saved_configuration=config, cache_files=files)
    return loaded


def save_phase_analysis_outputs(loaded, zres, reliability_results, significant_results,
                                configuration, output_root=None, significant_run=None,
                                review_figures=None):
    """Save canonical caches and timing provenance; optionally redirect for validation.

    Significant columns follow ``kept_neuron_indices`` in merged-neuron order.
    ``significant_results`` is the canonical detector's eight-item result.
    Existing canonical caches are replaced only when this writer is called.
    Optional ``significant_run`` captures the detector's actual settings, session,
    source, fps and original neuron IDs. Reject changed settings or selections
    before writing. Optional review figures are saved beside the canonical NPZ.
    """
    import json

    kept = np.asarray(reliability_results["kept_neuron_indices"])
    expected = (loaded["dfof"].shape[0], len(kept))
    if (np.asarray(zres["z_traces"]).shape != loaded["dfof"].shape
            or np.asarray(significant_results[6]).shape != expected
            or np.asarray(significant_results[1]).shape != expected):
        raise ValueError("Cache shapes do not match this recording and retained neuron mapping.")
    if (kept.ndim != 1 or not np.issubdtype(kept.dtype, np.integer)
            or len(np.unique(kept)) != len(kept) or (kept < 0).any()
            or (kept >= loaded["dfof"].shape[1]).any()):
        raise ValueError("Invalid retained merged-neuron IDs.")
    if significant_run is not None:
        if (significant_run["fish"] != loaded["paths"]["fish"]
                or significant_run["session_id"] != loaded["session_id"]
                or Path(significant_run["source_file"]).resolve() != Path(loaded["paths"]["dfof_file"]).resolve()
                or significant_run["fps"] != loaded["fps_2p"]
                or not np.array_equal(significant_run["neuron_indices"], kept)
                or significant_run["settings"] != configuration.get("significant")):
            raise ValueError("Significant-trace inputs/settings changed: rerun detection and review before saving.")
    figures = dict(review_figures or {})
    if any(Path(name).name != name or Path(name).suffix != ".png" for name in figures):
        raise ValueError("Review figure names must be plain PNG filenames.")
    root = Path(output_root) if output_root is not None else loaded["paths"]["plots_path"]
    prefix = loaded["paths"]["prefix"]
    for name in ("z_core", "filtered_neurons_by_stimuli", "significant_traces", "phase_alignment"):
        (root / name).mkdir(parents=True, exist_ok=True)
    np.savez(root / "z_core" / f"{prefix}_zcore.npz", **{
        k: zres[k] for k in ("z_traces", "baseline_mean", "baseline_std", "valid_onsets")})
    np.save(root / "filtered_neurons_by_stimuli" / f"{prefix}_kept_neuron_indices.npy", kept)
    np.savez(root / "significant_traces" / f"{prefix}_significant_traces.npz",
             raster=significant_results[6], deltaF_center=significant_results[1],
             kept_neuron_indices=kept, fps=loaded["fps_2p"],
             source_file=str(loaded["paths"]["dfof_file"]),
             settings_json=json.dumps(significant_run["settings"] if significant_run is not None
                                      else configuration.get("significant", {})))
    for name, figure in figures.items():
        figure.savefig(root / "significant_traces" / name, dpi=180, bbox_inches="tight")
    loaded["trials"].to_csv(root / "phase_alignment" / f"{prefix}_trial_timing.csv", index=False)
    provenance = dict(session_id=loaded["session_id"], paths=loaded["paths"],
        metadata=loaded["metadata"], stimulus_aliases=loaded["stimulus_aliases"],
        selected_blocks=loaded["selected_blocks"], configuration=configuration,
        timing_method="shared block trigger; recorded presentation timestamps; nominal 2-Hz imaging samples",
        reliability_threshold=float(reliability_results["otsu_threshold"]),
        kept_neuron_indices=kept.tolist(), significant_run=significant_run,
        review_figures=list(figures))
    out = root / "phase_alignment" / f"{prefix}_analysis_configuration.json"
    out.write_text(json.dumps(provenance, indent=2, default=str) + "\n", encoding="utf-8")
    return out


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


def _resolve_merged_map_file(dfof_dir: Path, prefix: str, expected_neurons: int):
    """
    Resolve a merged map with one row per dFoF neuron. Prefer the canonical
    writer name when it matches; a timestamped writer fallback may be newer.
    """
    preferred_path = dfof_dir / f"{prefix}_dFoF_merged_map.csv"
    candidates = list(dfof_dir.glob(f"{prefix}_dFoF_merged_map*.csv"))
    if not candidates:
        print(
            f"Merged map CSV not found in {dfof_dir} "
            f"(looked for '{preferred_path.name}' or compatibility variants). "
            "Continuing without map-dependent plane metadata."
        )
        return None

    ordered = ([preferred_path] if preferred_path in candidates else []) + sorted(
        (path for path in candidates if path != preferred_path),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    row_counts = {}
    for path in ordered:
        try:
            row_counts[path.name] = len(pd.read_csv(path))
        except Exception as exc:
            raise ValueError(f"Could not read merged map CSV '{path}'.") from exc
        if row_counts[path.name] == expected_neurons:
            label = "canonical" if path == preferred_path else "compatibility"
            print(f"Using {label} merged map file: {path}")
            return path
    raise ValueError(
        f"No merged map CSV in {dfof_dir} matches the merged dFoF's "
        f"{expected_neurons} neuron columns; map row counts: {row_counts}."
    )


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


def _load_plane_metadata(paths: Dict[str, Any], expected_neurons: int) -> Dict[str, Any]:
    """Load optional merged map and its Suite2p plane images and ROI records."""
    dfof_dir = paths["dfof_dir"]
    prefix = paths["prefix"]
    planes_dir = paths["planes_dir"]
    merged_map_file = _resolve_merged_map_file(dfof_dir, prefix, expected_neurons)
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
    An available map must have one row per merged neuron. A matching timestamped
    map is accepted when the canonical map is stale.
    """
    if selected_blocks is None:
        selected_blocks = [f"B{n}" for n in range(1, 3)]

    paths = _experiment_paths(fish_id, experiment_name, main_path, stimuli_main_path)
    caches = _load_optional_caches(paths)
    imaging = _load_merged_dfof(paths, selected_blocks, fps_2p)
    planes = _load_plane_metadata(paths, imaging["dfof"].shape[1])
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

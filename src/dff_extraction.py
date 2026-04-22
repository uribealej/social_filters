import json
import time
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d


def _find_unique_file_by_patterns(folder: Path, patterns: tuple[str, ...], label: str) -> Path:
    """Resolve one Suite2p input file from a small set of allowed filename patterns."""
    matches = []
    for pattern in patterns:
        matches.extend(folder.glob(pattern))

    unique_matches = sorted({path.resolve() for path in matches})
    if len(unique_matches) == 0:
        raise FileNotFoundError(
            f"No {label} file found in {folder}. Tried patterns: {patterns}"
        )
    if len(unique_matches) > 1:
        raise RuntimeError(
            f"Multiple {label} files found in {folder}: {unique_matches}"
        )
    return unique_matches[0]


def _resolve_suite2p_inputs(folder: Path) -> Tuple[Path, Path]:
    """Find fluorescence and cell-mask inputs for one Suite2p plane."""
    fluorescence_file = _find_unique_file_by_patterns(
        folder,
        ("F.npy", "*_F.npy"),
        "fluorescence",
    )
    iscell_file = _find_unique_file_by_patterns(
        folder,
        ("iscell.npy", "*_iscell.npy"),
        "iscell",
    )
    return fluorescence_file, iscell_file


def load_fluorescence_data(path):
    """
    Load raw fluorescence time-series data from Suite2p output.

    Parameters:
    - path (Path): Path to F.npy file.

    Returns:
    - np.ndarray: Fluorescence trace data (T x N).
    """
    fluorescence_trace = np.load(path).T
    return fluorescence_trace


def filter_dim_rois(fluorescence_trace, threshold_std=2):
    """
    Remove low-intensity (dim) ROIs based on mean fluorescence level.

    Parameters:
    - fluorescence_trace (np.ndarray): Fluorescence trace (T x N).
    - threshold_std (float): Threshold in standard deviations.

    Returns:
    - np.ndarray: Filtered fluorescence trace (T x N_filtered).
    - np.ndarray: Boolean mask indicating retained ROIs.
    """
    mean_fluo = np.mean(fluorescence_trace, axis=0)
    mu, sigma = np.mean(mean_fluo), np.std(mean_fluo)
    bright_rois_mask = mean_fluo >= mu - threshold_std * sigma
    return fluorescence_trace[:, bright_rois_mask], bright_rois_mask


def compute_percentile_baseline(
        fluorescence_trace,
        fps,
        tau,
        percentile=8,
        instability_ratio=0.1,
        min_window_s=15,
        window_tau_multiplier=40,
):
    """
    Compute smooth F0_baseline (F0) using sliding percentile window and stability filtering.

    Parameters:
    - fluorescence_trace (np.ndarray): Fluorescence data (T x N)
    - fps (float): Imaging rate in Hz.
    - tau (float): Indicator decay time constant (seconds).
    - percentile (int): Percentile for F0_baseline estimation.
    - instability_ratio (float): If F0 drops more than this ratio, ROI is unstable.
    - min_window_s (float): Minimum window size (seconds).
    - window_tau_multiplier (float): Multiplier of tau to compute window size.

    Returns:
    - np.ndarray: Baseline matrix F0 (T x N), NaN for unstable ROIs.
    """
    T, N = fluorescence_trace.shape
    window_s = max(min_window_s, window_tau_multiplier * tau)
    window_frames = int(window_s * fps)

    F0_baseline = np.full_like(fluorescence_trace, np.nan)

    for n in range(N):
        trace = fluorescence_trace[:, n]
        local_baseline = np.zeros_like(trace)

        for t in range(T):
            start = max(0, t - window_frames)
            end = min(T, t + window_frames + 1)
            local_baseline[t] = np.percentile(trace[start:end], percentile)

        if np.min(local_baseline) < instability_ratio * np.max(local_baseline):
            continue

        F0_baseline[:, n] = uniform_filter1d(local_baseline, size=window_frames)

    return F0_baseline


def compute_dff(fluorescence_trace, F0_baseline):
    """
    Compute delta F over F0.

    Parameters:
    - fluorescence_trace (np.ndarray): Cleaned fluorescence trace (T x N).
    - baseline (np.ndarray): Baseline F0 estimate (T x N).

    Returns:
    - np.ndarray: dF/F0 traces (T x N).
    """
    baseline_safe = np.where(F0_baseline == 0, np.finfo(float).eps, F0_baseline)
    return (fluorescence_trace - F0_baseline) / baseline_safe


def _filter_cell_rois(
        fluorescence_trace: np.ndarray,
        iscell_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep only ROIs classified as cells and return original Suite2p indices."""
    original_indices = np.where(iscell_mask)[0]
    cell_trace = fluorescence_trace[:, iscell_mask]
    return cell_trace, original_indices


def _filter_stable_rois(
        filtered_trace: np.ndarray,
        fps: float,
        tau: float,
        percentile: float,
        instability_ratio: float,
        min_window_s: float,
        window_tau_multiplier: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute baselines, drop unstable ROIs, and return clean traces plus kept mask."""
    F0_baseline = compute_percentile_baseline(
        filtered_trace,
        fps,
        tau,
        percentile,
        instability_ratio,
        min_window_s,
        window_tau_multiplier,
    )
    stable_rois_mask = ~np.isnan(F0_baseline).all(axis=0)
    clean_trace = filtered_trace[:, stable_rois_mask]
    clean_baseline = F0_baseline[:, stable_rois_mask]
    return clean_trace, clean_baseline, stable_rois_mask


def _reconstruct_retained_indices(
        original_indices: np.ndarray,
        bright_rois_mask: np.ndarray,
        stable_rois_mask: np.ndarray,
) -> np.ndarray:
    """Map retained ROIs back to original Suite2p ROI indices."""
    retained_indices = original_indices[bright_rois_mask]
    return retained_indices[stable_rois_mask]


def filter_inactive_rois_by_std_or_z(
        deltaF_F: np.ndarray,
        kept_index: np.ndarray,
        min_std: float = 0.01,
        z_event: float = 3.0,
        min_event_frac: float = 0.05,
        center_with_median: bool = True,
        verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Keep ROI if (std > min_std) OR (at least `min_event_frames` samples with z > z_event).
    If min_event_frames is None, it's set to ceil(min_event_frac * T), min 1.
    z = (dF/F - center) / std, per ROI. NaNs ignored.

    Returns:
        deltaF_F_active : (T, N_active)
        final_kept_active : (N_active,)   # original ROI IDs kept
    """
    T, N = deltaF_F.shape
    assert kept_index.shape[0] == N, "final_kept length mismatch"

    min_event_frames = max(1, int(np.ceil(min_event_frac * T)))

    center = (
        np.nanmedian(deltaF_F, axis=0)
        if center_with_median
        else np.nanmean(deltaF_F, axis=0)
    )
    stds = np.nanstd(deltaF_F, axis=0)

    stds_safe = np.where(stds > 0, stds, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = (deltaF_F - center) / stds_safe

    event_counts = np.sum(Z > z_event, axis=0)

    active_mask = (stds > float(min_std)) | (event_counts >= int(min_event_frames))
    inactive_mask = ~active_mask

    deltaF_F_active = deltaF_F[:, active_mask]
    final_index_active = kept_index[active_mask]

    if verbose:
        kept_count = int(active_mask.sum())
        removed_count = int(inactive_mask.sum())
        print(
            f"Mean std: {np.nanmean(stds):.5f} | min_std={min_std}, "
            f"z_event={z_event}, min_event_frames={min_event_frames} "
            f"(= ceil({min_event_frac * 100:.1f}% of {T}))"
        )
        print(f"Removed {removed_count} | Remaining: {kept_count}")

    return deltaF_F_active, final_index_active


def process_suite2p_fluorescence(
        f_path,
        fps,
        tau,
        percentile=8,
        instability_ratio=0.1,
        min_window_s=15,
        window_tau_multiplier=40,
        min_std=0.01,
):
    """
    Complete extraction pipeline: from Suite2p raw output to dF/F traces.

    Parameters:
    - f_path (Path): Path to plane folder containing Suite2p files.
    - fps (float): Imaging rate in Hz.
    - tau (float): Calcium decay constant (seconds).
    - percentile (int): Percentile for baseline estimation.
    - instability_ratio (float): Instability rejection threshold.

    Returns:
    - np.ndarray: dF/F traces (T x N_final).
    - np.ndarray: Retained ROI indices relative to full Suite2p ROI list.
    """
    f_path = Path(f_path)
    fluorescence_file, iscell_file = _resolve_suite2p_inputs(f_path)

    fluorescence_trace = load_fluorescence_data(fluorescence_file)
    iscell_mask = np.load(iscell_file)[:, 0].astype(bool)
    fluorescence_trace, original_indices = _filter_cell_rois(fluorescence_trace, iscell_mask)
    print(
        f"Excluded {np.sum(~iscell_mask)} non-cell ROIs. "
        f"Remaining: {fluorescence_trace.shape[1]} cells."
    )

    filtered_trace, bright_rois_mask = filter_dim_rois(fluorescence_trace)
    print(f"Removed {np.sum(~bright_rois_mask)} dim ROIs.")

    clean_trace, clean_baseline, stable_rois_mask = _filter_stable_rois(
        filtered_trace,
        fps,
        tau,
        percentile,
        instability_ratio,
        min_window_s,
        window_tau_multiplier,
    )
    print(f"Removed {np.sum(~stable_rois_mask)} unstable ROIs.")

    deltaF_F = compute_dff(clean_trace, clean_baseline)
    print(f"dF/F computed. Final ROIs: {deltaF_F.shape[1]}")

    retained_indices_stability = _reconstruct_retained_indices(
        original_indices,
        bright_rois_mask,
        stable_rois_mask,
    )

    deltaF_F_active, final_indices = filter_inactive_rois_by_std_or_z(
        deltaF_F,
        retained_indices_stability,
        min_std,
        verbose=True,
    )

    return deltaF_F_active, final_indices


if __name__ == "__main__":
    base_data_path = Path("/Volumes/LAB-MATI/Lausanne/2p/speed_groupsize_thalamus_exp03")
    fish_selected = np.arange(1, 3)

    n_planes = 5
    fps = 2.0
    tau = 6.0
    percentile = 8
    instability_ratio = 0.1

    for fish in fish_selected:
        fish_id = f"f{fish:02d}"
        segmented_path = base_data_path / fish_id / "04_segmented"
        print(f"\nProcessing {fish_id} in {segmented_path}")

        for i in range(n_planes):
            print(f"\nProcessing plane {i}")
            f_path = segmented_path / f"plane{i}"
            if (f_path / "F.npy").exists():
                deltaF_F, final_indices = process_suite2p_fluorescence(
                    f_path,
                    fps,
                    tau,
                    percentile=percentile,
                    instability_ratio=instability_ratio,
                )

                out_dir = base_data_path / fish_id / "05_dFoF" / f"plane{i}"
                out_dir.mkdir(parents=True, exist_ok=True)

                np.save(out_dir / "dFoF.npy", deltaF_F)
                np.save(out_dir / "roi_filtered.npy", final_indices)

                meta = {
                    "fish_id": fish_id,
                    "plane_index": i,
                    "source_folder": str(f_path),
                    "params": {
                        "fps": fps,
                        "tau": tau,
                        "percentile": percentile,
                        "instability_ratio": instability_ratio,
                    },
                    "shapes": {
                        "dFoF_TxN": [int(deltaF_F.shape[0]), int(deltaF_F.shape[1])],
                        "roi_indices_len": int(len(final_indices)),
                    },
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)

                print(f"Saved to {out_dir}")
            else:
                print(f"File not found: {f_path}")

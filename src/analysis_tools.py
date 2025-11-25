import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.filters import threshold_otsu

def build_trial_aligned_traces(
    dfof,
    stimuli_trace_60,
    fps_2p,
    t_pre_s=5.0,
    t_post_s=29.0,
    stimuli_id_map=None,
    verbose=True,
):
    """
    Build trial-aligned neural activity windows for each stimulus ID.

    Parameters
    ----------
    dfof : array, shape (T, n_neurons) or (n_neurons, T) depending on how you store it
        ΔF/F traces. In this function we assume dfof is (T, n_neurons)
        and we transpose it to (n_neurons, T), like in your original code.
    stimuli_trace_60 : array, shape (T_60,)
        Stimulus trace sampled at 60 Hz. Values: 0 (no stim) or integer IDs.
    fps_2p : float
        Frame rate of 2P imaging (Hz).
    t_pre_s : float
        Seconds before onset to include in each window.
    t_post_s : float
        Seconds after onset to include in each window.
    stimuli_id_map : dict, optional
        Mapping from stimulus name -> ID. If provided, used to generate stimuli_names.
        Otherwise stimuli_names will be ['stim_<id>', ...]
    verbose : bool
        If True, print sanity checks and per-stim summaries.

    Returns
    -------
    result : dict
        {
            'cell_traces'        : (n_neurons, T) array
            'stimuli_trace'      : (T,) array, 2P-matched stimulus IDs
            'stimuli_ids'        : list of int
            'stimuli_names'      : list of str
            'trial_aligned_traces': dict {stim_id -> (n_neurons, win_length, n_trials)}
            'onsets_by_id'       : dict {stim_id -> onsets (frames), n_trials }
            'pre_frames'         : int
            'post_frames'        : int
            'win_length'         : int
        }
    """

    # --- 1) Match stimulus trace (60 Hz) to 2P sampling ---
    # Assume your original convention: dfof shape (T, n_neurons)
    cell_traces = dfof.T  # -> (n_neurons, T)
    n_neurons, T = cell_traces.shape

    t_2p_sec = np.arange(T) / float(fps_2p)         # time vector at 2P rate
    idx60 = np.floor(t_2p_sec * 60.0).astype(int)  # map each 2P frame to 60 Hz index
    idx60 = np.clip(idx60, 0, len(stimuli_trace_60) - 1)
    stimuli_trace = stimuli_trace_60[idx60]

    # --- 2) Define peri-stimulus window (in frames) ---
    pre_frames  = int(round(t_pre_s  * fps_2p))
    post_frames = int(round(t_post_s * fps_2p))
    win_length  = pre_frames + post_frames

    # --- 3) Stimulus IDs and names ---
    stimuli_ids = sorted(int(x) for x in np.unique(stimuli_trace) if x != 0)

    if stimuli_id_map is not None:
        # user provided mapping name -> id; invert to get id -> name
        id_to_name = {v: k for k, v in stimuli_id_map.items()}
        stimuli_names = [id_to_name.get(sid, f"stim_{sid}") for sid in stimuli_ids]
    else:
        stimuli_names = [f"stim_{sid}" for sid in stimuli_ids]

    # --- 4) Sanity checks ---
    assert cell_traces.ndim == 2, "traces must be (n_neurons, T)"
    assert stimuli_trace.ndim == 1, "stimuli_trace must be (T,)"
    assert cell_traces.shape[1] == stimuli_trace.shape[0], "time dimension mismatch"

    if verbose:
        print(f"Number of neurons: {n_neurons}")
        print(f"Number of time points: {T}")
        print("2p duration (s):", T / fps_2p)
        print("60Hz duration from data (s):", len(stimuli_trace_60) / 60.0)
        print("Output stim trace shape:", stimuli_trace.shape)
        print("Stimuli IDs:", stimuli_ids)
        print("Stimuli names:", stimuli_names)

    # --- 5) Build trial-aligned traces per stimulus ---
    onsets_by_id = {}
    trial_aligned_traces = {}  # sid -> (n_neurons, win_length, n_trials)

    for stim in stimuli_ids:
        # 0→1 transitions: find onsets in the 2P-matched stimulus trace
        active = (stimuli_trace == stim).astype(np.int8)
        transitions = np.diff(active, prepend=0)
        onsets = np.flatnonzero(transitions == 1)
        onsets_by_id[stim] = onsets

        starts = onsets - pre_frames
        ends   = onsets + post_frames
        keep   = (starts >= 0) & (ends <= T)

        if not np.any(keep):
            arr = np.empty((n_neurons, win_length, 0), dtype=float)
        else:
            arr = np.stack(
                [cell_traces[:, s:e] for s, e in zip(starts[keep], ends[keep])],
                axis=2
            )

        trial_aligned_traces[stim] = arr

        if verbose:
            print(
                f"stim {stim}: {onsets.size} onsets | "
                f"kept {arr.shape[2]} trials | "
                f"dropped {np.count_nonzero(~keep)}"
            )

    # --- 6) Pack everything in a dict ---
    result = {
        "cell_traces": cell_traces,
        "stimuli_trace": stimuli_trace,
        "stimuli_ids": stimuli_ids,
        "stimuli_names": stimuli_names,
        "trial_aligned_traces": trial_aligned_traces,
        "onsets_by_id": onsets_by_id,
        "pre_frames": pre_frames,
        "post_frames": post_frames,
        "win_length": win_length,
    }

    return result

# # Example usage:
# res = build_trial_aligned_traces(
#     dfof=dfof,
#     stimuli_trace_60=stimuli_trace_60,
#     fps_2p=fps_2p,
#     t_pre_s=5.0,
#     t_post_s=29.0,
#     stimuli_id_map=stimuli_id_map,  # or None
#     verbose=True
# )
#
# trial_aligned_traces = res["trial_aligned_traces"]
# stimuli_ids          = res["stimuli_ids"]
# stimuli_names        = res["stimuli_names"]
# win_length           = res["win_length"]
# stimuli_trace        = res["stimuli_trace"]
# onsets_by_id         = res["onsets_by_id"]

def plot_accepted_rejected_rasters(
    dfof: np.ndarray,             # (n_neurons, n_frames)
    t=None,                       # (n_frames,) OR None OR scalar dt
    kept_mask: np.ndarray=None,   # (n_neurons,), boolean
    vmax: float = None,
    vmin: float = 0.0,
    perc_for_vmax: float = 99.0,
    sort_by_peak_time: bool = False,
    share_color_scale: bool = True,
):
    assert dfof.ndim == 2, "dfof must be (n_neurons, n_frames)"
    n_neurons, n_frames = dfof.shape

    # --- Build time axis / extent ---
    if t is None:
        x0, x1 = 0.0, float(n_frames - 1)
        x_label = "Frame"
    elif np.isscalar(t):  # t is a sampling interval (dt in seconds)
        dt = float(t)
        x0, x1 = 0.0, dt * (n_frames - 1)
        x_label = "Time (s)"
    else:
        t = np.asarray(t)
        assert t.ndim == 1 and t.size == n_frames, "t must be 1-D with length n_frames"
        x0, x1 = float(t[0]), float(t[-1])
        x_label = "Time"

    if kept_mask is None:
        kept_mask = np.ones(n_neurons, dtype=bool)
    else:
        kept_mask = np.asarray(kept_mask, dtype=bool)
        assert kept_mask.shape[0] == n_neurons, "kept_mask length must match n_neurons"

    kept_idx = np.flatnonzero(kept_mask)
    rej_idx  = np.setdiff1d(np.arange(n_neurons), kept_idx)

    # --- Color scaling ---
    if vmax is None:
        finite_vals = dfof[np.isfinite(dfof)]
        vmax = np.percentile(finite_vals, perc_for_vmax) if finite_vals.size else 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = np.nanmax(dfof) if np.isfinite(np.nanmax(dfof)) else 1.0

    vmax_kept = vmax
    vmax_rej  = vmax
    if not share_color_scale:
        if kept_idx.size:
            tmp = np.percentile(dfof[kept_idx], perc_for_vmax)
            vmax_kept = tmp if np.isfinite(tmp) and tmp > 0 else vmax
        if rej_idx.size:
            tmp = np.percentile(dfof[rej_idx], perc_for_vmax)
            vmax_rej  = tmp if np.isfinite(tmp) and tmp > 0 else vmax

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    def mat_for(idx):
        M = dfof[idx] if idx.size else np.zeros((1, n_frames))
        if sort_by_peak_time and idx.size > 1:
            order = np.argsort(np.argmax(M, axis=1))
            M = M[order]
        return M

    for (title, idx, vmax_here, ax) in [
        ("Accepted", kept_idx, vmax_kept, axes[0]),
        ("Rejected", rej_idx,  vmax_rej,  axes[1]),
    ]:
        M = mat_for(idx)
        im = ax.imshow(
            M,
            aspect='auto',
            interpolation='nearest',
            origin='lower',
            extent=[x0, x1, 0, M.shape[0]],
            vmin=vmin,
            vmax=vmax_here,
            cmap='gray_r',       # high = dark
        )
        ax.set_title(f"{title} (n={idx.size})")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Neuron #")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("ΔF/F")

    if share_color_scale:
        fig.suptitle(f"ΔF/F rasters (shared vmin={vmin:.3g}, vmax={vmax:.3g})", y=1.02)
    else:
        fig.suptitle(
            f"ΔF/F rasters (vmin={vmin:.3g}, kept vmax={vmax_kept:.3g}, rejected vmax={vmax_rej:.3g})",
            y=1.02
        )
    return fig, axes

#%%


def filter_neurons_by_trial_reliability(
    dfof: np.ndarray,                    # (T, n_neurons)
    trial_aligned_traces: dict,         # stim_id -> (n_neurons, win_length, n_trials)
    stimuli_ids: list,
    fps_2p: float,
    plots_path: Path,
    prefix: str = "",
    folder_name: str | None = None,
    make_plots: bool = True,
    save_indices: bool = True,
    hist_bins: int = 70,
):
    """

    Compute trial-to-trial reliability for each neuron, select neurons with
    high reliability using an Otsu threshold, optionally plot, and save
    kept neuron indices to disk.

    For each neuron and each stimulus condition, we compute the trial-to-trial
    reliability as the mean pairwise Pearson correlation across all trials
    in a peri-stimulus time window (based on trial_aligned_traces). This yields,
    for every neuron, one reliability value per stimulus. We then take, for each
    neuron, the maximum reliability across stimuli to obtain a single reliability
    score per neuron. To define an objective threshold separating reliable from
    unreliable neurons, we apply Otsu’s method on the distribution of these
    maximum reliability scores. Otsu’s method finds the threshold that best
    separates the distribution into two classes by maximizing the between-class
    variance. Neurons with a reliability score greater than or equal to this
    Otsu threshold are kept for subsequent analyses.

    Parameters
    ----------
    dfof : array, shape (T, n_neurons)
        ΔF/F traces over time (T) for each neuron.
    trial_aligned_traces : dict
        Mapping stim_id -> array of shape (n_neurons, win_length, n_trials),
        containing trial-aligned neural responses.
    stimuli_ids : list
        List of stimulus IDs present in trial_aligned_traces.
    fps_2p : float
        2P sampling rate in Hz, used to convert frames to seconds in rasters.
    plots_path : Path
        Base path used only to save kept neuron indices (.npy).
    prefix : str
        Optional string to prepend in the saved filename.
    filename_mid : str
        Optional middle part of the saved filename (e.g. session ID).
    folder_name : str or None
        Optional subfolder inside plots_path where indices are saved.
        If None, indices are saved directly in plots_path.
    make_plots : bool
        If True, show histogram, curve of #ROIs vs threshold, and accepted/
        rejected rasters. Plots are not saved to disk.
    save_indices : bool
        If True, save kept neuron indices as a .npy file in the chosen folder.
    hist_bins : int
        Number of bins to use for the reliability histogram.

    Returns
    -------
    result : dict
        Dictionary containing:
        - reliability_per_stim
        - max_stimuli_correlation
        - nanfiltered_max
        - otsu_threshold
        - kept_mask
        - kept_neuron_indices

    """

    T, n_neurons = dfof.shape

    # --- reliability per neuron x stimulus ---
    reliability_per_stim = np.full((n_neurons, len(stimuli_ids)), np.nan, dtype=float)

    for j, stim in enumerate(stimuli_ids):
        aligned_neural_traces = trial_aligned_traces[stim]  # (n_neurons, win_length, n_trials)
        _, _, n_trials = aligned_neural_traces.shape

        if n_trials < 2:
            # can't compute correlations with 1 trial → leave as NaN
            continue

        for i in range(n_neurons):
            neuron_trace = aligned_neural_traces[i, :, :]  # (time, trials)

            # Compute trial-to-trial correlation matrix (trials x trials)
            corr_mat = np.corrcoef(neuron_trace.T)

            # Ignore self-correlations by setting diagonal to NaN
            np.fill_diagonal(corr_mat, np.nan)

            # Mean correlation (reliability) across all trial pairs
            reliability_per_stim[i, j] = np.nanmean(corr_mat)

    # --- max reliability across stimuli & Otsu threshold ---
    max_stimuli_correlation = np.nanmax(reliability_per_stim, axis=1)
    nanfiltered_max = max_stimuli_correlation[np.isfinite(max_stimuli_correlation)]

    if nanfiltered_max.size > 0:
        otsu_threshold = float(threshold_otsu(nanfiltered_max))
    else:
        otsu_threshold = np.nan

    kept_mask = np.isfinite(max_stimuli_correlation) & (max_stimuli_correlation >= otsu_threshold)
    kept_neuron_indices = np.flatnonzero(kept_mask)
    kept_pct = (100.0 * kept_neuron_indices.size / n_neurons) if n_neurons else 0.0

    # --- save indices (but not figures) ---
    if save_indices:
        if plots_path is None:
            raise ValueError("plots_path must be provided if save_indices=True")

        # Decide where to save
        out_dir = plots_path / folder_name if folder_name is not None else plots_path
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save kept indices
        out_idx = out_dir / f"{prefix}_kept_neuron_indices.npy"
        np.save(out_idx, kept_neuron_indices)
        print(f"Saved kept neuron indices to: {out_idx}")

    # --- plots (purely for visualization, not saved) ---
    if make_plots and nanfiltered_max.size > 0:
        # 1) Histogram of reliability + threshold
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(nanfiltered_max, bins=hist_bins)
        ax.axvline(otsu_threshold, linestyle="--", color="k")
        ax.set(
            title=(
                "Reliability of response across stimuli\n"
                f"Otsu {otsu_threshold:.2f} — kept {kept_pct:.1f}% "
                f"({kept_neuron_indices.size} ROIs)"
            ),
            xlabel="max avg intertrial correlation",
            ylabel="neuron count",
        )
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.show()

        # 2) Curve: #ROIs kept vs threshold
        low = max(0.0, float(np.nanmin(nanfiltered_max)))
        high = min(1.0, max(float(np.nanmax(nanfiltered_max)), low + 1e-6))
        thr_grid = np.linspace(low, high, 51)
        counts = [np.sum(nanfiltered_max >= thr) for thr in thr_grid]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(thr_grid, counts, marker="o", linewidth=1)
        ax.set(
            title="ROIs kept vs threshold",
            xlabel="Threshold",
            ylabel="# ROIs kept",
        )
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.show()

        # 3) Rasters accepted vs rejected, with X-axis in seconds
        # dfof is (T, n_neurons) → transpose for raster function
        dt = 1.0 / float(fps_2p)
        _fig_r, _axes_r = plot_accepted_rejected_rasters(
            dfof=dfof.T,
            t=dt,  # scalar dt → seconds on X
            kept_mask=kept_mask,
            sort_by_peak_time=True,
            share_color_scale=True,
        )
        plt.show()

    return {
        "reliability_per_stim": reliability_per_stim,
        "max_stimuli_correlation": max_stimuli_correlation,
        "nanfiltered_max": nanfiltered_max,
        "otsu_threshold": otsu_threshold,
        "kept_mask": kept_mask,
        "kept_neuron_indices": kept_neuron_indices,
    }

# # Example usage:
# results = filter_neurons_by_trial_reliability(
#     dfof=dfof,                        # shape (T, n_neurons)
#     trial_aligned_traces=trial_aligned_traces,
#     stimuli_ids=stimuli_ids,
#     fps_2p=fps_2p,
#     plots_path=paths["plots_path"],
#     prefix=paths["prefix"],
#     folder_name="reliability_filter",
#     make_plots=True,      # switch to False if you don't want any plots
#     save_indices=True,    # switch to False if you don't want to save the .npy
# )
#
# kept_neuron_indices = results["kept_neuron_indices"]
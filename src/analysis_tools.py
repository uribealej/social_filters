import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.filters import threshold_otsu
from matplotlib_venn import venn3


def find_file_with_suffix(folder: Path, suffix: str) -> Path:
    matches = list(folder.glob(f"*{suffix}"))
    if len(matches) == 0:
        raise FileNotFoundError(f"No file ending with '{suffix}' in {folder}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple files ending with '{suffix}' in {folder}: {matches}")
    return matches[0]

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
    costom_threshold :[float] = None,
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

    if costom_threshold is None:
        otsu_threshold = otsu_threshold
    else:
        otsu_threshold = costom_threshold

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



def inspect_obj(obj, name="obj", max_items=5, level=0):
    """
    Quick inspection of a Python object:
    - prints type
    - for numpy arrays: shape + dtype
    - for dicts: shows keys and value types (and shapes if arrays)
    - for lists/tuples: length and types of first items
    """
    indent = "  " * level
    t = type(obj)
    print(f"{indent}{name}: {t}")

    # --- numpy array ---
    if isinstance(obj, np.ndarray):
        print(f"{indent}  shape={obj.shape}, dtype={obj.dtype}")
        return

    # --- dict ---
    if isinstance(obj, dict):
        print(f"{indent}  dict with {len(obj)} keys:")
        for i, (k, v) in enumerate(obj.items()):
            if i >= max_items:
                print(f"{indent}    ... ({len(obj) - max_items} more keys)")
                break
            line = f"{indent}    key={repr(k)} -> type={type(v)}"
            if isinstance(v, np.ndarray):
                line += f", shape={v.shape}, dtype={v.dtype}"
            elif isinstance(v, (list, tuple)):
                line += f", len={len(v)}"
            print(line)
        return

    # --- list / tuple ---
    if isinstance(obj, (list, tuple)):
        print(f"{indent}  {t.__name__} of length {len(obj)}")
        for i, item in enumerate(obj[:max_items]):
            line = f"{indent}    [{i}]: type={type(item)}"
            if isinstance(item, np.ndarray):
                line += f", shape={item.shape}, dtype={item.dtype}"
            elif isinstance(item, (list, tuple)):
                line += f", len={len(item)}"
            print(line)
        if len(obj) > max_items:
            print(f"{indent}    ... ({len(obj) - max_items} more items)")
        return

    # --- fallback: just print value ---
    try:
        print(f"{indent}  value={repr(obj)}")
    except Exception:
        pass

# # Example usage:
# inspect_obj(onset_sec_by_id, "onset_sec_by_id")

# Classify responses based on raster (significant_traces) and stimulus onsets, IT evaluates for each neuron
# whether it has evoked or tardive responses to each stimulus
# based on criteria like minimum run length and fraction of trials.
# criteria: For each neuron and stimulus, it checks if there are runs of significant activity
# it evaluates in the response_window_sec, if there are significant concecutive runs of activity at least by "min_run_sec". The activity should began during the stimulus presentation (MOTION(evoked)) or after (tardive).
# It also requires that the neuron shows this response type in at least min_fraction_trials of trials for that stimulus to be classified as such.
# it also returns the median onset time of the response type for each neuron and stimulus.

def classify_responses_from_raster(
    raster,
    onsets_by_id,
    stimuli_id_map,
    stimuli_durations,
    fps_2p=2.0,
    response_window_sec=20.0,
    min_run_sec=3.0,
    min_fraction_trials=0.5,
):
    """
    Parameters
    ----------
    raster : np.ndarray
        (n_frames_total, n_neurons) with 0/1 significant traces.
    onsets_by_id : dict[int -> np.ndarray]
        For each stim_id, a 1D array of onset frames (global indices), one per trial.
    stimuli_id_map : dict[str -> int]
        Mapping from stimulus name (e.g. 'FL1') to stim_id (1..10).
    stimuli_durations : dict[str -> dict]
        Has entry stimuli_durations[stim_name]['motion_sec'].

    Returns
    -------
    response_type_by_id : dict[int -> np.ndarray]
        For each stim_id, an array (n_neurons,) with:
            0 = no response
            1 = evoked response (onset during stimulus)
            2 = tardive response (onset only after stimulus)
    onset_sec_by_id : dict[int -> np.ndarray]
        For each stim_id, an array (n_neurons,) with:
            - median evoked onset in seconds if type 1
            - median tardive onset in seconds if type 2
            - np.nan if type 0
    """

    raster = np.asarray(raster)
    n_frames_total, n_neurons = raster.shape

    # id -> name to get motion_sec
    id_to_name = {v: k for k, v in stimuli_id_map.items()}

    # analysis window: 0 .. response_window_sec
    n_frames_window = int(round(response_window_sec * fps_2p))
    # minimum length of run of 1s for >= min_run_sec
    min_run_frames = int(np.ceil(min_run_sec * fps_2p))

    response_type_by_id = {}
    onset_sec_by_id = {}

    def find_runs(bool_vec):
        """
        Given 1D boolean array, return list of (start_idx, length)
        for each contiguous run of True.
        """
        bool_vec = np.asarray(bool_vec, dtype=bool)
        if bool_vec.size == 0:
            return []

        diff = np.diff(bool_vec.astype(int))

        # starts where 0 -> 1
        run_starts = np.where(diff == 1)[0] + 1
        if bool_vec[0]:
            run_starts = np.r_[0, run_starts]

        # ends where 1 -> 0
        run_ends = np.where(diff == -1)[0] + 1
        if bool_vec[-1]:
            run_ends = np.r_[run_ends, len(bool_vec)]

        return [(s, e - s) for s, e in zip(run_starts, run_ends)]

    # -------------------------------------------------------------------------
    # Loop over stimuli
    # -------------------------------------------------------------------------
    for stim_id, onset_frames in onsets_by_id.items():
        onset_frames = np.asarray(onset_frames, dtype=int)
        n_trials = len(onset_frames)

        # Which stimulus name is this?
        stim_name = id_to_name[stim_id]
        motion_sec = stimuli_durations[stim_name]["motion_sec"]
        static_before_sec = stimuli_durations[stim_name]["static_before_sec"]

        # convert static-before period to frames
        static_before_frames = int(round(static_before_sec * fps_2p))

        # Outputs for this stimulus: one vector per neuron (what you asked)
        resp_type = np.zeros(n_neurons, dtype=int)          # 0/1/2
        onset_sec = np.full(n_neurons, np.nan, dtype=float) # in seconds

        # ---------------------------------------------------------------------
        # For each neuron
        # ---------------------------------------------------------------------
        for neuron_idx in range(n_neurons):
            evoked_onsets = []   # list of onset times (sec) across trials
            tardive_onsets = []  # same, for tardive

            for trial_idx in range(n_trials):
                start_global = onset_frames[trial_idx]+static_before_frames
                if start_global >= n_frames_total:
                    # onset beyond recorded frames -> skip
                    continue

                end_global = min(start_global + n_frames_window, n_frames_total)

                # 0/1 trace for this neuron in this trial's window
                sig_vec = raster[start_global:end_global, neuron_idx].astype(bool)
                if not sig_vec.any():
                    # no significant frames in this trial window
                    continue

                n_frames_this = sig_vec.shape[0]
                frame_times = np.arange(n_frames_this) / fps_2p  # 0, 0.5, 1.0, ...

                runs = find_runs(sig_vec)

                trial_evoked_onset = None
                trial_tardive_onset = None

                for start_idx, length in runs:
                    # only consider runs long enough (>= min_run_sec)
                    if length < min_run_frames:
                        continue

                    onset_time_sec = frame_times[start_idx]

                    # Evoked: onset while stimulus is present (0 .. motion_sec)
                    if onset_time_sec <= motion_sec:
                        if trial_evoked_onset is None or onset_time_sec < trial_evoked_onset:
                            trial_evoked_onset = onset_time_sec

                    # Tardive: onset after stimulus is over
                    elif onset_time_sec > motion_sec:
                        if trial_tardive_onset is None or onset_time_sec < trial_tardive_onset:
                            trial_tardive_onset = onset_time_sec

                # Store earliest onset for this trial, if any
                if trial_evoked_onset is not None:
                    evoked_onsets.append(trial_evoked_onset)
                if trial_tardive_onset is not None:
                    tardive_onsets.append(trial_tardive_onset)

            # ---------------- neuron-level classification ------------------
            if n_trials == 0:
                resp_type[neuron_idx] = 0
                onset_sec[neuron_idx] = np.nan
                continue

            n_evoked = len(evoked_onsets)
            n_tardive = len(tardive_onsets)

            frac_evoked = n_evoked / n_trials
            frac_tardive = n_tardive / n_trials

            # 1) Evoked response (dominant)
            if frac_evoked >= min_fraction_trials:
                resp_type[neuron_idx] = 1
                onset_sec[neuron_idx] = np.median(evoked_onsets)

            # 2) Tardive response if no evoked by criterion
            elif frac_tardive >= min_fraction_trials:
                resp_type[neuron_idx] = 2
                onset_sec[neuron_idx] = np.median(tardive_onsets)

            # 3) No response
            else:
                resp_type[neuron_idx] = 0
                onset_sec[neuron_idx] = np.nan

        response_type_by_id[stim_id] = resp_type   # vector per stimulus
        onset_sec_by_id[stim_id] = onset_sec       # vector per stimulus

    return response_type_by_id, onset_sec_by_id

# response_type_by_id, onset_sec_by_id = classify_responses_from_raster(
#     raster=raster,
#     onsets_by_id=onsets_by_id,
#     stimuli_id_map=stimuli_id_map,
#     stimuli_durations=stimuli_durations,
#     fps_2p=2.0,
#     response_window_sec=20.0,  # 0–20 s after each onset
#     min_run_sec=2.0,           # ≥ 3 s consecutive 1s
#     min_fraction_trials=0.5,   # ≥ 50% of trials
# )


def compute_left_right_index(mean_traces,
                             left_stim,
                             right_stim,
                             frame_window,
                             kept_cells=None,
                             eps=1e-9):
    """
    mean_traces[stim]: (n_cells, n_frames)
    frame_window: (start_frame, end_frame) for averaging
    kept_cells: optional index/mask of cells to use
    """
    start, end = frame_window

    M_left  = mean_traces[left_stim][:, start:end]   # (n_cells, win_len)
    M_right = mean_traces[right_stim][:, start:end]

    # Optional subset of cells
    if kept_cells is not None:
        M_left  = M_left[kept_cells]
        M_right = M_right[kept_cells]

    # Scalar response per neuron (mean over time window)
    L = np.nanmean(M_left,  axis=1)   # (n_cells_sel,)
    R = np.nanmean(M_right, axis=1)

    # Tuning index in [-1, 1]
    TI = (R - L) / (R + L + eps)

    return L, R, TI

# mode to use
#{'FL1': 1, 'FL2': 2, 'FL3': 3, 'FLB': 4, 'FR1': 5, 'FR2': 6, 'FR3': 7, 'FRB': 8, 'LLB': 9, 'RLB': 10}
# L, R, TI = compute_left_right_index(
#     mean_traces=mean_traces,
#     left_stim=4,   # e.g., FLB = 4
#     right_stim=8,
#     frame_window=frame_window,
#     kept_cells=None,   # or None
# )

'''First by big groups (response pattern to the 3 stimuli): In this order: Evoked in all 3 → ev_123
Evoked in 1&2 only → ev_12_only Evoked in 1&3 only → ev_13_only Evoked in 2&3 only → ev_23_only
Evoked only in 1 → ev_1_only Evoked only in 2 → ev_2_only Evoked only in 3 → ev_3_only then the same 7 patterns for tardive (td_123, td_12_only, …, td_3_only), and finally: no_response → neurons that are not evoked or tardive to any of the 3 stimuli.
Then, inside each group:  For all evoked/tardive groups: neurons are sorted by their mean onset across the 3 stimuli (earliest → latest; neurons with no onset = NaN go at the end of that group).

For the no_response group: neurons are sorted by onset to the extra stimulus stim_id_for_noresp_sort
(again: earliest → latest; NaN onsets go at the end of the no_response group).

'''''

def build_neuron_order_groupwise_onset(
    response_type_by_id,
    onset_sec_by_stim,
    stim_ids_for_pattern,
    stim_id_for_noresp_sort=None,
):
    """
    Group neurons by response pattern to three stimuli, and sort within each
    group by onset:

    - For evoked/tardive groups (defined by `stim_ids_for_pattern`), sort by
      the neuron's *mean onset* across those stimuli where it responds.
      (Implemented via np.nanmean across the 3 onset arrays.)
    - For the no_response group (no evoked/tardive to those 3), sort by onset
      to another stimulus `stim_id_for_noresp_sort` (e.g. 10).
    - Within each group: finite onset first (ascending), then NaNs.

    Parameters
    ----------
    response_type_by_id : dict[int, np.ndarray]
        stim_id -> (n_neurons,) with values:
          0 = no response, 1 = evoked, 2 = tardive.
    onset_sec_by_stim : dict[int, np.ndarray]
        stim_id -> (n_neurons,) onset latencies (seconds).
        Use np.nan where no onset / no response.
        Must contain all `stim_ids_for_pattern` and (if used)
        `stim_id_for_noresp_sort`.
    stim_ids_for_pattern : list[int]
        3 stimulus IDs used to define response patterns, e.g. [5, 6, 7].
    stim_id_for_noresp_sort : int or None
        Stimulus ID whose onsets are used to sort the no_response group,
        e.g. 10. If None, no_response will also use the mean onset across
        `stim_ids_for_pattern` (usually they’ll all be NaN).

    Returns
    -------
    neuron_order : np.ndarray
        Final neuron order (indices 0..n_neurons-1 in the chosen order).
    groups : dict[str, np.ndarray]
        Mapping group_name -> sorted indices (in that group).
    mean_onset_pattern : np.ndarray
        (n_neurons,) mean onset across `stim_ids_for_pattern` (nanmean).
    """

    # --- stack response types for pattern stimuli ---
    resp_matrix = np.stack(
        [response_type_by_id[sid] for sid in stim_ids_for_pattern],
        axis=1
    )  # shape: (n_neurons, 3)

    n_neurons = resp_matrix.shape[0]

    # Boolean matrices: evoked and tardive for each of the 3 stimuli
    ev = (resp_matrix == 1)
    td = (resp_matrix == 2)

    # --- compute mean onset across the 3 pattern stimuli, per neuron ---
    # shape: (n_neurons, 3)
    onset_stack_pattern = np.stack(
        [onset_sec_by_stim[sid] for sid in stim_ids_for_pattern],
        axis=1
    )

    # nanmean: ignores NaNs. If all 3 are NaN, result is NaN.
    mean_onset_pattern = np.nanmean(onset_stack_pattern, axis=1)

    # onset used for sorting no_response group
    onset_noresp = None
    if stim_id_for_noresp_sort is not None:
        onset_noresp = onset_sec_by_stim[stim_id_for_noresp_sort]

    used = np.zeros(n_neurons, dtype=bool)
    groups = {}
    order_list = []

    def add_group(name, mask, use_noresp_onset=False):
        nonlocal used, groups, order_list

        idx = np.where(mask & ~used)[0]
        if idx.size == 0:
            return

        # choose which onset to use for this group
        if use_noresp_onset and (onset_noresp is not None):
            base_onset = onset_noresp
        else:
            base_onset = mean_onset_pattern

        group_onset = base_onset[idx]

        # finite onset first, then NaNs
        finite_mask = np.isfinite(group_onset)
        finite_local = np.where(finite_mask)[0]
        nan_local   = np.where(~finite_mask)[0]

        if finite_local.size > 0:
            order_finite = np.argsort(group_onset[finite_mask])
            sorted_local = np.concatenate([
                finite_local[order_finite],
                nan_local,
            ])
        else:
            # all NaN: keep original relative order (or just nan_local)
            sorted_local = nan_local

        idx_sorted = idx[sorted_local]

        groups[name] = idx_sorted
        order_list.append(idx_sorted)
        used[idx_sorted] = True

    # indices in second axis
    s0, s1, s2 = 0, 1, 2

    # ========= EVOKED (type 1) GROUPS =========
    add_group("ev_123",      ev[:, s0] & ev[:, s1] & ev[:, s2])
    add_group("ev_12_only",  ev[:, s0] & ev[:, s1] & ~ev[:, s2])
    add_group("ev_13_only",  ev[:, s0] & ~ev[:, s1] & ev[:, s2])
    add_group("ev_23_only",  ~ev[:, s0] & ev[:, s1] & ev[:, s2])
    add_group("ev_1_only",   ev[:, s0] & ~ev[:, s1] & ~ev[:, s2])
    add_group("ev_2_only",   ~ev[:, s0] & ev[:, s1] & ~ev[:, s2])
    add_group("ev_3_only",   ~ev[:, s0] & ~ev[:, s1] & ev[:, s2])

    # ========= TARDIVE (type 2) GROUPS =========
    add_group("td_123",      td[:, s0] & td[:, s1] & td[:, s2])
    add_group("td_12_only",  td[:, s0] & td[:, s1] & ~td[:, s2])
    add_group("td_13_only",  td[:, s0] & ~td[:, s1] & td[:, s2])
    add_group("td_23_only",  ~td[:, s0] & td[:, s1] & td[:, s2])
    add_group("td_1_only",   td[:, s0] & ~td[:, s1] & ~td[:, s2])
    add_group("td_2_only",   ~td[:, s0] & td[:, s1] & ~td[:, s2])
    add_group("td_3_only",   ~td[:, s0] & ~td[:, s1] & td[:, s2])

    # ========= NON-RESPONSIVE TO THESE 3 STIMULI =========
    no_resp_mask = ~(ev.any(axis=1) | td.any(axis=1))
    # For this group, use onset to the extra stimulus (e.g. stim 10)
    add_group("no_response", no_resp_mask, use_noresp_onset=True)

    # Final order
    if len(order_list) > 0:
        neuron_order = np.concatenate(order_list)
    else:
        neuron_order = np.arange(n_neurons)

    return neuron_order, groups, mean_onset_pattern

# def plot_venn_3stim(response_type_by_id, stim_ids=(5, 6, 7), ):
#     # 1) Boolean masks: neuron is responsive (1 or 2) to each stim
#     resp = {sid: (response_type_by_id[sid] != 0) for sid in stim_ids}
#
#     # 2) Convert to sets of neuron indices
#     sets = [set(np.where(resp[sid])[0]) for sid in stim_ids]
#     A, B, C = sets
#     s1, s2, s3 = stim_ids
#
#     # 3) Print counts per region
#     only_A = A - B - C
#     only_B = B - A - C
#     only_C = C - A - B
#     AB_only = (A & B) - C
#     AC_only = (A & C) - B
#     BC_only = (B & C) - A
#     ABC     = A & B & C
#
#     print(f"Only {s1}: {len(only_A)}")
#     print(f"Only {s2}: {len(only_B)}")
#     print(f"Only {s3}: {len(only_C)}")
#     print(f"{s1} & {s2} only: {len(AB_only)}")
#     print(f"{s1} & {s3} only: {len(AC_only)}")
#     print(f"{s2} & {s3} only: {len(BC_only)}")
#     print(f"{s1} & {s2} & {s3}: {len(ABC)}")
#
#     # 4) Plot Venn diagram
#     plt.figure(figsize=(5, 5))
#     venn3(sets, set_labels=[f"stim {s1}", f"stim {s2}", f"stim {s3}"])
#     plt.title("Responsive neurons to stimuli 5, 6, 7")
#     plt.show()
#
# from matplotlib_venn import venn3
# import matplotlib.pyplot as plt
# import numpy as np

def plot_venn_3stim(
    response_type_by_id,
    stim_ids=(5, 6, 7),
    stim_labels=None,          # NEW: optional list of names, one per stim_id
    title=None,                # NEW: optional custom title
):
    # 1) Boolean masks: neuron is responsive (1 or 2) to each stim
    resp = {sid: (response_type_by_id[sid] != 0) for sid in stim_ids}

    # 2) Convert to sets of neuron indices
    sets = [set(np.where(resp[sid])[0]) for sid in stim_ids]
    A, B, C = sets
    s1, s2, s3 = stim_ids

    # 3) Print counts per region
    only_A = A - B - C
    only_B = B - A - C
    only_C = C - A - B
    AB_only = (A & B) - C
    AC_only = (A & C) - B
    BC_only = (B & C) - A
    ABC     = A & B & C

    print(f"Only {s1}: {len(only_A)}")
    print(f"Only {s2}: {len(only_B)}")
    print(f"Only {s3}: {len(only_C)}")
    print(f"{s1} & {s2} only: {len(AB_only)}")
    print(f"{s1} & {s3} only: {len(AC_only)}")
    print(f"{s2} & {s3} only: {len(BC_only)}")
    print(f"{s1} & {s2} & {s3}: {len(ABC)}")

    # --- NEW: build labels from stim_labels if provided ---
    if stim_labels is None:  # fallback: just use the IDs
        stim_labels = [f"stim {s1}", f"stim {s2}", f"stim {s3}"]
    else:
        assert len(stim_labels) == 3, "stim_labels must have length 3"

    # --- Plot Venn diagram ---
    plt.figure(figsize=(5, 5))
    venn3(sets, set_labels=stim_labels)

    if title is None:  # NEW: nicer default title using names
        title = "Responsive neurons to: " + ", ".join(stim_labels)
    plt.title(title)

    plt.show()  # CHANGED: added parentheses!


import numpy as np


def zscore_dfof_from_prestim_baseline(
    dfof,
    onsets_by_id,
    fps_2p,
    baseline_s=10.0,
    assume_time_by_neuron=True,
    ddof=0,
    verbose=True,
):
    """
    Z-score ΔF/F traces using concatenated pre-stimulus baseline windows
    from all stimulus onsets, ignoring stimulus identity.

    Parameters
    ----------
    dfof : np.ndarray
        ΔF/F traces.
        Expected shape:
        - (T, n_neurons) if assume_time_by_neuron=True
        - (n_neurons, T) if assume_time_by_neuron=False
    onsets_by_id : dict
        Dictionary like {stim_id: array_of_onset_frames}.
        Onsets must be in 2P frame indices.
    fps_2p : float
        2P imaging frame rate in Hz.
    baseline_s : float, default=10.0
        Duration of the pre-stimulus baseline window in seconds.
    assume_time_by_neuron : bool, default=True
        If True, assumes dfof is (T, n_neurons) and transposes internally
        to (n_neurons, T). If False, assumes dfof is already (n_neurons, T).
    ddof : int, default=0
        Delta degrees of freedom for std calculation.
        Use ddof=0 for population std, ddof=1 for sample std.
    verbose : bool, default=True
        If True, prints summary information.

    Returns
    -------
    result : dict
        {
            'z_traces'              : np.ndarray, same orientation as input dfof
            'cell_traces'           : np.ndarray, shape (n_neurons, T)
            'baseline_mean'         : np.ndarray, shape (n_neurons,)
            'baseline_std'          : np.ndarray, shape (n_neurons,)
            'baseline_frames'       : int
            'all_onsets'            : np.ndarray, sorted all onsets
            'valid_onsets'          : np.ndarray, onsets used for baseline
            'invalid_onsets'        : np.ndarray, onsets too early for baseline
            'baseline_segments'     : np.ndarray, shape (n_neurons, total_baseline_frames)
        }
    """

    # --- 1) Arrange traces as (n_neurons, T) ---
    if assume_time_by_neuron:
        cell_traces = dfof.T
    else:
        cell_traces = dfof

    if cell_traces.ndim != 2:
        raise ValueError("dfof must be a 2D array")

    n_neurons, T = cell_traces.shape

    # --- 2) Flatten onsets across all stimulus IDs ---
    onset_arrays = []
    for stim_id, arr in onsets_by_id.items():
        arr = np.asarray(arr, dtype=int).ravel()
        if arr.size > 0:
            onset_arrays.append(arr)

    if len(onset_arrays) == 0:
        raise ValueError("onsets_by_id contains no onsets")

    all_onsets = np.sort(np.concatenate(onset_arrays))

    # Optional: remove exact duplicates in case the same onset appears twice
    all_onsets = np.unique(all_onsets)

    # --- 3) Define baseline window ---
    baseline_frames = int(round(baseline_s * fps_2p))
    if baseline_frames <= 0:
        raise ValueError("baseline_s is too small; baseline_frames must be > 0")

    # --- 4) Keep only onsets with a full pre-stimulus baseline available ---
    valid_mask = all_onsets >= baseline_frames
    valid_onsets = all_onsets[valid_mask]
    invalid_onsets = all_onsets[~valid_mask]

    if valid_onsets.size == 0:
        raise ValueError(
            "No valid onsets remain after baseline filtering. "
            "Try reducing baseline_s or check onset timing."
        )

    # --- 5) Extract and concatenate baseline segments across all valid onsets ---
    baseline_chunks = []
    for onset in valid_onsets:
        start = onset - baseline_frames
        end = onset
        baseline_chunks.append(cell_traces[:, start:end])  # (n_neurons, baseline_frames)

    # Concatenate along time
    baseline_segments = np.concatenate(baseline_chunks, axis=1)  # (n_neurons, total_frames)

    # --- 6) Compute baseline mean and std per neuron ---
    baseline_mean = np.mean(baseline_segments, axis=1)
    baseline_std = np.std(baseline_segments, axis=1, ddof=ddof)

    # Protect against division by zero
    zero_std_mask = baseline_std == 0
    if np.any(zero_std_mask):
        if verbose:
            print(
                f"[warn] {np.count_nonzero(zero_std_mask)} neuron(s) had baseline std = 0. "
                "Their std will be set to NaN, and z-scores will become NaN for those neurons."
            )
        baseline_std = baseline_std.astype(float)
        baseline_std[zero_std_mask] = np.nan

    # --- 7) Z-score full trace per neuron ---
    z_cell_traces = (cell_traces - baseline_mean[:, None]) / baseline_std[:, None]

    # Return in same orientation as input
    if assume_time_by_neuron:
        z_traces = z_cell_traces.T
    else:
        z_traces = z_cell_traces

    if verbose:
        print(f"n_neurons: {n_neurons}")
        print(f"T: {T}")
        print(f"fps_2p: {fps_2p}")
        print(f"baseline_s: {baseline_s}")
        print(f"baseline_frames: {baseline_frames}")
        print(f"total onsets found: {all_onsets.size}")
        print(f"valid onsets used: {valid_onsets.size}")
        print(f"invalid onsets dropped: {invalid_onsets.size}")
        print(f"baseline_segments shape: {baseline_segments.shape}")
        print(f"z_traces shape: {z_traces.shape}")

    return {
        "z_traces": z_traces,
        "cell_traces": cell_traces,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "baseline_frames": baseline_frames,
        "all_onsets": all_onsets,
        "valid_onsets": valid_onsets,
        "invalid_onsets": invalid_onsets,
        "baseline_segments": baseline_segments,
    }

#EXAMPLE FOR USAGE:
# zres = zscore_dfof_from_prestim_baseline(
#     dfof=dfof,
#     onsets_by_id=onsets_by_id,
#     fps_2p=fps_2p,
#     baseline_s=10.0,
#     assume_time_by_neuron=True,
#     verbose=True,
# )
#
# z_traces = zres["z_traces"]
# baseline_mean = zres["baseline_mean"]
# baseline_std = zres["baseline_std"]
# valid_onsets = zres["valid_onsets"]
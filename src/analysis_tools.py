import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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


def compute_trial_mean_response_metrics(
    trial_aligned_traces,
    stimuli_ids=None,
    fps_2p=2.0,
    t_pre_s=5.0,
    t_post_s=None,
    kept_neuron_indices=None,
):
    """
    Build per-stimulus trial-mean traces plus simple response metrics.

    ``trial_aligned_traces`` is expected to map stimulus IDs to arrays shaped
    ``(n_neurons, n_time, n_trials)``. The returned ``mean_traces`` dict keeps
    the plotting convention used by ``src.plotting.plot_stimulus_means``:
    each value is shaped ``(n_selected_neurons, n_time)``.
    """
    if stimuli_ids is None:
        stimuli_ids = list(trial_aligned_traces.keys())

    pre_frames = int(round(float(t_pre_s) * float(fps_2p)))
    dt = 1.0 / float(fps_2p)

    if kept_neuron_indices is not None:
        kept_neuron_indices = np.asarray(kept_neuron_indices, dtype=int).ravel()

    mean_traces = {}
    peaks = {}
    aucs = {}
    averages = {}

    for stim in stimuli_ids:
        if stim not in trial_aligned_traces:
            raise KeyError(f"Stimulus {stim!r} not found in trial_aligned_traces.")

        trace = np.asarray(trial_aligned_traces[stim], dtype=float)
        if trace.ndim != 3:
            raise ValueError(
                "Each trial-aligned trace must be shaped "
                "(n_neurons, n_time, n_trials)."
            )

        if kept_neuron_indices is not None:
            if kept_neuron_indices.size and (
                kept_neuron_indices.min() < 0
                or kept_neuron_indices.max() >= trace.shape[0]
            ):
                raise IndexError(
                    f"kept_neuron_indices contains values outside stimulus {stim!r} "
                    f"neuron axis of length {trace.shape[0]}."
                )
            trace = trace[kept_neuron_indices, :, :]

        mean_trace = np.nanmean(trace, axis=2)
        n_time = mean_trace.shape[1]
        if t_post_s is None:
            stop_frame = n_time
        else:
            stop_frame = min(n_time, pre_frames + int(round(float(t_post_s) * float(fps_2p))))
        start_frame = min(pre_frames, n_time)

        response_trace = mean_trace[:, start_frame:stop_frame]
        mean_traces[stim] = mean_trace

        if response_trace.shape[1] == 0:
            n_neurons = mean_trace.shape[0]
            peaks[stim] = np.full(n_neurons, np.nan, dtype=float)
            aucs[stim] = np.full(n_neurons, np.nan, dtype=float)
            averages[stim] = np.full(n_neurons, np.nan, dtype=float)
        else:
            peaks[stim] = np.nanmax(response_trace, axis=1)
            aucs[stim] = np.trapezoid(np.nan_to_num(response_trace), dx=dt, axis=1)
            averages[stim] = np.nanmean(response_trace, axis=1)

    return {
        "mean_traces": mean_traces,
        "peaks": peaks,
        "aucs": aucs,
        "averages": averages,
        "pre_frames": pre_frames,
        "response_start_frame": pre_frames,
    }

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


def _id_to_stimulus_name(stim_key, stimuli_id_map):
    if stimuli_id_map is None:
        return str(stim_key)

    id_to_name = {v: k for k, v in stimuli_id_map.items()}
    if stim_key in id_to_name:
        return id_to_name[stim_key]

    try:
        stim_int = int(stim_key)
    except (TypeError, ValueError):
        return str(stim_key)

    return id_to_name.get(stim_int, str(stim_key))


def _stimulus_duration_entry(
    stim_key,
    stimuli_durations,
    stimuli_id_map=None,
    motion_duration_key="motion_sec",
):
    if stimuli_durations is None:
        raise ValueError("stimuli_durations is required.")

    candidate_names = [stim_key, str(stim_key)]
    stim_name = _id_to_stimulus_name(stim_key, stimuli_id_map)
    candidate_names.append(stim_name)

    for candidate in candidate_names:
        if candidate in stimuli_durations:
            duration = stimuli_durations[candidate]
            missing = [
                key
                for key in (motion_duration_key,)
                if key not in duration or duration[key] is None
            ]
            if missing:
                raise ValueError(
                    f"Stimulus {candidate!r} is missing required timing field(s): "
                    f"{', '.join(missing)}."
                )
            return candidate, duration

    raise ValueError(
        f"No stimulus timing metadata found for stimulus {stim_key!r} "
        f"(resolved name {stim_name!r})."
    )


def resolve_selected_stimuli(selected_stimuli, stimuli_id_map, available_stimuli=None):
    """
    Resolve an ordered stimulus selection from names or IDs.

    Parameters
    ----------
    selected_stimuli : sequence
        Stimulus names or IDs in the requested analysis order.
    stimuli_id_map : dict
        Mapping from stimulus name to integer ID.
    available_stimuli : iterable, optional
        Stimulus keys available in the current trace or active-matrix object.

    Returns
    -------
    dict
        Ordered ``stimulus_ids`` and ``stimulus_labels`` plus a name-to-ID map.
    """
    if stimuli_id_map is None:
        stimuli_id_map = {}
    if selected_stimuli is None:
        raise ValueError("selected_stimuli is required.")

    selected_stimuli = list(selected_stimuli)
    if not selected_stimuli:
        raise ValueError("selected_stimuli must contain at least one stimulus.")

    id_to_name = {int(value): name for name, value in stimuli_id_map.items()}
    available_values = None
    if available_stimuli is not None:
        available_values = set(available_stimuli)
        available_values.update(str(value) for value in available_stimuli)
        for value in list(available_stimuli):
            try:
                available_values.add(int(value))
            except (TypeError, ValueError):
                pass

    stimulus_ids = []
    stimulus_labels = []
    seen_ids = set()

    for stimulus in selected_stimuli:
        if isinstance(stimulus, (np.integer, int)):
            stim_id = int(stimulus)
            label = id_to_name.get(stim_id, str(stim_id))
        elif stimulus in stimuli_id_map:
            stim_id = int(stimuli_id_map[stimulus])
            label = str(stimulus)
        else:
            try:
                stim_id = int(stimulus)
            except (TypeError, ValueError) as exc:
                raise KeyError(
                    f"Stimulus {stimulus!r} is not in stimuli_id_map and is not an ID."
                ) from exc
            label = id_to_name.get(stim_id, str(stimulus))

        if stim_id in seen_ids:
            raise ValueError(f"Stimulus ID {stim_id!r} was selected more than once.")
        seen_ids.add(stim_id)

        if available_values is not None and stim_id not in available_values:
            raise KeyError(
                f"Selected stimulus {stimulus!r} resolved to ID {stim_id}, "
                "which is not available in the provided object."
            )

        stimulus_ids.append(stim_id)
        stimulus_labels.append(label)

    return {
        "stimulus_ids": stimulus_ids,
        "stimulus_labels": stimulus_labels,
        "stimulus_id_map": dict(zip(stimulus_labels, stimulus_ids)),
    }


def compute_response_pair_index(response_matrix, left_stimulus, right_stimulus, eps=1e-12):
    """
    Compute a paired preference index from a neuron-by-stimulus response matrix.

    The returned index is ``(left - right) / (left + right)``. Missing
    configured stimulus columns return an all-NaN vector so notebooks can keep
    running with ``left_right_filter_mode='none'`` while still reporting the
    missing control pair.
    """
    response_matrix = pd.DataFrame(response_matrix)
    n_rows = response_matrix.shape[0]
    index_values = np.full(n_rows, np.nan, dtype=float)

    if left_stimulus not in response_matrix.columns or right_stimulus not in response_matrix.columns:
        return index_values

    left = response_matrix[left_stimulus].to_numpy(dtype=float)
    right = response_matrix[right_stimulus].to_numpy(dtype=float)
    denom = left + right
    finite = np.isfinite(left) & np.isfinite(right) & np.isfinite(denom) & (np.abs(denom) > eps)
    index_values[finite] = (left[finite] - right[finite]) / denom[finite]
    return index_values


def build_response_index_keep_mask(index_values, mode="none", threshold=0.3, value_range=(-0.3, 0.3)):
    """
    Build a neuron keep mask from a paired response index.

    Modes match the reusable notebook settings:
    ``none``, ``abs``, ``left``, ``right``, and ``range``.
    """
    index_values = np.asarray(index_values, dtype=float)
    mode = str(mode)

    if mode == "none":
        return np.ones(index_values.shape[0], dtype=bool)
    if mode == "abs":
        return np.isfinite(index_values) & (np.abs(index_values) >= float(threshold))
    if mode == "left":
        return np.isfinite(index_values) & (index_values >= float(threshold))
    if mode == "right":
        return np.isfinite(index_values) & (index_values <= -float(threshold))
    if mode == "range":
        lo, hi = value_range
        return np.isfinite(index_values) & (index_values >= float(lo)) & (index_values <= float(hi))

    raise ValueError("mode must be 'none', 'abs', 'left', 'right', or 'range'.")


def compute_response_window_frames(
    n_time,
    fps_2p,
    t_pre_s=5.0,
    motion_onset_s=8.0,
    motion_duration_s=None,
    tau_s=6.0,
    stimulus=None,
    stimuli_durations=None,
    stimuli_id_map=None,
    motion_duration_key="motion_sec",
):
    """
    Compute clipped frame indices for a stimulus-specific response window.

    The aligned trace time base is ``np.arange(n_time) / fps_2p - t_pre_s``.
    The response window starts at ``motion_onset_s`` and ends at
    ``motion_onset_s + motion_duration_s + tau_s * 2``. When
    ``motion_duration_s`` is omitted, it is read from ``stimuli_durations`` for
    ``stimulus`` using ``motion_duration_key``.
    """
    if n_time is None or int(n_time) <= 0:
        raise ValueError("n_time must be a positive integer.")
    if fps_2p <= 0:
        raise ValueError("fps_2p must be > 0")
    if tau_s is None:
        raise ValueError("tau_s is required.")
    tau_s = float(tau_s)
    if tau_s < 0:
        raise ValueError("tau_s must be >= 0")

    resolved_stimulus = stimulus
    if motion_duration_s is None:
        if stimulus is None:
            raise ValueError(
                "stimulus is required when motion_duration_s is not provided."
            )
        resolved_stimulus, duration = _stimulus_duration_entry(
            stimulus,
            stimuli_durations=stimuli_durations,
            stimuli_id_map=stimuli_id_map,
            motion_duration_key=motion_duration_key,
        )
        motion_duration_s = duration[motion_duration_key]

    motion_duration_s = float(motion_duration_s)
    if motion_duration_s < 0:
        raise ValueError("motion_duration_s must be >= 0")

    time_s = _trial_aligned_time_axis(
        int(n_time),
        fps_2p=fps_2p,
        t_pre_s=t_pre_s,
    )
    start_s = float(motion_onset_s)
    requested_end_s = start_s + motion_duration_s + tau_s * 2.0
    frame_period_s = 1.0 / float(fps_2p)
    clipped_end_s = min(requested_end_s, time_s[-1] + frame_period_s)

    response_mask = (time_s >= start_s) & (time_s < requested_end_s)
    frame_indices = np.flatnonzero(response_mask)
    if frame_indices.size == 0:
        raise ValueError(
            "No frames found in the response window "
            f"{start_s}..{requested_end_s} s after clipping to n_time={n_time}. "
            "Check fps_2p, t_pre_s, motion_onset_s, and stimulus duration."
        )

    return {
        "time_s": time_s,
        "response_mask": response_mask,
        "frame_indices": frame_indices,
        "start_frame": int(frame_indices[0]),
        "stop_frame": int(frame_indices[-1] + 1),
        "n_frames": int(frame_indices.size),
        "start_s": start_s,
        "requested_end_s": requested_end_s,
        "end_s": clipped_end_s,
        "motion_duration_s": motion_duration_s,
        "tau_s": tau_s,
        "stimulus": resolved_stimulus,
    }


def compute_zscore_response_auc(trial_aligned_trace, frame_indices, time_s=None, fps_2p=None):
    """
    Compute mean per-neuron z-score AUC across repetitions for one stimulus.

    Parameters
    ----------
    trial_aligned_trace : array
        Shape ``(n_neurons, n_time, n_reps)``.
    frame_indices : array-like
        Response-window frame indices along the time axis.
    time_s : array-like, optional
        Full aligned time axis. When provided, integration uses the selected
        times. Otherwise ``fps_2p`` supplies a constant frame interval.
    fps_2p : float, optional
        Frame rate used when ``time_s`` is not provided.

    Returns
    -------
    np.ndarray
        One mean AUC response per neuron.
    """
    arr = np.asarray(trial_aligned_trace, dtype=float)
    if arr.ndim != 3:
        raise ValueError(
            f"trial_aligned_trace has shape {arr.shape}; "
            "expected (n_neurons, n_time, n_reps)."
        )

    frame_indices = np.asarray(frame_indices, dtype=int).ravel()
    if frame_indices.size == 0:
        raise ValueError("frame_indices must contain at least one frame.")
    if np.any(frame_indices < 0) or np.any(frame_indices >= arr.shape[1]):
        raise IndexError(
            "frame_indices are outside the trace time axis "
            f"with n_time={arr.shape[1]}."
        )

    response = arr[:, frame_indices, :]
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

    if time_s is not None:
        time_s = np.asarray(time_s, dtype=float)
        if time_s.ndim != 1 or time_s.shape[0] != arr.shape[1]:
            raise ValueError(
                "time_s must be a 1D array with one entry per trace frame."
            )
        x = time_s[frame_indices]
        if x.size == 1:
            if fps_2p is None:
                raise ValueError("fps_2p is required for single-frame AUC windows.")
            auc_by_rep = response[:, 0, :] / float(fps_2p)
        else:
            auc_by_rep = trapz(response, x=x, axis=1)
    else:
        if fps_2p is None:
            raise ValueError("Either time_s or fps_2p is required.")
        if fps_2p <= 0:
            raise ValueError("fps_2p must be > 0")
        if frame_indices.size == 1:
            auc_by_rep = response[:, 0, :] / float(fps_2p)
        else:
            auc_by_rep = trapz(response, dx=1.0 / float(fps_2p), axis=1)

    return np.nanmean(auc_by_rep, axis=1)


def compute_trial_auc_by_neuron(trial_aligned_trace, frame_indices, time_s=None, fps_2p=None):
    """
    Return per-neuron, per-trial AUC values for a response window.

    Parameters mirror ``compute_zscore_response_auc``, but this keeps the
    repetition axis instead of averaging over repetitions. The returned array
    has shape ``(n_neurons, n_reps)``.
    """
    arr = np.asarray(trial_aligned_trace, dtype=float)
    if arr.ndim != 3:
        raise ValueError(
            f"trial_aligned_trace has shape {arr.shape}; "
            "expected (n_neurons, n_time, n_reps)."
        )

    frame_indices = np.asarray(frame_indices, dtype=int).ravel()
    if frame_indices.size == 0:
        raise ValueError("frame_indices must contain at least one frame.")
    if np.any(frame_indices < 0) or np.any(frame_indices >= arr.shape[1]):
        raise IndexError(
            "frame_indices are outside the trace time axis "
            f"with n_time={arr.shape[1]}."
        )

    response = arr[:, frame_indices, :]
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

    if time_s is not None:
        time_s = np.asarray(time_s, dtype=float)
        if time_s.ndim != 1 or time_s.shape[0] != arr.shape[1]:
            raise ValueError(
                "time_s must be a 1D array with one entry per trace frame."
            )
        x = time_s[frame_indices]
        if x.size == 1:
            if fps_2p is None:
                raise ValueError("fps_2p is required for single-frame AUC windows.")
            return response[:, 0, :] / float(fps_2p)
        return trapz(response, x=x, axis=1)

    if fps_2p is None:
        raise ValueError("Either time_s or fps_2p is required.")
    if fps_2p <= 0:
        raise ValueError("fps_2p must be > 0")
    if frame_indices.size == 1:
        return response[:, 0, :] / float(fps_2p)
    return trapz(response, dx=1.0 / float(fps_2p), axis=1)


def _longest_true_run(values):
    """Return the longest consecutive True run in a one-dimensional array."""
    values = np.asarray(values, dtype=bool).ravel()
    if values.size == 0:
        return 0
    padded = np.r_[False, values, False].astype(np.int8)
    transitions = np.flatnonzero(np.diff(padded))
    return int(np.max(transitions[1::2] - transitions[::2], initial=0))


def compute_static_flicker_trial_metrics(
    trial_aligned_traces_zscore,
    trial_aligned_traces_raster,
    side_stimuli,
    stimuli_durations,
    stimuli_id_map,
    fps_2p=2.0,
    t_pre_s=5.0,
    static_window_s=4.0,
    flicker_window_s=4.0,
    static_center_offset_s=-4.0,
    flicker_center_offset_s=4.0,
    min_consecutive_active_frames=2,
    min_active_trial_fraction=0.5,
    kept_neuron_indices=None,
    fish_id=None,
):
    """Build per-trial static--flicker metrics for one fish.

    Each window is centred on an editable offset from flicker onset. With the
    defaults, 4-s static and flicker windows are ``[onset - 6, onset - 2)``
    and ``[onset + 2, onset + 6)`` respectively. Input arrays use the
    repository's ``(neurons, time, trials)`` trial-aligned convention.
    """
    if static_window_s <= 0 or flicker_window_s <= 0:
        raise ValueError("static_window_s and flicker_window_s must be > 0.")
    if min_consecutive_active_frames < 1:
        raise ValueError("min_consecutive_active_frames must be >= 1.")
    if not 0 < min_active_trial_fraction <= 1:
        raise ValueError("min_active_trial_fraction must be in (0, 1].")

    side_stimuli = {str(side): list(stimuli) for side, stimuli in side_stimuli.items()}
    if set(side_stimuli) != {"left", "right"}:
        raise ValueError("side_stimuli must contain exactly 'left' and 'right'.")

    available = set(trial_aligned_traces_zscore)
    available_raster = set(trial_aligned_traces_raster)
    resolved_by_side = {}
    for side, selection in side_stimuli.items():
        resolved = resolve_selected_stimuli(
            selection,
            stimuli_id_map=stimuli_id_map,
            available_stimuli=available,
        )
        missing_raster = [stim for stim in resolved["stimulus_ids"] if stim not in available_raster]
        if missing_raster:
            raise KeyError(f"Significant raster is missing {side} stimulus IDs: {missing_raster}.")
        resolved_by_side[side] = resolved

    id_to_name = {int(value): str(name) for name, value in stimuli_id_map.items()}
    trial_rows = []
    window_rows = []
    display_rows = []

    for side, resolved in resolved_by_side.items():
        n_neurons_side = None
        for stim_id, stimulus in zip(resolved["stimulus_ids"], resolved["stimulus_labels"]):
            zscore = np.asarray(trial_aligned_traces_zscore[stim_id], dtype=float)
            raster = np.asarray(trial_aligned_traces_raster[stim_id], dtype=float)
            if zscore.ndim != 3 or raster.ndim != 3:
                raise ValueError(
                    f"Stimulus {stim_id!r} needs matching (neurons, time, trials) z-score and raster arrays."
                )
            if zscore.shape != raster.shape and kept_neuron_indices is not None:
                kept = np.asarray(kept_neuron_indices, dtype=int).ravel()
                if (
                    raster.shape[0] == kept.size
                    and kept.size > 0
                    and kept.min() >= 0
                    and kept.max() < zscore.shape[0]
                    and zscore.shape[1:] == raster.shape[1:]
                ):
                    zscore = zscore[kept, :, :]
            if zscore.shape != raster.shape:
                raise ValueError(
                    f"Stimulus {stim_id!r} has incompatible z-score and raster shapes: "
                    f"{zscore.shape} versus {raster.shape}."
                )
            if n_neurons_side is None:
                n_neurons_side = zscore.shape[0]
            elif n_neurons_side != zscore.shape[0]:
                raise ValueError(f"{side} stimuli do not share the same neuron count.")

            stimulus_name = id_to_name.get(int(stim_id), str(stimulus))
            duration = stimuli_durations.get(stimulus_name)
            if duration is None or "static_before_sec" not in duration:
                raise ValueError(f"Missing static_before_sec timing for stimulus {stimulus_name!r}.")
            onset_s = float(duration["static_before_sec"])
            stimulus_offset_s = onset_s + float(duration.get("motion_sec", 0.0))
            time_s = _trial_aligned_time_axis(zscore.shape[1], fps_2p=fps_2p, t_pre_s=t_pre_s)
            static_center_s = onset_s + float(static_center_offset_s)
            flicker_center_s = onset_s + float(flicker_center_offset_s)
            static_start_s = static_center_s - float(static_window_s) / 2.0
            static_stop_s = static_center_s + float(static_window_s) / 2.0
            flicker_start_s = flicker_center_s - float(flicker_window_s) / 2.0
            flicker_stop_s = flicker_center_s + float(flicker_window_s) / 2.0
            static_frames = np.flatnonzero((time_s >= static_start_s) & (time_s < static_stop_s))
            flicker_frames = np.flatnonzero((time_s >= flicker_start_s) & (time_s < flicker_stop_s))
            expected_static = int(round(float(static_window_s) * float(fps_2p)))
            expected_flicker = int(round(float(flicker_window_s) * float(fps_2p)))
            if static_frames.size != expected_static or flicker_frames.size != expected_flicker:
                raise ValueError(
                    f"Stimulus {stimulus_name!r} cannot provide requested static/flicker windows "
                    f"at {fps_2p:g} Hz."
                )

            static_auc = compute_trial_auc_by_neuron(zscore, static_frames, time_s=time_s, fps_2p=fps_2p)
            flicker_auc = compute_trial_auc_by_neuron(zscore, flicker_frames, time_s=time_s, fps_2p=fps_2p)
            static_events = raster[:, static_frames, :] > 0
            flicker_events = raster[:, flicker_frames, :] > 0
            static_active = np.apply_along_axis(
                lambda trace: _longest_true_run(trace) >= min_consecutive_active_frames,
                1,
                static_events,
            )
            flicker_active = np.apply_along_axis(
                lambda trace: _longest_true_run(trace) >= min_consecutive_active_frames,
                1,
                flicker_events,
            )

            for neuron_id in range(zscore.shape[0]):
                for trial_id in range(zscore.shape[2]):
                    trial_rows.append({
                        "fish_id": fish_id,
                        "side": side,
                        "stim_id": int(stim_id),
                        "stimulus": stimulus_name,
                        "neuron_id": neuron_id,
                        "trial_id": trial_id,
                        "static_auc": static_auc[neuron_id, trial_id],
                        "flicker_auc": flicker_auc[neuron_id, trial_id],
                        "static_active_trial": bool(static_active[neuron_id, trial_id]),
                        "flicker_active_trial": bool(flicker_active[neuron_id, trial_id]),
                    })
            window_rows.append({
                "fish_id": fish_id,
                "side": side,
                "stim_id": int(stim_id),
                "stimulus": stimulus_name,
                "flicker_onset_s": onset_s,
                "stimulus_offset_s": stimulus_offset_s,
                "static_center_s": static_center_s,
                "flicker_center_s": flicker_center_s,
                "static_start_s": static_start_s,
                "static_stop_s": static_stop_s,
                "flicker_start_s": flicker_start_s,
                "flicker_stop_s": flicker_stop_s,
                "static_n_frames": int(static_frames.size),
                "flicker_n_frames": int(flicker_frames.size),
            })
            static_display = np.nanmean(raster[:, static_frames, :], axis=2)
            flicker_display = np.nanmean(raster[:, flicker_frames, :], axis=2)
            for neuron_id in range(n_neurons_side):
                display_rows.append({
                    "fish_id": fish_id,
                    "side": side,
                    "stim_id": int(stim_id),
                    "stimulus": stimulus_name,
                    "neuron_id": neuron_id,
                    "static_raster": static_display[neuron_id],
                    "flicker_raster": flicker_display[neuron_id],
                    "static_time_s": time_s[static_frames] - onset_s,
                    "flicker_time_s": time_s[flicker_frames] - onset_s,
                    "stimulus_offset_relative_s": stimulus_offset_s - onset_s,
                    "static_window_s": float(static_window_s),
                    "flicker_window_s": float(flicker_window_s),
                })

    trial_metrics = pd.DataFrame(trial_rows)
    stimulus_metrics = (
        trial_metrics.groupby(["fish_id", "side", "stim_id", "stimulus", "neuron_id"], dropna=False)
        .agg(
            static_auc=("static_auc", "mean"),
            flicker_auc=("flicker_auc", "mean"),
            static_active_trial_fraction=("static_active_trial", "mean"),
            flicker_active_trial_fraction=("flicker_active_trial", "mean"),
        )
        .reset_index()
    )
    neuron_stimulus_metrics = stimulus_metrics.copy()
    neuron_stimulus_metrics["valid_neuron"] = (
        np.isfinite(neuron_stimulus_metrics["static_auc"]) & np.isfinite(neuron_stimulus_metrics["flicker_auc"])
    )
    neuron_stimulus_metrics["static_active"] = (
        neuron_stimulus_metrics["static_active_trial_fraction"] >= float(min_active_trial_fraction)
    ) & neuron_stimulus_metrics["valid_neuron"]
    neuron_stimulus_metrics["flicker_active"] = (
        neuron_stimulus_metrics["flicker_active_trial_fraction"] >= float(min_active_trial_fraction)
    ) & neuron_stimulus_metrics["valid_neuron"]
    category_conditions = [
        ~neuron_stimulus_metrics["static_active"] & ~neuron_stimulus_metrics["flicker_active"],
        neuron_stimulus_metrics["static_active"] & ~neuron_stimulus_metrics["flicker_active"],
        neuron_stimulus_metrics["static_active"] & neuron_stimulus_metrics["flicker_active"],
        ~neuron_stimulus_metrics["static_active"] & neuron_stimulus_metrics["flicker_active"],
    ]
    category_labels = ["non-responsive", "static-only", "shared", "newly recruited"]
    neuron_stimulus_metrics["category"] = np.select(category_conditions, category_labels, default="invalid")
    neuron_stimulus_metrics.loc[~neuron_stimulus_metrics["valid_neuron"], "category"] = "invalid"
    neuron_stimulus_metrics["delta_auc"] = neuron_stimulus_metrics["flicker_auc"] - neuron_stimulus_metrics["static_auc"]

    display_data = pd.DataFrame(display_rows).merge(
        neuron_stimulus_metrics[["fish_id", "side", "stim_id", "neuron_id", "category", "valid_neuron"]],
        on=["fish_id", "side", "stim_id", "neuron_id"],
        how="left",
        validate="one_to_one",
    )
    return {
        "trial_metrics": trial_metrics,
        "stimulus_metrics": stimulus_metrics,
        "neuron_stimulus_metrics": neuron_stimulus_metrics,
        "shared_neuron_metrics": neuron_stimulus_metrics.loc[
            neuron_stimulus_metrics["category"] == "shared"
        ].copy(),
        "window_validation": pd.DataFrame(window_rows),
        "classification_raster_data": display_data,
    }


def validate_static_flicker_recruitment_result(result, fps_2p=2.0):
    """Raise concise errors when one-fish static--flicker outputs are inconsistent."""
    windows = pd.DataFrame(result["window_validation"])
    neuron_metrics = pd.DataFrame(result["neuron_stimulus_metrics"])
    raster_data = pd.DataFrame(result["classification_raster_data"])
    if windows.empty or neuron_metrics.empty or raster_data.empty:
        raise ValueError("Static--flicker smoke check received empty analysis outputs.")
    static_duration = windows["static_stop_s"] - windows["static_start_s"]
    flicker_duration = windows["flicker_stop_s"] - windows["flicker_start_s"]
    expected_static_frames = np.rint(static_duration.to_numpy() * float(fps_2p)).astype(int)
    expected_flicker_frames = np.rint(flicker_duration.to_numpy() * float(fps_2p)).astype(int)
    if not np.array_equal(windows["static_n_frames"].to_numpy(), expected_static_frames):
        raise ValueError("Static window frame count does not match its duration.")
    if not np.array_equal(windows["flicker_n_frames"].to_numpy(), expected_flicker_frames):
        raise ValueError("Flicker window frame count does not match its duration.")
    valid = neuron_metrics.loc[neuron_metrics["valid_neuron"]]
    expected_categories = {"non-responsive", "static-only", "shared", "newly recruited"}
    if not set(valid["category"]).issubset(expected_categories):
        raise ValueError("Neuron categories are not mutually exclusive expected labels.")
    if raster_data.duplicated(["fish_id", "side", "stim_id", "neuron_id"]).any():
        raise ValueError("Classification raster data has duplicate neuron rows.")


def compute_stimulus_selectivity_metrics(response_values, stimulus_labels):
    """
    Compute raw-response preference and non-negative selectivity metrics.

    ``preferred_stimulus``, ``max_response``, and ``mean_response`` use raw
    response values. ``selectivity_index`` uses raw response values only when
    the preferred response is positive and the mean other response is
    non-negative; this avoids unstable ratios when excitation is cancelled by
    negative responses. ``simple_selectivity`` and ``lifetime_sparseness`` use
    responses clipped at zero for selectivity only.
    """
    response_values = np.asarray(response_values, dtype=float).ravel()
    stimulus_labels = list(stimulus_labels)

    if response_values.ndim != 1:
        raise ValueError("response_values must be 1D.")
    if response_values.size != len(stimulus_labels):
        raise ValueError("response_values and stimulus_labels must have the same length.")
    if response_values.size < 2:
        raise ValueError("At least two stimuli are required for selectivity metrics.")

    if np.all(np.isnan(response_values)):
        preferred_idx = None
        preferred_stimulus = np.nan
        max_response = np.nan
        mean_response = np.nan
    else:
        preferred_idx = int(np.nanargmax(response_values))
        preferred_stimulus = stimulus_labels[preferred_idx]
        max_response = float(np.nanmax(response_values))
        mean_response = float(np.nanmean(response_values))

    if preferred_idx is None:
        selectivity_index = np.nan
    else:
        other_responses = np.delete(response_values, preferred_idx)
        finite_other_responses = other_responses[np.isfinite(other_responses)]
        if finite_other_responses.size == 0:
            selectivity_index = np.nan
        else:
            preferred_response = float(response_values[preferred_idx])
            mean_other_response = float(np.mean(finite_other_responses))
            denominator = preferred_response + mean_other_response
            if (
                preferred_response <= 0.0
                or mean_other_response < 0.0
                or not np.isfinite(denominator)
                or np.isclose(denominator, 0.0)
            ):
                selectivity_index = np.nan
            else:
                selectivity_index = float(
                    (preferred_response - mean_other_response) / denominator
                )

    positive_responses = np.where(
        np.isfinite(response_values),
        np.maximum(response_values, 0.0),
        0.0,
    )
    total_positive = float(np.sum(positive_responses))

    if total_positive <= 0.0:
        simple_selectivity = np.nan
        lifetime_sparseness = np.nan
    else:
        simple_selectivity = float(np.max(positive_responses) / total_positive)
        n_stimuli = positive_responses.size
        mean_r = float(np.mean(positive_responses))
        mean_r2 = float(np.mean(positive_responses ** 2))
        if mean_r2 <= 0.0:
            lifetime_sparseness = np.nan
        else:
            lifetime_sparseness = float(
                (1.0 - ((mean_r ** 2) / mean_r2)) / (1.0 - (1.0 / n_stimuli))
            )

    return {
        "preferred_stimulus": preferred_stimulus,
        "max_response": max_response,
        "mean_response": mean_response,
        "simple_selectivity": simple_selectivity,
        "selectivity_index": selectivity_index,
        "lifetime_sparseness": lifetime_sparseness,
    }


def classify_stimulus_specificity_neuron(
    lifetime_sparseness,
    n_active_stimuli,
    response_breadth,
    max_positive_response,
    high_lifetime_sparseness=0.70,
    intermediate_lifetime_sparseness=0.40,
    broad_breadth_threshold=0.80,
    strong_response_threshold=np.nan,
    weak_response_threshold=np.nan,
):
    """
    Classify one neuron from selectivity metrics and positive response strength.
    """
    if (
        not np.isfinite(max_positive_response)
        or max_positive_response <= 0.0
        or not np.isfinite(lifetime_sparseness)
    ):
        return "Weak/unclear"

    if np.isfinite(weak_response_threshold) and max_positive_response <= weak_response_threshold:
        return "Weak/unclear"

    if (
        response_breadth >= broad_breadth_threshold
        and np.isfinite(strong_response_threshold)
        and max_positive_response >= strong_response_threshold
    ):
        return "Strong broad responder"

    if n_active_stimuli == 1 and lifetime_sparseness >= high_lifetime_sparseness:
        return "Stimulus-specific neuron"

    if (
        n_active_stimuli > 1
        and response_breadth < broad_breadth_threshold
        and lifetime_sparseness >= intermediate_lifetime_sparseness
    ):
        return "Subset-selective neuron"

    if (
        response_breadth >= broad_breadth_threshold
        and lifetime_sparseness < intermediate_lifetime_sparseness
    ):
        return "Broadly active neuron"

    return "Weak/unclear"


def _has_consecutive_true(bool_vec, min_run_frames):
    bool_vec = np.asarray(bool_vec, dtype=bool)
    if min_run_frames <= 0:
        raise ValueError("min_run_frames must be > 0")
    if bool_vec.size < min_run_frames:
        return False

    run_length = 0
    for value in bool_vec:
        if value:
            run_length += 1
            if run_length >= min_run_frames:
                return True
        else:
            run_length = 0
    return False


def build_active_neuron_matrix_from_trial_raster(
    trial_aligned_traces_raster,
    stimuli_durations,
    stimuli_id_map=None,
    stim_order=None,
    fps_2p=2.0,
    t_pre_s=5.0,
    motion_onset_s=8.0,
    active_fraction_threshold=0.30,
    min_epoch_s=2.0,
    min_active_reps=2,
    expected_reps=4,
    require_expected_reps=True,
    tau_s=6.0,
    motion_duration_key="motion_sec",
    return_counts=False,
):
    """
    Build a binary neuron-by-stimulus active matrix from trial-aligned rasters.

    Each stimulus array is expected to have shape
    (n_neurons, n_time, n_reps), with binary values where 1 means significant.
    A repetition is active when it passes both the significant-frame fraction
    and consecutive-epoch rules inside the response window after motion onset.
    """
    if fps_2p <= 0:
        raise ValueError("fps_2p must be > 0")
    if active_fraction_threshold < 0 or active_fraction_threshold > 1:
        raise ValueError("active_fraction_threshold must be between 0 and 1")
    if min_active_reps < 1:
        raise ValueError("min_active_reps must be >= 1")
    if expected_reps is not None and expected_reps < 1:
        raise ValueError("expected_reps must be >= 1 when provided")
    if tau_s is None:
        raise ValueError("tau_s is required.")
    tau_s = float(tau_s)
    if tau_s < 0:
        raise ValueError("tau_s must be >= 0")

    if stim_order is None:
        stim_order = list(trial_aligned_traces_raster.keys())

    min_run_frames = int(np.ceil(float(min_epoch_s) * float(fps_2p)))
    if min_run_frames <= 0:
        raise ValueError("min_epoch_s is too small; it must span at least one frame")

    matrices = []
    count_matrices = []
    n_neurons_expected = None

    for stim in stim_order:
        key = stim if stim in trial_aligned_traces_raster else str(stim)
        if key not in trial_aligned_traces_raster:
            raise KeyError(f"Stimulus {stim!r} not found in trial_aligned_traces_raster.")

        arr = np.asarray(trial_aligned_traces_raster[key])
        if arr.ndim != 3:
            raise ValueError(
                f"Stimulus {stim!r} has shape {arr.shape}; "
                "expected (n_neurons, n_time, n_reps)."
            )

        n_neurons, n_time, n_reps = arr.shape
        if n_neurons_expected is None:
            n_neurons_expected = n_neurons
        elif n_neurons != n_neurons_expected:
            raise ValueError(
                f"Stimulus {stim!r} has {n_neurons} neurons; expected "
                f"{n_neurons_expected} to preserve neuron order."
            )

        if require_expected_reps and expected_reps is not None and n_reps != expected_reps:
            raise ValueError(
                f"Stimulus {stim!r} has {n_reps} repetitions; "
                f"expected {expected_reps}."
            )

        stim_name, duration = _stimulus_duration_entry(
            stim,
            stimuli_durations=stimuli_durations,
            stimuli_id_map=stimuli_id_map,
            motion_duration_key=motion_duration_key,
        )
        motion_duration_s = float(duration[motion_duration_key])

        time_s = _trial_aligned_time_axis(n_time, fps_2p=fps_2p, t_pre_s=t_pre_s)
        response_end_s = float(motion_onset_s) + motion_duration_s + tau_s * 2.0
        response_mask = (time_s >= float(motion_onset_s)) & (time_s < response_end_s)
        if not np.any(response_mask):
            raise ValueError(
                f"Stimulus {stim_name!r} has no frames in the response window "
                f"{motion_onset_s}..{response_end_s} s. Check t_pre_s, fps_2p, "
                "and the aligned trace length."
            )

        response = arr[:, response_mask, :].astype(bool, copy=False)
        active_reps = np.zeros((n_neurons, n_reps), dtype=bool)

        for rep_idx in range(n_reps):
            rep_values = response[:, :, rep_idx]
            frac_active = np.mean(rep_values, axis=1)
            has_epoch = np.array(
                [
                    _has_consecutive_true(rep_values[neuron_idx], min_run_frames)
                    for neuron_idx in range(n_neurons)
                ],
                dtype=bool,
            )
            active_reps[:, rep_idx] = (
                (frac_active >= active_fraction_threshold) & has_epoch
            )

        active_counts = np.sum(active_reps, axis=1)
        matrices.append((active_counts >= min_active_reps).astype(int))
        count_matrices.append(active_counts.astype(int))

    if n_neurons_expected is None:
        result = pd.DataFrame(columns=stim_order, dtype=int)
        counts = pd.DataFrame(columns=stim_order, dtype=int)
    else:
        result = pd.DataFrame(
            np.column_stack(matrices),
            index=np.arange(n_neurons_expected),
            columns=stim_order,
            dtype=int,
        )
        counts = pd.DataFrame(
            np.column_stack(count_matrices),
            index=np.arange(n_neurons_expected),
            columns=stim_order,
            dtype=int,
        )
    result.index.name = "neuron_id"
    counts.index.name = "neuron_id"

    if return_counts:
        return result, counts
    return result


def _parse_side_segment(stimulus, segments):
    for side, prefixes in (("left", ("Le", "Left")), ("right", ("Ri", "Right"))):
        for prefix in prefixes:
            if stimulus.startswith(prefix):
                suffix = stimulus[len(prefix):]
                if suffix in segments:
                    return side, suffix
    return None, None


def _motion_onset_for_stimulus(stimulus, stimuli_durations, pre_motion_fixed_s):
    duration = None if stimuli_durations is None else stimuli_durations.get(stimulus)
    if duration is None:
        return float(pre_motion_fixed_s)

    return float(duration.get("static_before_sec", pre_motion_fixed_s))



def _trial_aligned_time_axis(n_time, fps_2p, t_pre_s):
    if fps_2p <= 0:
        raise ValueError("fps_2p must be > 0")
    return np.arange(n_time, dtype=float) / float(fps_2p) - float(t_pre_s)


def _validate_motion_windows(time_s, motion_onset_s):
    fixed_mask = (time_s >= 0.0) & (time_s <= motion_onset_s)
    motion_mask = time_s >= motion_onset_s
    if not np.any(fixed_mask):
        raise ValueError(
            "No samples found in fixed-before-motion window. "
            "Check fps_2p, t_pre_s, and pre_motion_fixed_s."
        )
    if not np.any(motion_mask):
        raise ValueError(
            "No samples found in motion window. "
            "Check fps_2p, t_pre_s, pre_motion_fixed_s, and stimuli_durations."
        )

    return fixed_mask, motion_mask


def _iter_motion_delta_blocks(
    trial_aligned_traces,
    fps_2p=2.0,
    t_pre_s=5.0,
    pre_motion_fixed_s=8.0,
    stimuli_id_map=None,
    stimuli_durations=None,
    fish_id=None,
    segments=("B1", "B2", "B3", "B4"),
):
    segments = tuple(segments)

    for stim_key, arr in trial_aligned_traces.items():
        stimulus = _id_to_stimulus_name(stim_key, stimuli_id_map)
        side, segment = _parse_side_segment(stimulus, segments)
        if side is None:
            side = "selected"
            segment = stimulus

        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 3:
            raise ValueError(
                f"Stimulus {stimulus!r} has shape {arr.shape}; "
                "expected (n_neurons, n_time, n_trials)."
            )

        n_neurons, n_time, n_trials = arr.shape
        time_s = _trial_aligned_time_axis(n_time, fps_2p=fps_2p, t_pre_s=t_pre_s)
        motion_onset_s = _motion_onset_for_stimulus(
            stimulus=stimulus,
            stimuli_durations=stimuli_durations,
            pre_motion_fixed_s=pre_motion_fixed_s,
        )
        fixed_mask, motion_mask = _validate_motion_windows(time_s, motion_onset_s)

        yield {
            "fish_id": fish_id,
            "stim_id": stim_key,
            "stimulus": stimulus,
            "segment": segment,
            "side": side,
            "arr": arr,
            "n_neurons": n_neurons,
            "n_trials": n_trials,
            "fixed_time_s": time_s[fixed_mask],
            "motion_time_s": time_s[motion_mask],
            "fixed_values": arr[:, fixed_mask, :],
            "motion_values": arr[:, motion_mask, :],
        }


def compute_motion_delta_integrals(
    trial_aligned_traces,
    fps_2p=2.0,
    t_pre_s=5.0,
    pre_motion_fixed_s=8.0,
    stimuli_id_map=None,
    stimuli_durations=None,
    fish_id=None,
    segments=("B1", "B2", "B3", "B4"),
):
    """
    Compute per-neuron, per-trial motion-minus-fixed integrals for stimuli.

    `trial_aligned_traces` is expected to map stimulus IDs to arrays shaped
    (n_neurons, n_time, n_trials). The fixed/control window starts at stimulus
    onset (t=0) and ends at motion onset. The motion window starts at motion
    onset and continues to the end of the aligned trace.
    """
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    rows = []

    for block in _iter_motion_delta_blocks(
        trial_aligned_traces=trial_aligned_traces,
        fps_2p=fps_2p,
        t_pre_s=t_pre_s,
        pre_motion_fixed_s=pre_motion_fixed_s,
        stimuli_id_map=stimuli_id_map,
        stimuli_durations=stimuli_durations,
        fish_id=fish_id,
        segments=segments,
    ):
        fixed_integral = trapz(block["fixed_values"], x=block["fixed_time_s"], axis=1)
        motion_integral = trapz(block["motion_values"], x=block["motion_time_s"], axis=1)
        delta_integral = motion_integral - fixed_integral

        for neuron_id in range(block["n_neurons"]):
            for trial_id in range(block["n_trials"]):
                rows.append(
                    {
                        "fish_id": block["fish_id"],
                        "trial_id": trial_id,
                        "neuron_id": neuron_id,
                        "stim_id": block["stim_id"],
                        "stimulus": block["stimulus"],
                        "segment": block["segment"],
                        "side": block["side"],
                        "fixed_before_motion_integral": fixed_integral[neuron_id, trial_id],
                        "motion_integral": motion_integral[neuron_id, trial_id],
                        "delta_integral": delta_integral[neuron_id, trial_id],
                    }
                )

    columns = [
        "fish_id",
        "trial_id",
        "neuron_id",
        "stim_id",
        "stimulus",
        "segment",
        "side",
        "fixed_before_motion_integral",
        "motion_integral",
        "delta_integral",
    ]
    return pd.DataFrame(rows, columns=columns)


def compute_motion_delta_peaks(
    trial_aligned_traces,
    fps_2p=2.0,
    t_pre_s=5.0,
    pre_motion_fixed_s=8.0,
    stimuli_id_map=None,
    stimuli_durations=None,
    fish_id=None,
    segments=("B1", "B2", "B3", "B4"),
):
    """
    Compute per-neuron, per-trial motion-minus-fixed max peaks for stimuli.
    """
    rows = []

    for block in _iter_motion_delta_blocks(
        trial_aligned_traces=trial_aligned_traces,
        fps_2p=fps_2p,
        t_pre_s=t_pre_s,
        pre_motion_fixed_s=pre_motion_fixed_s,
        stimuli_id_map=stimuli_id_map,
        stimuli_durations=stimuli_durations,
        fish_id=fish_id,
        segments=segments,
    ):
        fixed_peak = np.nanmax(block["fixed_values"], axis=1)
        motion_peak = np.nanmax(block["motion_values"], axis=1)
        delta_peak = motion_peak - fixed_peak

        for neuron_id in range(block["n_neurons"]):
            for trial_id in range(block["n_trials"]):
                rows.append(
                    {
                        "fish_id": block["fish_id"],
                        "trial_id": trial_id,
                        "neuron_id": neuron_id,
                        "stim_id": block["stim_id"],
                        "stimulus": block["stimulus"],
                        "segment": block["segment"],
                        "side": block["side"],
                        "fixed_before_motion_peak": fixed_peak[neuron_id, trial_id],
                        "motion_peak": motion_peak[neuron_id, trial_id],
                        "delta_peak": delta_peak[neuron_id, trial_id],
                    }
                )

    columns = [
        "fish_id",
        "trial_id",
        "neuron_id",
        "stim_id",
        "stimulus",
        "segment",
        "side",
        "fixed_before_motion_peak",
        "motion_peak",
        "delta_peak",
    ]
    return pd.DataFrame(rows, columns=columns)


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
    costom_threshold: float | None = None,
    folder_name: str | None = None,
    make_plots: bool = True,
    save_indices: bool = True,
    hist_bins: int = 70,
    custom_threshold: float | None = None,
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
    custom_threshold : float or None
        Optional fixed reliability threshold. If None, Otsu's threshold is used.
    costom_threshold : float or None
        Deprecated spelling of custom_threshold, kept for older notebooks.
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
    if custom_threshold is not None and costom_threshold is not None:
        raise ValueError("Use only one of custom_threshold or deprecated costom_threshold")
    threshold_override = custom_threshold if custom_threshold is not None else costom_threshold

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

    if threshold_override is not None:
        otsu_threshold = float(threshold_override)

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

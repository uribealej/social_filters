"""Stimulus selection, trial alignment, and response-window semantics."""

import numpy as np


def select_event_window(time_s, start_s, stop_s):
    """Select native sample centers in [start, stop), without interpolation.

    Times are seconds relative to the recorded event, strictly increasing.
    Equal bounds represent an absent phase and return no indices. Report
    selected sample coverage separately from requested scientific duration.
    """
    times = np.asarray(time_s, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError("Sample times must be finite, 1D and strictly increasing.")
    if not np.isfinite([start_s, stop_s]).all() or stop_s < start_s:
        raise ValueError("Invalid explicit response window.")
    indices = np.flatnonzero((times >= start_s) & (times < stop_s))
    return dict(frame_indices=indices, requested_start_s=float(start_s),
                requested_stop_s=float(stop_s), n_samples=len(indices),
                first_sample_s=float(times[indices[0]]) if len(indices) else np.nan,
                last_sample_s=float(times[indices[-1]]) if len(indices) else np.nan)


def build_event_aligned_traces(dfof, trials, fps_2p, alignment="sequence",
                               t_pre_s=5.0, recovery_s=10.0):
    """Extract trials from recorded event seconds, without a resampled ID trace.

    Input traces are (time, neurons). Output arrays preserve the established
    (neurons, time, trials) contract. Conditions retain first appearance in the
    input table; trials retain row order. Sequence windows have condition-specific
    lengths; bout windows share a length when bouts do. The anchor is rounded
    to the nearest imaging sample (half up). No interpolation is performed.
    Return nominal axes and the actual per-trial sample times relative to each
    unrounded anchor, so subframe alignment error remains visible. Any window
    crossing a block fails; no incomplete trials are silently dropped.
    """
    data = np.asarray(dfof)
    if data.ndim != 2 or fps_2p <= 0 or t_pre_s < 0 or recovery_s < 0:
        raise ValueError("Expected (time, neurons), positive rate and nonnegative windows.")
    if not np.isfinite([fps_2p, t_pre_s, recovery_s]).all():
        raise ValueError("Nonfinite alignment settings.")
    if alignment not in ("sequence", "bout") or trials.empty:
        raise ValueError("Choose sequence or bout alignment with nonempty trials.")
    anchor_key = "onset_s" if alignment == "sequence" else "bout_onset_s"
    pre = int(round(t_pre_s * fps_2p))
    arrays, axes, actual_axes, onsets, errors = {}, {}, {}, {}, {}
    for sid, group in trials.groupby("stimulus_id", sort=False):
        anchors = group[anchor_key].to_numpy(dtype=float)
        durations = group.end_s.to_numpy(dtype=float) - anchors
        if not np.isfinite(anchors).all() or not np.allclose(durations, durations[0]) or durations[0] <= 0:
            raise ValueError(f"Condition {sid}: invalid or inconsistent phase duration.")
        after = int(np.ceil(round(float(durations[0] + recovery_s) * fps_2p, 9)))
        offsets = np.arange(-pre, after)
        frames = np.floor(anchors * fps_2p + 0.5).astype(int)
        indices = frames[:, None] + offsets
        if ((indices[:, 0] < group.block_start_frame.to_numpy()) |
            (indices[:, -1] >= group.block_stop_frame.to_numpy())).any():
            raise ValueError(f"Condition {sid}: extraction window crosses an imaging block boundary.")
        if indices.min() < 0 or indices.max() >= len(data):
            raise ValueError(f"Condition {sid}: event falls outside the data.")
        arrays[sid] = data[indices].transpose(2, 1, 0)
        axes[sid] = offsets / float(fps_2p)
        actual_axes[sid] = indices.T / float(fps_2p) - anchors
        onsets[sid] = frames
        errors[sid] = frames / float(fps_2p) - anchors
    return dict(trial_aligned_traces=arrays, time_s=axes,
                actual_time_s=actual_axes, onsets_by_id=onsets,
                alignment_error_s=errors, alignment=alignment,
                fps_2p=float(fps_2p), pre_frames=pre, trials=trials.copy())

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


# Shared stimulus-name lookup for motion metrics. Keep the historical name.
id_to_stimulus_name = _id_to_stimulus_name


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


# Shared lookup for response-classification callers. Keep the historical name.
stimulus_duration_entry = _stimulus_duration_entry

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

def _trial_aligned_time_axis(n_time, fps_2p, t_pre_s):
    if fps_2p <= 0:
        raise ValueError("fps_2p must be > 0")
    return np.arange(n_time, dtype=float) / float(fps_2p) - float(t_pre_s)


# Shared time base for response metrics and classification. Keep the historical name.
trial_aligned_time_axis = _trial_aligned_time_axis

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

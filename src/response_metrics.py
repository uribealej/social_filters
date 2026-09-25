"""Trial response, AUC, and static-flicker response metrics."""

import numpy as np
import pandas as pd

from src.trial_alignment import _trial_aligned_time_axis, resolve_selected_stimuli

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

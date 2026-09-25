"""Motion-versus-fixed response-delta metrics."""

import numpy as np
import pandas as pd

from src.trial_alignment import (
    _id_to_stimulus_name,
    _trial_aligned_time_axis,
)

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

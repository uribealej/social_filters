"""Active-neuron and raster-response classification helpers."""

import numpy as np
import pandas as pd

from src.trial_alignment import _stimulus_duration_entry, _trial_aligned_time_axis

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

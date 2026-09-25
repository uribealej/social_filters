"""Baseline normalization for continuous calcium traces."""

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

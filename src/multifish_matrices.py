"""Core repetition combination and pooled trace and response matrices."""

import numpy as np
import pandas as pd

from src.response_metrics import compute_zscore_response_auc
from src.trial_alignment import compute_response_window_frames, resolve_selected_stimuli


def combine_reps_one_stim(arr, mode="concat"):
    """
    Combine repetitions for one stimulus.

    Parameters
    ----------
    arr : np.ndarray
        Trial-aligned traces with shape (n_neurons, n_time, n_reps).
    mode : {"concat", "mean"}
        If "concat", repetitions are laid out along time. If "mean",
        repetitions are averaged.

    Returns
    -------
    np.ndarray
        Shape (n_neurons, n_time * n_reps) for "concat", or
        (n_neurons, n_time) for "mean".
    """
    assert arr.ndim == 3, "Expected 3D array (neurons, time, reps)"
    n_neurons, n_time, n_reps = arr.shape

    if mode == "concat":
        arr_swapped = np.swapaxes(arr, 1, 2)
        out = arr_swapped.reshape(n_neurons, n_reps * n_time)
        return out

    elif mode == "mean":
        out = np.nanmean(arr, axis=2)
        return out

    else:
        raise ValueError(f"Unknown mode '{mode}', use 'concat' or 'mean'")


def build_matrix_for_fish(
    trial_aligned_traces,
    stim_order,
    kept_neuron_indices=None,
    combine_mode="concat",
):
    """
    Build one fish matrix by concatenating stimulus blocks along time.

    Parameters
    ----------
    trial_aligned_traces : dict
        Mapping stim_id -> array with shape (n_neurons, n_time, n_reps).
        Integer stimulus keys are tried first, then string keys.
    stim_order : list
        Stimulus IDs in the order to concatenate.
    kept_neuron_indices : array-like or None
        Optional neuron indices to keep before combining repetitions.
    combine_mode : {"concat", "mean"}
        Passed to `combine_reps_one_stim`.

    Returns
    -------
    np.ndarray
        Shape (n_selected_neurons, sum_time_across_stimuli).
    """
    blocks = []

    for stim in stim_order:
        if stim in trial_aligned_traces:
            arr = trial_aligned_traces[stim]
        else:
            arr = trial_aligned_traces[str(stim)]

        if kept_neuron_indices is not None:
            arr = arr[kept_neuron_indices, :, :]

        arr_block = combine_reps_one_stim(arr, mode=combine_mode)
        blocks.append(arr_block)

    fish_mat = np.concatenate(blocks, axis=1)
    return fish_mat


def _stim_key(trial_aligned_traces, stim):
    if stim in trial_aligned_traces:
        return stim
    str_stim = str(stim)
    if str_stim in trial_aligned_traces:
        return str_stim
    raise KeyError(stim)


def describe_matrix_widths(
    all_fish_data,
    stim_order,
    fish_ids=None,
    combine_mode="concat",
    trace_type="dfof",
):
    """Return per-fish matrix widths and per-stimulus source shapes."""
    if fish_ids is None:
        fish_ids = list(all_fish_data.keys())

    rows = []
    for fid in fish_ids:
        if trace_type == "raster":
            trial_key = "trial_aligned_traces_raster"
        elif trace_type == "zscore":
            trial_key = "trial_aligned_traces_z_core"
        elif trace_type == "dfof":
            trial_key = "trial_aligned_traces"
        else:
            raise ValueError(
                f"Unknown trace_type={trace_type!r}. "
                "Use 'dfof', 'raster', or 'zscore'."
            )

        trial_aligned_traces = all_fish_data[fid][trial_key]
        total_width = 0
        stim_shapes = {}
        for stim in stim_order:
            key = _stim_key(trial_aligned_traces, stim)
            arr = trial_aligned_traces[key]
            if arr.ndim != 3:
                raise ValueError(
                    f"Fish {fid!r}, stimulus {stim!r} has shape {arr.shape}; "
                    "expected (neurons, time, reps)."
                )
            _, n_time, n_reps = arr.shape
            width = n_time * n_reps if combine_mode == "concat" else n_time
            total_width += width
            stim_shapes[stim] = arr.shape

        rows.append(
            {
                "fish_id": fid,
                "matrix_width": total_width,
                "stim_shapes": stim_shapes,
            }
        )

    return rows


def build_matrix_all_fish(
    all_fish_data,
    stim_order,
    fish_ids=None,
    combine_mode="concat",
    trace_type="dfof",
):
    """
    Stack per-fish matrices into one all-fish matrix.

    Parameters
    ----------
    all_fish_data : dict
        Per-fish data container from the several-fish notebooks.
    stim_order : list
        Stimulus IDs in the order to concatenate.
    fish_ids : list or None
        Fish IDs to include. If None, uses all keys in `all_fish_data`.
    combine_mode : {"concat", "mean"}
        Passed to `build_matrix_for_fish`.
    trace_type : {"dfof", "raster", "zscore"}
        Selects the per-fish trial-aligned trace key. "dfof" and "zscore"
        apply `kept_neuron_indices`; "raster" uses the raster traces as-is.

    Returns
    -------
    np.ndarray
        Shape (sum_selected_neurons_all_fish, sum_time_across_stimuli).
    """
    if fish_ids is None:
        fish_ids = list(all_fish_data.keys())

    mats = []
    fish_widths = []
    for fid in fish_ids:
        if trace_type == "raster":
            trial_key = "trial_aligned_traces_raster"
            kept_idx = None

        elif trace_type == "zscore":
            trial_key = "trial_aligned_traces_z_core"
            kept_idx = all_fish_data[fid]["kept_neuron_indices"]

        elif trace_type == "dfof":
            trial_key = "trial_aligned_traces"
            kept_idx = all_fish_data[fid]["kept_neuron_indices"]

        else:
            raise ValueError(
                f"Unknown trace_type={trace_type!r}. "
                "Use 'dfof', 'raster', or 'zscore'."
            )

        trial_aligned_traces = all_fish_data[fid][trial_key]

        M_fish = build_matrix_for_fish(
            trial_aligned_traces=trial_aligned_traces,
            stim_order=stim_order,
            kept_neuron_indices=kept_idx,
            combine_mode=combine_mode,
        )
        mats.append(M_fish)
        fish_widths.append((fid, M_fish.shape[1]))

    widths = {width for _, width in fish_widths}
    if len(widths) != 1:
        details = describe_matrix_widths(
            all_fish_data=all_fish_data,
            stim_order=stim_order,
            fish_ids=fish_ids,
            combine_mode=combine_mode,
            trace_type=trace_type,
        )
        detail_lines = [
            f"{row['fish_id']}: width={row['matrix_width']}, "
            f"stim_shapes={row['stim_shapes']}"
            for row in details
        ]
        raise ValueError(
            "Cannot stack fish matrices because their time widths differ. "
            "All fish must have the same aligned window and repetition count "
            f"for stim_order={stim_order!r}, combine_mode={combine_mode!r}, "
            f"trace_type={trace_type!r}.\n"
            + "\n".join(detail_lines)
        )

    M_all = np.vstack(mats)

    return M_all


def build_zscore_response_matrix_for_fish(
    trial_aligned_traces_z_core,
    selected_stimuli,
    stimuli_durations,
    stimuli_id_map=None,
    kept_neuron_indices=None,
    fps_2p=2.0,
    t_pre_s=5.0,
    motion_onset_s=8.0,
    tau_s=6.0,
    motion_duration_key="motion_sec",
):
    """
    Build one neuron-by-stimulus z-score AUC response matrix for one fish.

    Z-score traces are subset with ``kept_neuron_indices`` before response
    extraction so matrix rows match downstream filtered-neuron rasters and
    active-neuron matrices.
    """
    selection = resolve_selected_stimuli(
        selected_stimuli,
        stimuli_id_map=stimuli_id_map,
        available_stimuli=trial_aligned_traces_z_core.keys(),
    )
    stimulus_ids = selection["stimulus_ids"]
    stimulus_labels = selection["stimulus_labels"]

    kept_idx = None
    if kept_neuron_indices is not None:
        kept_idx = np.asarray(kept_neuron_indices, dtype=int).ravel()
        if kept_idx.size == 0:
            raise ValueError("kept_neuron_indices is empty.")

    response_columns = {}
    n_neurons_expected = None

    for stim_id, stim_label in zip(stimulus_ids, stimulus_labels):
        key = _stim_key(trial_aligned_traces_z_core, stim_id)
        arr = np.asarray(trial_aligned_traces_z_core[key], dtype=float)
        if arr.ndim != 3:
            raise ValueError(
                f"Stimulus {stim_id!r} has shape {arr.shape}; "
                "expected (n_neurons, n_time, n_reps)."
            )

        if kept_idx is not None:
            if np.any(kept_idx < 0) or np.any(kept_idx >= arr.shape[0]):
                raise IndexError(
                    f"kept_neuron_indices contains values outside stimulus "
                    f"{stim_id!r} neuron axis with n_neurons={arr.shape[0]}."
                )
            arr = arr[kept_idx, :, :]

        if n_neurons_expected is None:
            n_neurons_expected = arr.shape[0]
        elif arr.shape[0] != n_neurons_expected:
            raise ValueError(
                f"Stimulus {stim_id!r} has {arr.shape[0]} selected neurons; "
                f"expected {n_neurons_expected} to preserve row alignment."
            )

        window = compute_response_window_frames(
            n_time=arr.shape[1],
            fps_2p=fps_2p,
            t_pre_s=t_pre_s,
            motion_onset_s=motion_onset_s,
            stimulus=stim_id,
            stimuli_durations=stimuli_durations,
            stimuli_id_map=stimuli_id_map,
            tau_s=tau_s,
            motion_duration_key=motion_duration_key,
        )
        response_columns[stim_label] = compute_zscore_response_auc(
            arr,
            frame_indices=window["frame_indices"],
            time_s=window["time_s"],
            fps_2p=fps_2p,
        )

    response_matrix = pd.DataFrame(response_columns, columns=stimulus_labels)
    response_matrix.index.name = "neuron_id"
    return response_matrix


def build_zscore_response_matrices_all_fish(
    all_fish_data,
    selected_stimuli,
    fish_ids=None,
    fps_2p=2.0,
    t_pre_s=5.0,
    motion_onset_s=8.0,
    tau_s=6.0,
    motion_duration_key="motion_sec",
):
    """
    Build per-fish and pooled z-score AUC response matrices.

    Returns per-fish matrices, a pooled matrix, and row metadata describing the
    pooled row order. Identity columns remain metadata here; Slice 3 promotes
    them into the neuron summary table.
    """
    if fish_ids is None:
        fish_ids = list(all_fish_data.keys())
    fish_ids = list(dict.fromkeys(fish_ids))

    response_matrices = {}
    pooled_matrices = []
    metadata_rows = []
    selected_stimulus_ids = None
    selected_stimulus_labels = None
    global_neuron_id = 0

    for fid in fish_ids:
        fish = all_fish_data[fid]
        response_matrix = build_zscore_response_matrix_for_fish(
            trial_aligned_traces_z_core=fish["trial_aligned_traces_z_core"],
            selected_stimuli=selected_stimuli,
            stimuli_durations=fish["stimuli_durations"],
            stimuli_id_map=fish.get("stimuli_id_map"),
            kept_neuron_indices=fish.get("kept_neuron_indices"),
            fps_2p=fps_2p,
            t_pre_s=t_pre_s,
            motion_onset_s=motion_onset_s,
            tau_s=tau_s,
            motion_duration_key=motion_duration_key,
        )

        selection = resolve_selected_stimuli(
            selected_stimuli,
            stimuli_id_map=fish.get("stimuli_id_map"),
            available_stimuli=fish["trial_aligned_traces_z_core"].keys(),
        )
        if selected_stimulus_ids is None:
            selected_stimulus_ids = selection["stimulus_ids"]
            selected_stimulus_labels = selection["stimulus_labels"]
        elif response_matrix.columns.tolist() != selected_stimulus_labels:
            raise ValueError(
                f"Fish {fid!r} resolved selected stimulus labels "
                f"{response_matrix.columns.tolist()} but expected "
                f"{selected_stimulus_labels}."
            )

        response_matrices[fid] = response_matrix
        pooled_matrices.append(response_matrix)

        kept_idx = fish.get("kept_neuron_indices")
        if kept_idx is None:
            source_neuron_ids = np.arange(response_matrix.shape[0])
        else:
            source_neuron_ids = np.asarray(kept_idx, dtype=int).ravel()

        if source_neuron_ids.shape[0] != response_matrix.shape[0]:
            raise ValueError(
                f"Fish {fid!r} kept_neuron_indices length "
                f"{source_neuron_ids.shape[0]} does not match response rows "
                f"{response_matrix.shape[0]}."
            )

        for neuron_id, source_neuron_id in enumerate(source_neuron_ids):
            metadata_rows.append(
                {
                    "fish_id": fid,
                    "neuron_id": neuron_id,
                    "source_neuron_id": int(source_neuron_id),
                    "global_neuron_id": global_neuron_id,
                }
            )
            global_neuron_id += 1

    if pooled_matrices:
        pooled_response_matrix = pd.concat(
            pooled_matrices,
            axis=0,
            ignore_index=True,
        )
    else:
        pooled_response_matrix = pd.DataFrame(columns=selected_stimulus_labels or [])

    pooled_response_matrix.index.name = "global_neuron_id"
    row_metadata = pd.DataFrame(
        metadata_rows,
        columns=["fish_id", "neuron_id", "source_neuron_id", "global_neuron_id"],
    )

    return {
        "response_matrices": response_matrices,
        "pooled_response_matrix": pooled_response_matrix,
        "row_metadata": row_metadata,
        "selected_stimulus_ids": selected_stimulus_ids or [],
        "selected_stimulus_labels": selected_stimulus_labels or [],
    }

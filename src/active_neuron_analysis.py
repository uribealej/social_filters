"""Multifish active-neuron decisions, overlaps and pooled diagnostic data."""

import numpy as np
import pandas as pd

from src.multifish_matrices import combine_reps_one_stim
from src.response_classification import build_active_neuron_matrix_from_trial_raster


def build_active_neuron_matrices_all_fish(
    all_fish_data,
    fish_ids=None,
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
    Build one binary neuron-by-stimulus active matrix per fish.

    Parameters controlling inclusion are exposed so notebooks can tune the
    active-neuron definition without changing the helper.
    """
    if fish_ids is None:
        fish_ids = list(all_fish_data.keys())
    fish_ids = list(dict.fromkeys(fish_ids))

    if stim_order is None:
        if not fish_ids:
            return ({}, {}) if return_counts else {}
        stim_order = all_fish_data[fish_ids[0]]["stimuli_ids"]

    active_matrices = {}
    active_counts = {}

    for fid in fish_ids:
        fish = all_fish_data[fid]
        result = build_active_neuron_matrix_from_trial_raster(
            trial_aligned_traces_raster=fish["trial_aligned_traces_raster"],
            stimuli_durations=fish["stimuli_durations"],
            stimuli_id_map=fish.get("stimuli_id_map"),
            stim_order=stim_order,
            fps_2p=fps_2p,
            t_pre_s=t_pre_s,
            motion_onset_s=motion_onset_s,
            active_fraction_threshold=active_fraction_threshold,
            min_epoch_s=min_epoch_s,
            min_active_reps=min_active_reps,
            expected_reps=expected_reps,
            require_expected_reps=require_expected_reps,
            tau_s=tau_s,
            motion_duration_key=motion_duration_key,
            return_counts=return_counts,
        )

        if return_counts:
            active_matrices[fid], active_counts[fid] = result
        else:
            active_matrices[fid] = result

    if return_counts:
        return active_matrices, active_counts
    return active_matrices


def _active_columns_as_bool(active_matrix, stim_order, labels):
    if len(stim_order) != len(labels):
        raise ValueError("stim_order and labels must have the same length.")

    columns = {}
    if hasattr(active_matrix, "columns"):
        for stim, label in zip(stim_order, labels):
            if stim in active_matrix.columns:
                key = stim
            elif str(stim) in active_matrix.columns:
                key = str(stim)
            else:
                raise KeyError(f"Stimulus {stim!r} not found in active matrix.")
            columns[label] = np.asarray(active_matrix[key]).astype(bool)
    else:
        arr = np.asarray(active_matrix)
        if arr.ndim != 2:
            raise ValueError("active_matrix must be 2D.")
        for stim, label in zip(stim_order, labels):
            columns[label] = arr[:, int(stim)].astype(bool)

    return pd.DataFrame(columns)


def compute_active_neuron_jaccard_overlap(
    active_matrix,
    stim_order,
    labels=None,
    empty_union_value=np.nan,
):
    """
    Compute pairwise Jaccard overlap between active-neuron stimulus sets.

    Nonzero values in active_matrix are treated as active. The returned
    DataFrame is indexed and columned by labels, or by stim_order if labels is
    not provided.
    """
    if labels is None:
        labels = list(stim_order)
    labels = list(labels)

    active = _active_columns_as_bool(active_matrix, list(stim_order), labels)
    values = np.empty((len(labels), len(labels)), dtype=float)

    for i, label_i in enumerate(labels):
        set_i = active[label_i].to_numpy(dtype=bool)
        for j, label_j in enumerate(labels):
            set_j = active[label_j].to_numpy(dtype=bool)
            union = np.count_nonzero(set_i | set_j)
            if union == 0:
                values[i, j] = empty_union_value
            else:
                intersection = np.count_nonzero(set_i & set_j)
                values[i, j] = intersection / union

    return pd.DataFrame(values, index=labels, columns=labels)


def build_active_neuron_overlap_matrices_all_fish(
    active_matrices,
    side_stimuli,
    condition_labels,
    empty_union_value=np.nan,
):
    """
    Build left/right active-neuron overlap matrices from per-fish matrices.

    Returns both pooled all-neuron overlaps and the mean of per-fish overlaps:
    result["pooled"][side] and result["mean_per_fish"][side].
    """
    condition_labels = list(condition_labels)
    results = {"pooled": {}, "mean_per_fish": {}}

    for side, stim_order in side_stimuli.items():
        stim_order = list(stim_order)
        per_fish = [
            compute_active_neuron_jaccard_overlap(
                active_matrix=active_matrix,
                stim_order=stim_order,
                labels=condition_labels,
                empty_union_value=empty_union_value,
            )
            for active_matrix in active_matrices.values()
        ]

        if per_fish:
            stacked = np.stack([matrix.to_numpy(dtype=float) for matrix in per_fish])
            valid = np.isfinite(stacked)
            sums = np.where(valid, stacked, 0.0).sum(axis=0)
            counts = valid.sum(axis=0)
            mean_values = np.full(sums.shape, np.nan, dtype=float)
            np.divide(sums, counts, out=mean_values, where=counts > 0)
            results["mean_per_fish"][side] = pd.DataFrame(
                mean_values,
                index=condition_labels,
                columns=condition_labels,
            )

            pooled_input = pd.concat(
                [
                    _active_columns_as_bool(active_matrix, stim_order, condition_labels)
                    for active_matrix in active_matrices.values()
                ],
                axis=0,
                ignore_index=True,
            )
        else:
            results["mean_per_fish"][side] = pd.DataFrame(
                np.nan,
                index=condition_labels,
                columns=condition_labels,
            )
            pooled_input = pd.DataFrame(columns=condition_labels, dtype=bool)

        results["pooled"][side] = compute_active_neuron_jaccard_overlap(
            active_matrix=pooled_input,
            stim_order=condition_labels,
            labels=condition_labels,
            empty_union_value=empty_union_value,
        )

    return results


def _active_column_as_bool(active_matrix, stim):
    if hasattr(active_matrix, "columns"):
        if stim in active_matrix.columns:
            key = stim
        elif str(stim) in active_matrix.columns:
            key = str(stim)
        else:
            raise KeyError(f"Stimulus {stim!r} not found in active matrix.")
        return np.asarray(active_matrix[key]).astype(bool)

    arr = np.asarray(active_matrix)
    if arr.ndim != 2:
        raise ValueError("active_matrix must be 2D.")
    return arr[:, int(stim)].astype(bool)


def _trial_trace_key(trial_aligned_traces, stim):
    if stim in trial_aligned_traces:
        return stim
    str_stim = str(stim)
    if str_stim in trial_aligned_traces:
        return str_stim
    raise KeyError(f"Stimulus {stim!r} not found in trial_aligned_traces.")


def _stimulus_label(stim, stimuli_id_map=None):
    if stimuli_id_map:
        for name, stim_id in stimuli_id_map.items():
            if stim_id == stim or str(stim_id) == str(stim):
                return name
    return str(stim)


def build_pooled_active_trace_diagnostic(
    all_fish_data,
    active_matrices,
    stim_order,
    fish_ids=None,
    combine_mode="mean",
    trial_key="trial_aligned_traces_raster",
):
    """
    Build pooled trace and final-decision matrices for active-neuron diagnostics.

    The trace matrix pools neurons across fish and concatenates selected
    stimulus blocks along time. The decision matrix uses the same pooled row
    order and has one strict active-neuron decision column per stimulus.
    """
    if combine_mode not in {"mean", "concat"}:
        raise ValueError("combine_mode must be 'mean' or 'concat'.")

    if fish_ids is None:
        fish_ids = list(all_fish_data.keys())
    fish_ids = list(dict.fromkeys(fish_ids))
    stim_order = list(stim_order)

    trace_mats = []
    decision_mats = []
    row_metadata = []
    block_widths = None
    block_timepoints = None
    block_reps = None
    stim_labels = None

    for fid in fish_ids:
        if fid not in active_matrices:
            raise KeyError(f"Fish {fid!r} not found in active_matrices.")

        fish = all_fish_data[fid]
        trial_aligned_traces = fish[trial_key]
        active_matrix = active_matrices[fid]

        if stim_labels is None:
            stim_labels = [
                _stimulus_label(stim, fish.get("stimuli_id_map"))
                for stim in stim_order
            ]

        fish_blocks = []
        fish_decisions = []
        n_neurons_expected = None
        fish_widths = []
        fish_timepoints = []
        fish_reps = []

        for stim in stim_order:
            key = _trial_trace_key(trial_aligned_traces, stim)
            arr = np.asarray(trial_aligned_traces[key])
            if arr.ndim != 3:
                raise ValueError(
                    f"Fish {fid!r}, stimulus {stim!r} has shape {arr.shape}; "
                    "expected (neurons, time, reps)."
                )

            n_neurons = arr.shape[0]
            if n_neurons_expected is None:
                n_neurons_expected = n_neurons
            elif n_neurons != n_neurons_expected:
                raise ValueError(
                    f"Fish {fid!r}, stimulus {stim!r} has {n_neurons} neurons; "
                    f"expected {n_neurons_expected} to preserve row alignment."
                )

            block = combine_reps_one_stim(arr, mode=combine_mode)
            fish_blocks.append(block)
            fish_widths.append(block.shape[1])
            fish_timepoints.append(arr.shape[1])
            fish_reps.append(arr.shape[2])

            decision = _active_column_as_bool(active_matrix, stim)
            if decision.shape[0] != n_neurons:
                raise ValueError(
                    f"Fish {fid!r}, stimulus {stim!r} active decision length "
                    f"{decision.shape[0]} does not match trace neurons {n_neurons}."
                )
            fish_decisions.append(decision.astype(int))

        if block_widths is None:
            block_widths = fish_widths
            block_timepoints = fish_timepoints
            block_reps = fish_reps
        elif fish_widths != block_widths:
            raise ValueError(
                f"Fish {fid!r} block widths {fish_widths} do not match expected "
                f"{block_widths}. All fish need matching time/repetition widths."
            )
        elif fish_timepoints != block_timepoints or fish_reps != block_reps:
            raise ValueError(
                f"Fish {fid!r} time/repetition shapes do not match expected values."
            )

        trace_mats.append(np.concatenate(fish_blocks, axis=1))
        decision_mats.append(np.column_stack(fish_decisions))
        row_metadata.extend(
            {"fish_id": fid, "neuron_id": neuron_id}
            for neuron_id in range(n_neurons_expected or 0)
        )

    if trace_mats:
        trace_matrix = np.vstack(trace_mats)
        decision_matrix = np.vstack(decision_mats).astype(int)
    else:
        block_widths = [0 for _ in stim_order]
        block_timepoints = [0 for _ in stim_order]
        block_reps = [0 for _ in stim_order]
        stim_labels = [str(stim) for stim in stim_order]
        trace_matrix = np.empty((0, 0), dtype=float)
        decision_matrix = np.empty((0, len(stim_order)), dtype=int)

    return {
        "trace_matrix": trace_matrix,
        "decision_matrix": decision_matrix,
        "row_metadata": pd.DataFrame(row_metadata),
        "stim_order": stim_order,
        "stim_labels": stim_labels,
        "trace_block_widths": block_widths,
        "trace_block_timepoints": block_timepoints,
        "trace_block_reps": block_reps,
        "combine_mode": combine_mode,
    }

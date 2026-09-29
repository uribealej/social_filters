"""Pure stimulus, response, and neuron-row preparation for several-fish analysis."""

import numpy as np
import pandas as pd

from src.response_selectivity import (
    build_response_index_keep_mask, compute_response_pair_index,
)
from src.stimulus_specificity import (
    add_selectivity_metrics_to_summary_table, build_neuron_stimulus_summary_table,
)
from src.trial_alignment import resolve_selected_stimuli


def resolve_stimulus_set(selection, reference_fish, fallback_labels=None):
    """Resolve a user-selected stimulus list against a reference fish.

    Args:
        selection (sequence or None): Ordered stimulus IDs or names.
        reference_fish (dict): Bundle with stimulus map, names, and aligned traces.
        fallback_labels (sequence or None): Ordered names used when selection is None.

    Returns:
        dict: Ordered stimulus IDs and labels from ``resolve_selected_stimuli``.
    """
    if selection is None:
        if fallback_labels is None:
            fallback_labels = reference_fish["stimuli_names"]
        selection = list(fallback_labels)
    return resolve_selected_stimuli(
        selection,
        stimuli_id_map=reference_fish["stimuli_id_map"],
        available_stimuli=reference_fish["trial_aligned_traces_z_core"].keys(),
    )


def _resolve_response_control_column(control, stimulus_ids, stimulus_labels, response_columns):
    """Resolve one control ID or name to a response-matrix column label.

    Args:
        control (object): Requested control ID or name.
        stimulus_ids (sequence): IDs corresponding to stimulus_labels.
        stimulus_labels (sequence): Labels corresponding to stimulus_ids.
        response_columns (sequence): Available matrix column labels.

    Returns:
        object: Matching response column label, preserving its original type.
    """
    response_columns = list(response_columns)
    if control in response_columns:
        return control

    control_text = str(control)
    for column in response_columns:
        if str(column) == control_text:
            return column

    for stim_id, stim_label in zip(stimulus_ids, stimulus_labels):
        if control == stim_id or control_text == str(stim_id):
            if stim_label in response_columns:
                return stim_label
            if str(stim_label) in response_columns:
                return str(stim_label)

    available = ", ".join(str(column) for column in response_columns)
    raise KeyError(
        f"left_right_controls value {control!r} could not be resolved to a "
        f"response column. Available response columns: {available}"
    )


def resolve_response_control_columns(
    preference_controls,
    stimulus_ids,
    stimulus_labels,
    response_columns,
):
    """Resolve two left/right control IDs or names to response columns.

    Args:
        preference_controls (sequence or None): Left and right controls in order.
        stimulus_ids (sequence): IDs corresponding to stimulus_labels.
        stimulus_labels (sequence): Labels corresponding to stimulus_ids.
        response_columns (sequence): Available response-matrix columns.

    Returns:
        tuple or None: Left and right column labels, or None when unconfigured.
    """
    if preference_controls is None:
        return None
    if len(preference_controls) != 2:
        raise ValueError("preference_controls must contain exactly two values.")
    return tuple(
        _resolve_response_control_column(
            control=control,
            stimulus_ids=stimulus_ids,
            stimulus_labels=stimulus_labels,
            response_columns=response_columns,
        )
        for control in preference_controls
    )


def build_selected_neuron_summary(
    response_matrices,
    pooled_response_matrix,
    response_row_metadata,
    active_matrices,
    response_stimulus_ids,
    response_stimulus_labels,
    plot_stimulus_ids,
    plot_stimulus_labels,
    analysis_label="",
    preference_controls=None,
    filter_mode="none",
    filter_threshold=0.3,
    filter_range=(-0.3, 0.3),
):
    """Build response summaries, left/right preference, and a plot keep mask.

    Args:
        response_matrices (dict): Fish ID to neuron-by-stimulus response table.
        pooled_response_matrix (DataFrame): Pooled neuron-by-stimulus responses.
        response_row_metadata (DataFrame): Pooled neuron rows in response order.
        active_matrices (dict): Fish ID to neuron-by-stimulus active decisions.
        response_stimulus_ids (sequence): IDs for all response columns.
        response_stimulus_labels (sequence): Labels for all response columns.
        plot_stimulus_ids (sequence): IDs selected for plotted summaries.
        plot_stimulus_labels (sequence): Labels selected for plotted summaries.
        analysis_label (str): Label included in each summary row.
        preference_controls (sequence or None): Left/right controls for the index.
        filter_mode (str): Response-index keep-mask mode.
        filter_threshold (float): Threshold used by the selected mode.
        filter_range (tuple): Inclusive range for ``range`` mode.

    Returns:
        dict: Summary tables, pooled preference values and keep mask, selected
        response rows, resolved controls, and the row-aligned filter table.
    """
    summary_all = build_neuron_stimulus_summary_table(
        response_matrices=response_matrices,
        active_matrices=active_matrices,
        row_metadata=response_row_metadata,
        selected_stimulus_ids=response_stimulus_ids,
        selected_stimulus_labels=response_stimulus_labels,
        analysis_label=analysis_label,
    )
    summary_plot = build_neuron_stimulus_summary_table(
        response_matrices=response_matrices,
        active_matrices=active_matrices,
        row_metadata=response_row_metadata,
        selected_stimulus_ids=plot_stimulus_ids,
        selected_stimulus_labels=plot_stimulus_labels,
        analysis_label=analysis_label,
    )
    summary_plot = add_selectivity_metrics_to_summary_table(
        summary_plot,
        selected_stimulus_labels=plot_stimulus_labels,
    )

    resolved_preference_controls = resolve_response_control_columns(
        preference_controls=preference_controls,
        stimulus_ids=response_stimulus_ids,
        stimulus_labels=response_stimulus_labels,
        response_columns=pd.DataFrame(pooled_response_matrix).columns,
    )

    if resolved_preference_controls is None:
        preference_index = np.full(pooled_response_matrix.shape[0], np.nan, dtype=float)
    else:
        preference_index = compute_response_pair_index(
            pooled_response_matrix,
            left_stimulus=resolved_preference_controls[0],
            right_stimulus=resolved_preference_controls[1],
        )
    keep_mask = build_response_index_keep_mask(
        preference_index,
        mode=filter_mode,
        threshold=filter_threshold,
        value_range=filter_range,
    )

    for table in (summary_plot, summary_all):
        table["left_right_index"] = preference_index
        table["plot_neuron_keep"] = keep_mask

    filter_table = response_row_metadata.copy().reset_index(drop=True)
    filter_table["left_right_index"] = preference_index
    filter_table["plot_neuron_keep"] = keep_mask
    if resolved_preference_controls is not None:
        filter_table["left_right_left_control"] = resolved_preference_controls[0]
        filter_table["left_right_right_control"] = resolved_preference_controls[1]

    return {
        "neuron_summary_table": summary_plot,
        "neuron_summary_all_stimuli_table": summary_all,
        "left_right_index": preference_index,
        "plot_neuron_keep_mask": keep_mask,
        "plot_neuron_keep_indices": np.flatnonzero(keep_mask),
        "resolved_preference_controls": resolved_preference_controls,
        "selected_auc_response_matrix": pd.DataFrame(pooled_response_matrix).loc[
            keep_mask,
            list(plot_stimulus_labels),
        ].copy(),
        "plot_filter_table": filter_table,
    }


def build_fish_keep_masks(response_row_metadata, keep_mask, fish_ids):
    """Split a pooled neuron keep mask into per-fish response-row masks.

    Args:
        response_row_metadata (DataFrame): One fish ID row per pooled neuron.
        keep_mask (array-like): Boolean pooled neuron mask in metadata row order.
        fish_ids (sequence): Fish IDs in desired result order.

    Returns:
        dict: Fish ID to its boolean neuron mask in response-row order.
    """
    keep_mask = np.asarray(keep_mask, dtype=bool)
    metadata = pd.DataFrame(response_row_metadata).reset_index(drop=True)
    if keep_mask.shape[0] != metadata.shape[0]:
        raise ValueError("keep_mask length does not match response_row_metadata rows.")
    if "fish_id" not in metadata.columns:
        raise ValueError("response_row_metadata is missing fish_id.")

    fish_column = metadata["fish_id"].to_numpy()
    return {fid: keep_mask[fish_column == fid] for fid in fish_ids}


def build_filtered_trial_aligned_traces_for_fish(
    fish_data,
    stimulus_ids,
    trace_key="trial_aligned_traces_z_core",
    fish_keep_mask=None,
):
    """Select one fish's trial traces after preprocessing and plot filtering.

    Args:
        fish_data (dict): Fish bundle with trial traces and kept neuron indices.
        stimulus_ids (sequence): Ordered IDs to select from the trace mapping.
        trace_key (str): Fish-bundle key for neuron-by-time-by-trial arrays.
        fish_keep_mask (array-like or None): Boolean mask after preprocessing.

    Returns:
        dict: Stimulus ID to filtered (neurons, time, trials) trace array.
    """
    kept = np.asarray(fish_data.get("kept_neuron_indices", []), dtype=int)
    traces = fish_data[trace_key]
    selected = {}
    for stim_id in stimulus_ids:
        key = stim_id if stim_id in traces else str(stim_id)
        arr = np.asarray(traces[key])
        if kept.size:
            arr = arr[kept, :, :]
        if fish_keep_mask is not None:
            fish_keep_mask = np.asarray(fish_keep_mask, dtype=bool)
            if arr.shape[0] != fish_keep_mask.shape[0]:
                raise ValueError(
                    f"Trace rows for stimulus {stim_id!r} do not match the filter rows: "
                    f"{arr.shape[0]} vs {fish_keep_mask.shape[0]}."
                )
            arr = arr[fish_keep_mask, :, :]
        selected[stim_id] = arr
    return selected


def subset_active_matrices(active_matrices, fish_ids, stim_order):
    """Subset per-fish active matrices to ordered stimulus columns.

    Args:
        active_matrices (dict): Fish ID to neuron-by-stimulus decision table.
        fish_ids (sequence): Fish IDs in desired result order.
        stim_order (sequence): Stimulus IDs in desired column order.

    Returns:
        dict: Fish ID to a copied neuron-by-stimulus table.
    """
    subset = {}
    for fid in fish_ids:
        active_matrix = active_matrices[fid]
        column_keys = []
        for stim in stim_order:
            if stim in active_matrix.columns:
                column_keys.append(stim)
            elif str(stim) in active_matrix.columns:
                column_keys.append(str(stim))
            else:
                raise KeyError(f"Stimulus {stim!r} was not found in active_matrices for {fid}.")
        subset[fid] = active_matrix.loc[:, column_keys].copy()
    return subset


def filter_active_matrices_by_keep_mask(active_matrices, response_row_metadata, keep_mask, fish_ids):
    """Apply a pooled neuron keep mask to fish active matrices.

    Args:
        active_matrices (dict): Fish ID to neuron-by-stimulus decision table.
        response_row_metadata (DataFrame): Pooled neuron rows with fish IDs.
        keep_mask (array-like): Boolean mask aligned to response metadata rows.
        fish_ids (sequence): Fish IDs in desired result order.

    Returns:
        dict: Fish ID to filtered active table with reset row index.
    """
    keep_mask = np.asarray(keep_mask, dtype=bool)
    if keep_mask.shape[0] != response_row_metadata.shape[0]:
        raise ValueError("keep_mask length does not match response_row_metadata rows.")

    filtered = {}
    fish_column = response_row_metadata["fish_id"].to_numpy()
    for fid in fish_ids:
        fish_rows = fish_column == fid
        fish_keep_mask = keep_mask[fish_rows]
        active_matrix = active_matrices[fid]
        if active_matrix.shape[0] != fish_keep_mask.shape[0]:
            raise ValueError(
                f"Active matrix rows for {fid} do not match response metadata rows: "
                f"{active_matrix.shape[0]} vs {fish_keep_mask.shape[0]}."
            )
        filtered[fid] = active_matrix.loc[fish_keep_mask].reset_index(drop=True)
    return filtered

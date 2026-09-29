"""Pure diagnostic and figure-input preparation for several-fish analysis."""

import numpy as np
import pandas as pd

from src.active_neuron_analysis import (
    build_active_neuron_overlap_matrices_all_fish,
    build_pooled_active_trace_diagnostic,
)
from src.multifish_matrices import (
    build_matrix_all_fish, build_zscore_response_matrices_all_fish,
)
from src.response_selectivity import (
    compute_response_pair_index, compute_stimulus_selectivity_metrics,
)
from src.several_fish_selection import (
    build_fish_keep_masks, filter_active_matrices_by_keep_mask,
    subset_active_matrices,
)
from src.trial_alignment import compute_response_window_frames, resolve_selected_stimuli


def build_all_fish_raster_inputs(
    all_fish_data, fish_ids, stim_order, reference_fish, timing, raster_settings,
):
    """Prepare pooled raster rows and optional left-right neuron sorting.

    Args:
        all_fish_data (dict): Fish ID to aligned trace bundles.
        fish_ids (sequence): Fish IDs in pooled row order.
        stim_order (sequence): Stimulus IDs in raster block order.
        reference_fish (dict): Stimulus map and aligned reference traces.
        timing (dict): Imaging frame rate and prestimulus duration.
        raster_settings (dict): Trace, repetition, and sort configuration.

    Returns:
        dict: Pooled neuron-by-time matrix, optional left-right index,
        optional neuron sort order, and sort label.
    """
    flat_matrix_all_fish = build_matrix_all_fish(
        all_fish_data=all_fish_data,
        stim_order=stim_order,
        fish_ids=fish_ids,
        combine_mode=raster_settings["combine_mode"],
        trace_type=raster_settings["trace_type"],
    )

    raster_neuron_order = None
    raster_sort_label = None
    raster_left_right_index = None
    if raster_settings["raster_sort_mode"] == "left_right_index":
        left_selection = resolve_selected_stimuli(
            [raster_settings["left_control"]],
            stimuli_id_map=reference_fish["stimuli_id_map"],
            available_stimuli=reference_fish["trial_aligned_traces_z_core"].keys(),
        )
        right_selection = resolve_selected_stimuli(
            [raster_settings["right_control"]],
            stimuli_id_map=reference_fish["stimuli_id_map"],
            available_stimuli=reference_fish["trial_aligned_traces_z_core"].keys(),
        )
        sort_response = build_zscore_response_matrices_all_fish(
            all_fish_data=all_fish_data,
            fish_ids=fish_ids,
            selected_stimuli=left_selection["stimulus_labels"] + right_selection["stimulus_labels"],
            fps_2p=timing["fps_2p"],
            t_pre_s=timing["t_pre_s"],
        )["pooled_response_matrix"]
        raster_left_right_index = compute_response_pair_index(
            pd.DataFrame(
                {
                    "left": sort_response[left_selection["stimulus_labels"][0]],
                    "right": sort_response[right_selection["stimulus_labels"][0]],
                }
            ),
            left_stimulus="left",
            right_stimulus="right",
        )
        sort_values = np.where(np.isfinite(raster_left_right_index), raster_left_right_index, np.inf)
        raster_neuron_order = np.argsort(sort_values, kind="stable")
        raster_sort_label = "left-right index"

    return {
        "flat_matrix_all_fish": flat_matrix_all_fish,
        "raster_left_right_index": raster_left_right_index,
        "raster_neuron_order": raster_neuron_order,
        "raster_sort_label": raster_sort_label,
    }


def build_lifetime_sparseness_summary(
    all_fish_data, fish_ids, reference_fish, timing, selected_stimuli,
):
    """Prepare pooled lifetime-sparseness rows for selected stimuli.

    Args:
        all_fish_data (dict): Fish ID to aligned trace bundles.
        fish_ids (sequence): Fish IDs in pooled response order.
        reference_fish (dict): Stimulus map and reference traces.
        timing (dict): Imaging frame rate and prestimulus duration.
        selected_stimuli (sequence): Stimulus IDs or names in desired order.

    Returns:
        dict: Resolved stimulus selection and pooled per-neuron metric table.
    """
    selection = resolve_selected_stimuli(
        selected_stimuli,
        stimuli_id_map=reference_fish["stimuli_id_map"],
        available_stimuli=reference_fish["trial_aligned_traces_z_core"].keys(),
    )
    response_results = build_zscore_response_matrices_all_fish(
        all_fish_data=all_fish_data,
        fish_ids=fish_ids,
        selected_stimuli=selection["stimulus_labels"],
        fps_2p=timing["fps_2p"],
        t_pre_s=timing["t_pre_s"],
    )
    response_matrix = response_results["pooled_response_matrix"]
    metric_rows = [
        compute_stimulus_selectivity_metrics(
            row.to_numpy(dtype=float),
            stimulus_labels=selection["stimulus_labels"],
        )
        for _, row in response_matrix.iterrows()
    ]
    summary_table = pd.concat(
        [
            response_results["row_metadata"].reset_index(drop=True),
            pd.DataFrame(metric_rows).reset_index(drop=True),
        ],
        axis=1,
    )
    return {"selection": selection, "summary_table": summary_table}


def build_motion_active_counts(active_matrices, fish_ids, stimulus_ids, stimulus_labels):
    """Count active neurons per fish and selected stimulus.

    Args:
        active_matrices (dict): Fish ID to neuron-by-stimulus decisions.
        fish_ids (sequence): Fish IDs in output row order.
        stimulus_ids (sequence): Stimulus IDs in output column order.
        stimulus_labels (sequence): Labels paired with stimulus_ids.

    Returns:
        DataFrame: Fish-by-stimulus active neuron counts.
    """
    return pd.DataFrame(
        {
            label: [active_matrices[fish_id][stim_id].sum() for fish_id in fish_ids]
            for stim_id, label in zip(stimulus_ids, stimulus_labels)
        },
        index=fish_ids,
    )


def build_response_window_validation(
    all_fish_data,
    fish_ids,
    stimulus_ids,
    stimulus_labels,
    fps_2p,
    t_pre_s,
    motion_onset_s=8.0,
    tau_s=6.0,
    motion_duration_key="motion_sec",
):
    """Build response-window rows for selected fish and stimuli.

    Args:
        all_fish_data (dict): Fish ID to aligned trace and stimulus metadata.
        fish_ids (sequence): Fish IDs in requested row order.
        stimulus_ids (sequence): Stimulus IDs in requested row order.
        stimulus_labels (sequence): Names paired with stimulus_ids.
        fps_2p (float): Imaging frames per second.
        t_pre_s (float): Seconds before stimulus onset in aligned traces.
        motion_onset_s (float): Motion onset after stimulus start, in seconds.
        tau_s (float): Extra response-window time constant, in seconds.
        motion_duration_key (str): Duration field in stimulus metadata.

    Returns:
        DataFrame: One row per fish and stimulus, with frame and second bounds.
    """
    rows = []
    for fid in fish_ids:
        fish = all_fish_data[fid]
        trial_aligned_z = fish["trial_aligned_traces_z_core"]
        for stim_id, stim_label in zip(stimulus_ids, stimulus_labels):
            stim_key = stim_id if stim_id in trial_aligned_z else str(stim_id)
            arr = trial_aligned_z[stim_key]
            window = compute_response_window_frames(
                n_time=arr.shape[1],
                fps_2p=fps_2p,
                t_pre_s=t_pre_s,
                motion_onset_s=motion_onset_s,
                stimulus=stim_id,
                stimuli_durations=fish["stimuli_durations"],
                stimuli_id_map=fish["stimuli_id_map"],
                tau_s=tau_s,
                motion_duration_key=motion_duration_key,
            )
            rows.append(
                {
                    "fish_id": fid,
                    "stimulus": stim_label,
                    "stimulus_id": stim_id,
                    "n_time": arr.shape[1],
                    "start_frame": window["start_frame"],
                    "stop_frame": window["stop_frame"],
                    "n_response_frames": window["n_frames"],
                    "start_s": window["start_s"],
                    "end_s": window["end_s"],
                }
            )
    return pd.DataFrame(rows)


def build_high_sparseness_raster_data(
    all_fish_data,
    fish_ids,
    summary_table,
    stim_order,
    preferred_stimulus_order,
    lifetime_sparseness_threshold=0.65,
    keep_mask=None,
    combine_mode="mean",
    trace_type="zscore",
):
    """Prepare a sorted high-sparseness raster matrix without plotting.

    Args:
        all_fish_data (dict): Fish ID to aligned trace bundles.
        fish_ids (sequence): Fish IDs in pooled neuron order.
        summary_table (DataFrame): Neuron metrics with global neuron IDs.
        stim_order (sequence): Stimuli in raster block order.
        preferred_stimulus_order (sequence): Preferred-stimulus sort order.
        lifetime_sparseness_threshold (float): Strict lower metric bound.
        keep_mask (array-like or None): Optional pooled neuron mask.
        combine_mode (str): Repetition combination mode.
        trace_type (str): Aligned trace family to pool.

    Returns:
        dict: Selected neuron-by-time matrix, sorted summary rows, global
        neuron IDs, and zero-based plot row order.
    """
    metric_table = pd.DataFrame(summary_table).copy()
    if keep_mask is not None and not metric_table.empty:
        keep_mask = np.asarray(keep_mask, dtype=bool)
        global_ids = metric_table["global_neuron_id"].to_numpy(dtype=int)
        if keep_mask.shape[0] <= global_ids.max(initial=-1):
            raise ValueError("keep_mask is shorter than summary_table global_neuron_id values.")
        metric_table = metric_table.loc[keep_mask[global_ids]].copy()

    response_matrix = build_matrix_all_fish(
        all_fish_data,
        stim_order,
        fish_ids=fish_ids,
        combine_mode=combine_mode,
        trace_type=trace_type,
    )
    preference_rank = {
        stimulus: rank
        for rank, stimulus in enumerate(preferred_stimulus_order)
    }
    selected_table = metric_table.loc[
        np.isfinite(metric_table["lifetime_sparseness"])
        & (metric_table["lifetime_sparseness"] > float(lifetime_sparseness_threshold))
    ].copy()
    selected_table["_preferred_stimulus_rank"] = (
        selected_table["preferred_stimulus"].map(preference_rank).fillna(len(preference_rank))
    )
    selected_table["_max_auc_sort"] = selected_table["max_response"].fillna(-np.inf)
    selected_table = selected_table.sort_values(
        by=["_preferred_stimulus_rank", "_max_auc_sort", "global_neuron_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    global_order = selected_table["global_neuron_id"].to_numpy(dtype=int)
    raster_matrix = response_matrix[global_order, :] if global_order.size else response_matrix[:0, :]
    return {
        "matrix": raster_matrix,
        "summary_table": selected_table,
        "global_neuron_order": global_order,
        "plot_neuron_order": np.arange(raster_matrix.shape[0], dtype=int),
    }


def build_pooled_mean_trace_by_stimulus(
    all_fish_data,
    fish_ids,
    stim_order,
    trace_type="zscore",
    use_kept_neurons=False,
    response_row_metadata=None,
    keep_mask=None,
):
    """Build per-fish mean traces for each selected stimulus.

    Args:
        all_fish_data (dict): Fish ID to aligned neuron-by-time-by-trial traces.
        fish_ids (sequence): Fish IDs in output row order.
        stim_order (sequence): Stimulus IDs in output key order.
        trace_type (str): ``dfof``, ``raster``, ``zscore``, or ``norm``.
        use_kept_neurons (bool): Apply preprocessing neuron indices.
        response_row_metadata (DataFrame or None): Pooled response-row fish IDs.
        keep_mask (array-like or None): Optional pooled response-row mask.

    Returns:
        dict: Stimulus ID to (fish, time) mean trace array.
    """
    trial_key_by_type = {
        "dfof": "trial_aligned_traces",
        "raster": "trial_aligned_traces_raster",
        "zscore": "trial_aligned_traces_z_core",
        "norm": "trial_aligned_traces_norm",
    }
    if trace_type not in trial_key_by_type:
        raise ValueError("trace_type must be 'dfof', 'raster', 'zscore', or 'norm'.")

    if keep_mask is not None:
        if response_row_metadata is None:
            raise ValueError("response_row_metadata is required when keep_mask is provided.")
        use_kept_neurons = True
        fish_keep_masks = build_fish_keep_masks(
            response_row_metadata=response_row_metadata,
            keep_mask=keep_mask,
            fish_ids=fish_ids,
        )
    else:
        fish_keep_masks = {fid: None for fid in fish_ids}

    trial_key = trial_key_by_type[trace_type]
    mean_traces = {stim: [] for stim in stim_order}
    for fid in fish_ids:
        fish = all_fish_data[fid]
        kept = np.asarray(fish.get("kept_neuron_indices", []), dtype=int)
        fish_keep_mask = fish_keep_masks[fid]
        for stim in stim_order:
            traces = fish[trial_key]
            key = stim if stim in traces else str(stim)
            arr = np.asarray(traces[key], dtype=float)
            if use_kept_neurons and kept.size:
                arr = arr[kept, :, :]
            if fish_keep_mask is not None:
                if arr.shape[0] != fish_keep_mask.shape[0]:
                    raise ValueError(
                        f"Trace rows for {fid}, stimulus {stim!r} do not match "
                        f"the filter rows: {arr.shape[0]} vs {fish_keep_mask.shape[0]}."
                    )
                arr = arr[fish_keep_mask, :, :]
            mean_by_neuron = np.nanmean(arr, axis=2)
            mean_traces[stim].append(np.nanmean(mean_by_neuron, axis=0))

    return {stim: np.vstack(rows) for stim, rows in mean_traces.items()}


def build_diagnostic_trace_source_data(all_fish_data, fish_ids, trial_key, stim_order, active_matrices_source):
    """Align z-score diagnostic traces to active-matrix neuron rows.

    Args:
        all_fish_data (dict): Fish ID to aligned trace bundles.
        fish_ids (sequence): Fish IDs in diagnostic order.
        trial_key (str): Trace family to use in the diagnostic.
        stim_order (sequence): Stimuli in diagnostic block order.
        active_matrices_source (dict): Fish ID to active neuron rows.

    Returns:
        dict: Fish bundles with row-aligned z-score traces, or the original
        input mapping for other trace families.
    """
    if trial_key != "trial_aligned_traces_z_core":
        return all_fish_data

    trace_source_data = {}
    for fid in fish_ids:
        fish = all_fish_data[fid]
        active_row_count = active_matrices_source[fid].shape[0]
        kept_indices = np.asarray(fish.get("kept_neuron_indices", []), dtype=int)
        z_traces = {}
        for stim in stim_order:
            traces = fish[trial_key]
            key = stim if stim in traces else str(stim)
            arr = np.asarray(traces[key])
            if arr.shape[0] == active_row_count:
                z_traces[key] = arr
            elif kept_indices.shape[0] == active_row_count and kept_indices.max(initial=-1) < arr.shape[0]:
                z_traces[key] = arr[kept_indices, :, :]
            else:
                raise ValueError(
                    f"Cannot align z-score rows for {fid}, stimulus {stim!r}: "
                    f"trace rows={arr.shape[0]}, active rows={active_row_count}, "
                    f"kept indices={kept_indices.shape[0]}."
                )
        fish_copy = fish.copy()
        fish_copy[trial_key] = z_traces
        trace_source_data[fid] = fish_copy
    return trace_source_data


def apply_diagnostic_keep_mask(diagnostic, keep_mask):
    """Subset a pooled diagnostic by its neuron keep mask.

    Args:
        diagnostic (dict): Pooled trace, decision, and metadata rows.
        keep_mask (array-like): Boolean mask in pooled diagnostic row order.

    Returns:
        dict: Shallow copy with trace, decision, and metadata rows filtered.
    """
    keep_mask = np.asarray(keep_mask, dtype=bool)
    diagnostic = diagnostic.copy()
    diagnostic["trace_matrix"] = diagnostic["trace_matrix"][keep_mask, :]
    diagnostic["decision_matrix"] = diagnostic["decision_matrix"][keep_mask, :]
    diagnostic["row_metadata"] = diagnostic["row_metadata"].loc[keep_mask].reset_index(drop=True)
    return diagnostic


def build_decision_then_mean_sort_order(diagnostic):
    """Sort rows by active decisions, then mean trace strength.

    Args:
        diagnostic (dict): Pooled trace and decision matrices with neuron rows.

    Returns:
        ndarray: Integer neuron row indices in descending signature/strength order.
    """
    trace_matrix = np.asarray(diagnostic["trace_matrix"], dtype=float)
    decision_matrix = np.asarray(diagnostic["decision_matrix"], dtype=int)
    if trace_matrix.shape[0] == 0:
        return np.array([], dtype=int)
    weights = 2 ** np.arange(decision_matrix.shape[1] - 1, -1, -1)
    signatures = decision_matrix @ weights
    mean_strength = np.nanmean(trace_matrix, axis=1)
    mean_strength = np.nan_to_num(mean_strength, nan=-np.inf)
    return np.lexsort((-mean_strength, -signatures))


def build_metric_sort_order(row_metadata, metric_table, metric_column, side="left"):
    """Sort diagnostic neuron rows using a summary-table metric.

    Args:
        row_metadata (DataFrame): Diagnostic rows with fish and neuron IDs.
        metric_table (DataFrame): Matching per-neuron metric rows.
        metric_column (str): Numeric metric to sort.
        side (str): ``right`` reverses the left-right-index preference.

    Returns:
        ndarray: Original diagnostic row indices in metric order.
    """
    if metric_column not in metric_table.columns:
        raise KeyError(f"metric_table is missing {metric_column!r}.")
    metric_source = metric_table[["fish_id", "neuron_id", metric_column]].copy()
    order_source = row_metadata.reset_index(names="_diagnostic_row").merge(
        metric_source,
        on=["fish_id", "neuron_id"],
        how="left",
        validate="one_to_one",
    )
    values = order_source[metric_column].to_numpy(dtype=float)
    original_rows = order_source["_diagnostic_row"].to_numpy(dtype=int)
    finite = np.isfinite(values)
    missing_rank = np.where(finite, 0, 1)
    if metric_column == "left_right_index" and side == "right":
        metric_rank = np.where(finite, values, np.inf)
    else:
        metric_rank = np.where(finite, -values, np.inf)
    return original_rows[np.lexsort((original_rows, metric_rank, missing_rank))]


def build_overlap_diagnostic_data(
    all_fish_data,
    active_matrices,
    fish_ids,
    side_stimulus_ids,
    side_stimulus_labels,
    response_row_metadata,
    keep_mask=None,
    side_to_plot="left",
    aggregation="mean_per_fish",
    plot_mode="significant_raster",
    combine_mode="mean",
    sort_mode="decision_then_mean",
    metric_table=None,
):
    """Build active-neuron overlap and pooled trace diagnostic inputs.

    Args:
        all_fish_data (dict): Fish ID to aligned trace bundles.
        active_matrices (dict): Fish ID to neuron-by-stimulus active decisions.
        fish_ids (sequence): Fish IDs in pooled row order.
        side_stimulus_ids (dict): Side to ordered stimulus IDs.
        side_stimulus_labels (dict): Side to matching stimulus labels.
        response_row_metadata (DataFrame): Pooled neuron identity rows.
        keep_mask (array-like or None): Optional pooled neuron mask.
        side_to_plot (str): Side selected for trace diagnostics.
        aggregation (str): ``pooled`` or ``mean_per_fish`` overlap view.
        plot_mode (str): ``significant_raster`` or ``z_score`` trace family.
        combine_mode (str): Repetition combination mode.
        sort_mode (str): Decision/mean or neuron-metric sorting mode.
        metric_table (DataFrame or None): Required for metric sorting.

    Returns:
        dict: Overlap tables, chosen matrix, trace/decision diagnostics,
        diagnostic row order, trace key, and stimulus order.
    """
    valid_plots = {"significant_raster", "z_score"}
    valid_aggregations = {"pooled", "mean_per_fish"}
    valid_sort_modes = {"decision_then_mean", "left_right_index", "lifetime_sparseness"}
    if plot_mode not in valid_plots:
        raise ValueError(f"plot_mode must be one of {sorted(valid_plots)}.")
    if aggregation not in valid_aggregations:
        raise ValueError(f"aggregation must be one of {sorted(valid_aggregations)}.")
    if sort_mode not in valid_sort_modes:
        raise ValueError(f"sort_mode must be one of {sorted(valid_sort_modes)}.")
    if side_to_plot not in side_stimulus_ids:
        raise KeyError(f"side_to_plot {side_to_plot!r} was not found in side_stimulus_ids.")

    combined_stim_order = []
    for stim_order in side_stimulus_ids.values():
        combined_stim_order.extend(stim_order)
    active_overlap = subset_active_matrices(active_matrices, fish_ids, combined_stim_order)
    if keep_mask is not None:
        active_overlap = filter_active_matrices_by_keep_mask(
            active_overlap,
            response_row_metadata=response_row_metadata,
            keep_mask=keep_mask,
            fish_ids=fish_ids,
        )

    overlap_results = {"pooled": {}, "mean_per_fish": {}}
    for side, stim_order in side_stimulus_ids.items():
        side_overlap = build_active_neuron_overlap_matrices_all_fish(
            active_matrices=active_overlap,
            side_stimuli={side: stim_order},
            condition_labels=side_stimulus_labels[side],
        )
        overlap_results["pooled"][side] = side_overlap["pooled"][side]
        overlap_results["mean_per_fish"][side] = side_overlap["mean_per_fish"][side]

    diagnostic_stim_order = side_stimulus_ids[side_to_plot]
    diagnostic_trial_key = {
        "significant_raster": "trial_aligned_traces_raster",
        "z_score": "trial_aligned_traces_z_core",
    }[plot_mode]
    active_diagnostic = subset_active_matrices(active_matrices, fish_ids, diagnostic_stim_order)
    diagnostic_source = build_diagnostic_trace_source_data(
        all_fish_data,
        fish_ids=fish_ids,
        trial_key=diagnostic_trial_key,
        stim_order=diagnostic_stim_order,
        active_matrices_source=active_diagnostic,
    )
    sort_source = build_pooled_active_trace_diagnostic(
        all_fish_data=all_fish_data,
        active_matrices=active_diagnostic,
        fish_ids=fish_ids,
        stim_order=diagnostic_stim_order,
        combine_mode=combine_mode,
        trial_key="trial_aligned_traces_raster",
    )
    diagnostic = build_pooled_active_trace_diagnostic(
        all_fish_data=diagnostic_source,
        active_matrices=active_diagnostic,
        fish_ids=fish_ids,
        stim_order=diagnostic_stim_order,
        combine_mode=combine_mode,
        trial_key=diagnostic_trial_key,
    )

    if keep_mask is not None:
        diagnostic = apply_diagnostic_keep_mask(diagnostic, keep_mask)
        sort_source = apply_diagnostic_keep_mask(sort_source, keep_mask)

    if sort_mode == "decision_then_mean":
        neuron_order = build_decision_then_mean_sort_order(sort_source)
    else:
        if metric_table is None:
            raise ValueError("metric_table is required for metric-based diagnostic sorting.")
        metric_column = "left_right_index" if sort_mode == "left_right_index" else "lifetime_sparseness"
        neuron_order = build_metric_sort_order(
            diagnostic["row_metadata"],
            metric_table=metric_table,
            metric_column=metric_column,
            side=side_to_plot,
        )

    return {
        "overlap_results": overlap_results,
        "matrix_to_plot": overlap_results[aggregation][side_to_plot],
        "active_trace_diagnostic": diagnostic,
        "sort_source_diagnostic": sort_source,
        "diagnostic_neuron_order": neuron_order,
        "diagnostic_trial_key": diagnostic_trial_key,
        "diagnostic_stim_order": diagnostic_stim_order,
    }

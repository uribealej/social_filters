"""Multifish active-neuron decisions, overlaps and pooled diagnostic data."""

import numpy as np
import pandas as pd

from src.multifish_matrices import combine_reps_one_stim
from src.response_classification import (
    build_active_neuron_matrix_from_trial_raster, classify_trial_raster_activity,
    first_persistent_response_onset,
)
from src.trial_alignment import resolve_selected_stimuli, trial_aligned_time_axis


def _condition_bin_activity(raster, n_retained, edges, fps_2p, t_pre_s,
                            program_onset_s, expected_reps, min_active_reps,
                            active_fraction_threshold, min_epoch_s, context):
    """Classify complete native-time bins without a whole-response gate."""
    values = np.asarray(raster)
    if (values.ndim != 3 or values.shape[0] != n_retained
            or values.shape[2] != expected_reps):
        raise ValueError(f"{context}: incompatible retained rows or repetitions.")
    if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
        raise ValueError(f"{context}: bin overlap needs finite binary rasters.")
    times = trial_aligned_time_axis(values.shape[1], fps_2p, t_pre_s) - program_onset_s
    if (not times.size or times[0] > edges[0] + 1e-9
            or times[-1] + 1 / fps_2p < edges[-1] - 1e-9):
        raise ValueError(f"{context}: traces do not cover the complete bin window.")
    decisions, windows = [], []
    for index, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (times >= start) & (times < end)
        if mask.sum() < int(np.ceil(min_epoch_s * fps_2p)):
            raise ValueError(f"{context}: bin {index} is shorter than the activity epoch.")
        per_trial = classify_trial_raster_activity(
            values[:, mask, :], fps_2p, active_fraction_threshold, min_epoch_s)
        decisions.append((per_trial == 1).sum(axis=1) >= min_active_reps)
        windows.append(dict(bin_index=index, bin_start_s=start, bin_end_s=end,
            n_frames=int(mask.sum()), first_sample_s=times[mask][0],
            last_sample_s=times[mask][-1]))
    return np.column_stack(decisions), windows


def _summarize_control_bin_overlap(fish_bins):
    """Average within-fish containment fractions, excluding empty controls."""
    keys = ["side", "control", "stimulus", "bin_index", "bin_start_s", "bin_end_s", "time_s"]
    return fish_bins.groupby(keys, sort=False, dropna=False).agg(
        fraction_shared=("fraction_shared", "mean"),
        sem_shared=("fraction_shared", "sem"),
        n_fish=("fraction_shared", "count"),
        n_control_total=("n_control", "sum"),
        n_shared_total=("n_shared", "sum"),
        mean_n_control=("n_control", "mean"),
    ).reset_index()


def build_control_bin_overlap(
    all_fish_data, fish_ids, side_conditions, bin_edges_s, fps_2p=2.0,
    t_pre_s=5.0, program_onset_s=8.0, min_epoch_s=1.0,
    min_active_reps=2, expected_reps=4, active_fraction_threshold=0.10,
):
    """Measure same-bin significant activity shared with same-side Control.

    Inputs are finite binary rasters (retained neurons, time, trials), aligned
    to each fish's unique ``kept_neuron_indices``. ``side_conditions`` maps side
    to ``control`` and ordered ``stimuli`` including that control exactly once.
    Bin edges are nonnegative seconds relative to ``program_onset_s`` after
    appearance. Each [left, right) bin must be covered and contain enough native
    samples for the epoch. Missing data and mismatched rows/repetitions raise.

    Apply the canonical per-trial fraction-and-consecutive-epoch rule separately
    inside each bin, then require ``min_active_reps`` trials. The epoch must fit
    wholly within the bin and within each passing trial. There is no onset or
    whole-response eligibility gate; a neuron can contribute in several bins.

    ``fish_bins`` returns counts and n_shared / n_control for each fish/bin/pair.
    This directional containment does not penalize stimulus-only recruitment.
    Empty controls yield NaN; ``bin_summary`` averages fractions equally across
    eligible fish, returns descriptive SEM (NaN below two fish) and valid counts.
    Summed counts are audits, not the denominator for the equal-fish percentage.
    Self-Control is identity, not independent repeatability or a statistical test.

    Also return ``bin_activity[side][fish_id][label]`` boolean arrays (retained
    neurons, bins), ``neuron_ids[fish_id]`` in matching row order, native sample
    ``bin_windows``, selections, edges and settings. No files are written.
    """
    fish_ids = list(fish_ids)
    edges = np.asarray(bin_edges_s, dtype=float)
    if not fish_ids or len(set(fish_ids)) != len(fish_ids):
        raise ValueError("Provide unique, nonempty fish_ids.")
    if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
            or edges[0] < 0 or not (np.diff(edges) > 0).all()):
        raise ValueError("bin_edges_s must be finite, increasing and nonnegative.")
    if (not side_conditions or not np.isfinite([fps_2p, t_pre_s, program_onset_s]).all()
            or fps_2p <= 0 or t_pre_s < 0 or program_onset_s < 0):
        raise ValueError("Provide sides, a positive imaging rate and nonnegative timing.")
    if (not np.isfinite([expected_reps, min_active_reps]).all()
            or int(expected_reps) != expected_reps or int(min_active_reps) != min_active_reps
            or not 1 <= min_active_reps <= expected_reps):
        raise ValueError("Require integer repetitions with 1 <= min_active_reps <= expected_reps.")
    expected_reps, min_active_reps = int(expected_reps), int(min_active_reps)
    classify_trial_raster_activity(np.empty((0, 0, expected_reps)), fps_2p,
        active_fraction_threshold, min_epoch_s)
    rows, window_rows, selections, activity, neuron_ids = [], [], {}, {}, {}
    for side, config in side_conditions.items():
        selections[side], activity[side] = None, {}
        for fid in fish_ids:
            fish = all_fish_data[fid]
            if not np.isclose(fish.get("fps_2p", fps_2p), fps_2p):
                raise ValueError(f"{fid}: imaging rate does not match fps_2p.")
            retained = np.asarray(fish["kept_neuron_indices"])
            if retained.ndim != 1 or len(np.unique(retained)) != len(retained):
                raise ValueError(f"{fid}: retained neuron identities must be unique and 1D.")
            neuron_ids[fid] = retained.copy()
            traces = fish["trial_aligned_traces_raster"]
            selection = resolve_selected_stimuli(config["stimuli"], fish["stimuli_id_map"], traces.keys())
            control = resolve_selected_stimuli([config["control"]], fish["stimuli_id_map"], traces.keys())
            labels, ids = selection["stimulus_labels"], selection["stimulus_ids"]
            control_label = control["stimulus_labels"][0]
            if control_label not in labels or len(set(labels)) != len(labels):
                raise ValueError("Ordered stimuli must include the control exactly once.")
            if selections[side] is not None and labels != selections[side]["stimulus_labels"]:
                raise ValueError("Fish must share the configured condition labels/order.")
            selections[side] = {**selection, "control_label": control_label}
            masks = activity[side][fid] = {}
            for sid, label in zip(ids, labels):
                masks[label], windows = _condition_bin_activity(
                    traces[sid if sid in traces else str(sid)], len(retained), edges,
                    fps_2p, t_pre_s, program_onset_s, expected_reps, min_active_reps,
                    active_fraction_threshold, min_epoch_s, f"{fid}/{label}")
                window_rows.extend(dict(side=side, fish_id=fid, stimulus=label, **w) for w in windows)
            reference = masks[control_label]
            for label in labels:
                comparison = masks[label]
                for index, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
                    n_control = int(reference[:, index].sum())
                    n_stimulus = int(comparison[:, index].sum())
                    n_shared = int((reference[:, index] & comparison[:, index]).sum())
                    rows.append(dict(side=side, fish_id=fid, control=control_label,
                        stimulus=label, bin_index=index, bin_start_s=start, bin_end_s=end,
                        time_s=(start + end) / 2, n_retained=len(retained),
                        n_control=n_control, n_stimulus=n_stimulus, n_shared=n_shared,
                        n_control_only=n_control - n_shared, n_stimulus_only=n_stimulus - n_shared,
                        fraction_shared=n_shared / n_control if n_control else np.nan))
    fish_bins = pd.DataFrame(rows)
    return dict(bin_edges_s=edges.copy(), side_selections=selections,
        fish_bins=fish_bins, bin_summary=_summarize_control_bin_overlap(fish_bins),
        bin_activity=activity, neuron_ids=neuron_ids, bin_windows=pd.DataFrame(window_rows),
        settings=dict(fps_2p=fps_2p, t_pre_s=t_pre_s, program_onset_s=program_onset_s,
            min_epoch_s=min_epoch_s, min_active_reps=min_active_reps,
            expected_reps=expected_reps, active_fraction_threshold=active_fraction_threshold))


def _condition_consensus_onsets(raster, n_retained, edges, fps_2p, t_pre_s,
                               motion_onset_s, expected_reps, persistence_frames,
                               min_significant_trial_fraction, context):
    """Validate one condition and return retained-row-aligned onset seconds."""
    raster = np.asarray(raster)
    if (raster.ndim != 3 or raster.shape[0] != n_retained
            or raster.shape[2] != expected_reps):
        raise ValueError(f"{context}: incompatible retained rows or repetitions.")
    if not np.isfinite(raster).all() or not np.isin(raster, [0, 1]).all():
        raise ValueError(f"{context}: onset comparison needs finite binary rasters.")
    times = trial_aligned_time_axis(raster.shape[1], fps_2p, t_pre_s) - motion_onset_s
    if (not times.size or times[0] > edges[0] + 1e-9
            or times[-1] + 1 / fps_2p < edges[-1] - 1e-9):
        raise ValueError(f"{context}: traces do not cover the complete onset window.")
    window = (times >= edges[0]) & (times < edges[-1])
    if window.sum() < persistence_frames:
        raise ValueError("Onset window is shorter than the persistence requirement.")
    consensus = raster[:, window, :].mean(axis=2)
    return np.asarray([first_persistent_response_onset(
        row, times[window], persistence_frames, min_significant_trial_fraction,
    ) for row in consensus])


def _control_onset_bin_counts(reference_bin, comparison_bin, n_bins):
    """Return raw counts/fractions and full transition rows for one fish/pair."""
    bins, transitions = [], []
    for cb in range(n_bins):
        selected = comparison_bin[reference_bin == cb]
        denominator = len(selected)
        counts = dict(n_shared=int((selected >= 0).sum()), n_same=int((selected == cb).sum()),
            n_earlier=int(((selected >= 0) & (selected < cb)).sum()),
            n_later=int((selected > cb).sum()), n_no_onset=int((selected < 0).sum()))
        fractions = {name.replace("n_", "fraction_", 1): value / denominator
                     if denominator else np.nan for name, value in counts.items()}
        bins.append(dict(control_bin=cb, n_control=denominator, **counts, **fractions))
        for sb in [*range(n_bins), -1]:
            count = int((selected == sb).sum())
            transitions.append(dict(control_bin=cb, stimulus_bin=sb,
                n_control=denominator, count=count,
                fraction=count / denominator if denominator else np.nan))
    return bins, transitions


def build_control_onset_comparison(
    all_fish_data, fish_ids, side_conditions, bin_edges_s, fps_2p=2.0,
    t_pre_s=5.0, motion_onset_s=8.0, persistence_frames=2,
    min_significant_trial_fraction=0.5, active_fraction_threshold=0.10,
    min_epoch_s=1.0, min_active_reps=2, expected_reps=4, tau_s=6.0,
):
    """Compare same-neuron onset bins against each side's control within fish.

    ``side_conditions`` maps side to ``control`` and ordered ``stimuli`` names
    or IDs (including the control). ``bin_edges_s`` are increasing, nonnegative
    seconds after the shared numeric ``motion_onset_s``. Bins are [left, right),
    including an exclusive final edge. Rasters are (retained neurons, time,
    trials); unique ``kept_neuron_indices`` must identify those exact rows.

    Apply the existing active-neuron rule, then locate the first sustained
    trial-fraction onset inside the bin span. Persistence must fit inside it.
    Control-active neurons without such an onset are audited and excluded from
    the reference population. Comparison-inactive/no-onset neurons go to the
    final no-detected-onset column. Missing/nonbinary data or incomplete temporal
    coverage raise rather than being counted as absent responses.

    Return neuron_table (identity, flags, onset seconds/bins), audit_table,
    fish_bins (raw counts and fractions), fish_transitions (all bin pairs plus
    comparison bin -1 for no onset), and equal-fish bin/transition summaries.
    Every fraction divides by all eligible control neurons in that fish/bin.
    Empty reference bins are NaN, excluded from means, with n_fish recorded.
    Control-versus-self is descriptive identity, not an independent benchmark.
    """
    fish_ids = list(fish_ids)
    edges = np.asarray(bin_edges_s, dtype=float)
    if not fish_ids or len(set(fish_ids)) != len(fish_ids):
        raise ValueError("Provide unique, nonempty fish_ids.")
    if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
            or edges[0] < 0 or not (np.diff(edges) > 0).all()):
        raise ValueError("bin_edges_s must be finite, increasing and nonnegative.")
    if (not np.isfinite([fps_2p, t_pre_s, motion_onset_s]).all()
            or fps_2p <= 0 or motion_onset_s < 0):
        raise ValueError("Provide a positive imaging rate and finite numeric timing.")
    if not side_conditions:
        raise ValueError("Provide at least one side and its control/stimuli.")
    # Validate onset settings even when no reference neurons are available.
    first_persistent_response_onset([], [], persistence_frames, min_significant_trial_fraction)
    bin_rows, transition_rows, neurons, audits, selections = [], [], [], [], {}
    n_bins = len(edges) - 1
    for side, config in side_conditions.items():
        selections[side] = None
        for fid in fish_ids:
            fish = all_fish_data[fid]
            if not np.isclose(fish.get("fps_2p", fps_2p), fps_2p):
                raise ValueError(f"{fid}: imaging rate does not match fps_2p.")
            traces = fish["trial_aligned_traces_raster"]
            selection = resolve_selected_stimuli(config["stimuli"], fish["stimuli_id_map"], traces.keys())
            control = resolve_selected_stimuli([config["control"]], fish["stimuli_id_map"], traces.keys())
            ids, labels = selection["stimulus_ids"], selection["stimulus_labels"]
            control_label = control["stimulus_labels"][0]
            if control_label not in labels or len(set(labels)) != len(labels):
                raise ValueError("Ordered stimuli must include the control exactly once.")
            if selections[side] is not None and labels != selections[side]["stimulus_labels"]:
                raise ValueError("Fish must share the configured condition labels/order.")
            selections[side] = {**selection, "control_label": control_label}
            retained = np.asarray(fish["kept_neuron_indices"])
            if retained.ndim != 1 or len(np.unique(retained)) != len(retained):
                raise ValueError(f"{fid}: retained neuron identities must be unique and 1D.")
            onsets = {}
            for sid, label in zip(ids, labels):
                onsets[label] = _condition_consensus_onsets(
                    traces[sid if sid in traces else str(sid)], len(retained), edges,
                    fps_2p, t_pre_s, motion_onset_s, expected_reps, persistence_frames,
                    min_significant_trial_fraction, f"{fid}/{label}",
                )
            active = build_active_neuron_matrix_from_trial_raster(
                traces, fish["stimuli_durations"], fish["stimuli_id_map"], ids,
                fps_2p=fps_2p, t_pre_s=t_pre_s, motion_onset_s=motion_onset_s,
                active_fraction_threshold=active_fraction_threshold, min_epoch_s=min_epoch_s,
                min_active_reps=min_active_reps, expected_reps=expected_reps,
                require_expected_reps=True, tau_s=tau_s,
            )
            for sid, label in zip(ids, labels):
                onsets[label][~active[sid].to_numpy(bool)] = np.inf
            reference_onset = onsets[control_label]
            eligible = np.isfinite(reference_onset)
            reference_bin = np.searchsorted(edges, reference_onset[eligible], side="right") - 1
            audits.append(dict(side=side, fish_id=fid, control=control_label,
                n_retained=len(retained), n_control_active=int(active[control["stimulus_ids"][0]].sum()),
                n_control_with_onset=int(eligible.sum()),
                n_control_active_no_onset=int(active[control["stimulus_ids"][0]].sum() - eligible.sum())))
            for sid, label in zip(ids, labels):
                comparison_onset = onsets[label][eligible]
                comparison_bin = np.where(np.isfinite(comparison_onset),
                    np.searchsorted(edges, comparison_onset, side="right") - 1, -1)
                for row, cb, sb, onset in zip(np.flatnonzero(eligible), reference_bin, comparison_bin, comparison_onset):
                    neurons.append(dict(side=side, fish_id=fid, neuron_id=retained[row],
                        retained_row=int(row), control=control_label, stimulus=label,
                        control_onset_s=reference_onset[row], stimulus_onset_s=onset,
                        control_bin=int(cb), stimulus_bin=int(sb),
                        stimulus_active=bool(active[sid].iloc[row])))
                bins, transitions = _control_onset_bin_counts(reference_bin, comparison_bin, n_bins)
                bin_rows.extend(dict(side=side, fish_id=fid, control=control_label,
                    stimulus=label, **row) for row in bins)
                transition_rows.extend(dict(side=side, fish_id=fid,
                    stimulus=label, **row) for row in transitions)
    fish_bins, fish_transitions = pd.DataFrame(bin_rows), pd.DataFrame(transition_rows)
    keys = ["side", "stimulus", "control_bin"]
    grouped = fish_bins.groupby(keys, sort=False, dropna=False)
    metrics = [c for c in fish_bins if c.startswith("fraction_")]
    bin_summary = grouped[metrics].mean().reset_index()
    bin_summary["n_fish"] = grouped["fraction_shared"].count().to_numpy()
    bin_summary["n_control_total"] = grouped["n_control"].sum().to_numpy()
    transition_summary = fish_transitions.groupby(keys + ["stimulus_bin"], sort=False)["fraction"].agg(
        fraction="mean", n_fish="count").reset_index()
    return dict(bin_edges_s=edges.copy(), side_selections=selections,
        neuron_table=pd.DataFrame(neurons, columns=["side", "fish_id", "neuron_id",
            "retained_row", "control", "stimulus", "control_onset_s", "stimulus_onset_s",
            "control_bin", "stimulus_bin", "stimulus_active"]), audit_table=pd.DataFrame(audits),
        fish_bins=fish_bins, fish_transitions=fish_transitions,
        bin_summary=bin_summary, transition_summary=transition_summary,
        settings=dict(fps_2p=fps_2p, t_pre_s=t_pre_s, motion_onset_s=motion_onset_s,
            persistence_frames=persistence_frames,
            min_significant_trial_fraction=min_significant_trial_fraction,
            active_fraction_threshold=active_fraction_threshold, min_epoch_s=min_epoch_s,
            min_active_reps=min_active_reps, expected_reps=expected_reps, tau_s=tau_s))


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


def compute_active_neuron_containment(
    active_matrix, stim_order, labels=None, empty_source_value=np.nan,
):
    """Return directional containment: row A, column B is |A & B| / |A|.

    Args:
        active_matrix (DataFrame or array): Neuron-by-stimulus decisions. Arrays
            follow the stimulus-index convention of the existing overlap helper.
            Finite nonzero values mean active; missing decisions raise.
        stim_order (sequence): Stimulus keys in requested row/column order.
        labels (sequence or None): Display labels paired with the keys.
        empty_source_value (float): Value for an empty source population.

    Returns:
        DataFrame: Fractions in [0, 1], with undefined source rows NaN by default.
            The input and neuron order are unchanged.
    """
    if not np.isfinite(np.asarray(active_matrix, dtype=float)).all():
        raise ValueError("Containment requires finite active decisions.")
    labels = list(stim_order) if labels is None else list(labels)
    active = _active_columns_as_bool(active_matrix, list(stim_order), labels)
    values = active.to_numpy(dtype=np.int64)
    intersections = values.T @ values
    counts = values.sum(axis=0)
    fractions = np.full(intersections.shape, empty_source_value, dtype=float)
    np.divide(intersections, counts[:, None], out=fractions, where=counts[:, None] > 0)
    return pd.DataFrame(fractions, index=labels, columns=labels)


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

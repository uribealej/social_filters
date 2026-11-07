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
from src.active_neuron_analysis import (
    build_active_neuron_matrices_all_fish, compute_active_neuron_containment,
    compute_active_neuron_jaccard_overlap,
)
from src.several_fish_selection import build_active_group_masks
from src.stimulus_similarity import build_stimulus_vector_similarity
from src.stimulus_specificity import build_active_membership_table, build_family_responsive_membership_fractions
from src.neuron_ordering import build_response_profile_sort_order


def _phase_sorting_profiles(fish, stimulus_keys, alignment, window_s):
    """Return neuron-by-time reference profiles averaged over all saved trials.

    Optional window bounds are seconds on the nominal display axis, [start,
    stop). This selects display samples without changing native-time metrics.
    Conditions are concatenated in requested order; no responses are rescaled.
    """
    from src.multifish_matrices import combine_reps_one_stim

    reference = fish["aligned"][alignment]["zscore"]
    ids = fish["stimuli_ids"] if stimulus_keys is None else [
        fish["stimuli_id_map"][key] for key in stimulus_keys]
    parts = []
    for sid in ids:
        values = combine_reps_one_stim(reference["trial_aligned_traces"][sid], "mean")
        if window_s is not None:
            times = reference["time_s"][sid]
            mask = (times >= window_s[0]) & (times < window_s[1])
            if not mask.any():
                raise ValueError(f"Sorting window has no display samples for stimulus {sid}.")
            values = values[:, mask]
        parts.append(values)
    return np.concatenate(parts, axis=1)


def build_phase_population_display(cohort, alignment="bout", sort_mode="corravg",
                                   sort_scope="within_fish", repetitions=None,
                                   sort_stimulus_keys=None, sort_alignment="sequence",
                                   sort_window_s=None):
    """Adapt reviewed phase bundles to existing pooled-matrix/mean helpers.

    Both trace types already contain retained neurons: positional indices are
    local here, while row_metadata retains original fish/neuron identities.
    sort_scope is 'within_fish' or 'pooled' (neurons may mix across fish).
    repetitions is None (all) or unique one-based repetition numbers in display
    order. Selection affects matrices (neurons, time), fish means (fish, time)
    and reference traces (neurons, time, selected trials), without mutating
    the cohort. Missing/out-of-range repetitions raise ValueError.
    Sorting uses all z-score repetitions, independent of display repetitions.
    sort_stimulus_keys optionally selects nonempty unique condition keys;
    sort_alignment and sort_window_s select the reference axis and optional
    nominal [start, stop) seconds. Defaults use every full sequence. A fixed
    reference keeps row order identical across displayed subsets/alignments.
    """
    from src.multifish_matrices import build_matrix_all_fish
    from src.neuron_ordering import compute_single_sort_order

    fish_ids = cohort["fish_ids"]
    if not fish_ids:
        raise ValueError("No reviewed fish are ready. Inspect the readiness table.")
    if sort_scope not in {"within_fish", "pooled"}:
        raise ValueError("sort_scope must be 'within_fish' or 'pooled'.")
    if sort_alignment not in {"sequence", "bout"}:
        raise ValueError("sort_alignment must be 'sequence' or 'bout'.")
    if sort_stimulus_keys is not None:
        sort_stimulus_keys = list(sort_stimulus_keys)
        if not sort_stimulus_keys or len(set(sort_stimulus_keys)) != len(sort_stimulus_keys):
            raise ValueError("sort_stimulus_keys must be nonempty and unique.")
        for fid in fish_ids:
            missing = set(sort_stimulus_keys) - set(cohort["all_fish_data"][fid]["stimuli_id_map"])
            if missing:
                raise ValueError(f"Unknown sorting conditions for {fid}: {sorted(missing)}.")
    if sort_window_s is not None:
        bounds = np.asarray(sort_window_s, dtype=float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[1] <= bounds[0]:
            raise ValueError("sort_window_s must contain finite increasing [start, stop) seconds.")
        sort_window_s = tuple(bounds)
    if repetitions is not None:
        repetitions = list(repetitions)
        if (not repetitions or any(isinstance(r, (bool, np.bool_)) or
                not isinstance(r, (int, np.integer)) or r < 1 for r in repetitions)
                or len(set(repetitions)) != len(repetitions)):
            raise ValueError("repetitions must contain unique positive one-based integers.")
    bundles, metadata, order, sorting_profiles = {}, [], [], []
    repetition_numbers = {}

    def selected_traces(traces):
        selected = {}
        for sid, values in traces.items():
            numbers = list(range(1, values.shape[2] + 1)) if repetitions is None else repetitions
            if max(numbers) > values.shape[2]:
                raise ValueError(f"Requested repetitions {numbers} unavailable for stimulus {sid}.")
            if sid in repetition_numbers and repetition_numbers[sid] != numbers:
                raise ValueError("Included fish must have matching repetition counts.")
            repetition_numbers[sid] = numbers
            selected[sid] = values[:, :, np.asarray(numbers) - 1]
        return selected

    def sorted_rows(profiles):
        usable = np.isfinite(profiles).all(axis=1) & (np.nanstd(profiles, axis=1) > 0)
        result = (compute_single_sort_order(profiles.T, sort_mode)
                  if usable.sum() >= 2 else None)
        return np.arange(len(profiles)) if result is None else result

    offset = 0
    for fid in fish_ids:
        fish = cohort["all_fish_data"][fid]
        aligned = fish["aligned"][alignment]
        ids = fish["kept_neuron_indices"]
        bundles[fid] = dict(trial_aligned_traces_z_core=selected_traces(aligned["zscore"]["trial_aligned_traces"]),
            trial_aligned_traces_raster=selected_traces(aligned["raster"]["trial_aligned_traces"]),
            kept_neuron_indices=np.arange(len(ids)))
        joined = _phase_sorting_profiles(fish, sort_stimulus_keys, sort_alignment, sort_window_s)
        sorting_profiles.append(joined)
        local_order = sorted_rows(joined) if sort_scope == "within_fish" else np.arange(len(ids))
        order.extend((offset + local_order).tolist())
        metadata.append(pd.DataFrame(dict(fish_id=fid, neuron_id=ids)))
        offset += len(ids)
    if sort_scope == "pooled":
        order = sorted_rows(np.concatenate(sorting_profiles, axis=0)).tolist()
    reference = cohort["all_fish_data"][fish_ids[0]]
    matrices = {}
    for kind in ("zscore", "raster"):
        for mode in ("mean", "concat"):
            matrices[kind, mode] = {sid: build_matrix_all_fish(bundles, [sid], fish_ids,
                combine_mode=mode, trace_type=kind) for sid in reference["stimuli_ids"]}
    means = build_pooled_mean_trace_by_stimulus(bundles, fish_ids, reference["stimuli_ids"], trace_type="zscore")
    row_metadata = pd.concat(metadata, ignore_index=True)
    aligned_reference = dict(reference["aligned"][alignment]["zscore"])
    aligned_reference["trial_aligned_traces"] = bundles[fish_ids[0]]["trial_aligned_traces_z_core"]
    for key in ("actual_time_s", "onsets_by_id", "alignment_error_s"):
        aligned_reference[key] = {sid: values[..., np.asarray(repetition_numbers[sid]) - 1]
                                  for sid, values in aligned_reference[key].items()}
    trials = aligned_reference["trials"]
    keep_trials = np.zeros(len(trials), dtype=bool)
    for sid, numbers in repetition_numbers.items():
        positions = np.flatnonzero(trials.stimulus_id.to_numpy() == sid)
        keep_trials[positions[np.asarray(numbers) - 1]] = True
    aligned_reference["trials"] = trials.iloc[np.flatnonzero(keep_trials)].copy()
    return dict(matrices=matrices, fish_means=means, row_metadata=row_metadata,
                displayed_row_metadata=row_metadata.iloc[order].reset_index(drop=True),
                neuron_order=np.asarray(order), fish_ids=fish_ids, alignment=alignment,
                reference=aligned_reference, sort_scope=sort_scope, sort_mode=sort_mode,
                sort_reference=dict(stimulus_keys=sort_stimulus_keys,
                    alignment=sort_alignment, window_s=sort_window_s, repetitions="all"),
                repetition_numbers=repetition_numbers,
                stimulus_keys=list(reference["stimuli_id_map"]), stimulus_ids=reference["stimuli_ids"])


def build_phase_stimulus_panel_displays(cohort, side_conditions, alignment="bout",
        repetitions=None, sort_mode="corravg", sort_stimulus_keys=None,
        sort_alignment="bout", sort_window_s=None, recruitment=None,
        left_right_controls=None):
    """Prepare side-separated pooled phase panels for both saved trace types.

    side_conditions maps side labels to logged condition names in display order.
    Input arrays are retained neurons by time by trials. Selected one-based
    repetitions affect display means only; ordinary sorting uses all z-score
    trials in the explicit reference. cell06 reuses each side's exact cached
    phase-recruitment order after checking original fish/neuron identities.
    left_right_index orders the signed mean z-score preference of two explicit
    controls within the reference window, with undefined values last.

    Return trace-type -> side -> stimulus-panel display dictionaries. Matrices
    are (pooled neurons, time), fish_means (fish, time), times seconds from the
    selected anchor. Significant values are selected-trial active fractions;
    no new normalization, thresholding or neuron selection is performed.
    All rows, original IDs, selected repetition numbers and phase trials remain
    available. No plotting, printing or saving.
    """
    if not side_conditions or any(not names or len(set(names)) != len(names)
                                  for names in side_conditions.values()):
        raise ValueError("side_conditions must contain nonempty unique conditions.")
    special = sort_mode in {"cell06", "left_right_index"}
    base = build_phase_population_display(cohort, alignment,
        sort_mode="unsorted" if special else sort_mode, sort_scope="pooled",
        repetitions=repetitions, sort_stimulus_keys=sort_stimulus_keys,
        sort_alignment=sort_alignment, sort_window_s=sort_window_s)
    rows = base["row_metadata"]
    trials = base["reference"]["trials"]
    lookup = trials.drop_duplicates("logged_name").set_index("logged_name")
    order = base["neuron_order"]
    if sort_mode == "left_right_index":
        if left_right_controls is None or len(left_right_controls) != 2:
            raise ValueError("left_right_controls must name the left and right reference controls.")
        profiles = {key: np.concatenate([_phase_sorting_profiles(
            cohort["all_fish_data"][fid], [key], sort_alignment, sort_window_s)
            for fid in cohort["fish_ids"]]) for key in left_right_controls}
        response = pd.DataFrame({key: np.nanmean(values, axis=1) for key, values in profiles.items()})
        index = compute_response_pair_index(response, *left_right_controls)
        order = np.argsort(np.where(np.isfinite(index), index, np.inf), kind="stable")
    offsets = np.cumsum([0] + [len(cohort["all_fish_data"][fid]["kept_neuron_indices"])
                              for fid in cohort["fish_ids"]])
    result = {kind: {} for kind in ("zscore", "raster")}
    for side, conditions in side_conditions.items():
        if not set(conditions) <= set(lookup.index):
            raise ValueError(f"{side}: unknown logged conditions in side_conditions.")
        ids = [int(lookup.loc[name, "stimulus_id"]) for name in conditions]
        side_order = order
        if sort_mode == "cell06":
            if recruitment is None or side not in recruitment["displays"]:
                raise ValueError("cell06 requires the prepared phase recruitment results for each side.")
            source = recruitment["displays"][side]
            if not np.array_equal(source["row_metadata"][["fish_id", "neuron_id"]].to_numpy(),
                                  rows[["fish_id", "neuron_id"]].to_numpy()):
                raise ValueError(f"{side}: Cell 06 fish/neuron identities differ; rerun preparation.")
            side_order = np.asarray(source["neuron_order"])
        if (side_order.shape != (len(rows),) or not np.issubdtype(side_order.dtype, np.integer)
                or not np.array_equal(np.sort(side_order), np.arange(len(rows)))):
            raise ValueError(f"{side}: order must contain every pooled neuron exactly once.")
        for kind in result:
            # Keep the full comparison set for common left/right trace limits;
            # stimulus_ids alone selects this side's plotted columns.
            matrices = base["matrices"][kind, "mean"]
            means = {sid: np.stack([np.nanmean(values[offsets[i]:offsets[i+1]], axis=0)
                for i in range(len(cohort["fish_ids"]))]) for sid, values in matrices.items()}
            result[kind][side] = dict(stimulus_ids=ids,
                stimulus_labels=dict(zip(ids, conditions)), matrices=matrices,
                fish_means=means, fish_ids=cohort["fish_ids"], trace_type=kind,
                time_s={sid: base["reference"]["time_s"][sid] for sid in ids},
                fps_2p=base["reference"]["fps_2p"], neuron_order=side_order,
                row_metadata=rows, displayed_row_metadata=rows.iloc[side_order].reset_index(drop=True),
                phase_trials={sid: trials.loc[trials.stimulus_id.eq(sid)].iloc[0] for sid in ids},
                alignment=alignment, sort_mode=sort_mode,
                repetition_numbers={sid: base["repetition_numbers"][sid] for sid in ids})
    return result


def build_phase_sequence_panel_displays(cohort, panel_displays, t_pre_s=5.0,
                                        recovery_s=15.0):
    """Rewindow cached phase panels to include conditioning, delay and recovery.

    ``panel_displays`` is the trace-type -> side mapping returned by
    ``build_phase_stimulus_panel_displays`` (both sides/types). Re-extract saved
    continuous traces using sequence onset, including ``t_pre_s`` seconds before
    and ``recovery_s`` seconds after bout offset. Keep cached neuron orders,
    identities, condition order and one-based repetition selections exactly.
    No sorting, normalization, significance or activity decisions are repeated.

    Return new panel dictionaries with matrices (pooled neurons, time),
    fish_means (fish, time), and nominal time_s in seconds from sequence onset.
    time_bounds_s records exact requested (start, stop) seconds for plotting,
    independently of the imaging sampling grid.
    Input dictionaries/arrays are not modified. Incompatible identities,
    repetitions or time axes raise ValueError; extraction crossing an imaging
    block fails rather than padding/truncating the requested recovery.
    """
    repetitions = _phase_sequence_cached_repetitions(cohort, panel_displays)
    result = {}
    for kind, sides in panel_displays.items():
        traces, times = _phase_sequence_trial_traces(cohort, kind, repetitions, t_pre_s, recovery_s)
        matrices, means = {}, {}
        for sid, numbers in repetitions.items():
            fish_values = [values[:, :, np.asarray(numbers) - 1] for values in traces[sid]]
            fish_values = [np.nanmean(values, axis=2) for values in fish_values]
            matrices[sid] = np.concatenate(fish_values, axis=0)
            means[sid] = np.stack([np.nanmean(values, axis=0) for values in fish_values])
        result[kind] = {side: _phase_sequence_panel(source, matrices, means, times,
                                                    t_pre_s, recovery_s)
                        for side, source in sides.items()}
    return result


def _phase_sequence_cached_repetitions(cohort, panel_displays):
    """Validate cached panel identities and return {stimulus_id: repetitions}.

    Every side/type display must keep the cohort's fish order and retained
    (fish_id, neuron_id) rows; one-based repetition lists must agree.
    """
    fish_ids = cohort["fish_ids"]
    rows = pd.concat([pd.DataFrame(dict(fish_id=fid,
        neuron_id=cohort["all_fish_data"][fid]["kept_neuron_indices"]))
        for fid in fish_ids], ignore_index=True)
    repetitions = {}
    for sides in panel_displays.values():
        for source in sides.values():
            if (source["fish_ids"] != fish_ids or not np.array_equal(
                    source["row_metadata"][["fish_id", "neuron_id"]].to_numpy(),
                    rows[["fish_id", "neuron_id"]].to_numpy())):
                raise ValueError("Panel fish/neuron identities differ; rerun Cell 04a.")
            for sid, numbers in source["repetition_numbers"].items():
                if sid in repetitions and list(numbers) != repetitions[sid]:
                    raise ValueError("Panel repetition selections must agree across sides/types.")
                repetitions[sid] = list(numbers)
    return repetitions


def _phase_sequence_trial_traces(cohort, kind, repetitions, t_pre_s, recovery_s):
    """Re-extract sequence-aligned single trials from saved continuous traces.

    kind is 'zscore' (retained columns of z_traces) or 'raster' (saved
    significant traces). repetitions maps stimulus IDs to one-based
    repetition numbers that every fish must provide. Return ({sid: [per-fish
    (neurons, time, trials)]}, {sid: time_s}) with time in seconds from
    sequence onset. Unavailable repetitions or mismatched time axes raise
    ValueError; extraction across an imaging block raises in the aligner.
    """
    from src.trial_alignment import build_event_aligned_traces

    traces, times = {sid: [] for sid in repetitions}, {}
    for fid in cohort["fish_ids"]:
        fish = cohort["all_fish_data"][fid]
        values = (fish["z_traces"][:, fish["kept_neuron_indices"]]
                  if kind == "zscore" else fish["raster"])
        aligned = build_event_aligned_traces(values, fish["trials"], fish["fps_2p"],
            alignment="sequence", t_pre_s=t_pre_s, recovery_s=recovery_s)
        for sid, numbers in repetitions.items():
            trial_values = aligned["trial_aligned_traces"][sid]
            if not numbers or min(numbers) < 1 or max(numbers) > trial_values.shape[2]:
                raise ValueError(f"{fid}: requested repetitions unavailable for stimulus {sid}.")
            if sid in times and not np.array_equal(times[sid], aligned["time_s"][sid]):
                raise ValueError("Fish have incompatible sequence display time axes.")
            times[sid] = aligned["time_s"][sid]
            traces[sid].append(trial_values)
    return traces, times


def _phase_sequence_panel(source, matrices, means, times, t_pre_s, recovery_s):
    """Return a copy of one cached side panel with sequence-window arrays."""
    return dict(source, matrices=matrices, fish_means=means,
        time_s={sid: times[sid] for sid in source["stimulus_ids"]}, alignment="sequence",
        time_bounds_s={sid: (-float(t_pre_s), float(source["phase_trials"][sid].end_s
            - source["phase_trials"][sid].onset_s + recovery_s))
            for sid in source["stimulus_ids"]})


def build_phase_sequence_repetition_displays(cohort, panel_displays, repetitions,
                                             t_pre_s=5.0, recovery_s=15.0):
    """Prepare one single-trial full-sequence panel display per repetition.

    Like ``build_phase_sequence_panel_displays`` (same window: ``t_pre_s``
    seconds before sequence onset to ``recovery_s`` seconds after bout
    offset, same cached neuron orders, identities, conditions and phase
    timing) but without averaging repetitions. ``panel_displays`` is the
    cached trace-type -> side mapping from ``build_phase_stimulus_panel_displays``;
    its own repetition selection is ignored. ``repetitions`` lists unique
    one-based repetition numbers, each required for every fish and stimulus.

    Return trace-type -> side -> repetition -> display. Each display's
    matrices are one trial's (pooled neurons, time) values (z-scores, or the
    saved binary 0/1 significant traces), fish_means (fish, time) average
    each fish's retained neurons for that trial, repetition_numbers maps every
    stimulus to [repetition] and ``repetition`` records the number. Missing
    samples stay NaN. No sorting, normalization, detection or measurement is
    repeated, and inputs are not modified.
    """
    repetitions = list(repetitions)
    if (not repetitions or len(set(repetitions)) != len(repetitions)
            or any(isinstance(r, (bool, np.bool_)) or not isinstance(r, (int, np.integer))
                   or r < 1 for r in repetitions)):
        raise ValueError("repetitions must contain unique positive one-based integers.")
    repetitions = [int(r) for r in repetitions]
    requested = {sid: repetitions for sid in _phase_sequence_cached_repetitions(cohort, panel_displays)}
    result = {}
    for kind, sides in panel_displays.items():
        traces, times = _phase_sequence_trial_traces(cohort, kind, requested, t_pre_s, recovery_s)
        result[kind] = {side: {} for side in sides}
        for repetition in repetitions:
            matrices = {sid: np.concatenate([values[:, :, repetition - 1] for values in fish_values])
                        for sid, fish_values in traces.items()}
            means = {sid: np.stack([np.nanmean(values[:, :, repetition - 1], axis=0)
                                    for values in fish_values])
                     for sid, fish_values in traces.items()}
            for side, source in sides.items():
                panel = _phase_sequence_panel(source, matrices, means, times, t_pre_s, recovery_s)
                panel["repetition_numbers"] = {sid: [repetition] for sid in source["stimulus_ids"]}
                panel["repetition"] = repetition
                result[kind][side][repetition] = panel
    return result


def _phase_trial_measurements(fid, fish, settings):
    """Measure every trial/window on native sequence-aligned samples.

    Windows: each configured bout window (seconds from bout onset) and the
    explicit preconditioning phase. A requested window that extends beyond the
    cached aligned samples raises ValueError rather than being truncated.
    """
    from src.trial_alignment import select_event_window
    from src.response_metrics import compute_explicit_window_metrics
    from src.response_classification import classify_trial_raster_activity

    z = fish["aligned"]["sequence"]["zscore"]
    raster = fish["aligned"]["sequence"]["raster"]
    frames, rows = [], []
    for sid, group in fish["trials"].groupby("stimulus_id", sort=False):
        for rep, trial in enumerate(group.itertuples(index=False)):
            times = z["actual_time_s"][sid][:, rep]
            values = z["trial_aligned_traces"][sid][:, :, rep:rep + 1]
            binary = raster["trial_aligned_traces"][sid][:, :, rep:rep + 1]
            bout = trial.bout_onset_s - trial.onset_s
            pre_end = trial.pre_end_s - trial.onset_s
            windows = {label: (bout + bounds[0], bout + bounds[1]) for label, bounds in settings["bout_windows"].items()}
            windows.update(preconditioning=(0, pre_end))
            for label, (start, stop) in windows.items():
                try:
                    window = select_event_window(times, start, stop, sample_period_s=1 / fish["fps_2p"])
                except ValueError as error:
                    raise ValueError(f"{fid}/{trial.logged_name} repetition {rep + 1}, "
                                     f"window '{label}': {error}") from error
                indices = window.pop("frame_indices")
                metrics = compute_explicit_window_metrics(values, indices, times, fish["fps_2p"])
                active = classify_trial_raster_activity(binary[:, indices, :], fish["fps_2p"],
                    settings["active_fraction_threshold"], settings["min_epoch_s"])
                frame = pd.DataFrame({key: metrics[key][:, 0] for key in ("auc_z_s", "peak_z", "mean_z")})
                frame["neuron_id"] = fish["kept_neuron_indices"]
                frame["active_trial"] = active[:, 0]
                frame["measurement_valid"] = np.isfinite(metrics["auc_z_s"][:, 0])
                frame["fish_id"], frame["condition"], frame["stimulus_key"] = fid, trial.logged_name, trial.stimulus_key
                frame["side"], frame["window"], frame["block"] = trial.hemifield, label, trial.block
                frame["repetition"], frame["trial_index"] = rep, trial.trial_index
                frame["applicable"] = stop > start
                frame["delay_s"] = trial.bout_onset_s - trial.pre_end_s
                frames.append(frame)
                rows.append(dict(fish_id=fid, condition=trial.logged_name, side=trial.hemifield,
                    window=label, block=trial.block, repetition=rep, applicable=stop > start, **window))
    return pd.concat(frames, ignore_index=True), pd.DataFrame(rows)


def build_phase_population_analysis(cohort, settings):
    """Build trial/neuron/fish descriptive tables from reviewed phase results.

    settings: bout_windows (label -> (start, stop) seconds from bout onset,
    e.g. {'bout_10s': (0, 10)}), active_fraction_threshold, min_epoch_s,
    min_active_reps and expected_reps. Each bout window and the explicit
    preconditioning phase are measured per trial on native sequence-aligned
    samples. A window extending beyond the cohort's aligned samples raises
    ValueError (use several_fish_loading.align_phase_cohort with a longer
    recovery_s); measurements are never silently truncated.

    Reuses explicit-window AUC/peak owners and the canonical activity rule.
    Every selected neuron remains in every condition's denominator. An active
    fraction with incomplete activity classifications is missing, with counts
    retained. Returns trial_metrics (one row per fish/condition/window/
    repetition/neuron, zero-based ``repetition``), neuron_metrics, fish_summary
    and window_coverage tables.
    """
    if not cohort["fish_ids"]:
        raise ValueError("No reviewed fish are ready. Inspect the readiness table.")
    if not 1 <= settings["min_active_reps"] <= settings["expected_reps"]:
        raise ValueError("Configure valid expected/active repetition counts.")
    trial_tables, window_tables = [], []
    for fid in cohort["fish_ids"]:
        fish = cohort["all_fish_data"][fid]
        if not fish["trials"].groupby("stimulus_id").size().eq(settings["expected_reps"]).all():
            raise ValueError(f"{fid}: incomplete trial count.")
        trial, window = _phase_trial_measurements(fid, fish, settings)
        trial_tables.append(trial)
        window_tables.append(window)
    trial = pd.concat(trial_tables, ignore_index=True)
    keys = ["fish_id", "side", "condition", "stimulus_key", "window", "neuron_id"]
    measurements = ["auc_z_s", "peak_z", "mean_z"]
    grouped = trial.groupby(keys, sort=False, dropna=False)
    neuron = grouped[measurements].mean().reset_index()
    counts = grouped.agg(valid_measurement_trials=("measurement_valid", "sum"),
        valid_activity_trials=("active_trial", "count"), active_trials=("active_trial", lambda x: x.sum(min_count=1)),
        applicable=("applicable", "first"), delay_s=("delay_s", "first")).reset_index()
    neuron = neuron.merge(counts, on=keys, validate="one_to_one")
    neuron["active"] = np.where(neuron.valid_activity_trials.eq(settings["expected_reps"]),
        (neuron.active_trials >= settings["min_active_reps"]).astype(float), np.nan)
    fish_keys = ["fish_id", "side", "condition", "window"]
    fish_summary = neuron.groupby(fish_keys, sort=False)[measurements].mean().reset_index()
    activity = neuron.groupby(fish_keys, sort=False).agg(n_neurons=("neuron_id", "size"),
        n_valid_active=("active", "count"), n_valid_auc=("auc_z_s", "count"),
        active_fraction=("active", lambda x: x.mean() if x.notna().all() else np.nan),
        mean_active_trials=("active_trials", "mean"), delay_s=("delay_s", "first"),
        applicable=("applicable", "first")).reset_index()
    fish_summary = fish_summary.merge(activity, on=fish_keys, validate="one_to_one")
    return dict(trial_metrics=trial, neuron_metrics=neuron, fish_summary=fish_summary,
        window_coverage=pd.concat(window_tables, ignore_index=True))


def _validate_phase_repetitions(repetitions, min_active_reps):
    """Return unique positive one-based repetitions; validate the passing count."""
    repetitions = list(repetitions)
    if (not repetitions or any(isinstance(r, (bool, np.bool_)) or not isinstance(r, (int, np.integer))
            or r < 1 for r in repetitions) or len(set(repetitions)) != len(repetitions)):
        raise ValueError("Use unique positive one-based repetition numbers.")
    if not 1 <= min_active_reps <= len(repetitions):
        raise ValueError("Invalid minimum active repetitions.")
    return repetitions


def _validate_phase_bounds(bounds):
    """Return finite increasing (start, stop) seconds as a tuple."""
    values = np.asarray(bounds, dtype=float)
    if values.shape != (2,) or not np.isfinite(values).all() or values[1] <= values[0]:
        raise ValueError("bout_bounds must be finite increasing seconds.")
    return tuple(float(v) for v in values)


def _phase_recruitment_decisions(trial_metrics, conditions, repetitions, min_active_reps, bout_window):
    """Aggregate saved native-window decisions over selected one-based trials."""
    keys = ["fish_id", "neuron_id", "condition", "window"]
    if not trial_metrics.window.eq(bout_window).any():
        raise ValueError(f"trial_metrics has no '{bout_window}' window; add it to the measurement settings.")
    selected = trial_metrics.loc[trial_metrics.condition.isin(conditions)
        & trial_metrics.window.isin(["preconditioning", bout_window])
        & (trial_metrics.repetition + 1).isin(repetitions)]
    if selected.duplicated(keys + ["repetition"]).any():
        raise ValueError("Duplicate neuron/condition/window/repetition decisions.")
    result = selected.groupby(keys, sort=False).active_trial.agg(["count", "sum"])
    result["active"] = np.where(result["count"].eq(len(repetitions)),
        (result["sum"] >= min_active_reps).astype(float), np.nan)
    return result.active


def _phase_recruitment_neurons(cohort, decisions, side_conditions, bout_window):
    """Build identity-preserving per-condition memberships; missing stays missing."""
    rows = []
    for fid in cohort["fish_ids"]:
        ids = cohort["all_fish_data"][fid]["kept_neuron_indices"]
        for side, conditions in side_conditions.items():
            def activity(condition, window):
                index = pd.MultiIndex.from_product([[fid], ids, [condition], [window]],
                    names=decisions.index.names)
                return decisions.reindex(index).to_numpy()

            control = activity(conditions[0], bout_window)
            for condition in conditions:
                bout = activity(condition, bout_window)
                pre = activity(condition, "preconditioning")
                shared_known = np.isfinite(control) & np.isfinite(bout)
                new_known = np.isfinite(pre) & np.isfinite(bout)
                rows.append(pd.DataFrame(dict(fish_id=fid, neuron_id=ids, side=side,
                    condition=condition, c1_active=control, pre_active=pre, bout_active=bout,
                    shared_c1=np.where(shared_known, (control == 1) & (bout == 1), np.nan),
                    new_after_pre=np.where(new_known, (pre == 0) & (bout == 1), np.nan),
                    new_relative_c1=np.where(shared_known, (control == 0) & (bout == 1), np.nan))))
    return pd.concat(rows, ignore_index=True)


def _phase_preconditioning_pool(cohort, trial_metrics, side_conditions, repetitions, min_fraction):
    """Select consistent preconditioning neurons from all conditioned trials.

    Count native-window per-trial decisions, not per-condition majorities.
    All expected decisions must be present; missing activity is not inactivity.
    """
    parts = []
    for side, conditions in side_conditions.items():
        expected = (len(conditions) - 1) * len(repetitions)
        minimum = int(np.ceil(min_fraction * expected))
        selected = trial_metrics.loc[trial_metrics.condition.isin(conditions[1:])
            & trial_metrics.window.eq("preconditioning")
            & (trial_metrics.repetition + 1).isin(repetitions)]
        counts = selected.groupby(["fish_id", "neuron_id"]).active_trial.agg(["count", "sum"])
        for fid in cohort["fish_ids"]:
            ids = cohort["all_fish_data"][fid]["kept_neuron_indices"]
            index = pd.MultiIndex.from_product([[fid], ids], names=["fish_id", "neuron_id"])
            values = counts.reindex(index)
            valid = values["count"].fillna(0).to_numpy(dtype=int)
            passing = values["sum"].fillna(0).to_numpy(dtype=int)
            parts.append(pd.DataFrame(dict(fish_id=fid, neuron_id=ids, side=side,
                n_expected_trials=expected, n_valid_trials=valid, n_passing_trials=passing,
                min_passing_trials=minimum, active_fraction=np.where(valid == expected, passing / expected, np.nan),
                active=np.where(valid == expected, passing >= minimum, np.nan))))
    return pd.concat(parts, ignore_index=True)


def _phase_condition_stimulus(fish, fid, condition, repetitions):
    """Return the condition's trial rows and stimulus ID, requiring the repetitions."""
    trials = fish["trials"].loc[fish["trials"].logged_name.eq(condition)]
    if len(trials) < max(repetitions):
        raise ValueError(f"{fid}/{condition}: selected repetitions unavailable.")
    return trials, int(trials.stimulus_id.iloc[0])


def _phase_bout_raster_mean(fish, sid, repetitions, bout_bounds):
    """Mean significant activity over selected one-based repetitions in a bout window.

    Returns (kept neurons, display samples) trial-active fractions and the
    nominal bout-relative axis in seconds. A window beyond the cached aligned
    samples raises ValueError rather than being truncated.
    """
    from src.trial_alignment import select_event_window

    aligned = fish["aligned"]["bout"]["raster"]
    times = aligned["time_s"][sid]
    selected = select_event_window(times, *bout_bounds, sample_period_s=1 / fish["fps_2p"])["frame_indices"]
    if not len(selected):
        raise ValueError("Bout display window has no samples.")
    values = aligned["trial_aligned_traces"][sid][:, selected, :][:, :, np.asarray(repetitions) - 1]
    return np.mean(values, axis=2), times[selected]


def _phase_pre_raster_mean(fish, trials, sid, repetitions):
    """Mean significant activity over the explicit preconditioning phase.

    Returns (kept neurons, samples) trial-active fractions over selected
    one-based repetitions and the sequence-relative axis [0, duration) s.
    """
    from src.trial_alignment import select_event_window

    durations = trials.pre_end_s.to_numpy() - trials.onset_s.to_numpy()
    if not np.allclose(durations, durations[0]) or durations[0] <= 0:
        raise ValueError("Preconditioning requires a consistent positive duration.")
    aligned = fish["aligned"]["sequence"]["raster"]
    times = aligned["time_s"][sid]
    selected = select_event_window(times, 0, durations[0], sample_period_s=1 / fish["fps_2p"])["frame_indices"]
    values = aligned["trial_aligned_traces"][sid][:, selected, :][:, :, np.asarray(repetitions) - 1]
    return np.mean(values, axis=2), times[selected]


def _phase_recruitment_display(cohort, neurons, pre_pool, side, conditions, repetitions, bout_bounds):
    """Prepare grayscale display means and membership groups, without interpolation."""
    pre_parts, bout_parts, metadata, codes, pre_codes = [], {c: [] for c in conditions}, [], {}, []
    pooled_codes, decisions = [], []
    axes = {}
    for fid in cohort["fish_ids"]:
        fish = cohort["all_fish_data"][fid]
        ids = fish["kept_neuron_indices"]
        lookup = neurons.loc[neurons.fish_id.eq(fid) & neurons.side.eq(side)]
        members = {c: lookup.loc[lookup.condition.eq(c)].set_index("neuron_id").reindex(ids)
                   for c in conditions}
        pre_conditions = []
        for condition in conditions:
            trials, sid = _phase_condition_stimulus(fish, fid, condition, repetitions)
            values, axis = _phase_bout_raster_mean(fish, sid, repetitions, bout_bounds)
            if condition in axes and not np.array_equal(axes[condition], axis):
                raise ValueError("Bout display axes differ between fish.")
            axes[condition] = axis
            bout_parts[condition].append(values)
            if condition != conditions[0]:
                values, axis = _phase_pre_raster_mean(fish, trials, sid, repetitions)
                if "preconditioning" in axes and not np.array_equal(axes["preconditioning"], axis):
                    raise ValueError("Cannot average different preconditioning display axes.")
                axes["preconditioning"] = axis
                pre_conditions.append(values)
        pre_parts.append(np.mean(pre_conditions, axis=0))
        bout_members = np.column_stack([members[c].bout_active for c in conditions])
        pre_members = np.column_stack([members[c].pre_active for c in conditions[1:]])
        pooled = pre_pool.loc[pre_pool.fish_id.eq(fid) & pre_pool.side.eq(side)].set_index("neuron_id").reindex(ids).active.to_numpy()
        pooled_codes.append(pooled)
        decisions.append(np.column_stack([pooled, bout_members]))
        known = np.isfinite(bout_members).all(axis=1) & np.isfinite(pooled)
        category = np.full(len(ids), len(conditions) + 1, dtype=int)
        for j in reversed(range(1, len(conditions))):
            category[bout_members[:, j] == 1] = j + 1
        category[bout_members[:, 0] == 1] = 1
        category[pooled == 1] = 0
        category[~known] = len(conditions) + 2
        metadata.append(pd.DataFrame(dict(fish_id=fid, neuron_id=ids, group=category)))
        pre_codes.append(np.where(known, np.any(pre_members == 1, axis=1), np.nan))
        for condition in conditions:
            shared, new = members[condition].shared_c1.to_numpy(), members[condition].new_after_pre.to_numpy()
            if condition == conditions[0]:
                code = np.where(np.isfinite(shared), shared, 4)
            else:
                code = np.where(np.isfinite(shared) & np.isfinite(new), shared + 2 * new, 4)
            codes.setdefault(condition, []).append(code)
    metadata = pd.concat(metadata, ignore_index=True)
    order = np.argsort(metadata.group.to_numpy(), kind="stable")
    matrices = {"preconditioning": np.concatenate(pre_parts)}
    matrices.update({c: np.concatenate(parts) for c, parts in bout_parts.items()})
    labels = ["Consistent preconditioning", "C1 bout active"] + [f"Additional in {c}" for c in conditions[1:]]
    labels += ["Other", "Unclassified"]
    return dict(side=side, conditions=conditions, matrices=matrices, time_s=axes,
        neuron_order=order, displayed_row_metadata=metadata.iloc[order].reset_index(drop=True),
        row_metadata=metadata, group_labels=labels, repetitions=list(repetitions),
        membership_codes={c: np.concatenate(parts) for c, parts in codes.items()},
        pre_active_any=np.concatenate(pre_codes), pre_active_pool=np.concatenate(pooled_codes),
        decision_matrix=np.concatenate(decisions), fps_2p=cohort["all_fish_data"][cohort["fish_ids"][0]]["fps_2p"],
        bout_bounds=bout_bounds)


def build_phase_recruitment_analysis(cohort, trial_metrics, side_conditions, bout_bounds,
                                     repetitions=(2, 3, 4), min_active_reps=2, bout_window="bout",
                                     pre_active_fraction=0.75):
    """Count significant-activity recruitment and prepare side-separated rasters.

    Reuse native-window ``active_trial`` decisions from phase population analysis;
    select one-based repetitions before applying the passing-trial count. Each
    side's first condition is its C1 control; remaining conditions define the
    preconditioning average (equal condition/repetition weights). Exclude unwanted
    conditions in side_conditions. The cohort and input tables are not modified.

    pre_active_fraction defines the display's consistent preconditioning pool:
    ceil(fraction * conditioned presentations) passing native trials, with all
    decisions known (0.75 across five conditions/three trials requires 12/15).
    This display selection does not change per-condition recruitment counts.
    Returns ``neurons``, ``fish_summary``, ``union_summary``, ``preconditioning_pool``
    tables and ``displays`` including binary neuron-by-phase decision matrices.
    Matrices are (pooled neurons, nominal display time); identity is fish/neuron.
    Counts/fractions with any missing required classification remain NaN and have
    explicit valid counts. C1 preconditioning/new-after-pre are not applicable.
    New-after-pre uses each condition's own preconditioning, not its pooled mean.
    fraction_c1_retained divides the C1/condition shared count by that fish's
    same-side C1-active count; stimulus-only neurons are excluded. An empty
    C1 population has an undefined (NaN) retained fraction.
    fraction_new_after_pre divides the bout-active, own-preconditioning-inactive
    count by all retained neurons in the fish, regardless of C1 membership.
    """
    repetitions = _validate_phase_repetitions(repetitions, min_active_reps)
    if not np.isfinite(pre_active_fraction) or not 0 < pre_active_fraction <= 1:
        raise ValueError("pre_active_fraction must be in (0, 1].")
    bounds = _validate_phase_bounds(bout_bounds)
    if not cohort["fish_ids"] or not side_conditions:
        raise ValueError("Recruitment needs a nonempty cohort and side conditions.")
    conditions = [c for names in side_conditions.values() for c in names]
    if any(len(names) < 2 for names in side_conditions.values()) or len(set(conditions)) != len(conditions):
        raise ValueError("Each side needs a control plus unique conditioned stimuli.")
    decisions = _phase_recruitment_decisions(trial_metrics, conditions, repetitions, min_active_reps, bout_window)
    neurons = _phase_recruitment_neurons(cohort, decisions, side_conditions, bout_window)
    pre_pool = _phase_preconditioning_pool(cohort, trial_metrics, side_conditions, repetitions, pre_active_fraction)
    summaries, unions = [], []
    flags = ["shared_c1", "new_after_pre", "new_relative_c1", "pre_active", "bout_active"]
    for (fid, side, condition), group in neurons.groupby(["fish_id", "side", "condition"], sort=False):
        record = dict(fish_id=fid, side=side, condition=condition, n_neurons=len(group))
        for flag in flags:
            record[f"n_valid_{flag}"] = int(group[flag].notna().sum())
            record[f"n_{flag}"] = float(group[flag].sum()) if group[flag].notna().all() else np.nan
            record[f"fraction_{flag}"] = record[f"n_{flag}"] / len(group)
        n_control = group.c1_active.sum() if group.c1_active.notna().all() else np.nan
        record["fraction_c1_retained"] = record["n_shared_c1"] / n_control if n_control > 0 else np.nan
        summaries.append(record)
    for (fid, side), group in neurons.groupby(["fish_id", "side"], sort=False):
        conditioned = group.loc[~group.condition.eq(side_conditions[side][0])]
        by_neuron = conditioned.groupby("neuron_id").new_after_pre
        known = by_neuron.count().eq(len(side_conditions[side]) - 1)
        unions.append(dict(fish_id=fid, side=side, n_neurons=len(known), n_valid=int(known.sum()),
            n_any_new_after_pre=float(by_neuron.max().sum()) if known.all() else np.nan))
    displays = {side: _phase_recruitment_display(cohort, neurons, pre_pool, side, list(names), repetitions, tuple(bounds))
                for side, names in side_conditions.items()}
    return dict(neurons=neurons, fish_summary=pd.DataFrame(summaries),
                union_summary=pd.DataFrame(unions), preconditioning_pool=pre_pool,
                pre_active_fraction=pre_active_fraction, min_active_reps=min_active_reps, displays=displays)


def _phase_decision_lookup(decisions, fid, ids, condition, window):
    """Return per-neuron 1/0/NaN decisions in kept-neuron order (missing stays NaN)."""
    index = pd.MultiIndex.from_product([[fid], ids, [condition], [window]], names=decisions.index.names)
    return decisions.reindex(index).to_numpy(dtype=float)


def _control_comparison_summary(fid, side, ids, reference, test, labels, ref_bout, test_pre, test_bout):
    """Per-fish counts/fractions of test-bout-active neurons and their overlaps."""
    test_known = bool(np.isfinite(test_bout).all())
    active = test_bout == 1
    n_test = float(active.sum()) if test_known else np.nan
    rows = []
    for category, condition, other in [("test_bout", f"{labels[1]} bout", None),
                                       ("test_pre", f"{labels[1]} pre", test_pre),
                                       ("reference_bout", f"{labels[0]} bout", ref_bout)]:
        if other is None:
            count = n_test
        else:
            known = test_known and bool(np.isfinite(other[active]).all())
            count = float((active & (other == 1)).sum()) if known else np.nan
        rows.append(dict(fish_id=fid, side=side, condition=condition, category=category,
            reference=reference, test=test, n_neurons=len(ids), n_valid_test=int(np.isfinite(test_bout).sum()),
            n_test_active=n_test, n_count=count,
            fraction_of_test=count / n_test if n_test > 0 else np.nan))
    return rows


def build_phase_control_comparison(cohort, trial_metrics, side_controls, bout_window, bout_bounds,
                                   repetitions=(2, 3, 4), min_active_reps=2,
                                   include_reference_only=False, labels=None):
    """Compare a test control's bout neurons with its own preconditioning and a reference bout.

    Designed for C2 (distant preconditioning) versus C1 (bout only), per side.
    Reuse native-window ``active_trial`` decisions in ``trial_metrics`` (from
    ``build_phase_population_analysis``): a phase is active when at least
    ``min_active_reps`` of the selected one-based ``repetitions`` pass; all
    selected decisions must be present, otherwise the decision stays NaN.

    Args:
        cohort (dict): Reviewed phase cohort; bout displays need aligned bout
            samples covering ``bout_bounds`` (see ``align_phase_cohort``).
        trial_metrics (DataFrame): Must contain ``bout_window`` and
            ``preconditioning`` rows for both conditions.
        side_controls (dict): side -> (reference, test) logged names, e.g.
            {'left': ('LeC1', 'LeC2')}. Conditions are unique across sides.
        bout_window (str): Measurement window label used for both bouts.
        bout_bounds (tuple): Displayed bout seconds from bout onset [start, stop).
        include_reference_only (bool): Append reference-bout-only neurons
            (test-bout inactive) as a final display group. Default False.
        labels (tuple or None): Compact (reference, test) labels for titles and
            summary categories, e.g. ('C1', 'C2'); None uses the logged names.

    Returns:
        dict: ``neurons`` (every retained neuron: reference_bout_active,
        test_pre_active, test_bout_active, display ``group`` or -1),
        ``fish_summary`` (long table per fish/side/category: test_bout count of
        test-active neurons, test_pre and reference_bout overlaps among them, with
        ``fraction_of_test`` = overlap / test-active count; empty pools give NaN,
        missing required decisions give NaN counts), ``categories`` (side -> x
        labels in plot order), and ``displays`` compatible with
        ``plot_phase_recruitment_rasters``/``plot_phase_recruitment_decisions``.
        Display rows are test-bout-active neurons grouped: test pre + reference
        bout, test pre only, reference bout shared (no pre), test bout only,
        unclassified (missing pre/reference decision), then optional reference
        only. Matrices are (displayed neurons, time) trial-active fractions: test
        preconditioning (sequence 0..duration s), reference and test bouts
        (``bout_bounds``). Decision columns are test pre, reference bout, test bout.
        Inputs are not modified.
    """
    repetitions = _validate_phase_repetitions(repetitions, min_active_reps)
    bounds = _validate_phase_bounds(bout_bounds)
    if not cohort["fish_ids"] or not side_controls:
        raise ValueError("The comparison needs a nonempty cohort and side_controls.")
    pairs = {side: tuple(pair) for side, pair in side_controls.items()}
    conditions = [c for pair in pairs.values() for c in pair]
    if any(len(pair) != 2 for pair in pairs.values()) or len(set(conditions)) != len(conditions):
        raise ValueError("side_controls must map each side to two unique (reference, test) conditions.")
    if labels is not None and (len(labels) != 2 or labels[0] == labels[1]):
        raise ValueError("labels must be two distinct (reference, test) names.")
    decisions = _phase_recruitment_decisions(trial_metrics, conditions, repetitions, min_active_reps, bout_window)
    trial_rule = f">= {min_active_reps}/{len(repetitions)} passing trials"
    neurons, summaries, displays, categories = [], [], {}, {}
    for side, (reference, test) in pairs.items():
        names = tuple(labels) if labels is not None else (reference, test)
        categories[side] = [f"{names[1]} bout", f"{names[1]} pre", f"{names[0]} bout"]
        parts = {key: [] for key in ("preconditioning", reference, test, "rows", "codes_ref",
                                     "codes_test", "pool", "decisions")}
        axes = {}
        for fid in cohort["fish_ids"]:
            fish = cohort["all_fish_data"][fid]
            ids = np.asarray(fish["kept_neuron_indices"])
            ref_bout = _phase_decision_lookup(decisions, fid, ids, reference, bout_window)
            test_bout = _phase_decision_lookup(decisions, fid, ids, test, bout_window)
            test_pre = _phase_decision_lookup(decisions, fid, ids, test, "preconditioning")
            on = test_bout == 1
            group = np.full(len(ids), -1)
            group[on & (test_pre == 1) & (ref_bout == 1)] = 0
            group[on & (test_pre == 1) & (ref_bout == 0)] = 1
            group[on & (test_pre == 0) & (ref_bout == 1)] = 2
            group[on & (test_pre == 0) & (ref_bout == 0)] = 3
            group[on & ~(np.isfinite(test_pre) & np.isfinite(ref_bout))] = 4
            if include_reference_only:
                group[(test_bout == 0) & (ref_bout == 1)] = 5
            neurons.append(pd.DataFrame(dict(fish_id=fid, neuron_id=ids, side=side, reference=reference,
                test=test, reference_bout_active=ref_bout, test_pre_active=test_pre,
                test_bout_active=test_bout, group=group)))
            summaries += _control_comparison_summary(fid, side, ids, reference, test, names,
                                                     ref_bout, test_pre, test_bout)
            shown = group >= 0
            ref_trials, ref_sid = _phase_condition_stimulus(fish, fid, reference, repetitions)
            test_trials, test_sid = _phase_condition_stimulus(fish, fid, test, repetitions)
            panels = {reference: _phase_bout_raster_mean(fish, ref_sid, repetitions, bounds),
                      test: _phase_bout_raster_mean(fish, test_sid, repetitions, bounds),
                      "preconditioning": _phase_pre_raster_mean(fish, test_trials, test_sid, repetitions)}
            for key, (values, axis) in panels.items():
                if key in axes and not np.array_equal(axes[key], axis):
                    raise ValueError("Display axes differ between fish.")
                axes[key] = axis
                parts[key].append(values[shown])
            r, t, p = ref_bout[shown], test_bout[shown], test_pre[shown]
            parts["rows"].append(pd.DataFrame(dict(fish_id=fid, neuron_id=ids[shown], group=group[shown])))
            parts["codes_ref"].append(np.where(~np.isfinite(r), 4, np.where(r == 1, np.where(t == 1, 1, 3), 0)))
            parts["codes_test"].append(np.where(~np.isfinite(t), 4, np.where(t == 1,
                np.where(r == 1, 1, np.where(r == 0, 2, 4)), 0)))
            parts["pool"].append(p)
            parts["decisions"].append(np.column_stack([p, r, t]))
        rows = pd.concat(parts["rows"], ignore_index=True)
        order = np.argsort(rows.group.to_numpy(), kind="stable")
        group_labels = [f"{names[1]} pre + {names[0]} bout", f"{names[1]} pre only",
                        f"{names[0]} bout shared (no {names[1]} pre)", f"{names[1]} bout only",
                        "Unclassified", f"{names[0]} bout only (not {names[1]})"]
        displays[side] = dict(side=side, conditions=[reference, test], fish_ids=list(cohort["fish_ids"]),
            matrices={key: np.concatenate(parts[key]) for key in ("preconditioning", reference, test)},
            time_s=axes, neuron_order=order, row_metadata=rows,
            displayed_row_metadata=rows.iloc[order].reset_index(drop=True), group_labels=group_labels,
            repetitions=repetitions, bout_bounds=bounds,
            membership_codes={reference: np.concatenate(parts["codes_ref"]),
                              test: np.concatenate(parts["codes_test"])},
            pre_active_pool=np.concatenate(parts["pool"]),
            decision_matrix=np.concatenate(parts["decisions"]).reshape(-1, 3),
            fps_2p=cohort["all_fish_data"][cohort["fish_ids"][0]]["fps_2p"],
            panel_titles={"preconditioning": f"{names[1]} own preconditioning",
                          reference: f"{names[0]} bout", test: f"{names[1]} bout"},
            decision_labels=[f"{names[1]} pre", f"{names[0]} bout", f"{names[1]} bout"],
            pre_strip_label=f"{names[1]} preconditioning-active ({trial_rule})",
            strip_labels={1: f"Active in {names[0]} and {names[1]} bouts", 2: f"{names[1]} bout only",
                          3: f"{names[0]} bout only", 4: "Unclassified"},
            row_description=(f"Rows: {names[1]}-bout-active neurons grouped by own preconditioning "
                             f"and {names[0]} bout activity"
                             + (f"; then {names[0]}-only neurons" if include_reference_only else "")),
            decision_title=f"all phases: {trial_rule}")
    return dict(neurons=pd.concat(neurons, ignore_index=True), fish_summary=pd.DataFrame(summaries),
                categories=categories, displays=displays, repetitions=repetitions,
                min_active_reps=min_active_reps, bout_window=bout_window, bout_bounds=bounds,
                side_controls=pairs, labels=labels)


def select_phase_panel_rows(panel, rows):
    """Restrict one prepared phase panel display to selected neurons, in order.

    Args:
        panel (dict): One trace-type/side display from
            ``build_phase_stimulus_panel_displays`` or
            ``build_phase_sequence_panel_displays`` (matrices are pooled
            (neurons, time) in cohort fish/kept-neuron order).
        rows (DataFrame): fish_id and neuron_id (original kept IDs) in the
            desired display order, e.g. a comparison display's
            ``displayed_row_metadata``. Extra columns are retained.

    Returns:
        dict: A new panel with matrices (selected neurons, time) in ``rows``
        order, identity ``neuron_order``, ``row_metadata`` = rows, and
        ``fish_means`` (fish, time) averaged over each fish's selected neurons
        only (NaN for fish without selected neurons). Pooled-neuron mean traces
        drawn from it therefore average exactly the selected rows. Unknown or
        duplicated identities raise ValueError; the input panel is unchanged.
    """
    source = pd.MultiIndex.from_frame(panel["row_metadata"][["fish_id", "neuron_id"]])
    target = pd.MultiIndex.from_frame(rows[["fish_id", "neuron_id"]])
    positions = source.get_indexer(target)
    if (positions < 0).any() or target.duplicated().any():
        raise ValueError("rows must contain unique fish/neuron identities present in the panel.")
    selected = rows.reset_index(drop=True)
    matrices = {sid: np.asarray(values)[positions] for sid, values in panel["matrices"].items()}
    fish = selected.fish_id.to_numpy()
    means = {}
    for sid, values in matrices.items():
        stacked = np.full((len(panel["fish_ids"]), values.shape[1]), np.nan)
        for index, fid in enumerate(panel["fish_ids"]):
            if (fish == fid).any():
                stacked[index] = np.nanmean(values[fish == fid], axis=0)
        means[sid] = stacked
    return dict(panel, matrices=matrices, fish_means=means, row_metadata=selected,
                displayed_row_metadata=selected, neuron_order=np.arange(len(selected)),
                source_positions=positions)


def _phase_neuron_mean_z(trial_metrics, conditions, window, repetitions, column="mean_z",
                         baseline_window=None):
    """Per-neuron window metric (default mean z-score) averaged over selected one-based repetitions.

    Returns a Series indexed by (fish_id, neuron_id, condition); missing trial
    values are skipped by the repetition mean (all missing gives NaN). With
    ``baseline_window`` (a measured window label), each trial's baseline value
    of ``column`` is first subtracted from the same trial's value (per neuron
    and repetition); a missing baseline makes that trial missing.
    """
    def select(label):
        return trial_metrics.loc[trial_metrics.condition.isin(conditions)
            & trial_metrics.window.eq(label) & (trial_metrics.repetition + 1).isin(repetitions)]

    selected = select(window)
    if baseline_window is not None:
        keys = ["fish_id", "neuron_id", "condition", "repetition"]
        baseline = select(baseline_window)
        if selected.duplicated(keys).any() or baseline.duplicated(keys).any():
            raise ValueError("Duplicate neuron/condition/repetition values for the baseline correction.")
        reference = baseline.set_index(keys)[column].reindex(pd.MultiIndex.from_frame(selected[keys]))
        selected = selected.assign(**{column: selected[column].to_numpy(dtype=float)
                                      - reference.to_numpy(dtype=float)})
    return selected.groupby(["fish_id", "neuron_id", "condition"], sort=False)[column].mean()


def reorder_phase_recruitment_rows(recruitment, trial_metrics, mode="pre_then_reference_auc",
                                   reference=None, window="bout_15s", metric="auc_z_s"):
    """Return a new recruitment result with an alternative raster/decision row order.

    mode='groups' returns the input order unchanged (a shallow copy). mode
    'pre_then_reference_auc' puts the consistent preconditioning pool (group 0)
    first, in its current relative order, then every other row (including
    unclassified) by the reference bout's per-neuron ``metric`` (default signed
    z-score AUC, z*s) in ``window``, averaged over the display's selected
    one-based repetitions, highest first; NaN last; ties keep the current order.
    ``reference`` is a {side: condition} dict; None uses each side's control
    (first condition). Rows are matched by (fish_id, neuron_id); the input
    recruitment and trial_metrics are not modified. Each new display gets
    ``row_sort``, ``row_segments`` [(label, start, stop)], a ``row_description``
    and ``sort_mode``/``sort_value`` columns in ``displayed_row_metadata``.
    """
    if mode == "groups":
        return dict(recruitment, displays=dict(recruitment["displays"]))
    if mode != "pre_then_reference_auc":
        raise ValueError("mode must be 'groups' or 'pre_then_reference_auc'.")
    if metric not in trial_metrics:
        raise ValueError(f"trial_metrics has no '{metric}' column.")
    displays = {}
    for side, display in recruitment["displays"].items():
        condition = (reference or {}).get(side, display["conditions"][0])
        if condition not in display["conditions"]:
            raise ValueError(f"{side}: reference '{condition}' is not a displayed condition.")
        metadata = display["row_metadata"]
        order = np.asarray(display["neuron_order"])
        if sorted(order.tolist()) != list(range(len(metadata))):
            raise ValueError(f"{side}: neuron_order is not a permutation of row_metadata.")
        values = _phase_neuron_mean_z(trial_metrics, [condition], window, display["repetitions"], metric)
        if values.empty:
            raise ValueError(f"{side}: no '{window}' {metric} values for {condition}.")
        index = pd.MultiIndex.from_arrays([metadata.fish_id, metadata.neuron_id, [condition] * len(metadata)])
        if not index.isin(values.index).any():
            raise ValueError(f"{side}: no {condition} values match the displayed fish/neuron identities.")
        score = values.reindex(index).to_numpy(dtype=float)
        groups = metadata.group.to_numpy()[order]
        pre, rest = order[groups == 0], order[groups != 0]
        key = np.where(np.isfinite(score[rest]), -score[rest], np.inf)
        new_order = np.concatenate([pre, rest[np.argsort(key, kind="stable")]]).astype(int)
        shown = metadata.iloc[new_order].reset_index(drop=True).assign(sort_mode=mode, sort_value=score[new_order])
        name = {"auc_z_s": "z-score AUC", "mean_z": "mean z-score", "peak_z": "peak z-score"}.get(metric, metric)
        rest_label = f"Other neurons by {condition} {name}, high to low"
        displays[side] = dict(display, neuron_order=new_order, displayed_row_metadata=shown, row_sort=mode,
            row_segments=[(display["group_labels"][0], 0, len(pre)), (rest_label, len(pre), len(new_order))],
            row_description=f"Rows, bottom to top: consistent preconditioning, then all other neurons by {condition} "
                            f"bout {name}, high to low")
    return dict(recruitment, displays=displays)


def _phase_selected_bout_trace(fish, condition, repetitions, window_s, selected, baseline_window_s=None):
    """Mean bout-aligned z-score trace of selected neurons for one fish.

    Averages the selected one-based repetitions per neuron, then the selected
    neurons (boolean mask in kept-neuron order). With ``baseline_window_s``
    ((start, stop) seconds from bout onset), each neuron/trial trace first has
    its mean over that span subtracted. Returns (trace, nominal time axis in
    seconds from bout onset); the trace is NaN when no neuron is selected. A
    trace or baseline window beyond the aligned samples raises ValueError.
    """
    import warnings
    from src.trial_alignment import select_event_window

    trials = fish["trials"].loc[fish["trials"].logged_name.eq(condition)]
    if trials.empty or len(trials) < max(repetitions):
        raise ValueError(f"{condition}: selected repetitions unavailable.")
    sid = int(trials.stimulus_id.iloc[0])
    aligned = fish["aligned"]["bout"]["zscore"]
    times = np.asarray(aligned["time_s"][sid], dtype=float)
    period = 1 / fish["fps_2p"]
    frames = select_event_window(times, *window_s, sample_period_s=period)["frame_indices"]
    if not len(frames):
        raise ValueError("The trace window has no samples.")
    base_frames = None
    if baseline_window_s is not None:
        base_frames = select_event_window(times, *baseline_window_s, sample_period_s=period)["frame_indices"]
        if not len(base_frames):
            raise ValueError("The baseline window has no samples.")
    if not selected.any():
        return np.full(len(frames), np.nan), times[frames]
    values = aligned["trial_aligned_traces"][sid][selected][:, :, np.asarray(repetitions) - 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if base_frames is not None:
            values = values - np.nanmean(values[:, base_frames, :], axis=1, keepdims=True)
        return np.nanmean(np.nanmean(values[:, frames, :], axis=2), axis=0), times[frames]


def _equal_fish_mean_sem(values):
    """Equal-fish mean, SEM and defined-fish count over axis 0, ignoring NaN fish."""
    values = np.asarray(values, dtype=float)
    count = np.isfinite(values).sum(axis=0)
    mean = np.full(values.shape[1:], np.nan)
    sem = np.full(values.shape[1:], np.nan)
    np.divide(np.nansum(values, axis=0), count, out=mean, where=count > 0)
    deviation = np.where(np.isfinite(values), values - mean, 0.0)
    variance = np.full(values.shape[1:], np.nan)
    np.divide((deviation ** 2).sum(axis=0), count - 1, out=variance, where=count > 1)
    np.divide(np.sqrt(variance), np.sqrt(np.maximum(count, 1)), out=sem, where=count > 1)
    return mean, sem, count


def build_phase_control_shared_nopre_response(cohort, trial_metrics, comparison, trace_window_s=(-5.0, 15.0),
                                              baseline_window=None, baseline_window_s=None):
    """Mean z-score of neurons active in both control bouts but not in the test's preconditioning.

    Designed for Exp 9 C1 (reference, bout only) versus C2 (test, distant
    preconditioning). Per side, a neuron is selected when the comparison's
    decisions are reference_bout_active == 1, test_bout_active == 1 and
    test_pre_active == 0 (the ``build_phase_control_comparison`` group 2,
    "reference bout shared (no test pre)"). Neurons with any missing decision
    are excluded and counted as unclassified, never treated as inactive.

    Measure: per neuron, the window mean z-score of ``comparison['bout_window']``
    (the classification window, e.g. 'bout_10s' = 0-10 s after bout onset)
    averaged over ``comparison['repetitions']``, for the reference and the test
    bout. Neurons need finite values in both conditions. Per fish, values are
    averaged over its selected neurons; fish without selected neurons are NaN
    (missing, not zero). Cohort values average defined fish equally. By default
    window means are not baseline-corrected: for a test condition with
    preconditioning, weak carried-over pre-bout activity (below the recruitment
    rule) is included. The opt-in ``baseline_window`` corrects for it: per
    neuron, the mean over the selected repetitions of (window mean_z - baseline
    window mean_z) of the same trial; traces then subtract each neuron/trial's
    mean over ``baseline_window_s`` before averaging.
    The window means use the native-window ``mean_z`` of ``trial_metrics``, while
    ``traces`` use the resampled bout-aligned z-score grid, so a trace averaged
    over the window can differ slightly from the bar value.

    Args:
        cohort (dict): Phase cohort whose bout-aligned z-scores cover
            ``trace_window_s`` (e.g. the 15 s-recovery ``align_phase_cohort`` copy);
            fish order defines the output order.
        trial_metrics (DataFrame): ``build_phase_population_analysis`` trial table
            containing ``comparison['bout_window']`` with ``mean_z``.
        comparison (dict): ``build_phase_control_comparison`` result built from
            the same cohort/trial_metrics.
        trace_window_s (tuple): Time-course window, seconds from bout onset
            [start, stop); beyond the aligned samples raises ValueError.
        baseline_window (str or None): Optional measured window label in
            ``trial_metrics`` (e.g. 'pre_bout' = -2 to 0 s from bout onset)
            subtracted per trial. None (default) keeps uncorrected values.
        baseline_window_s (tuple or None): The same baseline span, seconds from
            bout onset [start, stop), for the traces; required with
            ``baseline_window`` and must describe the same measured window.

    Returns:
        dict: ``neurons`` (selected neurons: fish_id, side, neuron_id, reference,
        test, reference_mean_z, test_mean_z, delta_mean_z = reference - test;
        baseline-corrected values when ``baseline_window`` is set),
        ``fish_summary`` (one row per fish/side: n_neurons retained, n_classified,
        n_unclassified, n_selected, n_finite_pairs, reference_mean_z, test_mean_z,
        delta_mean_z), ``cohort_summary`` (one row per side: n_fish, n_fish_defined,
        n_selected_total, equal-fish means and SEMs of the three values),
        ``traces`` (side -> time_s (samples,), fish ``reference``/``test`` arrays
        (fish, samples), their equal-fish ``*_mean``/``*_sem``/``*_n_fish``,
        and per-condition ``bout_duration_s``), plus fish_ids, labels
        (reference, test), side_controls, repetitions, min_active_reps,
        bout_window, bout_bounds, trace_window_s, baseline_window and
        baseline_window_s (both None when uncorrected). Inputs are not modified.
    """
    repetitions = _validate_phase_repetitions(comparison["repetitions"], comparison["min_active_reps"])
    trace_window = _validate_phase_bounds(trace_window_s)
    bout_window = comparison["bout_window"]
    if not cohort["fish_ids"]:
        raise ValueError("Needs a nonempty cohort.")
    if not trial_metrics.window.eq(bout_window).any():
        raise ValueError(f"trial_metrics has no '{bout_window}' window.")
    if (baseline_window is None) != (baseline_window_s is None):
        raise ValueError("Set baseline_window and baseline_window_s together (or neither).")
    if baseline_window is not None:
        baseline_window_s = _validate_phase_bounds(baseline_window_s)
        if not trial_metrics.window.eq(baseline_window).any():
            raise ValueError(f"trial_metrics has no '{baseline_window}' baseline window; "
                             "add it to the measurement settings.")
    table = comparison["neurons"]
    expected = table.reference_bout_active.eq(1) & table.test_bout_active.eq(1) & table.test_pre_active.eq(0)
    if not expected.equals(table.group.eq(2)):
        raise ValueError("Selection differs from the comparison's shared-no-pre group (code 2).")
    conditions = [c for pair in comparison["side_controls"].values() for c in pair]
    response = _phase_neuron_mean_z(trial_metrics, conditions, bout_window, repetitions,
                                    baseline_window=baseline_window)
    fish_ids = list(cohort["fish_ids"])
    neuron_rows, fish_rows, cohort_rows, traces = [], [], [], {}
    for side, (reference, test) in comparison["side_controls"].items():
        labels = tuple(comparison["labels"]) if comparison.get("labels") else (reference, test)
        curves = {"reference": [], "test": []}
        axes, durations = {}, {"reference": [], "test": []}
        for fid in fish_ids:
            fish = cohort["all_fish_data"][fid]
            ids = np.asarray(fish["kept_neuron_indices"])
            rows = table.loc[table.fish_id.eq(fid) & table.side.eq(side)].set_index("neuron_id")
            if len(rows) != len(ids) or not rows.index.isin(ids).all():
                raise ValueError(f"{fid}/{side}: comparison neurons differ from the cohort's kept neurons.")
            chosen = (rows.reference_bout_active.eq(1) & rows.test_bout_active.eq(1)
                      & rows.test_pre_active.eq(0)).reindex(ids).to_numpy()
            known = rows[["reference_bout_active", "test_bout_active", "test_pre_active"]].notna().all(
                axis=1).reindex(ids).to_numpy()
            values = {key: response.reindex(pd.MultiIndex.from_product([[fid], ids, [name]],
                names=response.index.names)).to_numpy(dtype=float)
                for key, name in (("reference", reference), ("test", test))}
            used = chosen & np.isfinite(values["reference"]) & np.isfinite(values["test"])
            for key, name in (("reference", reference), ("test", test)):
                trace, axis = _phase_selected_bout_trace(fish, name, repetitions, trace_window, used,
                                                         baseline_window_s)
                if key in axes and not np.allclose(axes[key], axis):
                    raise ValueError("Bout-aligned time axes differ between fish.")
                axes[key] = axis
                curves[key].append(trace)
                trials = fish["trials"].loc[fish["trials"].logged_name.eq(name)]
                durations[key].append(float((trials.end_s - trials.bout_onset_s).mean()))
            delta = values["reference"] - values["test"]
            neuron_rows.append(pd.DataFrame(dict(fish_id=fid, side=side, neuron_id=ids[used],
                reference=reference, test=test, reference_mean_z=values["reference"][used],
                test_mean_z=values["test"][used], delta_mean_z=delta[used])))
            fish_rows.append(dict(fish_id=fid, side=side, reference=reference, test=test,
                n_neurons=len(ids), n_classified=int(known.sum()), n_unclassified=int((~known).sum()),
                n_selected=int(chosen.sum()), n_finite_pairs=int(used.sum()),
                reference_mean_z=float(values["reference"][used].mean()) if used.any() else np.nan,
                test_mean_z=float(values["test"][used].mean()) if used.any() else np.nan,
                delta_mean_z=float(delta[used].mean()) if used.any() else np.nan))
        if not np.allclose(axes["reference"], axes["test"]):
            raise ValueError(f"{side}: reference and test bout time axes differ.")
        side_fish = pd.DataFrame(fish_rows[-len(fish_ids):])
        record = dict(side=side, reference=reference, test=test, reference_label=labels[0],
                      test_label=labels[1], n_fish=len(fish_ids),
                      n_fish_defined=int(side_fish.delta_mean_z.notna().sum()),
                      n_selected_total=int(side_fish.n_finite_pairs.sum()))
        for column in ("reference_mean_z", "test_mean_z", "delta_mean_z"):
            mean, sem, _ = _equal_fish_mean_sem(side_fish[[column]].to_numpy())
            record[column], record[column.replace("_mean_z", "_sem_z")] = float(mean[0]), float(sem[0])
        cohort_rows.append(record)
        side_traces = dict(time_s=axes["reference"], labels=labels,
                           bout_duration_s={key: float(np.mean(v)) for key, v in durations.items()})
        for key, parts in curves.items():
            stacked = np.vstack(parts)
            mean, sem, count = _equal_fish_mean_sem(stacked)
            side_traces.update({key: stacked, f"{key}_mean": mean, f"{key}_sem": sem, f"{key}_n_fish": count})
        traces[side] = side_traces
    return dict(neurons=pd.concat(neuron_rows, ignore_index=True), fish_summary=pd.DataFrame(fish_rows),
                cohort_summary=pd.DataFrame(cohort_rows), traces=traces, fish_ids=fish_ids,
                labels=comparison.get("labels"), side_controls=dict(comparison["side_controls"]),
                repetitions=repetitions, min_active_reps=comparison["min_active_reps"],
                bout_window=bout_window, bout_bounds=tuple(comparison["bout_bounds"]),
                trace_window_s=trace_window, baseline_window=baseline_window,
                baseline_window_s=baseline_window_s)


def build_phase_recruitment_strength_inputs(cohort, trial_metrics, side_conditions, condition_labels,
                                            bout_window, repetitions=(2, 3, 4), min_active_reps=2):
    """Adapt phase measurements to ``plot_recruitment_strength_summary`` inputs.

    Per side, the first condition is the control (e.g. C1). For each fish and
    condition, over the selected one-based ``repetitions`` in ``bout_window``:
    population activity = mean over all retained neurons of each neuron's mean
    z-score (window mean z, averaged over selected trials); recruitment =
    active fraction of retained neurons (>= ``min_active_reps`` passing
    trials; NaN when any decision is missing); shared-neuron strength = mean
    over neurons active in both control and condition of (control mean z -
    condition mean z), NaN with no shared neurons or any missing decision. The
    control's self-comparison is absent (shown as n/a by the plot).

    Args:
        cohort (dict): Reviewed phase cohort (fish order and kept neurons).
        trial_metrics (DataFrame): ``build_phase_population_analysis`` trial
            table containing ``bout_window``.
        side_conditions (dict): side -> logged conditions, control first.
        condition_labels (sequence): Compact labels shared by every side, same
            length/order as each side's conditions (e.g. ['C1', 'C2', 'S1']).
        bout_window (str): Window label, e.g. 'bout_10s'.

    Returns:
        dict: ``analysis`` and ``shared_auc`` dictionaries with the fields read
        by ``plot_recruitment_strength_summary`` (fish_ids, condition_labels,
        side_selections, tables). Stimulus IDs are the logged condition names.
        Window mean z-scores are stored as ``mean_normalized_auc`` (AUC divided
        by window span) and ``mean_delta_normalized_auc`` for plot
        compatibility, plus explicit ``mean_z``/``mean_delta_z`` and per-neuron
        tables. Fish are weighted equally by the plot; inputs are not modified.
    """
    repetitions = _validate_phase_repetitions(repetitions, min_active_reps)
    labels = list(condition_labels)
    if not cohort["fish_ids"] or not side_conditions:
        raise ValueError("Needs a nonempty cohort and side_conditions.")
    conditions = [c for names in side_conditions.values() for c in names]
    if (len(set(conditions)) != len(conditions) or len(set(labels)) != len(labels)
            or any(len(names) != len(labels) or len(names) < 2 for names in side_conditions.values())):
        raise ValueError("Each side needs unique conditions (control first, >= 2) matching condition_labels.")
    decisions = _phase_recruitment_decisions(trial_metrics, conditions, repetitions, min_active_reps, bout_window)
    response = _phase_neuron_mean_z(trial_metrics, conditions, bout_window, repetitions)
    neuron_rows, recruitment, shared_rows, neuron_pairs = [], [], [], []
    selections, shared_selections = {}, {}
    for side, names in side_conditions.items():
        names = list(names)
        selections[side] = dict(stimulus_ids=names, stimulus_labels=names, condition_labels=labels,
                                groups={"control": [names[0]]})
        shared_selections[side] = dict(control_id=names[0], control_label=labels[0],
                                       stimulus_ids=names[1:], condition_labels=labels[1:])
        for fid in cohort["fish_ids"]:
            ids = np.asarray(cohort["all_fish_data"][fid]["kept_neuron_indices"])
            values = {}
            for index, (name, label) in enumerate(zip(names, labels)):
                active = _phase_decision_lookup(decisions, fid, ids, name, bout_window)
                z = response.reindex(pd.MultiIndex.from_product([[fid], ids, [name]],
                    names=response.index.names)).to_numpy(dtype=float)
                values[name] = (active, z)
                known = bool(np.isfinite(active).all())
                n_active = float((active == 1).sum()) if known else np.nan
                recruitment.append(dict(fish_id=fid, side=side, stimulus_id=name, stimulus=name,
                    condition=label, condition_index=index, n_neurons=len(ids),
                    n_valid_active=int(np.isfinite(active).sum()), n_active=n_active,
                    active_fraction=n_active / len(ids), n_finite_responses=int(np.isfinite(z).sum()),
                    mean_z=pd.Series(z).mean(), mean_normalized_auc=pd.Series(z).mean()))
                neuron_rows.append(pd.DataFrame(dict(fish_id=fid, side=side, stimulus_id=name,
                    condition=label, neuron_id=ids, active=active, mean_z=z)))
            control_active, control_z = values[names[0]]
            for name, label in zip(names[1:], labels[1:]):
                active, z = values[name]
                known = bool(np.isfinite(control_active).all() and np.isfinite(active).all())
                shared = (control_active == 1) & (active == 1)
                finite = shared & np.isfinite(control_z) & np.isfinite(z)
                delta = control_z - z
                mean_delta = float(delta[finite].mean()) if known and finite.any() else np.nan
                shared_rows.append(dict(fish_id=fid, side=side, stimulus_id=name, condition=label,
                    control_id=names[0], control=labels[0],
                    n_control_active=float((control_active == 1).sum()) if known else np.nan,
                    n_stimulus_active=float((active == 1).sum()) if known else np.nan,
                    n_shared=float(shared.sum()) if known else np.nan,
                    n_finite_pairs=int(finite.sum()) if known else 0,
                    mean_control_z=float(control_z[finite].mean()) if known and finite.any() else np.nan,
                    mean_stimulus_z=float(z[finite].mean()) if known and finite.any() else np.nan,
                    mean_delta_z=mean_delta, mean_delta_normalized_auc=mean_delta))
                neuron_pairs.append(pd.DataFrame(dict(fish_id=fid, side=side, stimulus_id=name,
                    condition=label, control_id=names[0], neuron_id=ids[shared],
                    control_mean_z=control_z[shared], stimulus_mean_z=z[shared],
                    delta_mean_z=delta[shared])))
    fish_ids = list(cohort["fish_ids"])
    analysis = dict(fish_ids=fish_ids, condition_labels=labels, side_selections=selections,
                    bout_window=bout_window, repetitions=repetitions, min_active_reps=min_active_reps,
                    tables=dict(recruitment=pd.DataFrame(recruitment),
                                neuron_responses=pd.concat(neuron_rows, ignore_index=True)))
    shared_auc = dict(fish_ids=fish_ids, side_selections=shared_selections,
                      tables=dict(shared_auc_per_fish=pd.DataFrame(shared_rows),
                                  shared_neuron_auc=pd.concat(neuron_pairs, ignore_index=True)))
    return dict(analysis=analysis, shared_auc=shared_auc)


def _family_sparseness_neurons(panel, memberships, labels, fid, side, family, population):
    """Reuse the canonical positive-response metric only for complete profiles."""
    vectors = panel["responses"].loc[:, labels]
    rows = panel["row_metadata"].copy().reset_index(drop=True)
    if vectors.index.tolist() != rows["global_neuron_id"].tolist():
        raise ValueError(f"{fid}, {side}, {population}: response and metadata identities differ.")
    lookup = memberships.loc[
        (memberships["fish_id"] == fid) & (memberships["side"] == side)
        & (memberships["family"] == family)
    ].set_index("global_neuron_id")
    if lookup.index.has_duplicates or not vectors.index.isin(lookup.index).all():
        raise ValueError(f"{fid}, {side}, {family}: membership identities do not match profiles.")
    rows["family"], rows["population"] = family, population
    rows["n_active_variants"] = lookup.loc[vectors.index, "n_active_stimuli"].to_numpy(dtype=int)
    values = vectors.to_numpy(dtype=float)
    for index, label in enumerate(labels):
        rows[f"signed_response_{label}"] = values[:, index]
        rows[f"positive_response_{label}"] = np.maximum(values[:, index], 0)
    complete = np.isfinite(values).all(axis=1)
    rows["n_finite_variants"] = np.isfinite(values).sum(axis=1)
    rows["n_negative_variants"] = (values < 0).sum(axis=1)
    rows["lifetime_sparseness"] = [
        np.clip(compute_stimulus_selectivity_metrics(value, labels)["lifetime_sparseness"], 0, 1)
        if valid else np.nan for value, valid in zip(values, complete)
    ]
    rows["sparseness_status"] = np.where(
        ~complete, "incomplete_profile",
        np.where(np.isfinite(rows["lifetime_sparseness"]), "defined", "no_positive_response"))
    return rows


def build_family_lifetime_sparseness(breadth_profiles, cdf_grid):
    """Describe positive-response concentration separately within each family.

    Args:
        breadth_profiles (dict): Output of build_response_breadth_profiles,
            with identity-aligned signed normalized AUC and membership tables.
        cdf_grid (sequence): Strictly increasing finite evaluation points from
            exactly 0 through 1, including both endpoints.

    Returns:
        dict: Fish/family/population order and report-ready neuron, per-fish,
        cohort, per-fish CDF, and equal-fish CDF tables. The canonical metric
        clips negative responses at zero for this calculation only. Complete
        family profiles are required; nonfinite entries are not imputed as zero.
        Zero/nonpositive profiles have NaN metrics. Metric roundoff is bounded
        to [0, 1]; signed inputs are retained.
        Per-fish means and CDFs use defined neurons; cohort means/CDFs weight
        defined fish equally and record valid counts. Activity flags remain
        descriptive; neither population receives an extra activity filter.
        Values near 0 indicate equal positive activity and 1 indicates activity
        confined to one tested variant. No reliability validation is implied.
        No loading, normalization, mutation, plotting, or file I/O occurs.
    """
    grid = np.asarray(cdf_grid, dtype=float)
    if (grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all()
            or grid[0] != 0 or grid[-1] != 1 or not (np.diff(grid) > 0).all()):
        raise ValueError("cdf_grid must increase strictly from 0 to 1, including endpoints.")
    neurons, summaries, cdfs = [], [], []
    memberships = breadth_profiles["tables"]["response_memberships"]
    for fid in breadth_profiles["fish_ids"]:
        for side in ("left", "right"):
            for population in breadth_profiles["populations"]:
                panel = breadth_profiles["profiles"][fid][side][population]
                for family in breadth_profiles["families"]:
                    labels = breadth_profiles["family_labels"][family]
                    rows = _family_sparseness_neurons(
                        panel, memberships, labels, fid, side, family, population)
                    neurons.append(rows)
                    values = rows["lifetime_sparseness"].to_numpy(dtype=float)
                    defined = np.sort(values[np.isfinite(values)])
                    n_defined = len(defined)
                    keys = {"fish_id": fid, "side": side, "family": family, "population": population}
                    summaries.append({
                        **keys, "n_population_neurons": len(rows), "n_defined_neurons": n_defined,
                        "n_incomplete_neurons": int((rows["sparseness_status"] == "incomplete_profile").sum()),
                        "n_no_positive_neurons": int((rows["sparseness_status"] == "no_positive_response").sum()),
                        "n_active_family_neurons": int((rows["n_active_variants"] > 0).sum()),
                        "n_defined_active_neurons": int(((rows["n_active_variants"] > 0) & np.isfinite(values)).sum()),
                        "n_neurons_with_negative_values": int((rows["n_negative_variants"] > 0).sum()),
                        "mean_sparseness": float(np.mean(defined)) if n_defined else np.nan,
                        "median_sparseness": float(np.median(defined)) if n_defined else np.nan,
                    })
                    cumulative = (np.searchsorted(defined, grid, side="right") / n_defined
                                  if n_defined else np.full(len(grid), np.nan))
                    cdfs.extend({**keys, "sparseness": float(x), "cumulative_fraction": float(y),
                                 "n_defined_neurons": n_defined} for x, y in zip(grid, cumulative))
    summary, cdf = pd.DataFrame(summaries), pd.DataFrame(cdfs)
    keys = ["population", "side", "family"]
    counts = [name for name in summary.columns if name.startswith("n_")]
    cohort = summary.groupby(keys, sort=False).agg(
        mean_sparseness=("mean_sparseness", "mean"), valid_fish=("mean_sparseness", "count"),
        **{name: (name, "sum") for name in counts},
    ).reset_index()
    mean_cdf = cdf.groupby(keys + ["sparseness"], sort=False).agg(
        cumulative_fraction=("cumulative_fraction", "mean"), valid_fish=("cumulative_fraction", "count"),
    ).reset_index()
    return {
        **{key: list(breadth_profiles[key]) for key in ("fish_ids", "families", "populations")},
        "population_labels": dict(breadth_profiles["population_labels"]),
        "family_labels": dict(breadth_profiles["family_labels"]), "cdf_grid": grid.copy(),
        "tables": {"family_sparseness_neurons": pd.concat(neurons, ignore_index=True),
                   "family_sparseness_per_fish": summary, "family_sparseness_summary": cohort,
                   "family_sparseness_cdf": cdf, "family_sparseness_mean_cdf": mean_cdf},
    }


def _prepare_breadth_profile_panel(analysis, similarity, fid, side, population, metadata):
    """Check cached population identities and retain the sorted display rows."""
    active = analysis["active_matrices"][fid]
    mask = (np.ones(len(active), dtype=bool) if population == "all_kept" else
            build_active_group_masks(active, analysis["side_selections"][side]["groups"])["rocking"]["all"])
    expected = metadata.loc[mask].reset_index(drop=True)
    cached = similarity["row_metadata"][fid][side][population].reset_index(drop=True)
    vectors = similarity["response_vectors"][fid][side][population]
    identities = ["fish_id", "neuron_id", "source_neuron_id", "global_neuron_id"]
    if (not expected[identities].equals(cached[identities])
            or list(vectors.columns) != list(analysis["condition_labels"])
            or vectors.index.tolist() != expected["global_neuron_id"].tolist()):
        raise ValueError(f"{fid}, {side}, {population}: cached response identities/order differ.")
    if len(metadata):
        prepared = _prepare_similarity_response_vectors(analysis, fid, side, metadata)
        if not vectors.equals(prepared.iloc[np.flatnonzero(mask)]):
            raise ValueError(f"{fid}, {side}, {population}: cached response values differ.")
    else:
        source = analysis["tables"]["neuron_responses"]
        if ((source["fish_id"] == fid) & (source["side"] == side)).any():
            raise ValueError(f"{fid}, {side}: empty metadata has nonempty responses.")
    order = build_response_profile_sort_order(vectors, analysis["condition_labels"][1:])
    rows = expected.iloc[order].copy().reset_index(drop=True)
    rows["side"], rows["population"] = side, population
    rows["display_row"] = np.arange(len(rows))
    return {"responses": vectors.iloc[order].copy(), "row_metadata": rows}


def build_response_breadth_profiles(analysis, similarity):
    """Prepare family memberships and signed response profiles from cached inputs.

    Args:
        analysis (dict): Prepared recruitment/containment bundle; kept-neuron
            decisions and identity metadata must remain aligned.
        similarity (dict): Matching response-similarity bundle, containing
            signed normalized AUC for all_kept and side-specific all_rocking.

    Returns:
        dict: Identity-preserving neuron memberships, complete per-fish breadth
        and combination tables, equal-fish summaries with counts/valid_fish,
        and profiles[fish][side][population] with sorted response DataFrames
        (neurons by conditions) and display-row metadata. profile_color_limit
        is the largest absolute finite response across both populations, or 1
        if no nonzero finite responses exist. Units remain z-score activity.
        Original fractions use all kept neurons. Additional response_breadth_family_responsive
        and response_combinations_family_responsive tables condition on any family
        response per fish/side, excluding zero-active categories. Empty denominators
        are NaN. No loading,
        classification, normalization, plotting, or input mutation occurs.
    """
    fish_ids, labels = list(analysis["fish_ids"]), list(analysis["condition_labels"])
    if fish_ids != list(similarity["fish_ids"]) or labels != list(similarity["condition_labels"]):
        raise ValueError("Analysis and similarity fish/condition orders differ.")
    families = ["rocking", "flicker"]
    populations = ["all_kept", "all_rocking"]
    memberships, breadth, combinations, profile_rows = [], [], [], []
    profiles, family_labels, category_orders = {}, {}, {}
    color_limit = 0.0
    for fid in fish_ids:
        active = analysis["active_matrices"][fid]
        metadata = analysis["responses"]["row_metadata"]
        metadata = metadata.loc[metadata["fish_id"] == fid].reset_index(drop=True)
        if len(active) != len(metadata) or not np.array_equal(metadata["neuron_id"], np.arange(len(active))):
            raise ValueError(f"{fid}: active rows differ from kept-neuron metadata.")
        profiles[fid] = {}
        for side in ("left", "right"):
            selection = analysis["side_selections"][side]
            for family in families:
                ids = selection["groups"][family]
                selected_labels = [labels[selection["stimulus_ids"].index(key)] for key in ids]
                result = build_active_membership_table(active, ids, selected_labels)
                family_labels.setdefault(family, selected_labels)
                if family_labels[family] != selected_labels:
                    raise ValueError("Mirrored families must have matching condition labels.")
                category_orders[family] = result["combinations"]["membership"].tolist()
                neurons = pd.concat([metadata, result["neurons"].reset_index(drop=True)], axis=1)
                neurons["side"], neurons["family"] = side, family
                memberships.append(neurons)
                for key, destination in (("breadth", breadth), ("combinations", combinations)):
                    table = result[key].copy()
                    table["fish_id"], table["side"], table["family"] = fid, side, family
                    destination.append(table)
            profiles[fid][side] = {}
            for population in populations:
                panel = _prepare_breadth_profile_panel(analysis, similarity, fid, side, population, metadata)
                profiles[fid][side][population] = panel
                values = panel["responses"].to_numpy(dtype=float)
                finite = np.abs(values[np.isfinite(values)])
                if finite.size:
                    color_limit = max(color_limit, float(finite.max()))
                wide = panel["row_metadata"].copy()
                for condition in labels:
                    wide[f"response_{condition}"] = panel["responses"][condition].to_numpy()
                profile_rows.append(wide)
    breadth, combinations = pd.concat(breadth, ignore_index=True), pd.concat(combinations, ignore_index=True)
    summaries = {}
    for name, table, category in (("breadth", breadth, "n_active_stimuli"),
                                   ("combinations", combinations, "membership")):
        summaries[f"response_{name}_summary"] = table.groupby(
            ["side", "family", category], sort=False,
        ).agg(fraction_kept=("fraction_kept", "mean"), valid_fish=("fraction_kept", "count"),
              n_neurons=("n_neurons", "sum"), n_kept_neurons=("n_kept_neurons", "sum")).reset_index()
    responsive_tables = {
        "response_breadth_family_responsive": build_family_responsive_membership_fractions(breadth),
        "response_combinations_family_responsive": build_family_responsive_membership_fractions(combinations),
    }
    return {
        "fish_ids": fish_ids, "condition_labels": labels, "families": families,
        "populations": populations, "population_labels": dict(similarity["population_labels"]),
        "family_labels": family_labels, "combination_orders": category_orders,
        "profiles": profiles, "profile_color_limit": color_limit or 1.0,
        "tables": {"response_memberships": pd.concat(memberships, ignore_index=True),
                   "response_breadth": breadth, "response_combinations": combinations,
                   "response_profile_rows": pd.concat(profile_rows, ignore_index=True), **summaries,
                   **responsive_tables},
    }


def build_pooled_response_profiles(breadth_profiles, sort_mode="pearson"):
    """Pool all-kept neurons and align both sides with one joint display order.

    Args:
        breadth_profiles (dict): Prepared profile bundle. Each fish/side's all_kept
            responses are (neurons, conditions), with identity-aligned row_metadata.
        sort_mode (str): pearson, euclidean, or preferred, delegated to the ordering
            owner. Sorting concatenates left/right non-control response features.

    Returns:
        dict: responses[left/right] retain raw signed normalized-AUC values in
        condition order, with identical neuron rows. row_metadata maps pooled
        display_row to fish/source/global identities. The existing shared full-range
        profile_color_limit is preserved. This is one row per neuron, not a fish mean;
        no clustering cut, response normalization, missing-value filling, or mutation.
    """
    labels = list(breadth_profiles["condition_labels"])
    sorting_conditions = labels[1:]
    responses, metadata = {}, {}
    identities = ["fish_id", "neuron_id", "source_neuron_id", "global_neuron_id"]
    for side in ("left", "right"):
        matrices, rows = [], []
        for fid in breadth_profiles["fish_ids"]:
            panel = breadth_profiles["profiles"][fid][side]["all_kept"]
            source = panel["row_metadata"]
            if (list(panel["responses"].columns) != labels
                    or panel["responses"].index.tolist() != source["global_neuron_id"].tolist()
                    or not (source["fish_id"] == fid).all()):
                raise ValueError(f"{fid}, {side}: pooled profile values and identities are not aligned.")
            ordered = source.sort_values("neuron_id", kind="stable")
            matrices.append(panel["responses"].loc[ordered["global_neuron_id"]])
            rows.append(ordered[identities])
        responses[side] = pd.concat(matrices)
        metadata[side] = pd.concat(rows, ignore_index=True)
        if responses[side].index.has_duplicates:
            raise ValueError("Pooled response profiles need unique global neuron IDs.")
    if not metadata["left"].equals(metadata["right"]):
        raise ValueError("Left and right all-kept profiles must contain the same neuron identities.")
    joint = pd.concat([responses[side][sorting_conditions] for side in ("left", "right")],
                      axis=1, keys=["left", "right"])
    order = build_response_profile_sort_order(joint, joint.columns.tolist(), sort_mode=sort_mode)
    rows = metadata["left"].iloc[order].copy().reset_index(drop=True)
    rows["display_row"] = np.arange(len(rows))
    return {"responses": {side: values.iloc[order].copy() for side, values in responses.items()},
            "row_metadata": rows, "sort_mode": sort_mode, "sorting_conditions": sorting_conditions,
            "profile_color_limit": breadth_profiles["profile_color_limit"],
            "fish_ids": list(breadth_profiles["fish_ids"])}


def _prepare_similarity_response_vectors(analysis, fid, side, metadata):
    """Pivot existing normalized responses without losing source-neuron identities."""
    labels = analysis["condition_labels"]
    table = analysis["tables"]["neuron_responses"]
    rows = table.loc[(table["fish_id"] == fid) & (table["side"] == side)]
    keys = ["global_neuron_id", "condition"]
    if rows.duplicated(keys).any() or len(rows) != len(metadata) * len(labels):
        raise ValueError(f"{fid}, {side}: need exactly one response per neuron/condition.")
    identities = rows[["global_neuron_id", "source_neuron_id", "neuron_id"]].drop_duplicates()
    identities = identities.set_index("global_neuron_id").reindex(metadata["global_neuron_id"])
    if not np.array_equal(identities[["source_neuron_id", "neuron_id"]].to_numpy(),
                          metadata[["source_neuron_id", "neuron_id"]].to_numpy()):
        raise ValueError(f"{fid}, {side}: source-neuron identities differ from response metadata.")
    vectors = rows.pivot(index="global_neuron_id", columns="condition", values="normalized_auc")
    if set(vectors.columns) != set(labels) or set(vectors.index) != set(metadata["global_neuron_id"]):
        raise ValueError(f"{fid}, {side}: response neurons/conditions do not match metadata.")
    return vectors.reindex(index=metadata["global_neuron_id"], columns=labels)


def _response_similarity_pair_rows(matrices, vectors, fid, side, population):
    """Record every ordered pair, including diagonal and valid-neuron counts."""
    finite = np.isfinite(vectors.to_numpy(dtype=float)).astype(np.int64)
    valid_counts = finite.T @ finite
    rows = []
    for i, source in enumerate(vectors.columns):
        for j, target in enumerate(vectors.columns):
            rows.append({
                "fish_id": fid, "side": side, "population": population,
                "source_condition": source, "target_condition": target,
                "n_population_neurons": len(vectors),
                "n_valid_response_neurons": int(valid_counts[i, j]),
                **{metric: matrix.iloc[i, j] for metric, matrix in matrices.items()},
            })
    return rows


def build_response_similarity_comparison(analysis):
    """Compare prepared response patterns in all kept neurons and the all-rocking core.

    Args:
        analysis (dict): Output of build_recruitment_containment_analysis. Uses
            its identity-aligned signed normalized_auc values and existing
            active decisions, without loading or reclassifying data.

    Returns:
        dict: Per-fish and equal-fish mean Jaccard/Pearson/cosine matrices for
        each side/population, ordered response vectors and neuron metadata, and
        report-ready pairs, matrix summaries, and population-size tables.
        Pearson/cosine delegate to the existing stimulus-vector helper and keep
        its pairwise finite-value/undefined-value semantics. Empty cores remain
        empty, with NaN similarities. Means are arithmetic over defined fish,
        with valid-fish counts; they are not pooled or Fisher transformed.
        Inputs and all existing stage-1 results are left unchanged.
    """
    populations = ["all_kept", "all_rocking"]
    metrics = ["jaccard", "pearson", "cosine"]
    fish_ids, labels = list(analysis["fish_ids"]), list(analysis["condition_labels"])
    metadata = analysis["responses"]["row_metadata"]
    per_fish, response_vectors, selected_metadata = {}, {}, {}
    pair_rows, population_rows = [], []
    for fid in fish_ids:
        fish_metadata = metadata.loc[metadata["fish_id"] == fid].reset_index(drop=True)
        active = analysis["active_matrices"][fid]
        if len(active) != len(fish_metadata) or not np.array_equal(
            fish_metadata["neuron_id"], np.arange(len(fish_metadata)),
        ):
            raise ValueError(f"{fid}: active decisions must follow kept-neuron metadata order.")
        per_fish[fid], response_vectors[fid], selected_metadata[fid] = {}, {}, {}
        for side in ("left", "right"):
            selection = analysis["side_selections"][side]
            vectors = _prepare_similarity_response_vectors(analysis, fid, side, fish_metadata)
            masks = build_active_group_masks(active, selection["groups"])
            population_masks = {"all_kept": np.ones(len(active), dtype=bool),
                                "all_rocking": masks["rocking"]["all"]}
            per_fish[fid][side], response_vectors[fid][side], selected_metadata[fid][side] = {}, {}, {}
            for population, mask in population_masks.items():
                selected = vectors.iloc[np.flatnonzero(mask)].copy()
                similarity = build_stimulus_vector_similarity(selected, labels)
                jaccard = (
                    analysis["per_fish_matrices"][fid][side]["jaccard"].copy()
                    if population == "all_kept" else compute_active_neuron_jaccard_overlap(
                        active.iloc[np.flatnonzero(mask)], selection["stimulus_ids"], labels)
                )
                matrices = {"jaccard": jaccard, "pearson": similarity["pearson_similarity_matrix"],
                            "cosine": similarity["cosine_similarity_matrix"]}
                per_fish[fid][side][population] = matrices
                response_vectors[fid][side][population] = selected
                selected_metadata[fid][side][population] = fish_metadata.loc[mask].copy()
                pair_rows.extend(_response_similarity_pair_rows(matrices, selected, fid, side, population))
                population_rows.append({
                    "fish_id": fid, "side": side, "population": population,
                    "n_kept_neurons": len(active), "n_population_neurons": int(mask.sum()),
                    "n_all_finite_response_neurons": int(np.isfinite(selected.to_numpy(dtype=float)).all(axis=1).sum()),
                    "n_rocking_variants": len(selection["groups"]["rocking"]),
                })
    pairs, counts = pd.DataFrame(pair_rows), pd.DataFrame(population_rows)
    aggregation = {metric: (metric, "mean") for metric in metrics}
    aggregation.update({f"{metric}_valid_fish": (metric, "count") for metric in metrics})
    summary = pairs.groupby(
        ["population", "side", "source_condition", "target_condition"], sort=False,
    ).agg(**aggregation).reset_index()
    means = {}
    for population in populations:
        means[population] = {}
        for side in ("left", "right"):
            rows = summary.loc[(summary["population"] == population) & (summary["side"] == side)]
            means[population][side] = {
                metric: rows.pivot(index="source_condition", columns="target_condition", values=metric)
                .reindex(index=labels, columns=labels) for metric in metrics
            }
    count_summary = counts.pivot(index=["fish_id", "side"], columns="population", values="n_population_neurons")
    count_summary = count_summary.reindex(
        index=pd.MultiIndex.from_product([fish_ids, ["left", "right"]], names=["fish_id", "side"]),
        columns=populations,
    ).rename_axis(columns=None).reset_index()
    return {
        "fish_ids": fish_ids, "condition_labels": labels, "populations": populations,
        "population_labels": {"all_kept": "All kept neurons", "all_rocking": "All-rocking responsive core"},
        "per_fish_matrices": per_fish, "mean_matrices": means,
        "response_vectors": response_vectors, "row_metadata": selected_metadata,
        "tables": {"response_similarity_pairs": pairs, "response_similarity_summary": summary,
                   "response_similarity_population_counts": counts,
                   "response_similarity_population_summary": count_summary},
    }


def _resolve_recruitment_groups(all_fish_data, fish_ids, side_groups, condition_labels):
    """Resolve mirrored, disjoint groups and check every fish's ID/name contract."""
    if set(side_groups) != {"left", "right"}:
        raise ValueError("side_groups must contain exactly left and right.")
    reference = all_fish_data[fish_ids[0]]
    resolved = {}
    all_labels = []
    group_sizes = None
    for side in ("left", "right"):
        groups = side_groups[side]
        if set(groups) != {"control", "rocking", "flicker"}:
            raise ValueError(f"{side}: configure control, rocking, and flicker groups.")
        names = [groups["control"], *groups["rocking"], *groups["flicker"]]
        sizes = (1, len(groups["rocking"]), len(groups["flicker"]))
        if min(sizes) < 1 or (group_sizes is not None and sizes != group_sizes):
            raise ValueError("Left/right groups must have matching, nonempty sizes.")
        group_sizes = sizes
        if len(names) != len(set(names)) or len(names) != len(condition_labels):
            raise ValueError("Each side needs unique stimuli matching condition_labels.")
        selection = resolve_selected_stimuli(
            names, reference["stimuli_id_map"], reference["trial_aligned_traces_z_core"].keys())
        ids = selection["stimulus_ids"]
        labels = selection["stimulus_labels"]
        for fid in fish_ids:
            fish = all_fish_data[fid]
            actual = resolve_selected_stimuli(
                names, fish["stimuli_id_map"], fish["trial_aligned_traces_z_core"].keys())
            if actual["stimulus_ids"] != ids or actual["stimulus_labels"] != labels:
                raise ValueError(f"{fid}: stimulus mapping differs from the reference fish.")
        resolved[side] = {
            **selection, "condition_labels": list(condition_labels),
            "groups": {"control": ids[:1], "rocking": ids[1:1 + sizes[1]],
                       "flicker": ids[1 + sizes[1]:]},
        }
        all_labels.extend(labels)
    if len(set(all_labels)) != len(all_labels):
        raise ValueError("Left and right stimulus groups must be disjoint.")
    return resolved, all_labels


def _check_recruitment_row_alignment(all_fish_data, fish_ids, stimulus_ids):
    """Require filtered raster rows to follow kept-neuron order and shared time/trial axes."""
    for fid in fish_ids:
        fish = all_fish_data[fid]
        kept = fish.get("kept_neuron_indices")
        if kept is not None:
            kept = np.asarray(kept)
            if kept.ndim != 1 or not kept.size or len(np.unique(kept)) != kept.size:
                raise ValueError(f"{fid}: kept_neuron_indices must be a nonempty unique vector.")
        for stimulus in stimulus_ids:
            z_key = stimulus if stimulus in fish["trial_aligned_traces_z_core"] else str(stimulus)
            r_key = stimulus if stimulus in fish["trial_aligned_traces_raster"] else str(stimulus)
            z = np.asarray(fish["trial_aligned_traces_z_core"][z_key])
            raster = np.asarray(fish["trial_aligned_traces_raster"][r_key])
            if z.ndim != 3 or raster.ndim != 3:
                raise ValueError(f"{fid}, stimulus {stimulus}: expected (neurons, time, trials).")
            n_kept = z.shape[0] if kept is None else kept.size
            if raster.shape[0] != n_kept or raster.shape[1:] != z.shape[1:]:
                raise ValueError(
                    f"{fid}, stimulus {stimulus}: raster {raster.shape} must have {n_kept} "
                    f"kept rows in kept-neuron order and time/trial axes {z.shape[1:]}.")


def _build_recruitment_activity_tables(all_fish_data, fish_ids, selections, responses, active, timing):
    """Summarize canonical signed AUCs, integration spans, and neuron identities."""
    neuron_tables, window_rows, summary_rows = [], [], []
    metadata = responses["row_metadata"]
    for fid in fish_ids:
        fish = all_fish_data[fid]
        fish_metadata = metadata.loc[metadata["fish_id"] == fid].reset_index(drop=True)
        n_neurons = len(fish_metadata)
        if not n_neurons or len(active[fid]) != n_neurons:
            raise ValueError(f"{fid}: active, response, and metadata rows must match and be nonempty.")
        for side, selection in selections.items():
            for index, (stimulus, label, condition) in enumerate(zip(
                selection["stimulus_ids"], selection["stimulus_labels"], selection["condition_labels"],
            )):
                key = stimulus if stimulus in fish["trial_aligned_traces_z_core"] else str(stimulus)
                trace = fish["trial_aligned_traces_z_core"][key]
                window = compute_response_window_frames(
                    n_time=trace.shape[1], stimulus=stimulus,
                    stimuli_durations=fish["stimuli_durations"],
                    stimuli_id_map=fish["stimuli_id_map"], **timing)
                times = window["time_s"][window["frame_indices"]]
                span = float(times[-1] - times[0]) if len(times) > 1 else 1 / timing["fps_2p"]
                raw_auc = responses["response_matrices"][fid][label].to_numpy(dtype=float)
                normalized = raw_auc / span
                decisions = active[fid][stimulus].to_numpy(dtype=int)
                info = {"fish_id": fid, "side": side, "stimulus_id": stimulus,
                        "stimulus": label, "condition": condition, "condition_index": index}
                rows = fish_metadata.copy()
                for column, value in info.items():
                    rows[column] = value
                rows["active"] = decisions
                rows["raw_auc"] = raw_auc
                rows["normalized_auc"] = normalized
                neuron_tables.append(rows)
                window_rows.append({
                    **info, "start_s": window["start_s"],
                    "requested_end_s": window["requested_end_s"], "observed_end_s": window["end_s"],
                    "requested_duration_s": window["requested_end_s"] - window["start_s"],
                    "observed_duration_s": window["end_s"] - window["start_s"],
                    "integration_span_s": span, "first_sample_s": float(times[0]),
                    "last_sample_s": float(times[-1]), "n_response_frames": window["n_frames"],
                    "n_repetitions": trace.shape[2],
                    "window_clipped": window["requested_end_s"] > window["end_s"] + 1e-9,
                })
                summary_rows.append({
                    **info, "n_neurons": n_neurons, "n_active": int(decisions.sum()),
                    "active_fraction": float(decisions.mean()),
                    "n_finite_responses": int(np.isfinite(raw_auc).sum()),
                    "mean_raw_auc": pd.Series(raw_auc).mean(),
                    "mean_normalized_auc": pd.Series(normalized).mean(),
                })
    return {"neuron_responses": pd.concat(neuron_tables, ignore_index=True),
            "response_windows": pd.DataFrame(window_rows),
            "recruitment": pd.DataFrame(summary_rows)}


def _build_recruitment_containment_tables(active, fish_ids, selections):
    """Prepare per-fish pair matrices and control/all-rocking containment counts."""
    pair_rows, summary_rows, per_fish = [], [], {}
    for fid in fish_ids:
        per_fish[fid] = {}
        for side, selection in selections.items():
            ids, labels = selection["stimulus_ids"], selection["condition_labels"]
            containment = compute_active_neuron_containment(active[fid], ids, labels)
            jaccard = compute_active_neuron_jaccard_overlap(active[fid], ids, labels)
            per_fish[fid][side] = {"containment": containment, "jaccard": jaccard}
            values = active[fid].loc[:, ids].to_numpy(dtype=bool)
            for i, source in enumerate(ids):
                for j, target in enumerate(ids):
                    pair_rows.append({
                        "fish_id": fid, "side": side, "source_id": source, "target_id": target,
                        "source_condition": labels[i], "target_condition": labels[j],
                        "n_source": int(values[:, i].sum()), "n_target": int(values[:, j].sum()),
                        "n_shared": int((values[:, i] & values[:, j]).sum()),
                        "n_union": int((values[:, i] | values[:, j]).sum()),
                        "containment": containment.iloc[i, j], "jaccard": jaccard.iloc[i, j],
                    })
            masks = build_active_group_masks(active[fid], selection["groups"])
            for index, (stimulus, label, condition) in enumerate(zip(
                ids, selection["stimulus_labels"], labels,
            )):
                if stimulus in selection["groups"]["control"]:
                    continue
                source = values[:, index]
                targets = {"to_control": masks["control"]["all"]}
                if stimulus in selection["groups"]["flicker"]:
                    targets["to_all_rocking"] = masks["rocking"]["all"]
                for summary_type, target in targets.items():
                    count = int(source.sum())
                    shared = int((source & target).sum())
                    fraction = shared / count if count else np.nan
                    summary_rows.append({
                        "fish_id": fid, "side": side, "stimulus_id": stimulus,
                        "stimulus": label, "condition": condition, "summary_type": summary_type,
                        "n_source": count, "n_contained": shared, "n_not_contained": count - shared,
                        "fraction_contained": fraction, "fraction_not_contained": 1 - fraction,
                    })
    return pd.DataFrame(pair_rows), pd.DataFrame(summary_rows), per_fish


def build_recruitment_containment_analysis(
    all_fish_data, fish_ids, side_groups, timing, active_settings, condition_labels,
):
    """Prepare mirrored recruitment, signed activity, and containment without plotting.

    Args:
        all_fish_data (dict): Aligned fish bundles. Continuous traces have shape
            (all neurons, time, trials); raster rows must already follow the
            unique kept_neuron_indices order, as in canonical detector caches.
        fish_ids (sequence): Nonempty, unique cohort order; each fish has equal
            weight in matrix summaries, independent of its neuron count.
        side_groups (dict): left/right mappings with control (one ID/name),
            rocking and flicker (nonempty ID/name lists). Groups are disjoint.
        timing (dict): fps_2p, t_pre_s, motion_onset_s, tau_s; seconds except fps.
            motion_onset_s=None uses per-stimulus static_before_sec; the tail
            after motion is 2 * tau_s seconds for both activity and AUC.
        active_settings (dict): Explicit canonical fraction/epoch/repetition rules.
        condition_labels (sequence): Unique common labels, ordered control,
            rocking variants, flicker variants; both sides have matching sizes.

    Returns:
        dict: Identity-aligned response/active matrices, per-fish and equal-fish
        mean matrices, resolved groups, and report-ready tables. Raw AUC is in
        z-score seconds; normalized AUC divides by the actual trapezoid span
        (one frame period for single-frame windows), preserving signed values.
        NaN responses remain missing with finite-response counts recorded;
        empty source populations yield NaN containment. Inputs are not mutated.
    """
    fish_ids, condition_labels = list(fish_ids), list(condition_labels)
    if not fish_ids or len(set(fish_ids)) != len(fish_ids):
        raise ValueError("fish_ids must be a nonempty unique cohort.")
    if not condition_labels or len(set(condition_labels)) != len(condition_labels):
        raise ValueError("condition_labels must be nonempty and unique.")
    selections, selected_labels = _resolve_recruitment_groups(
        all_fish_data, fish_ids, side_groups, condition_labels)
    selected_ids = [stimulus for selection in selections.values() for stimulus in selection["stimulus_ids"]]
    _check_recruitment_row_alignment(all_fish_data, fish_ids, selected_ids)
    scientific_timing = {key: timing[key] for key in ("fps_2p", "t_pre_s", "motion_onset_s", "tau_s")}
    required = {"active_fraction_threshold", "min_epoch_s", "min_active_reps",
                "expected_reps", "require_expected_reps"}
    if set(active_settings) != required:
        raise ValueError(f"active_settings must explicitly configure {sorted(required)}.")
    responses = build_zscore_response_matrices_all_fish(
        all_fish_data, selected_labels, fish_ids=fish_ids, **scientific_timing)
    active = build_active_neuron_matrices_all_fish(
        all_fish_data, fish_ids=fish_ids, stim_order=selected_ids, **scientific_timing, **active_settings)
    tables = _build_recruitment_activity_tables(
        all_fish_data, fish_ids, selections, responses, active, scientific_timing)
    pairs, summaries, per_fish = _build_recruitment_containment_tables(active, fish_ids, selections)
    tables.update(pairwise=pairs, containment_summary=summaries)
    matrix_summary = pairs.groupby(
        ["side", "source_condition", "target_condition"], sort=False,
    ).agg(containment=("containment", "mean"), jaccard=("jaccard", "mean"),
          containment_valid_fish=("containment", "count"), jaccard_valid_fish=("jaccard", "count")).reset_index()
    tables["matrix_summary"] = matrix_summary
    tables["window_summary"] = tables["response_windows"].groupby(
        ["side", "condition"], sort=False,
    ).agg(requested_duration_s=("requested_duration_s", "max"),
          observed_duration_s=("observed_duration_s", "min"),
          integration_span_s=("integration_span_s", "min"),
          fish_with_clipped_window=("window_clipped", "sum")).reset_index()
    means = {}
    for side in selections:
        rows = matrix_summary.loc[matrix_summary["side"] == side]
        means[side] = {metric: rows.pivot(
            index="source_condition", columns="target_condition", values=metric,
        ).reindex(index=condition_labels, columns=condition_labels) for metric in ("containment", "jaccard")}
    return {"fish_ids": fish_ids, "condition_labels": condition_labels,
            "side_selections": selections, "active_matrices": active, "responses": responses,
            "per_fish_matrices": per_fish, "mean_matrices": means, "tables": tables}


def _shared_auc_pair_table(fid, side, sid, control, label, control_label, kept,
                           decisions, measures):
    """Pair the same retained neurons using the existing binary decisions."""
    shared = build_active_group_masks(decisions, {"shared": [control, sid]})["shared"]["all"]
    indices = np.flatnonzero(shared)
    info = {"fish_id": fid, "side": side, "stimulus_id": sid, "condition": label,
            "control_id": control, "control": control_label}
    rows = pd.DataFrame({"neuron_id": indices, "source_neuron_index": kept[indices]})
    for name, value in info.items():
        rows[name] = value
    for kind, stimulus in (("control", control), ("stimulus", sid)):
        measure = measures[stimulus]
        for metric in ("raw_auc", "normalized_auc", "n_valid_trials"):
            rows[f"{kind}_{metric}"] = measure[metric][indices]
    finite = np.isfinite(rows[["control_raw_auc", "stimulus_raw_auc"]]).all(axis=1)
    rows["finite_pair"] = finite
    summary = {**info, "n_control_active": int(decisions[control].sum()),
               "n_stimulus_active": int(decisions[sid].sum()),
               "n_shared": len(rows), "n_finite_pairs": int(finite.sum())}
    for metric in ("raw_auc", "normalized_auc"):
        rows[f"delta_{metric}"] = rows[f"control_{metric}"] - rows[f"stimulus_{metric}"]
        rows.loc[~finite, f"delta_{metric}"] = np.nan
        for kind in ("control", "stimulus", "delta"):
            summary[f"mean_{kind}_{metric}"] = rows.loc[finite, f"{kind}_{metric}"].mean()
    return rows, summary


def build_shared_control_auc_analysis(all_fish_data, fish_ids, overlap_results,
                                      side_controls, timing, decay_s=5.0):
    """Compare control minus other-stimulus AUC within pairwise shared neurons.

    overlap_results is the exact section-6 plot_left_right_active_overlap_diagnostics
    output: use its unsorted decision_matrix and row_metadata, never reclassify
    neurons or threshold continuous z-scores. Full per-fish rows must match
    kept_neuron_indices in source order. Each comparison has its own intersection.
    Controls are explicit IDs/names per side. Measurement windows use each
    stimulus's canonical motion onset/end plus decay_s seconds, independently
    of the wider activity-selection windows. Shapes are (neurons, time, trials).

    Returns fish_ids, side_selections, and tables: shared_neuron_auc (paired
    source identities and signed raw/normalized AUC), shared_auc_per_fish
    (mean paired differences, shared/finite counts), shared_auc_summary
    (equally weighted fish means/counts), shared_auc_windows (actual/requested
    bounds, integration spans and clipping). Empty intersections/missing pairs
    remain NaN; fish with no finite pairs do not enter the corresponding mean.
    Trial responses average all fully finite trials, not only active trials.
    This is descriptive within the selected observed-trial population.
    """
    from src.response_metrics import compute_motion_decay_auc

    fish_ids = list(fish_ids)
    if not fish_ids or len(set(fish_ids)) != len(fish_ids):
        raise ValueError("Configure unique, nonempty fish_ids.")
    if set(overlap_results) != set(side_controls) or not overlap_results:
        raise ValueError("side_controls must match the nonempty overlap-result sides.")
    expected_rows = [(fid, index) for fid in fish_ids
                     for index in range(len(all_fish_data[fid]["kept_neuron_indices"]))]
    neurons, summaries, windows, selections = [], [], [], {}
    for side, result in overlap_results.items():
        diagnostic = result["active_trace_diagnostic"]
        metadata = diagnostic["row_metadata"]
        if list(metadata[["fish_id", "neuron_id"]].itertuples(index=False, name=None)) != expected_rows:
            raise ValueError(f"{side}: section-6 decision rows do not match current fish/kept-neuron order.")
        ids, labels = list(diagnostic["stim_order"]), list(diagnostic["stim_labels"])
        binary = np.asarray(diagnostic["decision_matrix"])
        if (binary.shape != (len(expected_rows), len(ids)) or not np.isin(binary, [0, 1]).all()
                or len(ids) != len(labels) or len(set(ids)) != len(ids)):
            raise ValueError(f"{side}: invalid section-6 binary decision columns/rows.")
        reference = all_fish_data[fish_ids[0]]
        control = resolve_selected_stimuli([side_controls[side]], reference["stimuli_id_map"], ids)["stimulus_ids"][0]
        comparisons = [sid for sid in ids if sid != control]
        if not comparisons:
            raise ValueError(f"{side}: at least one non-control comparison is required.")
        names = dict(zip(ids, labels))
        selections[side] = {"control_id": control, "control_label": names[control],
                            "stimulus_ids": comparisons, "condition_labels": [names[sid] for sid in comparisons]}
        offset = 0
        for fid in fish_ids:
            fish = all_fish_data[fid]
            kept = np.asarray(fish["kept_neuron_indices"])
            if (kept.ndim != 1 or not np.issubdtype(kept.dtype, np.integer)
                    or len(np.unique(kept)) != len(kept) or (kept < 0).any()):
                raise ValueError(f"{fid}: invalid kept-neuron source indices.")
            decisions = pd.DataFrame(binary[offset:offset + len(kept)], columns=ids)
            offset += len(kept)
            measures = {}
            for index, sid in enumerate(ids):
                label = names[sid]
                if fish["stimuli_id_map"].get(label) != sid:
                    raise ValueError(f"{fid}, {label}: stimulus identity differs from section 6.")
                traces = fish["trial_aligned_traces_z_core"]
                arr = np.asarray(traces[sid if sid in traces else str(sid)])
                if (arr.ndim != 3 or (kept.size and kept.max() >= len(arr))
                        or arr.shape[1] != diagnostic["trace_block_timepoints"][index]
                        or arr.shape[2] != diagnostic["trace_block_reps"][index]):
                    raise ValueError(f"{fid}, {label}: continuous/raster source rows or time/trial axes differ.")
                measures[sid] = compute_motion_decay_auc(arr[kept], fish["stimuli_durations"][label],
                    timing["fps_2p"], timing["t_pre_s"], decay_s=decay_s)
                windows.append({"fish_id": fid, "side": side, "stimulus_id": sid,
                                "stimulus": label, **measures[sid]["window"]})
            for sid in comparisons:
                rows, summary = _shared_auc_pair_table(fid, side, sid, control,
                    names[sid], names[control], kept, decisions, measures)
                neurons.append(rows)
                summaries.append(summary)
    per_fish = pd.DataFrame(summaries)
    group = ["side", "stimulus_id", "condition", "control_id", "control"]
    aggregations = {name: (name, "mean") for name in per_fish if name.startswith("mean_")}
    cohort = per_fish.groupby(group, sort=False).agg(
        n_fish=("mean_delta_raw_auc", "count"), n_shared_total=("n_shared", "sum"),
        n_shared_min=("n_shared", "min"), n_shared_max=("n_shared", "max"),
        n_finite_pairs_total=("n_finite_pairs", "sum"), **aggregations).reset_index()
    return {"fish_ids": fish_ids, "side_selections": selections, "decay_s": float(decay_s),
            "tables": {"shared_neuron_auc": pd.concat(neurons, ignore_index=True),
                       "shared_auc_per_fish": per_fish, "shared_auc_summary": cohort,
                       "shared_auc_windows": pd.DataFrame(windows)}}


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


def build_stimulus_panel_display(all_fish_data, fish_ids, stim_order, reference_fish,
                                 timing, raster_settings, neuron_order=None,
                                 trace_type="zscore"):
    """Prepare trial-mean traces for independent stimulus panels.

    Inputs follow build_all_fish_raster_inputs. Force mean repetitions for this
    display only. trace_type is zscore (default) or raster (saved significant
    activity, already in retained-neuron order); raster means are fractions of
    repetitions active, without thresholding again. Use a full pooled-row
    permutation if supplied, otherwise
    compute the configured order once across stim_order (both controls for the
    left/right index). All panels retain every kept neuron in the same order.

    Returns a dict with stimulus-keyed matrices (neurons, time), fish_means
    (fish, time), time_s from stimulus appearance, canonical phase intervals,
    original row identities, and a shared neuron_order. Trials, per-condition
    shapes, maps and phase timings must match across fish; values are not
    normalized again. Means average trials per neuron, then neurons per fish;
    pooled_means averages all retained neurons with equal neuron weights.
    """
    from src.neuron_ordering import compute_single_sort_order
    from src.stimuli_timeline import stimulus_phase_intervals

    fish_ids, stim_order = list(fish_ids), list(stim_order)
    if trace_type not in {"zscore", "raster"}:
        raise ValueError("trace_type must be 'zscore' or 'raster'.")
    trial_key = ("trial_aligned_traces_z_core" if trace_type == "zscore"
                 else "trial_aligned_traces_raster")
    if not fish_ids or len(set(fish_ids)) != len(fish_ids):
        raise ValueError("Stimulus panels require unique, nonempty fish_ids.")
    selection = resolve_selected_stimuli(stim_order, reference_fish["stimuli_id_map"],
        reference_fish[trial_key].keys())
    ids, labels = selection["stimulus_ids"], selection["stimulus_labels"]
    fps, pre = float(timing["fps_2p"]), float(timing["t_pre_s"])
    if not np.isfinite([fps, pre]).all() or fps <= 0 or pre < 0:
        raise ValueError("Stimulus panels require positive fps_2p and nonnegative t_pre_s.")
    shapes, phases = {}, {}
    for sid, label in zip(ids, labels):
        ref = np.asarray(reference_fish[trial_key][sid])
        shapes[sid] = ref.shape[1:]
        phases[sid] = stimulus_phase_intervals(reference_fish["stimuli_durations"][label])
        for fid in fish_ids:
            fish = all_fish_data[fid]
            arr = np.asarray(fish[trial_key][sid])
            if (arr.ndim != 3 or arr.shape[1:] != shapes[sid]
                    or min(arr.shape[1:]) == 0 or fish["stimuli_id_map"].get(label) != sid):
                raise ValueError(f"{fid}, {label}: incompatible stimulus identity or time/trial shape.")
            if stimulus_phase_intervals(fish["stimuli_durations"][label]) != phases[sid]:
                raise ValueError(f"{fid}, {label}: inconsistent stimulus phase timings.")
            kept = np.asarray(fish["kept_neuron_indices"])
            if (kept.ndim != 1 or not kept.size or not np.issubdtype(kept.dtype, np.integer)
                    or len(np.unique(kept)) != len(kept) or kept.min() < 0
                    or (trace_type == "zscore" and kept.max() >= len(arr))):
                raise ValueError(f"{fid}: invalid or empty kept-neuron indices.")
            if trace_type == "raster" and len(arr) != len(kept):
                raise ValueError(f"{fid}, {label}: significant raster rows must match retained-neuron order.")
    config = {**raster_settings, "combine_mode": "mean", "trace_type": trace_type}
    prepared = build_all_fish_raster_inputs(
        all_fish_data, fish_ids, ids, reference_fish, timing, config)
    matrix = prepared["flat_matrix_all_fish"]
    order = neuron_order
    if order is None:
        order = prepared["raster_neuron_order"]
        if order is None:
            order = compute_single_sort_order(matrix.T, config["raster_sort_mode"], n_clusters=8, random_state=0)
    order = np.arange(len(matrix)) if order is None else np.asarray(order)
    if (order.shape != (len(matrix),) or not np.issubdtype(order.dtype, np.integer)
            or not np.array_equal(np.sort(order), np.arange(len(matrix)))):
        raise ValueError("neuron_order must contain every pooled neuron exactly once.")
    rows = pd.DataFrame([(fid, index, int(source)) for fid in fish_ids
        for index, source in enumerate(all_fish_data[fid]["kept_neuron_indices"])],
        columns=["fish_id", "neuron_id", "source_neuron_index"])
    rows["global_neuron_id"] = np.arange(len(rows))
    offsets = np.cumsum([0] + [shapes[sid][0] for sid in ids])
    matrices = {sid: matrix[:, offsets[i]:offsets[i + 1]] for i, sid in enumerate(ids)}
    fish_offsets = np.cumsum([0] + [len(all_fish_data[fid]["kept_neuron_indices"]) for fid in fish_ids])
    return {
        "fish_ids": fish_ids, "stimulus_ids": ids, "stimulus_labels": dict(zip(ids, labels)),
        "matrices": matrices,
        "fish_means": {sid: np.stack([np.nanmean(values[fish_offsets[i]:fish_offsets[i + 1]], axis=0)
            for i in range(len(fish_ids))]) for sid, values in matrices.items()},
        "pooled_means": {sid: np.nanmean(values, axis=0) for sid, values in matrices.items()},
        "trace_type": trace_type,
        "time_s": {sid: np.arange(shapes[sid][0]) / fps - pre for sid in ids},
        "phases": phases, "repetitions": {sid: shapes[sid][1] for sid in ids},
        "fps_2p": fps, "neuron_order": order, "row_metadata": rows,
        "displayed_row_metadata": rows.iloc[order].reset_index(drop=True),
        "sort_mode": config["raster_sort_mode"],
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


def build_left_right_active_overlap_diagnostics(
    all_fish_data, fish_ids, reference_fish, timing, side_stimuli,
    active_fraction_threshold=0.10, min_epoch_s=1.0, min_active_reps=2,
    expected_reps=4, require_expected_reps=True, motion_onset_s=8.0,
    tau_s=6.0, motion_duration_key="motion_sec",
):
    """Prepare both Cell 06 diagnostics and their exact row orders once.

    Fish/stimulus order, activity rules and seconds-based timing match
    plot_left_right_active_overlap_diagnostics. Input rasters are retained
    neurons by time by trials; continuous traces may contain all source rows.
    Return left/right overlap tables, pooled binary diagnostics, the original
    decision-then-mean permutations, and source_row_metadata with fish_id,
    neuron_id (within retained rows) and source_neuron_index. The latter makes
    reuse in z-score panels checkable against their retained source identities.
    Missing kept_neuron_indices implies the original raster row indices.
    No plotting, saving or printing; inputs remain unchanged.
    """
    expected_sides = {"left", "right"}
    if set(side_stimuli) != expected_sides:
        raise ValueError("side_stimuli must contain exactly 'left' and 'right'.")

    selections = {
        side: resolve_selected_stimuli(
            stimuli,
            stimuli_id_map=reference_fish["stimuli_id_map"],
            available_stimuli=reference_fish["trial_aligned_traces_raster"].keys(),
        )
        for side, stimuli in side_stimuli.items()
    }
    side_stimulus_ids = {side: selection["stimulus_ids"] for side, selection in selections.items()}
    side_stimulus_labels = {side: selection["stimulus_labels"] for side, selection in selections.items()}
    active_stim_order = list(dict.fromkeys(side_stimulus_ids["left"] + side_stimulus_ids["right"]))
    active_matrices = build_active_neuron_matrices_all_fish(
        all_fish_data=all_fish_data,
        fish_ids=fish_ids,
        stim_order=active_stim_order,
        fps_2p=timing["fps_2p"],
        t_pre_s=timing["t_pre_s"],
        motion_onset_s=motion_onset_s,
        active_fraction_threshold=active_fraction_threshold,
        min_epoch_s=min_epoch_s,
        min_active_reps=min_active_reps,
        expected_reps=expected_reps,
        require_expected_reps=require_expected_reps,
        tau_s=tau_s,
        motion_duration_key=motion_duration_key,
    )

    source_rows = []
    for fid in fish_ids:
        kept = all_fish_data[fid].get("kept_neuron_indices")
        if kept is None:
            kept = np.arange(len(active_matrices[fid]))
        source_rows.extend((fid, index, int(source)) for index, source in enumerate(kept))
    source_rows = pd.DataFrame(source_rows,
        columns=["fish_id", "neuron_id", "source_neuron_index"])
    results = {}
    for side in ("left", "right"):
        diagnostic_data = build_overlap_diagnostic_data(
            all_fish_data=all_fish_data,
            active_matrices=active_matrices,
            fish_ids=fish_ids,
            side_stimulus_ids=side_stimulus_ids,
            side_stimulus_labels=side_stimulus_labels,
            response_row_metadata=None,
            keep_mask=None,
            side_to_plot=side,
            aggregation="mean_per_fish",
            plot_mode="significant_raster",
            combine_mode="mean",
            sort_mode="decision_then_mean",
        )
        diagnostic_data["source_row_metadata"] = source_rows.copy()
        results[side] = diagnostic_data

    return results


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

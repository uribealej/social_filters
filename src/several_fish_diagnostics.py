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


def _phase_trial_measurements(fid, fish, settings):
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
            pre_indices = select_event_window(times, bout - settings["pre_bout_s"], bout)["frame_indices"]
            level = compute_explicit_window_metrics(values, pre_indices, times, fish["fps_2p"])["mean_z"]
            windows = {label: (bout + bounds[0], bout + bounds[1]) for label, bounds in settings["bout_windows"].items()}
            windows.update(preconditioning=(0, pre_end), preconditioning_early=(0, pre_end / 2),
                           preconditioning_late=(pre_end / 2, pre_end),
                           pre_bout=(bout - settings["pre_bout_s"], bout))
            for label, (start, stop) in windows.items():
                window = select_event_window(times, start, stop)
                indices = window.pop("frame_indices")
                metrics = compute_explicit_window_metrics(values, indices, times, fish["fps_2p"],
                    baseline=level if label in settings["bout_windows"] else None)
                active = classify_trial_raster_activity(binary[:, indices, :], fish["fps_2p"],
                    settings["active_fraction_threshold"], settings["min_epoch_s"])
                frame = pd.DataFrame({key: val[:, 0] for key, val in metrics.items()})
                frame["neuron_id"] = fish["kept_neuron_indices"]
                frame["active_trial"] = active[:, 0]
                frame["measurement_valid"] = np.isfinite(metrics["auc_z_s"][:, 0])
                frame["pre_bout_level_z"] = level[:, 0]
                frame["fish_id"], frame["condition"], frame["stimulus_key"] = fid, trial.logged_name, trial.stimulus_key
                frame["side"], frame["window"], frame["block"] = trial.hemifield, label, trial.block
                frame["repetition"], frame["trial_index"] = rep, trial.trial_index
                frame["applicable"] = stop > start
                frame["delay_s"] = trial.bout_onset_s - trial.pre_end_s
                frames.append(frame)
                rows.append(dict(fish_id=fid, condition=trial.logged_name, side=trial.hemifield,
                    window=label, block=trial.block, repetition=rep, applicable=stop > start, **window))
    return pd.concat(frames, ignore_index=True), pd.DataFrame(rows)


def build_phase_population_analysis(cohort, settings, comparison_pairs):
    """Build trial/neuron/fish descriptive tables from reviewed phase results.

    Reuses explicit-window AUC/peak owners and the canonical activity rule.
    Every selected neuron remains in every condition's denominator. An active
    fraction with incomplete activity classifications is missing, with counts
    retained. Measurements use native recorded-event-relative sample times.
    """
    from scipy.stats import spearmanr

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
    measurements = ["auc_z_s", "peak_z", "mean_z", "delta_auc_z_s", "delta_peak_z", "delta_mean_z", "pre_bout_level_z"]
    grouped = trial.groupby(keys, sort=False, dropna=False)
    neuron = grouped[measurements].mean().reset_index()
    counts = grouped.agg(valid_measurement_trials=("measurement_valid", "sum"),
        valid_activity_trials=("active_trial", "count"), active_trials=("active_trial", lambda x: x.sum(min_count=1)),
        applicable=("applicable", "first"), delay_s=("delay_s", "first")).reset_index()
    neuron = neuron.merge(counts, on=keys, validate="one_to_one")
    neuron["active"] = np.where(neuron.valid_activity_trials.eq(settings["expected_reps"]),
        (neuron.active_trials >= settings["min_active_reps"]).astype(float), np.nan)
    identity = ["fish_id", "side", "condition", "neuron_id"]
    early = neuron.loc[neuron.window.eq("preconditioning_early"), identity + ["mean_z"]].rename(columns={"mean_z": "early_z"})
    late = neuron.loc[neuron.window.eq("preconditioning_late"), identity + ["mean_z"]].rename(columns={"mean_z": "late_z"})
    changes = early.merge(late, on=identity, validate="one_to_one")
    changes["early_late_change_z"] = changes.late_z - changes.early_z
    neuron = neuron.merge(changes[identity + ["early_late_change_z"]], on=identity, validate="many_to_one")
    fish_keys = ["fish_id", "side", "condition", "window"]
    fish_summary = neuron.groupby(fish_keys, sort=False)[measurements + ["early_late_change_z"]].mean().reset_index()
    activity = neuron.groupby(fish_keys, sort=False).agg(n_neurons=("neuron_id", "size"),
        n_valid_active=("active", "count"), n_valid_auc=("auc_z_s", "count"),
        active_fraction=("active", lambda x: x.mean() if x.notna().all() else np.nan),
        mean_active_trials=("active_trials", "mean"), delay_s=("delay_s", "first"),
        applicable=("applicable", "first")).reset_index()
    fish_summary = fish_summary.merge(activity, on=fish_keys, validate="one_to_one")
    relationships, correlations, memberships = [], [], []
    pre = neuron.loc[neuron.window.eq("preconditioning") & neuron.applicable,
        identity + ["auc_z_s", "active"]].rename(columns={"auc_z_s": "preconditioning_auc", "active": "preconditioning_active"})
    for label in settings["bout_windows"]:
        bout = neuron.loc[neuron.window.eq(label), identity + ["auc_z_s", "active"]].rename(
            columns={"auc_z_s": "bout_auc", "active": "bout_active"})
        pairs = pre.merge(bout, on=identity, validate="one_to_one")
        pairs["window"] = label
        relationships.append(pairs)
        for key, group in pairs.groupby(["fish_id", "side", "condition"], sort=False):
            valid = group[["preconditioning_auc", "bout_auc"]].dropna()
            rho = spearmanr(valid.preconditioning_auc, valid.bout_auc).statistic if (
                len(valid) >= 3 and valid.preconditioning_auc.nunique() > 1 and valid.bout_auc.nunique() > 1) else np.nan
            correlations.append(dict(zip(["fish_id", "side", "condition"], key), window=label,
                                     spearman_rho=rho, n_valid_pairs=len(valid)))
            known = group.preconditioning_active.notna() & group.bout_active.notna()
            for category, a, b in [("preconditioning_only", 1, 0), ("bout_only", 0, 1), ("both", 1, 1), ("neither", 0, 0)]:
                n = ((group.preconditioning_active == a) & (group.bout_active == b)).sum()
                memberships.append(dict(zip(["fish_id", "side", "condition"], key), window=label,
                    category=category, n_neurons=len(group), n_valid=int(known.sum()),
                    fraction=float(n / len(group)) if known.all() else np.nan))
    differences = []
    for name, (condition, reference) in comparison_pairs.items():
        left = fish_summary.loc[fish_summary.condition.eq(condition)]
        right = fish_summary.loc[fish_summary.condition.eq(reference)]
        pairs = left.merge(right, on=["fish_id", "side", "window"], suffixes=("", "_reference"), validate="one_to_one")
        for metric in measurements + ["active_fraction"]:
            differences.append(pd.DataFrame(dict(fish_id=pairs.fish_id, side=pairs.side,
                window=pairs.window, comparison=name, condition=condition, reference=reference,
                metric=metric, difference=pairs[metric] - pairs[metric + "_reference"])))
    return dict(trial_metrics=trial, neuron_metrics=neuron, fish_summary=fish_summary,
        window_coverage=pd.concat(window_tables, ignore_index=True),
        phase_relationships=pd.concat(relationships, ignore_index=True),
        correlations=pd.DataFrame(correlations), phase_memberships=pd.DataFrame(memberships),
        paired_differences=pd.concat(differences, ignore_index=True) if differences else pd.DataFrame())


def _phase_recruitment_decisions(trial_metrics, conditions, repetitions, min_active_reps, bout_window):
    """Aggregate saved native-window decisions over selected one-based trials."""
    keys = ["fish_id", "neuron_id", "condition", "window"]
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


def _phase_recruitment_display(cohort, neurons, pre_pool, side, conditions, repetitions, bout_bounds):
    """Prepare grayscale display means and membership groups, without interpolation."""
    pre_parts, bout_parts, metadata, codes, pre_codes = [], {c: [] for c in conditions}, [], {}, []
    pooled_codes, decisions = [], []
    indices = np.asarray(repetitions) - 1
    axes = {}
    for fid in cohort["fish_ids"]:
        fish = cohort["all_fish_data"][fid]
        ids = fish["kept_neuron_indices"]
        lookup = neurons.loc[neurons.fish_id.eq(fid) & neurons.side.eq(side)]
        members = {c: lookup.loc[lookup.condition.eq(c)].set_index("neuron_id").reindex(ids)
                   for c in conditions}
        pre_conditions = []
        for condition in conditions:
            trials = fish["trials"].loc[fish["trials"].logged_name.eq(condition)]
            if len(trials) < max(repetitions):
                raise ValueError(f"{fid}/{condition}: selected repetitions unavailable.")
            sid = int(trials.stimulus_id.iloc[0])
            aligned = fish["aligned"]["bout"]["raster"]
            times = aligned["time_s"][sid]
            mask = (times >= bout_bounds[0]) & (times < bout_bounds[1])
            if not mask.any():
                raise ValueError("Bout display window has no samples.")
            values = aligned["trial_aligned_traces"][sid][:, mask, :][:, :, indices]
            if condition in axes and not np.array_equal(axes[condition], times[mask]):
                raise ValueError("Bout display axes differ between fish.")
            axes[condition] = times[mask]
            bout_parts[condition].append(np.mean(values, axis=2))
            if condition != conditions[0]:
                aligned = fish["aligned"]["sequence"]["raster"]
                durations = trials.pre_end_s.to_numpy() - trials.onset_s.to_numpy()
                if not np.allclose(durations, durations[0]) or durations[0] <= 0:
                    raise ValueError("Preconditioning requires a consistent positive duration.")
                times = aligned["time_s"][sid]
                mask = (times >= 0) & (times < durations[0])
                if "preconditioning" in axes and not np.array_equal(axes["preconditioning"], times[mask]):
                    raise ValueError("Cannot average different preconditioning display axes.")
                axes["preconditioning"] = times[mask]
                values = aligned["trial_aligned_traces"][sid][:, mask, :][:, :, indices]
                pre_conditions.append(np.mean(values, axis=2))
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
    """
    repetitions = list(repetitions)
    if (not repetitions or any(isinstance(r, (bool, np.bool_)) or not isinstance(r, (int, np.integer))
            or r < 1 for r in repetitions) or len(set(repetitions)) != len(repetitions)):
        raise ValueError("Use unique positive one-based repetition numbers.")
    if not 1 <= min_active_reps <= len(repetitions):
        raise ValueError("Invalid minimum active repetitions.")
    if not np.isfinite(pre_active_fraction) or not 0 < pre_active_fraction <= 1:
        raise ValueError("pre_active_fraction must be in (0, 1].")
    bounds = np.asarray(bout_bounds, dtype=float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[1] <= bounds[0]:
        raise ValueError("bout_bounds must be finite increasing seconds.")
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

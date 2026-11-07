"""Shared population PCA and native-time, within-fish condition comparisons."""

from itertools import combinations, product

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.trial_alignment import resolve_selected_stimuli, select_event_window
from src.stimulus_similarity import compute_population_pattern_correlation


def _crossvalidated_squared_distance(differences):
    """Return trial-pair cross-products per neuron; input (neurons,time,trials)."""
    n_neurons, _, n_trials = differences.shape
    return (np.square(differences.sum(axis=2)).sum(axis=0)
            - np.square(differences).sum(axis=(0, 2))) / (
                n_neurons * n_trials * (n_trials - 1))


def _pattern_correlation(a, b):
    return compute_population_pattern_correlation(a, b)


def _finite_average(values):
    values = np.asarray(values, dtype=float)
    counts = np.isfinite(values).sum(axis=0)
    return np.divide(np.nansum(values, axis=0), counts,
                     out=np.full(values.shape[1:], np.nan), where=counts > 0), counts


def _balanced_half_splits(n_reps):
    if n_reps < 2 or n_reps % 2 or n_reps > 10:
        raise ValueError('Split-half similarity requires an even repetition count from 2 through 10.')
    trials = set(range(n_reps))
    return [(list(first), sorted(trials - set(first)))
            for first in combinations(range(n_reps), n_reps // 2) if 0 in first]


def _same_time_similarity(stimulus, control, estimator):
    if estimator == 'trial_mean':
        return compute_population_pattern_correlation(stimulus.mean(axis=2), control.mean(axis=2))
    curves = []
    for first, second in _balanced_half_splits(control.shape[2]):
        for a, b in ((first, second), (second, first)):
            curves.append(compute_population_pattern_correlation(
                stimulus[:, :, a].mean(axis=2), control[:, :, b].mean(axis=2)))
    return _finite_average(curves)[0]


def build_control_pattern_similarity(analysis, controls, *, time_window_s=(0, 14),
                                     estimator='split_half', include_repeatability=True):
    """Build same-time population-pattern correlation curves versus Control.

    Use unscaled selected (neurons,time,trials) responses from the prepared analysis,
    preserving within-fish neuron identity. Display-window endpoints are inclusive
    native centers. Compute Pearson across neurons, then average fish equally
    (arithmetic r, no Fisher transform). Constant vectors are NaN, with fish counts
    returned per time sample; no neuron-as-replicate statistics.

    ``split_half`` averages curves across all balanced repetition splits and both
    directions. For four blocks, three unique 2+2 splits yield six curves. Every
    comparison uses independent-block two-trial means. The optional Control-vs-
    Control reference uses this same estimator/sample count; it is descriptive
    repeatability, not a formal ceiling or equivalence threshold. ``trial_mean``
    correlates all-trial means; its reference is omitted because a two-trial
    reference would not have matched precision. Both modes use the full neuron
    space without PCA, smoothing, rectification or temporal warping.

    Return fish_time_table, curve_summary (mean_r,sem_r,n_fish),
    repeatability_fish/summary, controls, estimator and time_s[side].
    """
    if (estimator not in {'split_half', 'trial_mean'} or not np.isfinite(time_window_s).all()
            or len(time_window_s) != 2 or time_window_s[0] >= time_window_s[1]):
        raise ValueError('Invalid similarity estimator or inclusive time window.')
    rows, reference_rows, axes = [], [], {}
    for side, config in analysis['side_conditions'].items():
        times = analysis['metrics'].loc[analysis['metrics'].side.eq(side), 'time_s'].drop_duplicates().to_numpy()
        keep = (times >= time_window_s[0]) & (times <= time_window_s[1])
        if (keep.sum() < 2 or time_window_s[0] < times[0] or time_window_s[1] > times[-1]):
            raise ValueError(f'{side}: similarity window needs at least two samples and complete native coverage.')
        axes[side] = times[keep]
        control_label = controls[side]
        if control_label not in config['stimuli']:
            raise ValueError(f'{side}: Control is absent from prepared conditions.')
        for fid in analysis['fish_ids']:
            arrays = analysis['responses'][side][fid]
            control = arrays[control_label][:, keep, :]
            if include_repeatability and estimator == 'split_half':
                reference = _same_time_similarity(control, control, 'split_half')
                reference_rows.append(pd.DataFrame(dict(side=side, fish_id=fid, control=control_label,
                    time_s=axes[side], correlation=reference)))
            for stimulus in config['stimuli']:
                if stimulus == control_label:
                    continue
                values = arrays[stimulus][:, keep, :]
                if values.shape != control.shape:
                    raise ValueError(f'{side}/{fid}/{stimulus}: response axes must match Control.')
                correlation = _same_time_similarity(values, control, estimator)
                rows.append(pd.DataFrame(dict(side=side, fish_id=fid, control=control_label,
                    stimulus=stimulus, time_s=axes[side], correlation=correlation, n_neurons=len(values))))
    fish_time = pd.concat(rows, ignore_index=True)
    curves = fish_time.groupby(['side', 'control', 'stimulus', 'time_s'], sort=False).correlation.agg(
        mean_r='mean', sem_r='sem', n_fish='count').reset_index()
    reference_fish = (pd.concat(reference_rows, ignore_index=True) if reference_rows else
                      pd.DataFrame(columns=['side', 'fish_id', 'control', 'time_s', 'correlation']))
    reference_summary = reference_fish.groupby(['side', 'control', 'time_s'], sort=False).correlation.agg(
        mean_r='mean', sem_r='sem', n_fish='count').reset_index()
    return dict(fish_time_table=fish_time, curve_summary=curves, time_s=axes,
                repeatability_fish=reference_fish, repeatability_summary=reference_summary,
                controls=dict(controls), estimator=estimator, time_window_s=tuple(time_window_s))


def _prepare_fish(fish, labels, fps, pre_s, anchor_s, window_s, expected_reps):
    selection = resolve_selected_stimuli(labels, stimuli_id_map=fish['stimuli_id_map'],
        available_stimuli=fish['trial_aligned_traces_z_core'].keys())
    kept = np.asarray(fish['kept_neuron_indices'])
    if (kept.ndim != 1 or kept.dtype.kind not in 'iu' or kept.size == 0
            or len(np.unique(kept)) != len(kept)):
        raise ValueError('Require nonempty, unique integer retained-neuron indices.')
    arrays, indices, times, shape = {}, None, None, None
    for label, sid in zip(labels, selection['stimulus_ids']):
        traces = fish['trial_aligned_traces_z_core']
        arr = np.asarray(traces[sid] if sid in traces else traces[str(sid)], dtype=float)
        if arr.ndim != 3 or arr.shape[2] != expected_reps:
            raise ValueError(f'{label}: expected (neurons,time,{expected_reps}), got {arr.shape}.')
        if kept.min() < 0 or kept.max() >= arr.shape[0]:
            raise ValueError(f'{label}: retained-neuron indices outside trace axis.')
        if shape is not None and arr.shape != shape:
            raise ValueError('All conditions must share neuron/time/trial axes.')
        shape = arr.shape
        times = (np.arange(arr.shape[1]) - int(round(pre_s * fps))) / fps - anchor_s
        if window_s[0] < times[0] - 1e-8 or window_s[1] > times[-1] + 1 / fps + 1e-8:
            raise ValueError(f'Requested window {window_s} exceeds aligned trace coverage.')
        indices = select_event_window(times, *window_s)['frame_indices']
        arrays[label] = arr[kept][:, indices, :]
    valid = np.logical_and.reduce([np.isfinite(a).all(axis=(1, 2)) for a in arrays.values()])
    if not valid.any():
        raise ValueError('No retained neuron has finite samples in every selected condition/trial.')
    return {label: a[valid] for label, a in arrays.items()}, times[indices], kept[valid], kept[~valid]


def _fit_pca(matrix, n_components, context):
    if min(matrix.shape) < n_components or not np.any(np.var(matrix, axis=0) > 0):
        raise ValueError(f'{context}: insufficient finite neurons/samples or no variance for PCA.')
    return PCA(n_components=n_components, svd_solver='full').fit(matrix)


def _pool_condition_means(responses, times, labels, fish_ids, neuron_ids,
                          fit_window_s, n_components):
    """Concatenate distinct neuron features with count-balanced fish weights."""
    blocks, metadata = {label: [] for label in labels}, []
    for fid in fish_ids:
        ids = neuron_ids[fid]
        weight = 1 / np.sqrt(len(fish_ids) * len(ids))
        for label in labels:
            blocks[label].append(responses[fid][label].mean(axis=2).T * weight)
        metadata.extend(dict(fish_id=fid, neuron_id=int(neuron_id), feature_weight=weight)
                        for neuron_id in ids)
    means = {label: np.concatenate(blocks[label], axis=1) for label in labels}
    indices = select_event_window(times, *fit_window_s)['frame_indices']
    model = _fit_pca(np.concatenate([means[label][indices] for label in labels]),
                     n_components, 'Pooled population')
    return dict(time_s=times, conditions=labels, pca=model,
                explained_variance_ratio=model.explained_variance_ratio_,
                means={label: model.transform(means[label]) for label in labels},
                neuron_metadata=pd.DataFrame(metadata), fit_window_s=tuple(fit_window_s))


def build_population_trajectory_analysis(all_fish_data, fish_ids, side_conditions, *,
        fps_2p, t_pre_s, program_onset_s, window_s=(-2.0, 14.0),
        fit_window_s=(0.0, 9.0), expected_reps=4, n_components=3, pca_scope='per_fish'):
    """Fit one shared PCA per fish/side and compare conditions at matching times.

    Fish bundles use the loader's (neurons,time,trials) z-score arrays and original
    ``kept_neuron_indices``. ``side_conditions`` maps side -> ``stimuli`` (ordered
    labels), ``pairs`` (label pairs). Seconds are relative to the common program
    anchor; windows are half-open and select native samples without interpolation.
    Baseline z-scores are preserved: no per-condition scaling, rectification,
    smoothing, whitening, or per-trial baseline subtraction. Nonfinite neurons
    are excluded jointly across all selected samples/conditions/trials, and audited.

    PCA fits equally sampled condition means on ``fit_window_s``, centered across
    all fitted rows. Return ``fits[side][fish]`` with shared loadings, explained
    variance, means (time,PC), trials (time,trials,PC), and original neuron IDs.
    ``metrics`` is a long fish/pair/time table: RMS distance in baseline-z units,
    crossvalidated squared distance in baseline-z squared per neuron, and Pearson
    correlation across neurons (NaN for a constant population vector). The squared
    estimate averages products of differences from distinct repetition blocks;
    negative estimates are retained, never square-rooted or clipped. Repetition
    indices must correspond to the same acquisition blocks for all conditions.
    PCA is descriptive; all distance calculations use the full neuron space.

    ``pca_scope`` is 'per_fish' (historical default), 'pooled', or 'both'. Pooled
    PCA uses condition means as separate neuron features across fish, weighted
    1/sqrt(number_of_fish * neurons_in_fish). This balances neuron counts, not
    biological response energy. Return ``pooled_fits[side]`` with (time,PC) paths
    and ordered fish/original-neuron metadata. No cross-fish single trials are
    fabricated. ``responses[side][fish][label]`` retains selected unscaled
    (neurons,time,trials) arrays for contrast and subsampling diagnostics.
    """
    fish_ids = list(fish_ids)
    if not fish_ids or len(set(fish_ids)) != len(fish_ids):
        raise ValueError('Require nonempty unique fish IDs.')
    if (not np.isfinite([fps_2p, t_pre_s, program_onset_s, *window_s, *fit_window_s]).all()
            or fps_2p <= 0 or t_pre_s < 0 or window_s[1] <= window_s[0]
            or fit_window_s[1] <= fit_window_s[0] or fit_window_s[0] < window_s[0]
            or fit_window_s[1] > window_s[1] or expected_reps < 2
            or int(expected_reps) != expected_reps or n_components < 2
            or int(n_components) != n_components or not side_conditions
            or pca_scope not in {'per_fish', 'pooled', 'both'}):
        raise ValueError('Invalid rate, windows, repetition count, component count, or sides.')
    fits, rows, audits, differences, responses, pooled_fits = {}, [], [], {}, {}, {}
    for side, config in side_conditions.items():
        labels = list(config['stimuli'])
        pairs = [tuple(pair) for pair in config['pairs']]
        if (not labels or len(set(labels)) != len(labels) or not pairs
                or len(set(pairs)) != len(pairs)
                or any(len(p) != 2 or p[0] == p[1] or not set(p) <= set(labels) for p in pairs)):
            raise ValueError(f'{side}: invalid ordered conditions or comparison pairs.')
        fits[side], responses[side], side_ids = {}, {}, {}
        side_times = None
        for fid in fish_ids:
            arrays, times, neuron_ids, excluded = _prepare_fish(all_fish_data[fid], labels,
                fps_2p, t_pre_s, program_onset_s, window_s, expected_reps)
            if side_times is not None and not np.array_equal(side_times, times):
                raise ValueError(f'{side}: fish must share the same native time samples.')
            side_times = times
            responses[side][fid], side_ids[fid] = arrays, neuron_ids
            fit_idx = select_event_window(times, *fit_window_s)['frame_indices']
            means = {label: arr.mean(axis=2).T for label, arr in arrays.items()}
            matrix = np.concatenate([means[label][fit_idx] for label in labels])
            model = None
            if pca_scope in {'per_fish', 'both'}:
                model = _fit_pca(matrix, n_components, f'{side}/{fid}')
                projected_trials = {label: model.transform(arr.transpose(1, 2, 0).reshape(-1, len(neuron_ids)))
                    .reshape(len(times), expected_reps, n_components) for label, arr in arrays.items()}
                fits[side][fid] = dict(time_s=times, neuron_ids=neuron_ids, excluded_neuron_ids=excluded,
                    conditions=labels, pca=model, explained_variance_ratio=model.explained_variance_ratio_,
                    means={label: model.transform(means[label]) for label in labels}, trials=projected_trials)
            audits.append(dict(side=side, fish_id=fid, n_neurons=len(neuron_ids),
                n_excluded_nonfinite=len(excluded), repetitions=expected_reps,
                first_sample_s=times[0], last_sample_s=times[-1], fit_samples_per_condition=len(fit_idx),
                variance_pc1=model.explained_variance_ratio_[0] if model is not None else np.nan,
                variance_shown=model.explained_variance_ratio_.sum() if model is not None else np.nan))
            for a, b in pairs:
                delta = arrays[a] - arrays[b]
                differences[(side, a, b, fid)] = delta
                rms = np.sqrt(np.mean(np.square(delta.mean(axis=2)), axis=0))
                cv = _crossvalidated_squared_distance(delta)
                corr = _pattern_correlation(means[a].T, means[b].T)
                rows.append(pd.DataFrame(dict(side=side, fish_id=fid, condition_a=a, condition_b=b,
                    time_s=times, rms_distance=rms, cv_squared_distance=cv, pattern_correlation=corr)))
        if pca_scope in {'pooled', 'both'}:
            pooled_fits[side] = _pool_condition_means(responses[side], side_times, labels,
                fish_ids, side_ids, fit_window_s, n_components)
    return dict(fits=fits, metrics=pd.concat(rows, ignore_index=True), audit=pd.DataFrame(audits),
        differences=differences, fish_ids=fish_ids, side_conditions=side_conditions,
        window_s=tuple(window_s), fit_window_s=tuple(fit_window_s), program_onset_s=program_onset_s,
        pooled_fits=pooled_fits, responses=responses, pca_scope=pca_scope)


def build_joint_population_pca(analysis, *, sides=None, n_components=3):
    """Fit one pooled PCA jointly across stimulus sides on matching neuron IDs.

    Input is a prepared analysis with pooled fits/responses. Intersect original
    neuron IDs across requested sides within each fish, preserving the first
    side's order; return an explicit joint-neuron audit. Never match neurons
    across fish or align independently fitted PC axes. All sides must share native
    times. Stack each side/condition's mean time rows in the original fit window,
    with 1/sqrt(number_of_fish * common_neurons_in_fish) feature weights. Every
    condition/time has equal weight; sides with more fitted conditions have more
    rows. Return the pooled-fit contract with means keyed by (side,label), sides,
    and joint_audit. No trial trajectories or distances are changed.
    """
    sides = list(analysis['side_conditions']) if sides is None else list(sides)
    if not sides or len(set(sides)) != len(sides) or not set(sides) <= set(analysis['pooled_fits']):
        raise ValueError('Joint PCA requires unique sides with prepared pooled fits.')
    if int(n_components) != n_components or n_components < 2:
        raise ValueError('Require an integer component count of at least two.')
    first = analysis['pooled_fits'][sides[0]]
    times = first['time_s']
    if any(not np.array_equal(analysis['pooled_fits'][side]['time_s'], times) for side in sides):
        raise ValueError('Joint PCA requires identical native times across sides.')
    labels = [(side, label) for side in sides for label in analysis['side_conditions'][side]['stimuli']]
    responses, identities, audit = {}, {}, []
    for fid in analysis['fish_ids']:
        side_ids = {}
        for side in sides:
            metadata = analysis['pooled_fits'][side]['neuron_metadata']
            side_ids[side] = metadata.loc[metadata.fish_id.eq(fid), 'neuron_id'].to_numpy()
        common = set.intersection(*(set(ids) for ids in side_ids.values()))
        ids = np.asarray([neuron for neuron in side_ids[sides[0]] if neuron in common], dtype=int)
        if not len(ids):
            raise ValueError(f'{fid}: no common finite retained neurons across sides.')
        responses[fid], identities[fid] = {}, ids
        for side in sides:
            lookup = {neuron: row for row, neuron in enumerate(side_ids[side])}
            indices = [lookup[neuron] for neuron in ids]
            for label in analysis['side_conditions'][side]['stimuli']:
                responses[fid][(side, label)] = analysis['responses'][side][fid][label][indices]
            audit.append(dict(fish_id=fid, side=side, side_neurons=len(side_ids[side]),
                              joint_neurons=len(ids), excluded_from_joint=len(side_ids[side]) - len(ids)))
    fit = _pool_condition_means(responses, times, labels, analysis['fish_ids'], identities,
                                analysis['fit_window_s'], n_components)
    fit.update(sides=sides, joint_audit=pd.DataFrame(audit))
    return fit


def test_population_separation(analysis, *, n_permutations=2000, seed=0, alpha=0.05):
    """Whole-block paired label swaps, equal-fish means, corrected across time/pairs.

    Each block's two condition labels are swapped together for all neurons and
    times. Enumerate each fish's 2**R swap patterns (R<=10); draw independent
    patterns across fish to form a group null. Use full-space crossvalidated
    squared distances. Max-statistic correction spans every displayed time sample
    within a pair, then Bonferroni correction spans all configured pairs/sides.
    Input is build_population_trajectory_analysis's result; output has one row per
    pair/time with mean_cv_squared_distance, p_corrected and significant. Monte
    Carlo p-values use (count+1)/(B+1). Assumes block-wise condition exchangeability
    and independence across fish/blocks; no neuron-as-replicate inference. A first
    significant sample is evidence of separation, not a precise change-point or
    proof of earlier equivalence. Negative squared estimates are valid.
    """
    if int(n_permutations) != n_permutations or n_permutations < 1 or not 0 < alpha < 1:
        raise ValueError('Require positive integer permutations and alpha in (0,1).')
    rng, rows = np.random.default_rng(seed), []
    n_pairs = sum(len(c['pairs']) for c in analysis['side_conditions'].values())
    fish_ids = analysis['fish_ids']
    if len(fish_ids) < 2:
        raise ValueError('Group separation testing requires at least two fish.')
    for side, config in analysis['side_conditions'].items():
        for a, b in config['pairs']:
            null = None
            observed = []
            for fid in fish_ids:
                delta = analysis['differences'][(side, a, b, fid)]
                if delta.shape[2] > 10:
                    raise ValueError('Exact within-fish swap enumeration supports at most 10 blocks.')
                signs = np.asarray(list(product((-1, 1), repeat=delta.shape[2])))
                candidates = np.stack([_crossvalidated_squared_distance(delta * s) for s in signs])
                sampled = candidates[rng.integers(len(signs), size=n_permutations)]
                null = sampled if null is None else null + sampled
                observed.append(_crossvalidated_squared_distance(delta))
            null /= len(fish_ids)
            mean = np.mean(observed, axis=0)
            maxima = null.max(axis=1)
            p = np.minimum(1.0, n_pairs * (1 + (maxima[:, None] >= mean).sum(axis=0))
                           / (n_permutations + 1))
            rows.append(pd.DataFrame(dict(side=side, condition_a=a, condition_b=b,
                time_s=analysis['metrics'].loc[
                    analysis['metrics'].side.eq(side) & analysis['metrics'].fish_id.eq(fish_ids[0]),
                    'time_s'].drop_duplicates().to_numpy(),
                mean_cv_squared_distance=mean, p_corrected=p, significant=(p < alpha) & (mean > 0))))
    return pd.concat(rows, ignore_index=True)


def evaluate_population_contrasts(analysis, controls, candidates, *, window_s,
                                  discovery_reps=(0, 2), validation_reps=(1, 3)):
    """Compare control/candidates on disjoint repetition sets, using unscaled z.

    ``controls`` maps side->label; ``candidates`` maps side->{candidate_name:label}.
    In each split, average its repetitions and correlate flattened neuron/time
    response profiles within the half-open window. Return a long fish/side/name/
    split table, with correlation and RMS. NaN denotes constant profiles. Choose
    a contrast using discovery only; validation is descriptive confirmation,
    not an independent population-level significance test. No condition is
    guaranteed to differ. Indices are zero-based and must be disjoint and valid.
    """
    discovery, validation = list(discovery_reps), list(validation_reps)
    if (not discovery or not validation or set(discovery) & set(validation)
            or len(set(discovery)) != len(discovery) or len(set(validation)) != len(validation)):
        raise ValueError('Require nonempty, disjoint, unique discovery/validation repetitions.')
    rows = []
    for side, fish_responses in analysis['responses'].items():
        times = analysis['metrics'].query('side == @side')['time_s'].drop_duplicates().to_numpy()
        indices = select_event_window(times, *window_s)['frame_indices']
        if not len(indices):
            raise ValueError('Contrast window contains no native samples.')
        for fid, arrays in fish_responses.items():
            for split, reps in [('discovery', discovery), ('validation', validation)]:
                r = np.asarray(reps)
                if r.dtype.kind not in 'iu' or r.min() < 0 or r.max() >= arrays[controls[side]].shape[2]:
                    raise ValueError('Contrast repetition indices must be in range.')
                reference = arrays[controls[side]][:, indices][:, :, r].mean(axis=2).ravel()
                for name, label in candidates[side].items():
                    comparison = arrays[label][:, indices][:, :, r].mean(axis=2).ravel()
                    correlation = _pattern_correlation(reference[:, None], comparison[:, None])[0]
                    rows.append(dict(side=side, fish_id=fid, candidate=name, stimulus=label,
                        split=split, pattern_correlation=correlation,
                        rms_distance=float(np.sqrt(np.mean((reference - comparison) ** 2)))))
    return pd.DataFrame(rows)


def check_population_subsampling(analysis, *, n_draws=50, neurons_per_fish=None, seed=0):
    """Repeated equal-count neuron draws check full-space distance robustness.

    Sample without replacement within each fish/side; preserve the same sampled
    neurons across every pair, time and repetition in that draw. Default count
    is the minimum jointly valid fish count within that side. Return one row per
    side/pair/time: all-neuron equal-fish crossvalidated squared distance and the
    median/5th/95th percentiles across draws, plus count. These percentiles describe
    neuron-sampling sensitivity, not a confidence interval across independent fish.
    No random PCA axes are averaged and no population significance is inferred.
    """
    if int(n_draws) != n_draws or n_draws < 1:
        raise ValueError('Require positive integer neuron-sampling draws.')
    rng, rows = np.random.default_rng(seed), []
    for side, config in analysis['side_conditions'].items():
        fish_ids = analysis['fish_ids']
        counts = [next(iter(analysis['responses'][side][fid].values())).shape[0] for fid in fish_ids]
        count = min(counts) if neurons_per_fish is None else neurons_per_fish
        if int(count) != count or count < 1 or count > min(counts):
            raise ValueError('Neuron count must be a positive integer no larger than every fish population.')
        samples = {fid: np.stack([rng.choice(n, count, replace=False) for _ in range(n_draws)])
                   for fid, n in zip(fish_ids, counts)}
        for a, b in config['pairs']:
            draws = np.mean([np.stack([_crossvalidated_squared_distance(
                analysis['differences'][(side, a, b, fid)][indices]) for indices in samples[fid]])
                for fid in fish_ids], axis=0)
            full = analysis['metrics'].query('side == @side & condition_a == @a & condition_b == @b')
            mean = full.groupby('time_s', sort=False).cv_squared_distance.mean()
            rows.append(pd.DataFrame(dict(side=side, condition_a=a, condition_b=b,
                time_s=mean.index, all_neuron_distance=mean.to_numpy(),
                subsample_median=np.median(draws, axis=0), subsample_p05=np.quantile(draws, 0.05, axis=0),
                subsample_p95=np.quantile(draws, 0.95, axis=0), neurons_per_fish=count, n_draws=n_draws)))
    return pd.concat(rows, ignore_index=True)

"""Stimulus-vector similarity and seeded segment-selectivity permutations."""

import numpy as np
import pandas as pd

from src.multifish_matrices import _stim_key
from src.response_metrics import compute_trial_auc_by_neuron
from src.trial_alignment import compute_response_window_frames


def build_stimulus_vector_similarity(response_matrix, selected_stimuli=None):
    """
    Compute stimulus-vector similarity from a neuron-by-stimulus response matrix.

    Returns Pearson and cosine similarity matrices plus a compact pair table
    with the positional distance between selected stimuli.
    """
    response_matrix = pd.DataFrame(response_matrix).copy()
    if selected_stimuli is None:
        selected_stimuli = list(response_matrix.columns)
    selected_stimuli = list(selected_stimuli)
    if not selected_stimuli:
        raise ValueError("selected_stimuli must contain at least one stimulus.")

    missing = [
        stimulus for stimulus in selected_stimuli
        if stimulus not in response_matrix.columns
    ]
    if missing:
        raise KeyError(
            "selected_stimuli contains label(s) missing from response_matrix: "
            + ", ".join(map(str, missing))
        )

    selected_response_matrix = response_matrix.loc[:, selected_stimuli]
    pearson_similarity_matrix = selected_response_matrix.corr(method="pearson")
    cosine_similarity_matrix = pd.DataFrame(
        np.nan,
        index=selected_stimuli,
        columns=selected_stimuli,
        dtype=float,
    )

    for stimulus_a in selected_stimuli:
        vector_a = selected_response_matrix[stimulus_a].to_numpy(dtype=float)
        for stimulus_b in selected_stimuli:
            vector_b = selected_response_matrix[stimulus_b].to_numpy(dtype=float)
            finite_mask = np.isfinite(vector_a) & np.isfinite(vector_b)
            if not np.any(finite_mask):
                continue
            a = vector_a[finite_mask]
            b = vector_b[finite_mask]
            denominator = np.linalg.norm(a) * np.linalg.norm(b)
            if denominator > 0:
                cosine_similarity_matrix.loc[stimulus_a, stimulus_b] = (
                    np.dot(a, b) / denominator
                )

    pair_rows = []
    for index_a, stimulus_a in enumerate(selected_stimuli):
        for index_b in range(index_a + 1, len(selected_stimuli)):
            stimulus_b = selected_stimuli[index_b]
            pair_rows.append(
                {
                    "stimulus_a": stimulus_a,
                    "stimulus_b": stimulus_b,
                    "segment_distance": abs(index_b - index_a),
                    "pearson_similarity": pearson_similarity_matrix.loc[
                        stimulus_a, stimulus_b
                    ],
                    "cosine_similarity": cosine_similarity_matrix.loc[
                        stimulus_a, stimulus_b
                    ],
                }
            )

    pair_similarity = pd.DataFrame(
        pair_rows,
        columns=[
            "stimulus_a",
            "stimulus_b",
            "segment_distance",
            "pearson_similarity",
            "cosine_similarity",
        ],
    )

    return {
        "selected_response_matrix": selected_response_matrix,
        "pearson_similarity_matrix": pearson_similarity_matrix,
        "cosine_similarity_matrix": cosine_similarity_matrix,
        "pair_similarity": pair_similarity,
    }


def resolve_segment_labels(segments, available_labels):
    """Resolve editable segment names against selected stimulus labels."""
    available_labels = list(available_labels)
    resolved = []
    for segment in segments:
        exact_matches = [label for label in available_labels if label == segment]
        if len(exact_matches) == 1:
            resolved.append(exact_matches[0])
            continue

        suffix_matches = [
            label for label in available_labels
            if str(label).endswith(str(segment))
        ]
        if len(suffix_matches) == 1:
            resolved.append(suffix_matches[0])
        elif len(suffix_matches) == 0:
            raise KeyError(
                f"Segment {segment!r} was not found in available labels "
                f"{available_labels!r}."
            )
        else:
            raise ValueError(
                f"Segment {segment!r} matched multiple labels: "
                f"{suffix_matches!r}. Use an exact label."
            )

    if len(set(resolved)) != len(resolved):
        raise ValueError(f"Resolved segment labels are not unique: {resolved!r}")
    return resolved


def _compute_segment_selectivity_index(segment_to_trials):
    means = {}
    for segment, values in segment_to_trials.items():
        values = np.asarray(values, dtype=float).ravel()
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return np.nan, np.nan, "missing_or_empty_trials"
        means[segment] = float(np.mean(finite_values))

    mean_values = np.asarray(list(means.values()), dtype=float)
    if not np.all(np.isfinite(mean_values)):
        return np.nan, np.nan, "invalid_segment_mean"

    preferred_idx = int(np.argmax(mean_values))
    preferred = list(means.keys())[preferred_idx]
    r_preferred = mean_values[preferred_idx]
    r_others = float(np.mean(np.delete(mean_values, preferred_idx)))
    denominator = r_preferred + r_others
    if r_preferred <= 0.0 or r_others < 0.0:
        return np.nan, preferred, "negative_or_nonpositive_response"
    if not np.isfinite(denominator) or denominator <= 0.0 or np.isclose(denominator, 0.0):
        return np.nan, preferred, "zero_or_invalid_denominator"

    return float((r_preferred - r_others) / denominator), preferred, None


def _shuffle_segment_trials(segment_to_trials, rng):
    segment_counts = {
        segment: np.asarray(values, dtype=float).ravel().size
        for segment, values in segment_to_trials.items()
    }
    pooled = np.concatenate(
        [np.asarray(values, dtype=float).ravel() for values in segment_to_trials.values()]
    )
    shuffled = rng.permutation(pooled)

    shuffled_segments = {}
    start = 0
    for segment, count in segment_counts.items():
        stop = start + count
        shuffled_segments[segment] = shuffled[start:stop]
        start = stop
    return shuffled_segments


def _run_segment_selectivity_permutations(neuron_segment_trials, n_permutations, rng):
    real_si, real_preferred, skip_reason = _compute_segment_selectivity_index(
        neuron_segment_trials
    )
    shuffled_si = np.full(int(n_permutations), np.nan, dtype=float)
    if skip_reason is not None:
        return real_si, real_preferred, shuffled_si, skip_reason

    for perm_idx in range(int(n_permutations)):
        shuffled_trials = _shuffle_segment_trials(neuron_segment_trials, rng)
        shuffled_si[perm_idx], _, _ = _compute_segment_selectivity_index(
            shuffled_trials
        )
    return real_si, real_preferred, shuffled_si, None


def build_segment_selectivity_permutation_summary(
    all_fish_data,
    fish_ids,
    selected_stimulus_ids,
    selected_stimulus_labels,
    segments_to_compare,
    fps_2p=2.0,
    t_pre_s=5.0,
    motion_onset_s=8.0,
    tau_s=6.0,
    motion_duration_key="motion_sec",
    n_permutations=1000,
    random_seed=42,
    alpha_percentile=95,
):
    """
    Run segment-selectivity permutations for selected z-score AUC responses.

    The summary table has one row per pooled neuron and can be joined to the
    neuron summary table on ``global_neuron_id``.
    """
    fish_ids = list(fish_ids)
    selected_stimulus_ids = list(selected_stimulus_ids)
    selected_stimulus_labels = list(selected_stimulus_labels)
    resolved_segment_labels = resolve_segment_labels(
        segments_to_compare,
        selected_stimulus_labels,
    )
    segment_display_labels = list(segments_to_compare)
    segment_label_to_display = dict(
        zip(resolved_segment_labels, segment_display_labels)
    )
    selected_label_to_id = dict(zip(selected_stimulus_labels, selected_stimulus_ids))
    segment_label_to_id = {
        label: selected_label_to_id[label] for label in resolved_segment_labels
    }

    rng = np.random.default_rng(random_seed)
    trial_auc_by_fish = {}
    trial_count_rows = []
    warning_rows = []

    for fid in fish_ids:
        fish = all_fish_data[fid]
        trial_aligned_z = fish["trial_aligned_traces_z_core"]
        kept_idx = np.asarray(fish["kept_neuron_indices"], dtype=int).ravel()
        if kept_idx.size == 0:
            raise ValueError(f"Fish {fid!r} has empty kept_neuron_indices.")

        trial_auc_by_fish[fid] = {}
        n_kept_expected = None
        fish_trial_counts = []

        for resolved_label in resolved_segment_labels:
            display_label = segment_label_to_display[resolved_label]
            stim_id = segment_label_to_id[resolved_label]
            stim_key = _stim_key(trial_aligned_z, stim_id)
            arr = np.asarray(trial_aligned_z[stim_key], dtype=float)
            if arr.ndim != 3:
                raise ValueError(
                    f"Fish {fid!r}, segment {resolved_label!r} has shape "
                    f"{arr.shape}; expected (n_neurons, n_time, n_reps)."
                )
            if np.any(kept_idx < 0) or np.any(kept_idx >= arr.shape[0]):
                raise IndexError(
                    f"Fish {fid!r} kept_neuron_indices contains values outside "
                    f"segment {resolved_label!r} neuron axis with n_neurons={arr.shape[0]}."
                )

            arr_kept = arr[kept_idx, :, :]
            if n_kept_expected is None:
                n_kept_expected = arr_kept.shape[0]
            elif arr_kept.shape[0] != n_kept_expected:
                raise ValueError(
                    f"Fish {fid!r}, segment {resolved_label!r} has "
                    f"{arr_kept.shape[0]} kept neurons; expected {n_kept_expected}."
                )

            if arr_kept.shape[2] == 0:
                warning_rows.append(
                    {
                        "fish_id": fid,
                        "segment": display_label,
                        "warning": "empty_response_array",
                    }
                )

            window = compute_response_window_frames(
                n_time=arr_kept.shape[1],
                fps_2p=fps_2p,
                t_pre_s=t_pre_s,
                motion_onset_s=motion_onset_s,
                stimulus=stim_id,
                stimuli_durations=fish["stimuli_durations"],
                stimuli_id_map=fish["stimuli_id_map"],
                tau_s=tau_s,
                motion_duration_key=motion_duration_key,
            )
            trial_auc_by_fish[fid][display_label] = compute_trial_auc_by_neuron(
                arr_kept,
                frame_indices=window["frame_indices"],
                time_s=window["time_s"],
                fps_2p=fps_2p,
            )
            fish_trial_counts.append(arr_kept.shape[2])
            trial_count_rows.append(
                {
                    "fish_id": fid,
                    "segment": display_label,
                    "resolved_label": resolved_label,
                    "stimulus_id": stim_id,
                    "n_kept_neurons": arr_kept.shape[0],
                    "n_trials": arr_kept.shape[2],
                    "n_response_frames": window["n_frames"],
                    "start_frame": window["start_frame"],
                    "stop_frame": window["stop_frame"],
                }
            )

        if len(set(fish_trial_counts)) > 1:
            warning_rows.append(
                {
                    "fish_id": fid,
                    "segment": "all",
                    "warning": f"unequal_trial_counts:{fish_trial_counts}",
                }
            )

    summary_rows = []
    si_shuffle_rows = []
    global_row = 0
    skip_reason_counts = {}

    for fid in fish_ids:
        kept_idx = np.asarray(all_fish_data[fid]["kept_neuron_indices"], dtype=int).ravel()
        for neuron_id in range(kept_idx.size):
            neuron_segment_trials = {
                segment: trial_auc_by_fish[fid][segment][neuron_id, :]
                for segment in segment_display_labels
            }
            real_si, real_preferred, shuffled_si, skip_reason = (
                _run_segment_selectivity_permutations(
                    neuron_segment_trials,
                    n_permutations=n_permutations,
                    rng=rng,
                )
            )
            if skip_reason is not None:
                skip_reason_counts[skip_reason] = skip_reason_counts.get(skip_reason, 0) + 1

            finite_shuffle = shuffled_si[np.isfinite(shuffled_si)]
            if finite_shuffle.size == 0:
                threshold = np.nan
                is_significant = False
            else:
                threshold = float(np.nanpercentile(finite_shuffle, alpha_percentile))
                is_significant = bool(np.isfinite(real_si) and real_si > threshold)

            summary_rows.append(
                {
                    "fish_id": fid,
                    "neuron_id": neuron_id,
                    "kept_neuron_index": int(kept_idx[neuron_id]),
                    "global_neuron_id": global_row,
                    "segment_selectivity_index": real_si,
                    "preferred_segment": real_preferred,
                    "segment_shuffle_threshold": threshold,
                    "segment_selective": is_significant,
                    "segment_skip_reason": skip_reason,
                }
            )
            si_shuffle_rows.append(shuffled_si)
            global_row += 1

    summary_df = pd.DataFrame(summary_rows)
    si_shuffle = (
        np.vstack(si_shuffle_rows)
        if si_shuffle_rows
        else np.empty((0, int(n_permutations)))
    )

    return {
        "segments_requested": segment_display_labels,
        "segments_resolved": resolved_segment_labels,
        "segment_label_to_id": segment_label_to_id,
        "trial_auc_by_fish": trial_auc_by_fish,
        "trial_counts": pd.DataFrame(trial_count_rows),
        "warnings": pd.DataFrame(warning_rows),
        "summary_df": summary_df,
        "si_real": summary_df["segment_selectivity_index"].to_numpy(dtype=float),
        "si_shuffle": si_shuffle,
        "preferred_segment": summary_df["preferred_segment"].to_numpy(dtype=object),
        "significant_segment_selective": summary_df["segment_selective"].to_numpy(dtype=bool),
        "skip_reason_counts": skip_reason_counts,
        "alpha_percentile": alpha_percentile,
        "n_permutations": n_permutations,
        "random_seed": random_seed,
    }

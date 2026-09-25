"""Response-index and stimulus-selectivity calculations."""

import numpy as np
import pandas as pd

def compute_response_pair_index(response_matrix, left_stimulus, right_stimulus, eps=1e-12):
    """
    Compute a paired preference index from a neuron-by-stimulus response matrix.

    The returned index is ``(left - right) / (left + right)``. Missing
    configured stimulus columns return an all-NaN vector so notebooks can keep
    running with ``left_right_filter_mode='none'`` while still reporting the
    missing control pair.
    """
    response_matrix = pd.DataFrame(response_matrix)
    n_rows = response_matrix.shape[0]
    index_values = np.full(n_rows, np.nan, dtype=float)

    if left_stimulus not in response_matrix.columns or right_stimulus not in response_matrix.columns:
        return index_values

    left = response_matrix[left_stimulus].to_numpy(dtype=float)
    right = response_matrix[right_stimulus].to_numpy(dtype=float)
    denom = left + right
    finite = np.isfinite(left) & np.isfinite(right) & np.isfinite(denom) & (np.abs(denom) > eps)
    index_values[finite] = (left[finite] - right[finite]) / denom[finite]
    return index_values

def build_response_index_keep_mask(index_values, mode="none", threshold=0.3, value_range=(-0.3, 0.3)):
    """
    Build a neuron keep mask from a paired response index.

    Modes match the reusable notebook settings:
    ``none``, ``abs``, ``left``, ``right``, and ``range``.
    """
    index_values = np.asarray(index_values, dtype=float)
    mode = str(mode)

    if mode == "none":
        return np.ones(index_values.shape[0], dtype=bool)
    if mode == "abs":
        return np.isfinite(index_values) & (np.abs(index_values) >= float(threshold))
    if mode == "left":
        return np.isfinite(index_values) & (index_values >= float(threshold))
    if mode == "right":
        return np.isfinite(index_values) & (index_values <= -float(threshold))
    if mode == "range":
        lo, hi = value_range
        return np.isfinite(index_values) & (index_values >= float(lo)) & (index_values <= float(hi))

    raise ValueError("mode must be 'none', 'abs', 'left', 'right', or 'range'.")

def compute_stimulus_selectivity_metrics(response_values, stimulus_labels):
    """
    Compute raw-response preference and non-negative selectivity metrics.

    ``preferred_stimulus``, ``max_response``, and ``mean_response`` use raw
    response values. ``selectivity_index`` uses raw response values only when
    the preferred response is positive and the mean other response is
    non-negative; this avoids unstable ratios when excitation is cancelled by
    negative responses. ``simple_selectivity`` and ``lifetime_sparseness`` use
    responses clipped at zero for selectivity only.
    """
    response_values = np.asarray(response_values, dtype=float).ravel()
    stimulus_labels = list(stimulus_labels)

    if response_values.ndim != 1:
        raise ValueError("response_values must be 1D.")
    if response_values.size != len(stimulus_labels):
        raise ValueError("response_values and stimulus_labels must have the same length.")
    if response_values.size < 2:
        raise ValueError("At least two stimuli are required for selectivity metrics.")

    if np.all(np.isnan(response_values)):
        preferred_idx = None
        preferred_stimulus = np.nan
        max_response = np.nan
        mean_response = np.nan
    else:
        preferred_idx = int(np.nanargmax(response_values))
        preferred_stimulus = stimulus_labels[preferred_idx]
        max_response = float(np.nanmax(response_values))
        mean_response = float(np.nanmean(response_values))

    if preferred_idx is None:
        selectivity_index = np.nan
    else:
        other_responses = np.delete(response_values, preferred_idx)
        finite_other_responses = other_responses[np.isfinite(other_responses)]
        if finite_other_responses.size == 0:
            selectivity_index = np.nan
        else:
            preferred_response = float(response_values[preferred_idx])
            mean_other_response = float(np.mean(finite_other_responses))
            denominator = preferred_response + mean_other_response
            if (
                preferred_response <= 0.0
                or mean_other_response < 0.0
                or not np.isfinite(denominator)
                or np.isclose(denominator, 0.0)
            ):
                selectivity_index = np.nan
            else:
                selectivity_index = float(
                    (preferred_response - mean_other_response) / denominator
                )

    positive_responses = np.where(
        np.isfinite(response_values),
        np.maximum(response_values, 0.0),
        0.0,
    )
    total_positive = float(np.sum(positive_responses))

    if total_positive <= 0.0:
        simple_selectivity = np.nan
        lifetime_sparseness = np.nan
    else:
        simple_selectivity = float(np.max(positive_responses) / total_positive)
        n_stimuli = positive_responses.size
        mean_r = float(np.mean(positive_responses))
        mean_r2 = float(np.mean(positive_responses ** 2))
        if mean_r2 <= 0.0:
            lifetime_sparseness = np.nan
        else:
            lifetime_sparseness = float(
                (1.0 - ((mean_r ** 2) / mean_r2)) / (1.0 - (1.0 / n_stimuli))
            )

    return {
        "preferred_stimulus": preferred_stimulus,
        "max_response": max_response,
        "mean_response": mean_response,
        "simple_selectivity": simple_selectivity,
        "selectivity_index": selectivity_index,
        "lifetime_sparseness": lifetime_sparseness,
    }

def classify_stimulus_specificity_neuron(
    lifetime_sparseness,
    n_active_stimuli,
    response_breadth,
    max_positive_response,
    high_lifetime_sparseness=0.70,
    intermediate_lifetime_sparseness=0.40,
    broad_breadth_threshold=0.80,
    strong_response_threshold=np.nan,
    weak_response_threshold=np.nan,
):
    """
    Classify one neuron from selectivity metrics and positive response strength.
    """
    if (
        not np.isfinite(max_positive_response)
        or max_positive_response <= 0.0
        or not np.isfinite(lifetime_sparseness)
    ):
        return "Weak/unclear"

    if np.isfinite(weak_response_threshold) and max_positive_response <= weak_response_threshold:
        return "Weak/unclear"

    if (
        response_breadth >= broad_breadth_threshold
        and np.isfinite(strong_response_threshold)
        and max_positive_response >= strong_response_threshold
    ):
        return "Strong broad responder"

    if n_active_stimuli == 1 and lifetime_sparseness >= high_lifetime_sparseness:
        return "Stimulus-specific neuron"

    if (
        n_active_stimuli > 1
        and response_breadth < broad_breadth_threshold
        and lifetime_sparseness >= intermediate_lifetime_sparseness
    ):
        return "Subset-selective neuron"

    if (
        response_breadth >= broad_breadth_threshold
        and lifetime_sparseness < intermediate_lifetime_sparseness
    ):
        return "Broadly active neuron"

    return "Weak/unclear"

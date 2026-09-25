"""Compatibility facade for calcium-analysis helpers.

Scientific implementations live in responsibility-specific modules. Existing
notebooks and modules may continue importing ``src.analysis_tools`` while later
consumer-migration slices adopt the narrower owners.
"""

from src.analysis_io import find_file_with_suffix, inspect_obj
from src.motion_metrics import compute_motion_delta_integrals, compute_motion_delta_peaks
from src.plotting import plot_accepted_rejected_rasters, plot_venn_3stim
from src.reliability import filter_neurons_by_trial_reliability
from src.response_classification import (
    _has_consecutive_true,
    build_active_neuron_matrix_from_trial_raster,
    build_neuron_order_groupwise_onset,
    classify_responses_from_raster,
    compute_left_right_index,
)
from src.response_metrics import (
    _longest_true_run,
    compute_static_flicker_trial_metrics,
    compute_trial_auc_by_neuron,
    compute_trial_mean_response_metrics,
    compute_zscore_response_auc,
    validate_static_flicker_recruitment_result,
)
from src.response_normalization import zscore_dfof_from_prestim_baseline
from src.response_selectivity import (
    build_response_index_keep_mask,
    classify_stimulus_specificity_neuron,
    compute_response_pair_index,
    compute_stimulus_selectivity_metrics,
)
from src.trial_alignment import (
    _id_to_stimulus_name,
    _stimulus_duration_entry,
    _trial_aligned_time_axis,
    build_trial_aligned_traces,
    compute_response_window_frames,
    resolve_selected_stimuli,
)


__all__ = [
    "build_active_neuron_matrix_from_trial_raster",
    "build_neuron_order_groupwise_onset",
    "build_response_index_keep_mask",
    "build_trial_aligned_traces",
    "classify_responses_from_raster",
    "classify_stimulus_specificity_neuron",
    "compute_left_right_index",
    "compute_motion_delta_integrals",
    "compute_motion_delta_peaks",
    "compute_response_pair_index",
    "compute_response_window_frames",
    "compute_static_flicker_trial_metrics",
    "compute_stimulus_selectivity_metrics",
    "compute_trial_auc_by_neuron",
    "compute_trial_mean_response_metrics",
    "compute_zscore_response_auc",
    "filter_neurons_by_trial_reliability",
    "find_file_with_suffix",
    "inspect_obj",
    "plot_accepted_rejected_rasters",
    "plot_venn_3stim",
    "resolve_selected_stimuli",
    "validate_static_flicker_recruitment_result",
    "zscore_dfof_from_prestim_baseline",
]

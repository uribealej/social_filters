"""Compatibility facade for multifish calcium analyses.

Scientific families live in narrow owners. Historical imports, including the
selectivity re-export, remain available for old callers; new consumers should
use the narrow owners.
"""

# Preserve historically exposed dependency aliases as well as helper imports.
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.multifish_matrices import (
    combine_reps_one_stim,
    build_matrix_for_fish,
    _stim_key,
    describe_matrix_widths,
    build_matrix_all_fish,
    build_zscore_response_matrix_for_fish,
    build_zscore_response_matrices_all_fish,
)
from src.static_flicker_analysis import (
    build_static_flicker_recruitment_analysis,
    summarize_pooled_static_flicker_categories,
    _exact_sign_flip_pvalue,
    _wilcoxon_signed_rank_pvalue,
    _bootstrap_median_ci,
    _holm_adjust,
    compute_static_flicker_fish_level_statistics,
)
from src.bout_flicker_analysis import (
    build_bout_flicker_position_metadata,
    _read_bout_flicker_trajectory,
    _require_bout_flicker_timing,
    _extract_visible_bout_positions,
    _stationary_flicker_coordinate,
    build_bout_flicker_position_analysis,
    _validate_bout_flicker_window,
    _trace_for_bout_flicker,
    _first_persistent_significant_time,
    _window_frames,
    _build_bout_flicker_side_data,
    _active_column,
    _stimulus_motion_onset,
)
from src.stimulus_specificity import (
    _active_column_values,
    build_neuron_stimulus_summary_table,
    add_selectivity_metrics_to_summary_table,
    classify_stimulus_specificity_summary_table,
    build_stimulus_specificity_neuron_order,
)
from src.stimulus_similarity import (
    build_stimulus_vector_similarity,
    resolve_segment_labels,
    _compute_segment_selectivity_index,
    _shuffle_segment_trials,
    _run_segment_selectivity_permutations,
    build_segment_selectivity_permutation_summary,
)
from src.active_neuron_analysis import (
    build_active_neuron_matrices_all_fish,
    _active_columns_as_bool,
    compute_active_neuron_jaccard_overlap,
    build_active_neuron_overlap_matrices_all_fish,
    _active_column_as_bool,
    _trial_trace_key,
    _stimulus_label,
    build_pooled_active_trace_diagnostic,
)

# Historical S06 helper re-exports remain part of the compatibility surface.
from src.response_classification import build_active_neuron_matrix_from_trial_raster
from src.response_metrics import (
    compute_static_flicker_trial_metrics,
    compute_trial_auc_by_neuron,
    compute_zscore_response_auc,
)
from src.response_selectivity import (
    classify_stimulus_specificity_neuron,
    compute_stimulus_selectivity_metrics,
)
from src.trial_alignment import compute_response_window_frames, resolve_selected_stimuli

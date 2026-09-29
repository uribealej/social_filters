"""Reusable orchestration helpers for several-fish calcium analysis notebooks."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import src.analysis_tools as at
import src.multifish_analysis as mfa
import src.plotting as plott
from src.several_fish_figures import (
    build_all_fish_raster_figure, build_plot_all_fish_mean_zscore_traces,
    plot_left_right_active_overlap_diagnostics, plot_lifetime_sparseness_analysis,
    plot_motion_active_neuron_counts,
)
from src.several_fish_diagnostics import (
    apply_diagnostic_keep_mask, build_decision_then_mean_sort_order,
    build_diagnostic_trace_source_data, build_high_sparseness_raster_data,
    build_metric_sort_order, build_overlap_diagnostic_data,
    build_pooled_mean_trace_by_stimulus, build_response_window_validation,
)
from src.several_fish_loading import load_and_preflight_fish_raster_inputs
from src.several_fish_reporting import (
    _slugify_label, export_notebook_report, save_analysis_report_run,
)
from src.several_fish_selection import (
    build_filtered_trial_aligned_traces_for_fish, build_fish_keep_masks,
    build_selected_neuron_summary, filter_active_matrices_by_keep_mask,
    resolve_response_control_columns, resolve_stimulus_set, subset_active_matrices,
)

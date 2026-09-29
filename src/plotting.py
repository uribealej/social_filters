"""Compatibility facade for calcium-imaging figure families.

Historical helper imports and dependency aliases remain available for old
callers. New consumers should use the narrow figure owners; neuron ordering
and timing preparation live in their own modules.
"""

# Preserve historical helper imports and dependency aliases.
from src.plotting_reliability import (
    plot_accepted_rejected_rasters,
    plot_venn_3stim,
    plot_reliability_diagnostics,
)
from src.plotting_lme import (
    _plot_lme_fixed_effects,
    _plot_lme_model_comparison,
    _plot_lme_observed_vs_fitted,
    _plot_lme_response_distribution,
    plot_lme_model_outputs,
)
from src.plotting_specificity import (
    plot_similarity_heatmaps,
    plot_similarity_by_distance,
    plot_stimulus_specificity_sparseness,
    plot_stimulus_specificity_selectivity_index,
    plot_active_stimuli_histogram,
    plot_preferred_stimulus_distribution,
    plot_stimulus_specificity_summary,
)
from src.plotting_bout_position import (
    plot_bout_flicker_position_figure,
    _add_bout_position_markers,
    _plot_bout_flicker_position_summary,
    plot_bout_flicker_position_cell06_style,
)
from src.plotting_static_flicker import (
    plot_static_flicker_classification_raster,
    plot_static_flicker_category_proportions,
    plot_pooled_static_flicker_category_proportions,
    plot_shared_static_flicker_auc_summary,
    plot_recruitment_amplification,
)
from src.plotting_diagnostics import (
    plot_motion_delta_distribution,
    plot_active_trace_decision_diagnostic,
)
from src.plotting_all_fish import (
    plot_stimulus_means,
    plot_allfish_flat_raster,
)
from src.plotting_single_fish import (
    add_stimuli_markers,
    raster_with_stimuli,
    save_sorted_rasters_for_all_modes,
    plot_sorted_chunks_single_mode,
)
from src.plotting_common import (
    _natural_sort_key,
    _vertical_reference_line,
    _horizontal_reference_line,
    list_stimulus_names,
    _stimulus_style_key,
    _is_right_side_stimulus,
    build_stimulus_style_maps,
    _resolve_analysis_label,
    _preferred_stimulus_colors,
    STATIC_FLICKER_CATEGORY_ORDER,
    STATIC_FLICKER_CATEGORY_COLORS,
    STATIC_FLICKER_STIMULUS_COLORS,
)
from src.stimuli_timeline import (
    summarize_durations,
    compute_move_lines_for_flat_matrix,
)
from src.neuron_ordering import (
    compute_sort_orders,
    compute_single_sort_order,
    _diagnostic_sort_order,
    _first_post_onset_raster_time,
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import pdist
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pathlib import Path
import re
import math
import src.stimuli_timeline as st
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pandas as pd
from matplotlib_venn import venn3

"""Deterministic inputs for every S08 figure family and its public callers."""

import copy
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import multifish_analysis as mfa
from tests.analysis.multifish_cases import (
    cohort, cases, trajectory_folder, FISH_IDS, STIM_IDS, STIM_LABELS, TIMING,
)


FAMILIES = ("common", "single_fish", "all_fish", "static_bout", "specificity",
            "diagnostics", "lme", "reliability")


def inputs():
    """Build ordered synthetic cohorts through the existing scientific owners."""
    with trajectory_folder() as folder:
        outputs = cases(mfa, folder)
    data = cohort()
    colors = dict(zip(STIM_LABELS, ["#0072B2", "#D55E00", "#CC79A7", "#009E73"]))
    styles = dict(zip(STIM_LABELS, ["-", "--", ":", "-."]))
    rng = np.random.default_rng(813)
    continuous = rng.uniform(0, 3, (100, 5))
    durations = {s: dict(static_before_sec=1., motion_sec=2., static_after_sec=1., total_sec=4.)
                 for s in ["FL2", "FR1"]}
    single = dict(dfof=continuous, exp_log=pd.DataFrame({
        "event": ["stim_FL2", "stim_FR1", "stim_FL2", "stim_FR1"],
        "timestamp": [3., 13., 23., 33.]}), stimuli_durations=durations,
        stimuli_colors={"FR1": "#CC79A7", "FL2": "#0072B2"},
        stimuli_linestyles={"FR1": "--", "FL2": "-"},
        stimuli_ordered=["FR1", "FL2"], fps_2p=2., window_pre=1., window_post=1.)
    return dict(outputs=outputs, data=data, colors=colors, styles=styles, single=single)


def lme_inputs():
    """Use fixed fitted values: figure tests do not refit a statistical model."""
    table = pd.DataFrame(dict(response=[1., 2., 3., 4., 2., 1.],
        stimulus_class=["flicker", "bout"] * 3, position_id=["B", "A", "B", "A", "B", "A"]))
    results = dict(fixed_effects=pd.DataFrame(dict(model_name=["full"] * 3,
        term=["Intercept", "position", "flicker"], coefficient=[1., -0.5, 0.8],
        ci_lower=[0.8, -0.8, 0.4], ci_upper=[1.2, -0.2, 1.1])),
        model_comparison=pd.DataFrame(dict(model_name=["full", "simple", "failed"],
            status=["success", "success", "failed"], aic=[12., 14., np.nan], bic=[15., 16., np.nan])),
        fit_results={"full": dict(status="success", result=SimpleNamespace(
            fittedvalues=np.array([0.9, 2.2, 2.8, 3.9, 2.1, 1.2])))})
    return results, table


def bout_four_panels(side):
    """Expand the CSV-backed one-position fixture for the historical four-panel API."""
    side = copy.deepcopy(side)
    bout = side["bout_stimulus"]
    flicker = [label for label in side["stimulus_labels"] if label != bout][0]
    labels = [flicker, "flick2", "flick3"]
    for label in labels[1:]:
        for key in ["pooled_traces", "time_relative_s", "panel_timing"]:
            side[key][label] = copy.deepcopy(side[key][flicker])
    side["stimulus_labels"] = [bout] + labels
    matches, summaries = [], []
    for i, label in enumerate(labels):
        match = pd.DataFrame(side["position_matches"]).copy()
        summary = pd.DataFrame(side["summary_table"]).copy()
        for table in [match, summary]:
            table["flicker_stimulus"] = label
            table["time_after_motion_onset_s"] = float(i + 1)
            table["nearest_bout_position_index"] = i + 1
        matches.append(match)
        summaries.append(summary)
    side["position_matches"] = pd.concat(matches, ignore_index=True)
    side["summary_table"] = pd.concat(summaries, ignore_index=True)
    return side


def figure_cases(api, fixture, family, folder):
    """Return named callables; each creates a bounded set of figures."""
    out, data = fixture["outputs"], fixture["data"]
    colors, styles = fixture["colors"], fixture["styles"]
    ref = data[FISH_IDS[0]]
    if family == "common":
        def markers():
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot([0, 5], [1, 4])
            api._vertical_reference_line(ax, 2, color="red", linestyle="--")
            api._horizontal_reference_line(ax, 3, color="blue")
            return fig, ax
        return {"reference_lines": markers, "styles": lambda: api.build_stimulus_style_maps(
            stimuli_names=["FR10", "FL2", "FR2", "FL10", "FL2"], color_overrides={"FR2": "red"})}
    if family == "single_fish":
        def raw(binary=False):
            fig, ax = plt.subplots(figsize=(10, 5))
            values = fixture["single"]["dfof"]
            image = api.raster_with_stimuli(ax, (values > 1.5) if binary else values,
                2., "synthetic", neuron_order=np.array([4, 1, 3, 0, 2]), is_binary=binary)
            handles = api.add_stimuli_markers(ax, fixture["single"]["exp_log"],
                fixture["single"]["stimuli_durations"], fixture["single"]["stimuli_colors"],
                stimuli_linestyles=fixture["single"]["stimuli_linestyles"])
            return fig, ax, image, handles
        result = {"raster": raw, "binary_raster": lambda: raw(True)}
        for binary in [False, True]:
            for average in [False, True]:
                result[f"chunks_{binary}_{average}"] = lambda b=binary, a=average: api.plot_sorted_chunks_single_mode(
                    **fixture["single"], sort_mode="max_intensity", is_binary=b,
                    average_across_repeats=a, figsize=(12, 6), fish_id="synthetic")
        result["custom_chunks"] = lambda: api.plot_sorted_chunks_single_mode(**fixture["single"],
            neuron_order=np.array([4, 3, 2, 1, 0]), sort_label="fixed", vmin=-1., vmax=3.)
        return result
    if family == "all_fish":
        result = {}
        for mode in ["mean", "concat"]:
            for binary in [False, True]:
                result[f"flat_{mode}_{binary}"] = lambda m=mode, b=binary: api.plot_allfish_flat_raster(
                    out[f"matrix_{m}_zscore"], ref["trial_aligned_traces_z_core"], STIM_IDS,
                    ref["stimuli_id_map"], ref["stimuli_durations"], colors, styles, 1., 1., m,
                    sort_mode="max_intensity", is_binary=b, show_mean_trace=b, figsize=(12, 7))
        means = {sid: np.nanmean(ref["trial_aligned_traces_z_core"][sid], axis=2) for sid in STIM_IDS}
        result["means"] = lambda: api.plot_stimulus_means(means, STIM_IDS, STIM_LABELS,
            1., 11., 1., ref["stimuli_durations"], stimuli_colors=colors, stimuli_linestyles=styles,
            title_prefix="Synthetic cohort", kept_cells=[3, 0, 4], figsize=(10, 5))
        result["means_saved"] = lambda: api.plot_stimulus_means(means, STIM_IDS, STIM_LABELS,
            1., 11., 1., ref["stimuli_durations"], stimuli_colors=colors, stimuli_linestyles=styles,
            show_sem=False, save=True, plots_path=folder, prefix="fixture", dpi=72,
            close_after=False, comment="selected", figsize=(10, 5))
        return result
    if family == "static_bout":
        static = out["static"]
        side = out["bout_False"]["sides"]["left"]
        return {
            "classification": lambda: api.plot_static_flicker_classification_raster(static["classification_raster_data"]),
            "categories": lambda: api.plot_static_flicker_category_proportions(static["fish_side_summary"]),
            "pooled_categories": lambda: api.plot_pooled_static_flicker_category_proportions(static["pooled_category_summary"]),
            "auc": lambda: api.plot_shared_static_flicker_auc_summary(static["shared_neuron_metrics"],
                static["fish_median_delta_auc"], mfa.compute_static_flicker_fish_level_statistics(static["fish_median_delta_auc"])),
            "recruitment": lambda: api.plot_recruitment_amplification(static["recruitment_amplification"]),
            "bout_cell06": lambda: api.plot_bout_flicker_position_cell06_style(side, fps_2p=1., t_pre_s=1.),
            "bout_four_panel": lambda: api.plot_bout_flicker_position_figure(bout_four_panels(side)),
        }
    if family == "specificity":
        summary = out["metrics"]
        similarity = out["similarity"]
        return {
            "summary": lambda: api.plot_stimulus_specificity_summary(summary, STIM_LABELS[::-1]),
            "selectivity": lambda: api.plot_stimulus_specificity_selectivity_index(summary, STIM_LABELS[::-1]),
            "heatmaps": lambda: api.plot_similarity_heatmaps(similarity["pearson_similarity_matrix"], similarity["cosine_similarity_matrix"]),
            "distance": lambda: api.plot_similarity_by_distance(similarity["pair_similarity"]),
        }
    if family == "diagnostics":
        delta = pd.DataFrame(dict(stimulus=["B", "A"] * 8, side=["left"] * 8 + ["right"] * 8,
                                 delta_integral=np.linspace(-2, 3, 16)))
        return {
            "motion_delta": lambda: api.plot_motion_delta_distribution(delta, stimuli=["B", "A"], side=None, sides=["right", "left"]),
            "active_mean": lambda: api.plot_active_trace_decision_diagnostic(out["diagnostic_mean"],
                fps_2p=1., t_pre_s=1., stimuli_durations=ref["stimuli_durations"], show_active_count_trace=True,
                row_separator=3, figsize=(14, 8)),
            "active_concat": lambda: api.plot_active_trace_decision_diagnostic(out["diagnostic_concat"],
                fps_2p=1., t_pre_s=1., stimuli_durations=ref["stimuli_durations"], sort_mode="none", figsize=(14, 8)),
        }
    if family == "lme":
        results, table = lme_inputs()
        return {"models": lambda: api.plot_lme_model_outputs(results, table),
                "intercept": lambda: api._plot_lme_fixed_effects(results["fixed_effects"], include_intercept=True),
                "empty": lambda: api.plot_lme_model_outputs({})}
    if family == "reliability":
        values = fixture["single"]["dfof"].T
        mask = np.array([True, False, True, False, True])
        return {
            "reliability": lambda: api.plot_reliability_diagnostics(values.T, np.array([.1, .2, .7, .8, .9]),
                .5, mask, np.flatnonzero(mask), 2., hist_bins=8),
            "independent_scales": lambda: api.plot_accepted_rejected_rasters(values,
                t=np.arange(100) / 2., kept_mask=mask, share_color_scale=False),
            "venn": lambda: api.plot_venn_3stim({5: np.array([1, 1, 0, 0, 1, 1, 0]),
                6: np.array([0, 1, 1, 0, 1, 0, 1]), 7: np.array([0, 0, 0, 1, 1, 1, 1])},
                stim_labels=["A", "B", "C"]),
        }
    raise ValueError(f"Unknown family {family!r}")

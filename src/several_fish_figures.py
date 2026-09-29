"""High-level several-fish figure orchestration for calcium notebooks."""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import src.active_neuron_analysis as active_neuron_analysis
import src.plotting_all_fish as plot_all_fish
import src.plotting_diagnostics as plot_diagnostics
import src.plotting_specificity as plot_specificity
import src.trial_alignment as trial_alignment
from src.several_fish_diagnostics import (
    build_all_fish_raster_inputs, build_high_sparseness_raster_data,
    build_lifetime_sparseness_summary, build_motion_active_counts,
    build_overlap_diagnostic_data, build_pooled_mean_trace_by_stimulus,
)
from src.several_fish_reporting import _slugify_label


def build_all_fish_raster_figure(
    all_fish_data,
    fish_ids,
    stim_order,
    reference_fish,
    timing,
    raster_settings,
    stimuli_colors,
    stimuli_linestyles,
    figsize=(8, 8),
):
    """Build and order a configured all-fish raster figure.

    Args:
        all_fish_data (dict): Fish ID to aligned trace bundles.
        fish_ids (sequence): Fish IDs in pooled neuron order.
        stim_order (sequence): Stimuli in raster block order.
        reference_fish (dict): Stimulus map, durations, and reference traces.
        timing (dict): Imaging frame rate and prestimulus duration.
        raster_settings (dict): Trace, repetition, and neuron-sort settings.
        stimuli_colors (dict): Stimulus labels to line colors.
        stimuli_linestyles (dict): Stimulus labels to line styles.
        figsize (tuple): Matplotlib figure size in inches.

    Returns:
        dict: Pooled matrix, optional left-right index, displayed neuron order,
        and the figure, axes, and raster image.
    """
    prepared = build_all_fish_raster_inputs(
        all_fish_data, fish_ids, stim_order, reference_fish, timing, raster_settings,
    )
    flat_matrix_all_fish = prepared["flat_matrix_all_fish"]
    raster_neuron_order = prepared["raster_neuron_order"]

    fig, ax, image, neuron_order = plot_all_fish.plot_allfish_flat_raster(
        data=flat_matrix_all_fish,
        trial_aligned_traces=reference_fish["trial_aligned_traces_z_core"],
        stim_order=stim_order,
        stimuli_id_map=reference_fish["stimuli_id_map"],
        stimuli_durations=reference_fish["stimuli_durations"],
        stimuli_colors=stimuli_colors,
        stimuli_linestyles=stimuli_linestyles,
        fps_2p=timing["fps_2p"],
        t_pre_s=timing["t_pre_s"],
        combine_mode=raster_settings["combine_mode"],
        sort_mode=("corravg" if raster_neuron_order is not None else raster_settings["raster_sort_mode"]),
        neuron_order=raster_neuron_order,
        sort_label=prepared["raster_sort_label"],
        is_binary=False,
        show_mean_trace=True,
        figsize=figsize,
        fish_id=f"all_fish_{_slugify_label(raster_settings.get('analysis_label', raster_settings.get('experiment_name', 'analysis')))}",
    )
    return {
        "flat_matrix_all_fish": flat_matrix_all_fish,
        "raster_left_right_index": prepared["raster_left_right_index"],
        "neuron_order": neuron_order,
        "figure": fig,
        "axes": ax,
        "image": image,
    }


def plot_lifetime_sparseness_analysis(
    all_fish_data,
    fish_ids,
    reference_fish,
    timing,
    selected_stimuli,
    lifetime_sparseness_threshold,
    stimuli_colors,
    stimuli_linestyles,
    combine_mode="concat",
    sparseness_figsize=(5, 4),
    raster_figsize=(12, 8),
    analysis_label="",
):
    """Plot lifetime sparseness and a high-sparseness z-score raster.

    Args:
        all_fish_data (dict): Fish ID to aligned trace bundles.
        fish_ids (sequence): Fish IDs in pooled neuron order.
        reference_fish (dict): Stimulus map, durations, and reference traces.
        timing (dict): Imaging frame rate and prestimulus duration.
        selected_stimuli (sequence): Stimulus IDs or names to analyze.
        lifetime_sparseness_threshold (float): Strict high-sparseness cutoff.
        stimuli_colors (dict): Stimulus labels to line colors.
        stimuli_linestyles (dict): Stimulus labels to line styles.
        combine_mode (str): Repetition combination mode for the raster.
        sparseness_figsize (tuple): Scatter figure size in inches.
        raster_figsize (tuple): Raster figure size in inches.
        analysis_label (str): Figure and filename label.

    Returns:
        dict: Summary table, stimulus selection, prepared raster data, and
        sparseness and optional raster figures.
    """
    if combine_mode not in {"mean", "concat"}:
        raise ValueError("combine_mode must be 'mean' or 'concat'.")
    prepared = build_lifetime_sparseness_summary(
        all_fish_data, fish_ids, reference_fish, timing, selected_stimuli,
    )
    selection = prepared["selection"]
    summary_table = prepared["summary_table"]

    fig_sparseness, ax_sparseness = plt.subplots(figsize=sparseness_figsize)
    plot_specificity.plot_stimulus_specificity_sparseness(
        summary_table,
        selected_stimulus_labels=selection["stimulus_labels"],
        analysis_label=analysis_label,
        ax=ax_sparseness,
    )
    plt.tight_layout()
    plt.show()

    high_sparseness = build_high_sparseness_raster_data(
        all_fish_data=all_fish_data,
        fish_ids=fish_ids,
        summary_table=summary_table,
        stim_order=selection["stimulus_ids"],
        preferred_stimulus_order=selection["stimulus_labels"],
        lifetime_sparseness_threshold=lifetime_sparseness_threshold,
        combine_mode=combine_mode,
        trace_type="zscore",
    )
    if high_sparseness["matrix"].shape[0] == 0:
        print("No neurons exceeded the selected lifetime-sparseness threshold.")
        high_sparseness_figure = None
    else:
        high_sparseness_figure, _, _, _ = plot_all_fish.plot_allfish_flat_raster(
            data=high_sparseness["matrix"],
            trial_aligned_traces=reference_fish["trial_aligned_traces_z_core"],
            stim_order=selection["stimulus_ids"],
            stimuli_id_map=reference_fish["stimuli_id_map"],
            stimuli_durations=reference_fish["stimuli_durations"],
            stimuli_colors=stimuli_colors,
            stimuli_linestyles=stimuli_linestyles,
            fps_2p=timing["fps_2p"],
            t_pre_s=timing["t_pre_s"],
            combine_mode=combine_mode,
            sort_mode="unsorted",
            neuron_order=high_sparseness["plot_neuron_order"],
            sort_label=f"lifetime sparseness > {lifetime_sparseness_threshold}",
            is_binary=False,
            show_mean_trace=True,
            figsize=raster_figsize,
            fish_id=f"all_fish_{_slugify_label(analysis_label or 'analysis')}_high_lifetime_sparseness_zscore",
            vmax=4,
        )
        plt.show()

    return {
        "summary_table": summary_table,
        "selection": selection,
        "sparseness_figure": fig_sparseness,
        "high_sparseness": high_sparseness,
        "high_sparseness_figure": high_sparseness_figure,
    }


def build_plot_all_fish_mean_zscore_traces(
    all_fish_data,
    fish_ids,
    stim_order,
    reference_fish,
    timing,
    stimuli_colors,
    stimuli_linestyles,
    figsize=(7, 4.5),
):
    """Plot all-fish mean z-score traces in selected stimulus order.

    Args:
        all_fish_data (dict): Fish ID to aligned trace bundles.
        fish_ids (sequence): Fish IDs in pooled mean row order.
        stim_order (sequence): Stimulus IDs or names in plotted order.
        reference_fish (dict): Stimulus map, durations, and reference traces.
        timing (dict): Imaging frame rate and peri-stimulus seconds.
        stimuli_colors (dict): Stimulus labels to line colors.
        stimuli_linestyles (dict): Stimulus labels to line styles.
        figsize (tuple): Matplotlib figure size in inches.

    Returns:
        dict: Mean traces, resolved stimulus IDs/labels, figure, axes, and colors.
    """
    selection = trial_alignment.resolve_selected_stimuli(
        stim_order,
        stimuli_id_map=reference_fish["stimuli_id_map"],
        available_stimuli=reference_fish["trial_aligned_traces_z_core"].keys(),
    )
    mean_traces = build_pooled_mean_trace_by_stimulus(
        all_fish_data=all_fish_data,
        fish_ids=fish_ids,
        stim_order=selection["stimulus_ids"],
        trace_type="zscore",
        use_kept_neurons=True,
    )
    fig, ax, colors_used, _ = plot_all_fish.plot_stimulus_means(
        mean_traces=mean_traces,
        stimuli_ids=selection["stimulus_ids"],
        stimuli_names=selection["stimulus_labels"],
        title_prefix="",
        fps_2p=timing["fps_2p"],
        t_post_s=timing["t_post_s"],
        t_pre_s=timing["t_pre_s"],
        stimuli_durations=reference_fish["stimuli_durations"],
        plots_path=None,
        prefix=None,
        save=False,
        stimuli_colors=stimuli_colors,
        stimuli_linestyles=stimuli_linestyles,
        close_after=False,
        kept_cells=None,
        comment="all_stimuli",
        figsize=figsize,
    )
    return {
        "mean_traces": mean_traces,
        "stimulus_ids": selection["stimulus_ids"],
        "stimulus_labels": selection["stimulus_labels"],
        "figure": fig,
        "axes": ax,
        "colors_used": colors_used,
    }


def plot_left_right_active_overlap_diagnostics(
    all_fish_data,
    fish_ids,
    reference_fish,
    timing,
    side_stimuli,
    show_overlap=True,
    show_significant_raster=True,
    active_fraction_threshold=0.10,
    min_epoch_s=1.0,
    min_active_reps=2,
    expected_reps=4,
    require_expected_reps=True,
    motion_onset_s=8.0,
    tau_s=6.0,
    motion_duration_key="motion_sec",
    overlap_figsize=(5, 4),
    raster_figsize=(12, 7),
):
    """Plot overlap heatmaps and significant-raster diagnostics per side.

    Args:
        all_fish_data (dict): Fish ID to aligned trace bundles.
        fish_ids (sequence): Fish IDs in pooled neuron order.
        reference_fish (dict): Stimulus map, durations, and reference traces.
        timing (dict): Imaging frame rate and prestimulus duration.
        side_stimuli (dict): Left/right sides to ordered stimulus selections.
        show_overlap (bool): Draw the overlap heatmaps.
        show_significant_raster (bool): Draw pooled active-trace diagnostics.
        active_fraction_threshold (float): Minimum active-trial fraction.
        min_epoch_s (float): Minimum active epoch in seconds.
        min_active_reps (int): Minimum active repetitions.
        expected_reps (int): Configured repetition count.
        require_expected_reps (bool): Require the configured repetition count.
        motion_onset_s (float): Motion onset after stimulus start, in seconds.
        tau_s (float): Response-window extension in seconds.
        motion_duration_key (str): Duration field in stimulus metadata.
        overlap_figsize (tuple): Heatmap size in inches.
        raster_figsize (tuple): Active diagnostic size in inches.

    Returns:
        dict: Side name to prepared overlap and trace diagnostic data.
    """
    expected_sides = {"left", "right"}
    if set(side_stimuli) != expected_sides:
        raise ValueError("side_stimuli must contain exactly 'left' and 'right'.")

    selections = {
        side: trial_alignment.resolve_selected_stimuli(
            stimuli,
            stimuli_id_map=reference_fish["stimuli_id_map"],
            available_stimuli=reference_fish["trial_aligned_traces_raster"].keys(),
        )
        for side, stimuli in side_stimuli.items()
    }
    side_stimulus_ids = {side: selection["stimulus_ids"] for side, selection in selections.items()}
    side_stimulus_labels = {side: selection["stimulus_labels"] for side, selection in selections.items()}
    active_stim_order = list(dict.fromkeys(side_stimulus_ids["left"] + side_stimulus_ids["right"]))
    active_matrices = active_neuron_analysis.build_active_neuron_matrices_all_fish(
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
        results[side] = diagnostic_data

        if show_overlap:
            fig, ax = plt.subplots(figsize=overlap_figsize)
            sns.heatmap(
                diagnostic_data["matrix_to_plot"],
                vmin=0,
                vmax=1,
                square=True,
                annot=True,
                fmt=".2f",
                cmap="viridis",
                cbar_kws={"label": "Jaccard overlap"},
                ax=ax,
            )
            ax.set(title=f"{side.capitalize()} active-neuron overlap", xlabel="", ylabel="")
            plt.tight_layout()
            plt.show()

        if show_significant_raster:
            diagnostic = diagnostic_data["active_trace_diagnostic"]
            if diagnostic["trace_matrix"].shape[0] == 0:
                print(f"No active neurons passed the {side} settings.")
            else:
                plot_diagnostics.plot_active_trace_decision_diagnostic(
                    diagnostic,
                    fps_2p=timing["fps_2p"],
                    stimuli_durations=reference_fish["stimuli_durations"],
                    t_pre_s=timing["t_pre_s"],
                    sort_mode=None,
                    neuron_order=diagnostic_data["diagnostic_neuron_order"],
                    trace_cmap="Greys",
                    show_active_count_trace=True,
                    active_count_threshold=0.5,
                    active_count_ylabel="# Active neurons",
                    active_count_color="black",
                    active_count_height_ratio=1.8,
                    figsize=raster_figsize,
                )
                plt.show()

    return results


def plot_motion_active_neuron_counts(
    all_fish_data,
    fish_ids,
    reference_fish,
    timing,
    selected_stimuli,
    stimuli_colors,
    figsize=(10, 4.5),
    active_fraction_threshold=0.10,
    min_epoch_s=1.0,
    min_active_reps=2,
    expected_reps=4,
    require_expected_reps=True,
    motion_onset_s=8.0,
    tau_s=6.0,
    motion_duration_key="motion_sec",
):
    """Plot mean motion-active counts with one point per fish.

    Args:
        all_fish_data (dict): Fish ID to aligned trace bundles.
        fish_ids (sequence): Fish IDs in count-table row order.
        reference_fish (dict): Stimulus map and reference traces.
        timing (dict): Imaging frame rate and prestimulus duration.
        selected_stimuli (sequence): Stimulus IDs or names in plotted order.
        stimuli_colors (dict): Stimulus labels to bar colors.
        figsize (tuple): Matplotlib figure size in inches.
        active_fraction_threshold (float): Minimum active-trial fraction.
        min_epoch_s (float): Minimum active epoch in seconds.
        min_active_reps (int): Minimum active repetitions.
        expected_reps (int): Configured repetition count.
        require_expected_reps (bool): Require the configured repetition count.
        motion_onset_s (float): Motion onset after stimulus start, in seconds.
        tau_s (float): Response-window extension in seconds.
        motion_duration_key (str): Duration field in stimulus metadata.

    Returns:
        dict: Selection, active decisions, fish-by-stimulus count table, and figure.
    """
    selection = trial_alignment.resolve_selected_stimuli(
        selected_stimuli,
        stimuli_id_map=reference_fish["stimuli_id_map"],
        available_stimuli=reference_fish["trial_aligned_traces_raster"].keys(),
    )
    active_matrices = active_neuron_analysis.build_active_neuron_matrices_all_fish(
        all_fish_data=all_fish_data,
        fish_ids=fish_ids,
        stim_order=selection["stimulus_ids"],
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
    counts = build_motion_active_counts(
        active_matrices, fish_ids, selection["stimulus_ids"], selection["stimulus_labels"],
    )

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(counts.shape[1])
    colors = [stimuli_colors.get(label, f"C{idx}") for idx, label in enumerate(counts.columns)]
    ax.bar(x, counts.mean(axis=0), color=colors, alpha=0.7)
    for offset, fish_id in zip(np.linspace(-0.12, 0.12, len(fish_ids)), fish_ids):
        ax.scatter(x + offset, counts.loc[fish_id], color="black", s=32, zorder=3)
    ax.set(
        xticks=x,
        xticklabels=counts.columns,
        ylabel="Active neurons per fish",
        title="Motion-period active neurons per stimulus",
    )
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="x", rotation=35)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    plt.show()
    return {"selection": selection, "active_matrices": active_matrices, "counts": counts, "figure": fig, "axes": ax}

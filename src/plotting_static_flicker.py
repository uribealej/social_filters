"""Static-flicker category, recruitment and paired-response figures."""

import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.lines import Line2D
import pandas as pd
from src.neuron_ordering import (
    _first_post_onset_raster_time,
)
from src.plotting_common import (
    STATIC_FLICKER_CATEGORY_ORDER,
    STATIC_FLICKER_CATEGORY_COLORS,
    STATIC_FLICKER_STIMULUS_COLORS,
)


def plot_static_flicker_classification_raster(
    classification_raster_data,
    static_label="Static",
    comparison_label="Flicker",
):
    """Return independent category rasters for an editable pair of time windows."""
    data = pd.DataFrame(classification_raster_data).copy()
    figures = {}
    for side in ("left", "right"):
        positions = data.loc[data["side"] == side, ["stim_id", "stimulus"]].drop_duplicates().sort_values("stim_id")
        n_positions = len(positions)
        n_cols = min(max(n_positions, 1), 2)
        n_rows = int(math.ceil(max(n_positions, 1) / n_cols))
        figure = plt.figure(figsize=(4.6 * n_cols, 4.8 * n_rows))
        axes = [figure.add_subplot(n_rows, n_cols, index + 1) for index in range(max(n_positions, 1))]
        for col_idx, position in enumerate(positions.itertuples()):
            ax = axes[col_idx]
            subset = data.loc[(data["side"] == side) & (data["stim_id"] == position.stim_id)].copy()
            subset["category"] = pd.Categorical(
                subset["category"], categories=STATIC_FLICKER_CATEGORY_ORDER, ordered=True
            )
            subset["motion_activity_onset_s"] = [
                _first_post_onset_raster_time(row.flicker_raster, row.flicker_time_s)
                for row in subset.itertuples()
            ]
            subset = subset.sort_values(
                ["category", "motion_activity_onset_s", "fish_id", "neuron_id"],
                kind="stable",
            )
            if subset.empty:
                ax.set_axis_off()
                continue
            matrix = np.vstack([
                np.concatenate([np.asarray(row.static_raster), np.asarray(row.flicker_raster)])
                for row in subset.itertuples()
            ])
            static_width = len(np.asarray(subset.iloc[0]["static_raster"]))
            activity_rgb = np.repeat((1.0 - matrix)[..., None], 3, axis=2)
            category_rgb = np.asarray(
                [
                    tuple(int(STATIC_FLICKER_CATEGORY_COLORS[name].lstrip("#")[offset:offset + 2], 16) / 255.0
                          for offset in (0, 2, 4))
                    for name in subset["category"].astype(str)
                ]
            )
            image_rgb = np.concatenate([category_rgb[:, None, :], activity_rgb], axis=1)
            ax.imshow(image_rgb, aspect="auto", interpolation="nearest")
            ax.set_title(f"{side.title()} {position.stimulus}")
            static_duration = float(subset.iloc[0]["static_window_s"])
            flicker_duration = float(subset.iloc[0]["flicker_window_s"])
            static_time_s = np.asarray(subset.iloc[0]["static_time_s"], dtype=float)
            flicker_time_s = np.asarray(subset.iloc[0]["flicker_time_s"], dtype=float)
            static_tick_indices = np.unique(np.linspace(0, static_time_s.size - 1, 3, dtype=int))
            flicker_tick_indices = np.unique(np.linspace(0, flicker_time_s.size - 1, 3, dtype=int))
            tick_positions = np.concatenate([
                1 + static_tick_indices,
                1 + static_width + flicker_tick_indices,
            ])
            tick_labels = [
                f"{time_s:g}" for time_s in np.concatenate([
                    static_time_s[static_tick_indices], flicker_time_s[flicker_tick_indices]
                ])
            ]
            ax.set_xticks(tick_positions, tick_labels)
            ax.set_xlabel(
                f"Time from onset (s): {static_label} {static_duration:g} s | "
                f"{comparison_label} {flicker_duration:g} s"
            )
            event_lines = [(0.0, "black", "--")]
            stimulus_offset_s = float(subset.iloc[0]["stimulus_offset_relative_s"])
            if stimulus_offset_s > 0:
                event_lines.append((stimulus_offset_s, "#d95f02", ":"))
            for event_s, color, linestyle in event_lines:
                for start, times in ((1, static_time_s), (1 + static_width, flicker_time_s)):
                    if times.size and times[0] <= event_s <= times[-1]:
                        event_x = start + np.interp(event_s, times, np.arange(times.size))
                        ax.axvline(event_x, color=color, linestyle=linestyle, linewidth=1.1, zorder=3)
            if col_idx == 0:
                ax.set_ylabel("Neurons (category, then motion-onset ordered)")
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)
        figure.suptitle(
            f"{side.title()} hemifield — categories then first post-onset raster activity: gray non-responsive | blue static-only | green shared | red newly recruited",
            fontsize=10,
        )
        figure.legend(
            handles=[
                Line2D([0], [0], color="black", linestyle="--", label="stimulus onset"),
                Line2D([0], [0], color="#d95f02", linestyle=":", label="stimulus offset"),
            ],
            loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2, frameon=False, fontsize=8,
        )
        figure.subplots_adjust(top=0.79, wspace=0.26, hspace=0.38, bottom=0.14)
        figures[side] = figure
    return figures


def plot_static_flicker_category_proportions(fish_side_summary, figsize=(12, 5)):
    """Return independent left/right per-fish category-proportion figures."""
    summary = pd.DataFrame(fish_side_summary).copy()
    figures = {}
    for side in ("left", "right"):
        positions = summary.loc[summary["side"] == side, ["stim_id", "stimulus"]].drop_duplicates().sort_values("stim_id")
        n_positions = len(positions)
        n_cols = min(max(n_positions, 1), 3)
        n_rows = int(math.ceil(max(n_positions, 1) / n_cols))
        fig = plt.figure(figsize=(4.6 * n_cols, 4.5 * n_rows))
        axes = [fig.add_subplot(n_rows, n_cols, index + 1) for index in range(max(n_positions, 1))]
        for col_idx, position in enumerate(positions.itertuples()):
            ax = axes[col_idx]
            subset = summary.loc[(summary["side"] == side) & (summary["stim_id"] == position.stim_id)].sort_values("fish_id")
            x = np.arange(len(subset))
            bottom = np.zeros(len(subset), dtype=float)
            for category in STATIC_FLICKER_CATEGORY_ORDER:
                values = subset[f"{category}_proportion"].fillna(0).to_numpy(float)
                ax.bar(x, values, bottom=bottom, color=STATIC_FLICKER_CATEGORY_COLORS[category], label=category)
                bottom += values
            ax.set_xticks(x, subset["fish_id"], rotation=45, ha="right")
            ax.set_ylim(0, 1)
            ax.set_title(f"{side.title()} {position.stimulus}")
            ax.set_ylabel("Proportion of valid neurons")
        if n_positions:
            axes[0].legend(loc="upper right", frameon=False, fontsize=7)
        fig.suptitle(f"{side.title()} hemifield: category proportions", fontsize=11)
        fig.subplots_adjust(top=0.86, bottom=0.22, wspace=0.28)
        figures[side] = fig
    return figures


def plot_pooled_static_flicker_category_proportions(pooled_category_summary):
    """Return left/right descriptive pooled-cell category-proportion figures."""
    summary = pd.DataFrame(pooled_category_summary).copy()
    figures = {}
    for side in ("left", "right"):
        subset = summary.loc[summary["side"] == side].sort_values("stim_id")
        figure, axis = plt.subplots(figsize=(max(6.4, 1.8 * len(subset)), 4.8))
        x = np.arange(len(subset))
        bottom = np.zeros(len(subset), dtype=float)
        # Draw in reverse so the visible top-to-bottom stack matches Cell 06.
        for category in reversed(STATIC_FLICKER_CATEGORY_ORDER):
            values = subset[f"{category}_proportion"].fillna(0).to_numpy(float)
            axis.bar(x, values, bottom=bottom, width=0.72, color=STATIC_FLICKER_CATEGORY_COLORS[category], label=category)
            bottom += values
        axis.set_xticks(x, subset["stimulus"])
        axis.set_ylim(0, 1)
        axis.set_xlabel("Stimulus position")
        axis.set_ylabel("Proportion of pooled valid neurons")
        axis.set_title(f"{side.title()} hemifield: pooled-cell recruitment categories", pad=12)
        handles, labels = axis.get_legend_handles_labels()
        ordered_handles = [handles[labels.index(category)] for category in STATIC_FLICKER_CATEGORY_ORDER]
        axis.legend(
            ordered_handles, STATIC_FLICKER_CATEGORY_ORDER,
            frameon=False, fontsize=8, loc="center left",
            bbox_to_anchor=(1.02, 0.5), borderaxespad=0,
        )
        figure.subplots_adjust(left=0.14, right=0.75, bottom=0.18, top=0.88)
        figures[side] = figure
    return figures


def plot_shared_static_flicker_auc_summary(
    shared_neuron_metrics, fish_median_delta_auc, fish_level_statistics, figsize=(12, 5)
):
    """Return descriptive neuron views plus fish-level ΔAUC inference views."""
    shared = pd.DataFrame(shared_neuron_metrics).copy()
    fish_medians = pd.DataFrame(fish_median_delta_auc).copy()
    statistics = pd.DataFrame(fish_level_statistics).copy()
    figures = {}
    for side in ("left", "right"):
        figure, (scatter_ax, summary_ax) = plt.subplots(1, 2, figsize=figsize)
        side_shared = shared.loc[shared["side"] == side].copy()
        side_medians = fish_medians.loc[fish_medians["side"] == side].copy()
        positions = side_medians[["stim_id", "stimulus"]].drop_duplicates().sort_values("stim_id")
        significance_symbols = []
        for index, position in enumerate(positions.itertuples()):
            rows = side_shared.loc[side_shared["stim_id"] == position.stim_id]
            color = STATIC_FLICKER_STIMULUS_COLORS[index % len(STATIC_FLICKER_STIMULUS_COLORS)]
            scatter_ax.scatter(rows["static_auc"], rows["flicker_auc"], s=13, alpha=0.55, color=color, label=position.stimulus)
            fish_values = side_medians.loc[side_medians["stim_id"] == position.stim_id, "median_delta_auc"].dropna().to_numpy(float)
            jitter = np.linspace(-0.10, 0.10, len(fish_values)) if len(fish_values) > 1 else np.array([0.0])
            summary_ax.scatter(np.full(len(fish_values), index) + jitter, fish_values, color=color, alpha=0.85)
            test_rows = statistics.loc[
                (statistics["side"] == side) & (statistics["test"] == "ΔAUC > 0")
                & (statistics["stimulus_a"] == position.stimulus)
            ]
            if len(fish_values) and not test_rows.empty:
                test_row = test_rows.iloc[0]
                summary_ax.errorbar(index, test_row["effect_median"], yerr=[[test_row["effect_median"] - test_row["ci_low"]], [test_row["ci_high"] - test_row["effect_median"]]], color="black", capsize=3, linewidth=1.2, zorder=3)
                summary_ax.scatter(index, test_row["effect_median"], color="black", marker="_", s=320, linewidths=2.5, zorder=4)
                raw_p = float(test_row["p_wilcoxon"])
                symbol = "***" if raw_p < 0.001 else "**" if raw_p < 0.01 else "*" if raw_p < 0.05 else ""
                if symbol:
                    significance_symbols.append((index, symbol))
        finite = side_shared[["static_auc", "flicker_auc"]].to_numpy(float)
        if finite.size:
            limits = np.nanpercentile(finite, [1, 99])
            scatter_ax.set_xlim(limits); scatter_ax.set_ylim(limits)
            scatter_ax.plot(limits, limits, color="black", linestyle="--", linewidth=1.1, label="no change")
        scatter_ax.set_title(f"{side.title()}: shared neurons"); scatter_ax.set_xlabel("Static AUC"); scatter_ax.set_ylabel("Motion AUC")
        scatter_ax.legend(title="Stimulus", frameon=False, fontsize=8)
        summary_ax.axhline(0, color="black", linestyle="--", linewidth=1.1, zorder=0)
        summary_ax.set_xticks(np.arange(len(positions)), positions["stimulus"])
        summary_ax.set_title("Fish median ΔAUC versus zero"); summary_ax.set_xlabel("Stimulus"); summary_ax.set_ylabel("Median motion − static AUC")
        y_bottom, y_top = summary_ax.get_ylim()
        summary_ax.set_ylim(y_bottom, y_top + 0.10 * (y_top - y_bottom))
        for index, symbol in significance_symbols:
            summary_ax.text(index, y_top + 0.04 * (y_top - y_bottom), symbol, ha="center", va="bottom", fontsize=11, fontweight="bold")
        figure.subplots_adjust(wspace=0.32, bottom=0.18)
        figures[side] = figure
    return figures


def plot_recruitment_amplification(recruitment_amplification, figsize=(12, 8)):
    """Return independent left/right recruitment-versus-amplification figures."""
    data = pd.DataFrame(recruitment_amplification).copy()
    figures = {}
    for side in ("left", "right"):
        positions = data.loc[data["side"] == side, ["stim_id", "stimulus"]].drop_duplicates().sort_values("stim_id")
        n_positions = len(positions)
        n_cols = min(max(n_positions, 1), 3)
        n_rows = int(math.ceil(max(n_positions, 1) / n_cols))
        fig = plt.figure(figsize=(4.6 * n_cols, 4.8 * n_rows))
        axes = [fig.add_subplot(n_rows, n_cols, index + 1) for index in range(max(n_positions, 1))]
        for col_idx, position in enumerate(positions.itertuples()):
            ax = axes[col_idx]
            subset = data.loc[(data["side"] == side) & (data["stim_id"] == position.stim_id)].sort_values("fish_id")
            x = np.arange(len(subset))
            width = 0.38
            ax.bar(x - width / 2, subset["recruitment"], width, label="Recruitment", color="#e15759")
            ax.bar(x + width / 2, subset["amplification"], width, label="Amplification", color="#59a14f")
            ax.set_xticks(x, subset["fish_id"], rotation=45, ha="right")
            ax.set_title(f"{side.title()} {position.stimulus}")
            ax.set_ylabel("Summed z-score AUC")
        if n_positions:
            axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{side.title()} hemifield: recruitment and amplification", fontsize=11)
        fig.subplots_adjust(top=0.86, bottom=0.22, wspace=0.30)
        figures[side] = fig
    return figures

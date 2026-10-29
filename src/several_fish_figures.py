"""High-level several-fish figure orchestration for calcium notebooks."""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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
from src.several_fish_reporting import slugify_label


def plot_phase_cohort_rasters(display, trace_type="zscore", combine_mode="mean",
                               figsize=(20, 9), z_limits=(-1, 4), raster_cmap="Greys",
                               zscore_cmap="magma"):
    """Plot pooled matrices, original repeat numbers and optional fish boundaries.

    raster_cmap controls significant activity (default: white=0, black=1).
    zscore_cmap controls z-score heatmaps without changing z_limits.
    Pooled sorting uses neuron positions without fish boundaries. Returns fig,
    axes; input display matrices and original neuron identities are unchanged.
    """
    from src.plotting_single_fish import add_phase_annotations
    from matplotlib.patches import Patch

    ids, keys = display["stimulus_ids"], display["stimulus_keys"]
    fig, axes = plt.subplots(2, int(np.ceil(len(ids) / 2)), figsize=figsize,
        sharex=True, sharey=True, squeeze=False, layout="constrained")
    reference = display["reference"]
    fps = reference["fps_2p"]
    order = display["neuron_order"]
    counts = display["row_metadata"].groupby("fish_id", sort=False).size()
    grouped = display.get("sort_scope", "within_fish") == "within_fish"
    largest = 0
    for ax, sid, key in zip(axes.flat, ids, keys):
        values = display["matrices"][trace_type, combine_mode][sid][order]
        times = reference["time_s"][sid]
        period = len(times) / fps
        repeats = 1 if combine_mode == "mean" else reference["trial_aligned_traces"][sid].shape[2]
        left = times[0] - 0.5 / fps if combine_mode == "mean" else 0
        right = left + values.shape[1] / fps
        largest = max(largest, right)
        im = ax.imshow(values, aspect="auto", interpolation="nearest", origin="lower",
            extent=[left, right, -0.5, len(order) - 0.5], cmap=raster_cmap if trace_type == "raster" else zscore_cmap,
            vmin=0 if trace_type == "raster" else z_limits[0],
            vmax=1 if trace_type == "raster" else z_limits[1])
        row = reference["trials"].loc[reference["trials"].stimulus_id == sid].iloc[0]
        anchor = row.onset_s if display["alignment"] == "sequence" else row.bout_onset_s
        numbers = display.get("repetition_numbers", {}).get(sid,
            list(range(1, reference["trial_aligned_traces"][sid].shape[2] + 1)))
        for rep in range(repeats):
            offset = 0 if combine_mode == "mean" else rep * period - times[0]
            add_phase_annotations(ax, row, anchor, offset)
            if combine_mode == "concat":
                ax.axvline(rep * period, color="gray", linewidth=0.5)
                ax.text((rep + 0.5) * period, 0.98, str(numbers[rep]), transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=7, color="#555555",
                        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1))
        for boundary in counts.cumsum().to_numpy()[:-1] if grouped else []:
            ax.axhline(boundary - 0.5, color="cyan", linewidth=0.7)
        ax.set_title(f"{key}\nTrials {', '.join(map(str, numbers))}", fontsize=10)
    for ax in axes.flat[len(ids):]:
        ax.set_visible(False)
    minimum = min(reference["time_s"][sid][0] - 0.5 / fps for sid in ids) if combine_mode == "mean" else 0
    for ax in axes.flat[:len(ids)]:
        ax.set_xlim(minimum, largest)
    for ax in axes[-1]:
        ax.set_xlabel(f"From {display['alignment']} onset (s)" if combine_mode == "mean" else "Concatenated trials (s)", fontsize=8)
    centers = counts.cumsum() - counts / 2
    for ax in axes[:, 0]:
        if grouped:
            ax.set(yticks=centers.to_numpy(), yticklabels=counts.index, ylabel="Fish / pooled neurons")
        else:
            ax.set_ylabel("Pooled neurons (sorted)")
    units = ("Trial activation fraction" if combine_mode == "mean" else "Event (0/1)") if trace_type == "raster" else "Z-score"
    fig.colorbar(im, ax=axes.ravel().tolist(), label=units, shrink=0.7)
    cohort_label = "SINGLE-FISH preview" if len(display["fish_ids"]) == 1 else f"{len(display['fish_ids'])} fish"
    scope_label = "within fish" if grouped else "across fish"
    sorting = display.get("sort_reference", {})
    keys_label = "+".join(sorting.get("stimulus_keys") or ["all conditions"])
    window_label = "full window" if sorting.get("window_s") is None else f"{sorting['window_s'][0]:g}-{sorting['window_s'][1]:g} s"
    fig.suptitle(f"{cohort_label} | {display['alignment']}-aligned | {trace_type} | {combine_mode} | {len(order)} reliable neurons\n"
                 f"{display.get('sort_mode', 'corravg')} sorting {scope_label}: {keys_label}, "
                 f"{sorting.get('alignment', 'sequence')} {window_label} (all trials)")
    fig.legend(handles=[Patch(color=color, alpha=0.5, label=name) for name, color in
        [("Preconditioning", "#b59bd6"), ("Blank delay", "#d9dde3"), ("Bout", "#52b9b1")]],
        loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=3, frameon=False)
    fig.canvas.draw()
    return fig, axes


def plot_phase_cohort_means(display, figsize=(20, 7)):
    """Individual fish population means with an equally weighted fish mean."""
    from src.plotting_single_fish import add_phase_annotations

    ids = display["stimulus_ids"]
    fig, axes = plt.subplots(2, int(np.ceil(len(ids) / 2)), figsize=figsize,
        sharex=True, sharey=True, squeeze=False, layout="constrained")
    colors = plt.get_cmap("tab10")
    reference = display["reference"]
    for ax, sid, key in zip(axes.flat, ids, display["stimulus_keys"]):
        values, times = display["fish_means"][sid], reference["time_s"][sid]
        for i, fid in enumerate(display["fish_ids"]):
            ax.plot(times, values[i], color=colors(i % 10), alpha=0.5, linewidth=0.7, label=fid)
        ax.plot(times, np.nanmean(values, axis=0), color="black", linewidth=1.5, label="Equal-fish mean")
        row = reference["trials"].loc[reference["trials"].stimulus_id == sid].iloc[0]
        add_phase_annotations(ax, row, row.onset_s if display["alignment"] == "sequence" else row.bout_onset_s)
        ax.axhline(0, color="gray", linewidth=0.5)
        numbers = display.get("repetition_numbers", {}).get(sid,
            list(range(1, reference["trial_aligned_traces"][sid].shape[2] + 1)))
        ax.set_title(f"{key}\nTrials {', '.join(map(str, numbers))}", fontsize=10)
    for ax in axes.flat[len(ids):]:
        ax.set_visible(False)
    xmin = min(reference["time_s"][sid][0] for sid in ids)
    xmax = max(reference["time_s"][sid][-1] for sid in ids)
    axes[0, 0].set_xlim(xmin, xmax)
    for ax in axes[-1]:
        ax.set_xlabel(f"From {display['alignment']} onset (s)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Population mean z-score")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=min(5, len(labels)), frameon=False)
    fig.suptitle(f"{'SINGLE-FISH preview' if len(display['fish_ids']) == 1 else str(len(display['fish_ids'])) + ' fish'} | {display['alignment']}-aligned | equal fish weights")
    fig.canvas.draw()
    return fig, axes


def plot_phase_fish_summary(table, value_column, side_conditions, fish_ids,
                            window, ylabel, figsize=(13, 4), title=""):
    """Use the existing paired-fish plot for side-separated phase metrics."""
    fig, axes = plt.subplots(1, len(side_conditions), figsize=figsize, sharey=True,
                              squeeze=False, layout="constrained")
    for ax, (side, conditions) in zip(axes.flat, side_conditions.items()):
        rows = table.loc[table.side.eq(side) & table.window.eq(window)]
        plot_diagnostics.plot_paired_fish_values(rows, value_column, conditions, fish_ids, ax, ylabel)
        ax.set_title(side.capitalize())
        if value_column == "active_fraction" or value_column == "fraction":
            ax.set_ylim(-0.03, 1.03)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=min(5, len(labels)), frameon=False)
    fig.suptitle(f"{title or value_column} | {window} | {len(fish_ids)} fish")
    fig.canvas.draw()
    return fig, axes


def plot_phase_recruitment_rasters(recruitment, side, figsize=(20, 9)):
    """One pooled preconditioning panel followed by control/condition bout panels.

    Grayscale is the fraction of selected trials with significant activity,
    averaged across conditioned stimuli only in the preconditioning panel.
    Membership strips use native-window classifications; rows stay identical
    across panels. Returns (figure, axes), without changing prepared inputs.
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    display = recruitment["displays"][side]
    conditions, order = display["conditions"], display["neuron_order"]
    panels = ["preconditioning", *conditions]
    fig, axes = plt.subplots(1, len(panels), figsize=figsize, sharey=True,
        squeeze=False, layout="constrained")
    axes = axes[0]
    colors = ["#eeeeee", "#377eb8", "#ff8c28", "#984ea3", "#777777"]
    groups = display["displayed_row_metadata"].group.to_numpy()
    boundaries = np.flatnonzero(np.diff(groups)) + 0.5
    for ax, panel in zip(axes, panels):
        values = display["matrices"][panel][order]
        times = display["time_s"][panel]
        dt = times[1] - times[0] if len(times) > 1 else 0.5
        im = ax.imshow(values, origin="lower", aspect="auto", interpolation="nearest",
            cmap="Greys", vmin=0, vmax=1,
            extent=[times[0] - dt / 2, times[-1] + dt / 2, -0.5, len(order) - 0.5])
        if panel == "preconditioning":
            codes = display["pre_active_pool"][order]
            codes = np.where(np.isfinite(codes), codes, 2)
            cmap = ListedColormap(["#eeeeee", "#4daf4a", "#777777"])
            limit = 2
            ax.set_title("Preconditioning\nConditioned-stimulus mean", fontsize=9)
            ax.set_xlabel("From preconditioning onset (s)", fontsize=8)
        else:
            codes = display["membership_codes"][panel][order]
            cmap, limit = ListedColormap(colors), 4
            ax.set_title(panel + " bout", fontsize=10)
            ax.set_xlabel("From bout onset (s)", fontsize=8)
        strip = ax.inset_axes([0.973, 0, 0.027, 1])
        strip.imshow(codes[:, None], aspect="auto", origin="lower", interpolation="nearest",
            cmap=cmap, vmin=0, vmax=limit)
        strip.set_axis_off()
        for boundary in boundaries:
            ax.axhline(boundary, color="#555555", linestyle=":", linewidth=0.5)
        ax.tick_params(labelsize=8)
    centers, labels = [], []
    for group in np.unique(groups):
        positions = np.flatnonzero(groups == group)
        centers.append(positions.mean())
        labels.append(f"{display['group_labels'][group]} ({len(positions)})")
    # Keep labels for small recruitment groups readable, with leaders to rows.
    label_positions = []
    for center in centers:
        label_positions.append(max(center, label_positions[-1] + len(order) * 0.025)
                               if label_positions else center)
    axes[0].set(yticks=label_positions, yticklabels=labels, ylabel="Same pooled neurons in every panel")
    from matplotlib.transforms import blended_transform_factory
    transform = blended_transform_factory(axes[0].transAxes, axes[0].transData)
    for actual, position in zip(centers, label_positions):
        if actual != position:
            axes[0].plot([-0.035, 0], [position, actual], transform=transform,
                color="#555555", linewidth=0.5, clip_on=False)
    fig.colorbar(im, ax=axes.tolist(), label="Significant-frame fraction", shrink=0.7, pad=0.015)
    pre_label = f"Preconditioning-active in >= {recruitment['pre_active_fraction']:.0%} of presentations"
    fig.legend(handles=[Patch(color="#4daf4a", label=pre_label)]
        + [Patch(color=colors[i], label=label) for i, label in
           [(1, "Shared with C1"), (2, "New after own preconditioning"),
            (3, "Both memberships"), (4, "Unclassified")]],
        loc="lower center", bbox_to_anchor=(0.5, -0.065), ncol=3, frameon=False, fontsize=8)
    n_fish = display["row_metadata"].fish_id.nunique()
    cohort_label = "SINGLE-FISH preview" if n_fish == 1 else f"{n_fish} fish"
    fig.suptitle(f"{side.capitalize()} | {cohort_label} | significant activity | repetitions "
        f"{', '.join(map(str, display['repetitions']))} | {len(order)} neurons\n"
        "Rows: consistent preconditioning, C1-active, additional bout recruitment by condition, other")
    fig.canvas.draw()
    return fig, axes


def plot_phase_recruitment_decisions(recruitment, side, figsize=(14, 8)):
    """Reuse the Exp 7 trace/decision diagnostic with strict 0/1 phase columns.

    Preconditioning decisions use the pooled trial fraction; bout decisions
    retain the per-condition passing-repetition rule. NaN is gray, not zero.
    Returns figure and (trace axis, decision axis) with the same row ordering
    and orientation as the recruitment rasters.
    """
    from matplotlib.colors import ListedColormap

    display = recruitment["displays"][side]
    keys = ["preconditioning", *display["conditions"]]
    blocks = [display["matrices"][key] for key in keys]
    diagnostic = dict(trace_matrix=np.concatenate(blocks, axis=1),
        decision_matrix=display["decision_matrix"],
        stim_labels=["Preconditioning", *display["conditions"]],
        trace_block_widths=[block.shape[1] for block in blocks], combine_mode="mean")
    decision_cmap = ListedColormap(["white", "black"]).with_extremes(bad="#999999")
    fig, axes, _ = plot_diagnostics.plot_active_trace_decision_diagnostic(
        diagnostic, fps_2p=display["fps_2p"], t_pre_s=0,
        neuron_order=display["neuron_order"], figsize=figsize,
        decision_cmap=decision_cmap, trace_title="Mean significant activity")
    axes[0].invert_yaxis()
    axes[0].get_subplotspec().get_gridspec().update(wspace=0.45)
    axes[0].images[0].colorbar.set_label("Mean significant activity")
    axes[1].set_title("Binary active decision")
    axes[1].images[0].colorbar.set_ticks([0, 1])
    axes[1].images[0].colorbar.set_label("0 inactive / 1 active")
    for boundary in np.arange(len(keys) - 1) + 0.5:
        axes[1].axvline(boundary, color="#cccccc", linewidth=0.5)
    groups = display["displayed_row_metadata"].group.to_numpy()
    for boundary in np.flatnonzero(np.diff(groups)) + 0.5:
        for ax in axes:
            ax.axhline(boundary, color="#555555", linestyle=":", linewidth=0.5)
    pool = recruitment["preconditioning_pool"].loc[lambda t: t.side.eq(side)]
    expected, minimum = int(pool.n_expected_trials.iloc[0]), int(pool.min_passing_trials.iloc[0])
    n_fish = display["row_metadata"].fish_id.nunique()
    cohort_label = "SINGLE-FISH preview" if n_fish == 1 else f"{n_fish} fish"
    fig.suptitle(f"{side.capitalize()} | {cohort_label} | preconditioning: >= {minimum}/{expected} passing trials | "
        f"bouts: >= {recruitment['min_active_reps']}/{len(display['repetitions'])} | "
        "repetitions " + ", ".join(map(str, display["repetitions"])))
    fig.canvas.draw()
    return fig, axes


def plot_phase_recruitment_counts(recruitment, fish_ids, side_conditions, figsize=(13, 8)):
    """Equal-fish mean count bars and individual fish points for two memberships."""
    fig, axes = plt.subplots(2, len(side_conditions), figsize=figsize,
        sharey="row", squeeze=False, layout="constrained")
    table = recruitment["fish_summary"]
    colors = plt.get_cmap("tab10")
    for j, (side, conditions) in enumerate(side_conditions.items()):
        for i, (metric, title) in enumerate([
                ("n_shared_c1", "Shared with C1 during bout"),
                ("n_new_after_pre", "New after own preconditioning")]):
            ax = axes[i, j]
            values = table.loc[table.side.eq(side)].pivot(index="fish_id", columns="condition", values=metric)
            values = values.reindex(index=fish_ids, columns=conditions)
            x = np.arange(len(conditions))
            means = values.mean(axis=0).to_numpy()
            bars = ax.bar(x, means, color="#cccccc", width=0.65, zorder=1)
            ax.bar_label(bars, labels=[f"{v:.1f}" if np.isfinite(v) else "N/A" for v in means],
                         padding=4, fontsize=8)
            for position in np.flatnonzero(~np.isfinite(means)):
                ax.text(position, 0.025, "N/A", transform=ax.get_xaxis_transform(),
                        ha="center", fontsize=8, color="#555555")
            for k, fid in enumerate(fish_ids):
                ax.plot(x + (k - (len(fish_ids) - 1) / 2) * 0.025,
                    values.loc[fid], ".-", linewidth=0.5, markersize=5,
                    color=colors(k % 10), alpha=0.75, label=fid, zorder=2)
            ax.set(xticks=x, xticklabels=conditions, ylabel="Neuron count", title=f"{side.capitalize()}: {title}")
            ax.set_xlim(-0.5, len(conditions) - 0.5)
            ax.set_ylim(bottom=0)
            ax.margins(y=0.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.07),
        ncol=min(4, len(fish_ids)), frameon=False, fontsize=8)
    repetitions = next(iter(recruitment["displays"].values()))["repetitions"]
    cohort_label = "SINGLE-FISH preview" if len(fish_ids) == 1 else f"{len(fish_ids)} fish"
    fig.suptitle(cohort_label + " | significant activity | repetitions " + ", ".join(map(str, repetitions))
        + " | bars: equal-fish means; points: individual fish")
    fig.canvas.draw()
    return fig, axes


def plot_phase_population_comparisons(analysis, fish_ids, side_conditions, bout_windows,
                                      comparison_pairs, delay_conditions, figsize=(14, 10)):
    """Paired differences and physical-delay curves through the existing fish plot."""
    figures = {}
    metrics = [("auc_z_s", "AUC (z-score s)"), ("peak_z", "Peak z-score"), ("active_fraction", "Active-neuron fraction")]
    for window in bout_windows:
        for kind in ("differences", "delay"):
            fig, axes = plt.subplots(3, len(side_conditions), figsize=figsize,
                                      sharey="row", squeeze=False, layout="constrained")
            for j, (side, conditions) in enumerate(side_conditions.items()):
                for i, (metric, label) in enumerate(metrics):
                    ax = axes[i, j]
                    if kind == "differences":
                        names = [name for name, (a, b) in comparison_pairs.items() if a in conditions and b in conditions]
                        table = analysis["paired_differences"].loc[lambda t: t.side.eq(side) & t.window.eq(window) & t.metric.eq(metric)].copy()
                        table["condition"] = table["comparison"]
                        plot_diagnostics.plot_paired_fish_values(table, "difference", names, fish_ids, ax, label + " difference")
                        ax.axhline(0, color="gray", linewidth=0.6)
                    else:
                        names = delay_conditions[side]
                        table = analysis["fish_summary"].loc[lambda t: t.side.eq(side) & t.window.eq(window)]
                        delays = table.groupby("condition").delay_s.first().reindex(names).to_numpy()
                        plot_diagnostics.plot_paired_fish_values(table, metric, names, fish_ids, ax, label, x_values=delays)
                        ax.set_xticklabels([f"{x:g}" for x in delays], rotation=45, ha="right", fontsize=8)
                        ax.set_xlabel("Blank delay (s)")
                        if metric == "active_fraction":
                            ax.set_ylim(-0.03, 1.03)
                    if i == 0:
                        ax.set_title(side.capitalize())
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=min(5, len(labels)), frameon=False)
            fig.suptitle(f"{kind.capitalize()} | {window} | {len(fish_ids)} fish")
            fig.canvas.draw()
            figures[f"{kind}_{window}"] = fig
    return figures


def plot_phase_population_relationships(analysis, fish_ids, side_conditions, bout_windows,
                                       membership_figsize=(14, 12), scatter_figsize=(20, 7)):
    """Descriptive phase memberships and within-fish neuron scatterplots."""
    figures = {}
    categories = ["preconditioning_only", "bout_only", "both", "neither"]
    colors = plt.get_cmap("tab10")
    for window in bout_windows:
        fig, axes = plt.subplots(4, len(side_conditions), figsize=membership_figsize,
                                  sharey=True, squeeze=False, layout="constrained")
        for i, category in enumerate(categories):
            for j, (side, conditions) in enumerate(side_conditions.items()):
                table = analysis["phase_memberships"].loc[lambda t: t.side.eq(side) & t.window.eq(window) & t.category.eq(category)]
                plot_diagnostics.plot_paired_fish_values(table, "fraction", conditions, fish_ids, axes[i, j], category.replace("_", " "))
                axes[i, j].set_ylim(-0.03, 1.03)
                if i == 0:
                    axes[i, j].set_title(side.capitalize())
        fig.suptitle(f"Phase memberships | {window} | C1 not applicable")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=min(5, len(labels)), frameon=False)
        fig.canvas.draw()
        figures[f"memberships_{window}"] = fig
        conditions = [c for group in side_conditions.values() for c in group]
        fig, axes = plt.subplots(2, int(np.ceil(len(conditions) / 2)), figsize=scatter_figsize,
                                 sharex=True, sharey=True, squeeze=False, layout="constrained")
        for ax, condition in zip(axes.flat, conditions):
            rows = analysis["phase_relationships"].loc[lambda t: t.condition.eq(condition) & t.window.eq(window)]
            for i, fid in enumerate(fish_ids):
                points = rows.loc[rows.fish_id.eq(fid)]
                ax.scatter(points.preconditioning_auc, points.bout_auc, s=3, alpha=0.25, color=colors(i % 10), label=fid)
            ax.set_title(condition, fontsize=10)
            if rows.empty:
                ax.text(0.5, 0.5, "Not applicable", ha="center", transform=ax.transAxes, fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("Preconditioning AUC (z s)", fontsize=8)
        for ax in axes[:, 0]:
            ax.set_ylabel("Bout-window AUC (z s)")
        fig.suptitle(f"Within-fish neuron relationships | {window}; descriptive correlations in tables")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=min(5, len(labels)), markerscale=3, frameon=False)
        fig.canvas.draw()
        figures[f"relationships_{window}"] = fig
    return figures


def _summary_axes(figsize, separate, sharey=True, sharex=False):
    """Create a 2-by-2 grid or four independent figures, with matching scales.

    Args:
        figsize (tuple): Grid size, or each separate figure's size, in inches.
        separate (bool): Create four figures instead of one grid.
        sharey (bool or str): True shares all y scales; row shares each pair.
        sharex (bool): Share all x scales.

    Returns:
        tuple: Grid Figure (None when separate) and a (2, 2) axes array.
    """
    if not separate:
        return plt.subplots(2, 2, figsize=figsize, sharey=sharey, sharex=sharex, squeeze=False)
    axes = np.empty((2, 2), dtype=object)
    for row, column in np.ndindex(axes.shape):
        _, axes[row, column] = plt.subplots(figsize=figsize)
        if sharey and (row or column):
            reference = axes[row, 0] if sharey == "row" else axes[0, 0]
            if axes[row, column] is not reference:
                axes[row, column].sharey(reference)
        if sharex and (row or column):
            axes[row, column].sharex(axes[0, 0])
    return None, axes


def _finish_separate_panels(axes, keys, cdf=False):
    """Lay out independently labeled compact panels in row/left-right order.

    Args:
        axes (ndarray): (2, 2) axes, each owned by its own figure.
        keys (sequence): Four stable export keys in row-major order.
        cdf (bool): Reserve space for two-line CDF titles and short x ticks.

    Returns:
        dict: Ordered panel key to (Figure, Axes); no saving or display.
    """
    panels = {}
    for key, ax in zip(keys, axes.ravel()):
        fig = ax.figure
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
                   handlelength=1.2, columnspacing=0.8, handletextpad=0.3, borderpad=0.1)
        ax.set_title(ax.get_title(), pad=2)
        ax.xaxis.labelpad = ax.yaxis.labelpad = 1
        ax.tick_params(axis="both", pad=1)
        if np.allclose(ax.get_ylim(), (-0.025, 1.025)) or np.allclose(ax.get_ylim(), (0, 1.025)):
            ax.set_yticks([0, 0.5, 1])
        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=2))
        bottom = 0.36 if cdf else (0.51 if "1 = active" in ax.get_xlabel() else 0.46)
        fig.subplots_adjust(left=0.28, right=0.94, bottom=bottom,
                            top=0.76 if cdf else 0.81)
        panels[key] = (fig, ax)
    return panels


def plot_family_lifetime_sparseness(sparseness, population, figsize=(14, 9),
                                   separate=False, individual_style="identified"):
    """Mirror family sparseness distributions for one fixed neuron population.

    Args:
        sparseness (dict): Output of build_family_lifetime_sparseness.
        population (str): all_kept or all_rocking.
        figsize (tuple): Figure size in inches, set beside the notebook call.
        separate (bool): Return four independently labeled figures when True.
        individual_style (str): identified or anonymous_points (gray fish dots).

    Returns:
        tuple: Figure and 2-by-2 axes, rocking/flicker rows and left/right
        columns. Fish CDFs are equally weighted; undefined neurons are excluded
        with counts retained in the bundle. With separate=True, returns an
        ordered dict of population_family_side -> (Figure, Axes), using figsize
        for each panel. No saving or forced display.
    """
    if population not in sparseness["populations"]:
        raise ValueError("Unknown sparseness population.")
    fig, axes = _summary_axes(figsize, separate, sharex=True)
    source = sparseness["tables"]["family_sparseness_cdf"]
    means = sparseness["tables"]["family_sparseness_mean_cdf"]
    for row, family in enumerate(sparseness["families"]):
        for column, side in enumerate(("left", "right")):
            selected = {"population": population, "family": family, "side": side}
            tables = []
            for table in (source, means):
                mask = np.ones(len(table), dtype=bool)
                for key, value in selected.items():
                    mask &= table[key].to_numpy() == value
                tables.append(table.loc[mask])
            plot_specificity.plot_family_sparseness_cdf(
                *tables, sparseness["fish_ids"], axes[row, column],
                f"{side.capitalize()} — {family.capitalize()}", individual_style=individual_style)
            if separate:
                ax = axes[row, column]
                population_title = "All kept" if population == "all_kept" else "All-rocking core"
                ax.set_title(f"{population_title}\n{side.capitalize()} — {family.capitalize()}")
                ax.set_xlabel("Lifetime sparseness")
                ax.set_ylabel("Cumulative\nfraction")
                ax.set_xticks([0, 0.5, 1])
            elif column:
                axes[row, column].set_ylabel("")
    if separate:
        keys = [f"{population}_{family}_{side}" for family in sparseness["families"]
                for side in ("left", "right")]
        return _finish_separate_panels(axes, keys, cdf=True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)), frameon=False)
    fig.suptitle(f"Lifetime sparseness — {sparseness['population_labels'][population]}")
    fig.tight_layout(rect=(0, 0.075, 1, 0.96), h_pad=2)
    return fig, axes


def plot_family_sparseness_summary(sparseness, figsize=(14, 9), separate=False,
                                  individual_style="identified"):
    """Compare per-fish mean sparseness within rocking and flicker.

    Args:
        sparseness (dict): Output of build_family_lifetime_sparseness.
        figsize (tuple): Figure size in inches, set beside the notebook call.
        separate (bool): Return four independently labeled figures when True.
        individual_style (str): identified or anonymous_points.

    Returns:
        tuple: Figure and 2-by-2 axes: population rows, left/right columns.
        Individual fish and equal-fish means use defined neurons only, and
        all axes have [0, 1] scales. With separate=True, returns an ordered dict
        of population_side -> (Figure, Axes), sized individually by figsize.
        No saving or forced display.
    """
    source = sparseness["tables"]["family_sparseness_per_fish"]
    specs = []
    for population in sparseness["populations"]:
        table = source.loc[source["population"] == population].rename(columns={"family": "condition"})
        specs.append({"table": table, "value": "mean_sparseness", "order": sparseness["families"],
                      "fraction": True, "title": sparseness["population_labels"][population],
                      "key": population, "short_title": "All kept" if population == "all_kept" else "All-rocking core",
                      "short_ylabel": "Mean\nsparseness", "xlabel": "Stimulus family",
                      "ylabel": "Mean lifetime sparseness\n(defined neurons)"})
    return _mirrored_fish_metric_figure(
        sparseness, specs, "Within-family lifetime sparseness — equal fish weight", figsize,
        separate=separate, individual_style=individual_style)


def plot_response_breadth_summary(breadth_profiles, mode="breadth", figsize=(16, 10),
                                 separate=False, individual_style="identified", denominator="all_kept"):
    """Mirror rocking/flicker membership fractions with paired fish and means.

    Args:
        breadth_profiles (dict): Output of build_response_breadth_profiles.
        mode (str): breadth (0 through family size) or combinations (all exact
            memberships).
        figsize (tuple): Figure size in inches, set beside the notebook call.
        separate (bool): Return four independently labeled figures when True.
        individual_style (str): identified or anonymous_points.
        denominator (str): all_kept (default) or family_responsive. The latter
            uses prepared conditional fractions and excludes none/zero categories.

    Returns:
        tuple: Figure and 2-by-2 axes, family rows and left/right columns, all
        using [0, 1] fraction scales. With separate=True, returns an ordered dict
        of mode_family_side -> (Figure, Axes), sized individually by figsize.
        Exact combinations use binary tick codes with the variant key on x.
        No saving or forced display.
    """
    if mode not in {"breadth", "combinations"}:
        raise ValueError("mode must be breadth or combinations.")
    if denominator not in {"all_kept", "family_responsive"}:
        raise ValueError("denominator must be all_kept or family_responsive.")
    responsive = denominator == "family_responsive"
    fig, axes = _summary_axes(figsize, separate)
    category = "n_active_stimuli" if mode == "breadth" else "membership"
    source = breadth_profiles["tables"][f"response_{mode}" + ("_family_responsive" if responsive else "")]
    value_column = "fraction_family_responsive" if responsive else "fraction_kept"
    for row, family in enumerate(breadth_profiles["families"]):
        order = (list(range(len(breadth_profiles["family_labels"][family]) + 1))
                 if mode == "breadth" else breadth_profiles["combination_orders"][family])
        if responsive:
            order = order[1:]
        for column, side in enumerate(("left", "right")):
            table = source.loc[(source["side"] == side) & (source["family"] == family)]
            table = table.rename(columns={category: "condition"})
            ax = axes[row, column]
            plot_diagnostics.plot_paired_fish_values(
                table, value_column, order, breadth_profiles["fish_ids"], ax,
                ("Fraction of family-\nresponsive neurons" if responsive else "Fraction of all\nkept neurons")
                if column == 0 or separate else "",
                individual_style=individual_style)
            ax.set_title(f"{side.capitalize()} — {family.capitalize()}")
            ax.set_ylim(-0.025, 1.025)
            if separate:
                ax.set_ylabel("Responsive\nfraction" if responsive else "Kept neuron\nfraction")
                ax.set_title(f"{side.capitalize()} — {family.capitalize()} {'breadth' if mode == 'breadth' else 'combinations'}")
            if mode == "breadth":
                ax.set_xlabel("Number of active variants")
                if separate:
                    ax.tick_params(axis="x", labelrotation=0)
                    for label in ax.get_xticklabels():
                        label.set_ha("center")
                    ax.set_xlabel("Active variants (count)")
            elif separate:
                variants = breadth_profiles["family_labels"][family]
                codes = ["".join("1" if variant in label.split(" + ") else "0" for variant in variants)
                         for label in order]
                ax.set_xticks(np.arange(len(order)), codes)
                ax.tick_params(axis="x", labelrotation=45)
                for label in ax.get_xticklabels():
                    label.set_ha("right")
                ax.set_xlabel(" / ".join(variants) + "\n1 = active; 0 = inactive")
            else:
                ax.set_xticks(np.arange(len(order)), [label.replace(" + ", "\n+ ") for label in order])
                # Rotation separates the multiline labels at final column width.
                compact = figsize[0] <= 7.5
                ax.tick_params(axis="x", labelrotation=45 if compact else 0)
                for label in ax.get_xticklabels():
                    label.set_ha("right" if compact else "center")
                ax.set_xlabel("Exact active-stimulus combination")
    if separate:
        keys = [f"{mode}_{family}_{side}" for family in breadth_profiles["families"]
                for side in ("left", "right")]
        return _finish_separate_panels(axes, keys)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)), frameon=False)
    title = "Response breadth" if mode == "breadth" else "Exact stimulus-response memberships"
    fig.suptitle(f"{title} — equal fish weight")
    fig.tight_layout(rect=(0, 0.065, 1, 0.96), h_pad=2.2)
    return fig, axes


def plot_response_profile_panels(breadth_profiles, population, figsize=(14, 19), cmap="RdBu_r", fish_ids=None):
    """Show one mirrored signed-activity panel per fish in a fixed population.

    Args:
        breadth_profiles (dict): Output of build_response_breadth_profiles.
        population (str): all_kept or all_rocking.
        figsize (tuple): Figure size in inches.
        cmap (str): Diverging palette; both population figures share the supplied
            bundle's full-range symmetric color limits. No per-neuron scaling.
        fish_ids (sequence or None): Optional nonempty ordered subset for compact
            figure blocks. None shows the whole cohort. Color limits stay global.

    Returns:
        tuple: Figure and (n_fish, 2) axes. Panel neuron orders are independent
        and preserved in the bundle, not correspondences between sides. No I/O.
    """
    if population not in breadth_profiles["populations"]:
        raise ValueError("Unknown response-profile population.")
    fish_ids = list(breadth_profiles["fish_ids"] if fish_ids is None else fish_ids)
    if not fish_ids or len(set(fish_ids)) != len(fish_ids):
        raise ValueError("fish_ids must be nonempty and contain no duplicates.")
    if any(fid not in breadth_profiles["fish_ids"] for fid in fish_ids):
        raise ValueError("fish_ids must be a subset of the prepared cohort.")
    fig, axes = plt.subplots(len(fish_ids), 2, figsize=figsize, layout="constrained", squeeze=False)
    for row, fid in enumerate(fish_ids):
        for column, side in enumerate(("left", "right")):
            responses = breadth_profiles["profiles"][fid][side][population]["responses"]
            image = plot_specificity.plot_signed_response_profiles(
                responses, axes[row, column], f"{fid} — {side.capitalize()} (n={len(responses)})",
                breadth_profiles["profile_color_limit"], cmap=cmap)
    fig.colorbar(image, ax=axes.ravel().tolist(),
                 label="Signed z-score activity\n(AUC / integration span)", shrink=0.8)
    fig.suptitle(f"{breadth_profiles['population_labels'][population]} — continuous response profiles")
    return fig, axes


def plot_pooled_response_profiles(pooled_profiles, figsize=(3.5, 2.75), cmap="RdBu_r", color_limit=None):
    """Draw separate left/right all-fish profiles with matched rows and color scales.

    Args:
        pooled_profiles (dict): Output of build_pooled_response_profiles; raw signed
            response matrices share identical pooled neuron order and condition order.
        figsize (tuple): Each independent panel's width and height in inches.
        cmap (str): Shared diverging palette, centered at zero.
        color_limit (float or None): Shared absolute display bound; None preserves
            the prepared full range. A smaller bound clips colors only, with colorbar
            extensions indicating clipping. It does not change the response values.

    Returns:
        dict: left/right -> (Figure, Axes). Fonts follow the caller's style. No I/O.
    """
    limit = pooled_profiles["profile_color_limit"] if color_limit is None else color_limit
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("color_limit must be finite and positive.")
    panels = {}
    for side in ("left", "right"):
        responses = pooled_profiles["responses"][side]
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        image = plot_specificity.plot_signed_response_profiles(
            responses, ax, f"All fish — {side.capitalize()} (n={len(responses)})", limit, cmap=cmap)
        ax.set_ylabel("Neurons (shared order)")
        fig.colorbar(image, ax=ax, label="Signed z-score activity\n(AUC / integration span)",
                     fraction=0.055, pad=0.025,
                     extend="both" if limit < pooled_profiles["profile_color_limit"] else "neither")
        panels[side] = (fig, ax)
    return panels


def plot_response_similarity_comparison(similarity, metric, figsize=(14, 12), cmap=None):
    """Compare all kept neurons with the all-rocking core, mirroring left/right.

    Args:
        similarity (dict): Output of build_response_similarity_comparison.
        metric (str): jaccard, pearson, or cosine.
        figsize (tuple): Figure size in inches, set beside the notebook call.
        cmap (str or None): Optional shared metric palette.

    Returns:
        tuple: Figure and 2-by-2 axes: population rows, left/right columns.
        The arithmetic fish mean is displayed; no files are saved or shown.
    """
    titles = {"jaccard": "Jaccard overlap", "pearson": "Pearson response-pattern similarity",
              "cosine": "Cosine response-pattern similarity"}
    if metric not in titles:
        raise ValueError("metric must be jaccard, pearson, or cosine.")
    fig, axes = plt.subplots(2, 2, figsize=figsize, layout="constrained", squeeze=False)
    for row, population in enumerate(similarity["populations"]):
        for column, side in enumerate(("left", "right")):
            image = plot_specificity.plot_stimulus_similarity_matrix(
                similarity["mean_matrices"][population][side][metric], axes[row, column],
                title=f"{side.capitalize()}\n{similarity['population_labels'][population]}",
                metric=metric, cmap=cmap)
    fig.colorbar(image, ax=axes.ravel().tolist(), label=f"{metric.capitalize()} (equal-fish mean)", shrink=0.85)
    fig.suptitle(titles[metric])
    return fig, axes


def _mirrored_fish_metric_figure(analysis, row_specs, title, figsize, separate=False,
                                individual_style="identified"):
    """Arrange paired per-fish measurements with common scales and cohort colors."""
    fig, axes = _summary_axes(figsize, separate, sharey="row")
    for row, spec in enumerate(row_specs):
        for column, side in enumerate(("left", "right")):
            table = spec["table"].loc[spec["table"]["side"] == side]
            if "summary_type" in spec:
                table = table.loc[table["summary_type"] == spec["summary_type"]]
            ax = axes[row, column]
            plot_diagnostics.plot_paired_fish_values(
                table, spec["value"], spec["order"], analysis["fish_ids"], ax, spec["ylabel"],
                individual_style=individual_style)
            ax.set_title(f"{side.capitalize()} — {spec['title']}")
            if spec.get("fraction", False):
                ax.set_ylim(-0.025, 1.025)
            else:
                ax.axhline(0, color="0.7", linewidth=0.7, zorder=0)
            if separate:
                ax.set_title(f"{side.capitalize()} — {spec['short_title']}")
                ax.set_ylabel(spec["short_ylabel"])
                ax.set_xlabel(spec.get("xlabel", "Stimulus"))
            elif column:
                ax.set_ylabel("")
    if separate:
        keys = [f"{spec['key']}_{side}" for spec in row_specs for side in ("left", "right")]
        return _finish_separate_panels(axes, keys)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)), frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.075, 1, 0.96), h_pad=2.0)
    return fig, axes


def plot_recruitment_activity_summary(analysis, figsize=(14, 9), separate=False,
                                       individual_style="identified"):
    """Plot mirrored recruitment fractions and signed activity, with one dot per fish.

    Args:
        analysis (dict): Output of build_recruitment_containment_analysis.
        figsize (tuple): Figure size in inches, kept beside the notebook call.
        separate (bool): Return four independently labeled figures when True.
        individual_style (str): identified or anonymous_points.

    Returns:
        tuple: Figure and 2-by-2 axes; rows share y limits across left/right.
        With separate=True, returns an ordered dict of metric_side ->
        (Figure, Axes), using figsize per panel. No saving or forced display.
    """
    table, order = analysis["tables"]["recruitment"], analysis["condition_labels"]
    specs = [
        {"table": table, "value": "active_fraction", "order": order, "fraction": True,
         "key": "recruitment", "short_title": "Recruitment", "short_ylabel": "Active\nfraction",
         "title": "recruitment", "ylabel": "Fraction of kept\nneurons active"},
        {"table": table, "value": "mean_normalized_auc", "order": order,
         "key": "response_window_activity", "short_title": "Response-window activity", "short_ylabel": "Mean\nz-score",
         "title": "response-window activity", "ylabel": "Mean z-score\n(AUC / integration span)"},
    ]
    return _mirrored_fish_metric_figure(
        analysis, specs, "Recruitment and activity — equal fish weight", figsize,
        separate=separate, individual_style=individual_style)


def plot_recruitment_containment_matrices(analysis, figsize=(14, 12), cmap="viridis"):
    """Plot equal-fish directional containment and Jaccard in mirrored columns.

    Args:
        analysis (dict): Output of build_recruitment_containment_analysis.
        figsize (tuple): Figure size in inches.
        cmap (str): Shared [0, 1] fraction color scale; missing cells are gray.

    Returns:
        tuple: Figure and 2-by-2 axes. Containment row A/column B is |A & B|/|A|.
        No files are saved and no interactive display is forced.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, layout="constrained", squeeze=False)
    for row, metric in enumerate(("containment", "jaccard")):
        for column, side in enumerate(("left", "right")):
            image = plot_diagnostics.plot_fraction_matrix(
                analysis["mean_matrices"][side][metric], axes[row, column],
                title=f"{side.capitalize()} — {'directional containment' if row == 0 else 'Jaccard overlap'}",
                xlabel="Target stimulus B" if row == 0 else "Stimulus B",
                ylabel="Source stimulus A" if row == 0 else "Stimulus A", cmap=cmap)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Fraction (equal-fish mean)", shrink=0.85)
    fig.suptitle("Response overlap and containment")
    return fig, axes


def plot_recruitment_containment_summary(analysis, figsize=(14, 9), separate=False,
                                        individual_style="identified"):
    """Plot per-fish containment in control and in the intersection of rocking variants.

    Args:
        analysis (dict): Output of build_recruitment_containment_analysis.
        figsize (tuple): Figure size in inches.
        separate (bool): Return four independently labeled figures when True.
        individual_style (str): identified or anonymous_points.

    Returns:
        tuple: Figure and 2-by-2 axes. Denominators are source-stimulus active
        populations, not the whole cohort. Empty sources leave gaps. With
        separate=True, returns an ordered dict of metric_side -> (Figure, Axes),
        using figsize per panel. No saving or forced display.
    """
    table, labels = analysis["tables"]["containment_summary"], analysis["condition_labels"]
    selection = analysis["side_selections"]["left"]
    n_rocking = len(selection["groups"]["rocking"])
    specs = [
        {"table": table, "value": "fraction_contained", "order": labels[1:], "fraction": True,
         "key": "to_control", "short_title": "Control containment", "short_ylabel": "Contained\nfraction",
         "summary_type": "to_control", "title": "contained in control",
         "ylabel": "Fraction also active\nto control"},
        {"table": table, "value": "fraction_contained", "order": labels[1 + n_rocking:],
         "key": "to_all_rocking", "short_title": "All-rocking containment", "short_ylabel": "Contained\nfraction",
         "fraction": True, "summary_type": "to_all_rocking", "title": "contained in all rocking variants",
         "ylabel": "Fraction also active to\nevery rocking variant"},
    ]
    return _mirrored_fish_metric_figure(
        analysis, specs, "Containment within shared response populations", figsize,
        separate=separate, individual_style=individual_style)


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
        mean_ylabel="Mean\nz-score",
        figsize=figsize,
        fish_id=f"all_fish_{slugify_label(raster_settings.get('analysis_label', raster_settings.get('experiment_name', 'analysis')))}",
    )
    if raster_settings["trace_type"] == "zscore":
        image.colorbar.set_label("Z-score")
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
    if sparseness_figsize[0] <= 3.6:
        ax_sparseness.set_title("Lifetime sparseness\nvs maximum response")
        handles, labels = ax_sparseness.get_legend_handles_labels()
        if ax_sparseness.get_legend() is not None:
            ax_sparseness.get_legend().remove()
        fig_sparseness.legend(handles, labels, title="Preferred stimulus", ncol=2,
                              loc="lower center", frameon=False)
        for collection in ax_sparseness.collections:
            collection.set_sizes([6])
            collection.set_rasterized(True)
        fig_sparseness.tight_layout(rect=(0, 0.40, 1, 1))
    else:
        fig_sparseness.tight_layout()
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
        high_sparseness_figure, _, high_sparseness_image, _ = plot_all_fish.plot_allfish_flat_raster(
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
            mean_ylabel="Mean\nz-score",
            figsize=raster_figsize,
            fish_id=f"all_fish_{slugify_label(analysis_label or 'analysis')}_high_lifetime_sparseness_zscore",
            vmax=4,
        )
        if raster_figsize[0] <= 7.5:
            high_sparseness_figure.axes[0].set_title(
                f"High lifetime sparseness (> {lifetime_sparseness_threshold})")
        high_sparseness_image.colorbar.set_label("Z-score")
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
        ylabel="Mean z-score",
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
        dict: Side name to prepared overlap and trace diagnostic data, plus
        overlap_figure and diagnostic_figure when those figures are drawn.
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
            diagnostic_data["overlap_figure"] = fig
            sns.heatmap(
                diagnostic_data["matrix_to_plot"],
                vmin=0,
                vmax=1,
                square=True,
                annot=True,
                annot_kws={"size": "small"},
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
                diagnostic_figure, _, _ = plot_diagnostics.plot_active_trace_decision_diagnostic(
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
                diagnostic_data["diagnostic_figure"] = diagnostic_figure
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

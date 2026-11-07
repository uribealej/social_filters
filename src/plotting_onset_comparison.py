"""Figures for same-neuron Control onset and time-bin sharing comparisons."""

import matplotlib.pyplot as plt
import numpy as np


def plot_control_bin_overlap(result, side, *, colors=None, label_map=None,
                             figsize=(9, 5.5), font_size=9, show_sem=True,
                             show_fish=False, program_end_s=None):
    """Plot same-bin Control containment (%) and reference counts per fish.

    Uses cached ``build_control_bin_overlap`` tables only. Ordered non-Control
    conditions get equal-fish mean curves and optional descriptive SEM; missing
    bins stay gaps. Control-self is omitted. The lower panel includes all fish's
    Control counts, including zeros. Return figure and two axes. This is a
    descriptive comparison, without a statistical departure-time estimate.
    """
    selection = result["side_selections"][side]
    control = selection["control_label"]
    labels = [label for label in selection["stimulus_labels"] if label != control]
    edges = result["bin_edges_s"]
    centers = (edges[:-1] + edges[1:]) / 2
    palette = plt.get_cmap("tab10")(np.arange(len(labels)) % 10)
    colors, label_map = colors or {}, label_map or {}
    raw = result["fish_bins"]
    summary = result["bin_summary"]
    with plt.rc_context({"font.size": font_size}):
        figure, axes = plt.subplots(2, 1, sharex=True, figsize=figsize,
            gridspec_kw={"height_ratios": [3, 1]}, layout="constrained")
        for label, fallback in zip(labels, palette):
            color = colors.get(label, fallback)
            table = summary[(summary.side == side) & (summary.stimulus == label)]
            table = table.set_index("bin_index").reindex(range(len(centers)))
            mean, sem = 100 * table.fraction_shared.to_numpy(), 100 * table.sem_shared.to_numpy()
            if show_fish:
                fish_table = raw[(raw.side == side) & (raw.stimulus == label)]
                for _, data in fish_table.groupby("fish_id", sort=False):
                    values = data.set_index("bin_index").fraction_shared.reindex(range(len(centers)))
                    axes[0].plot(centers, 100 * values, color=color, alpha=0.12, linewidth=0.6)
            axes[0].plot(centers, mean, color=color, marker="o", markersize=3,
                linewidth=1.6, label=label_map.get(label, label))
            if show_sem:
                axes[0].fill_between(centers, mean - sem, mean + sem, color=color, alpha=0.15, linewidth=0)
        control_rows = raw[(raw.side == side) & (raw.stimulus == control)]
        for _, data in control_rows.groupby("fish_id", sort=False):
            values = data.set_index("bin_index").n_control.reindex(range(len(centers)))
            axes[1].plot(centers, values, color="gray", alpha=0.3, linewidth=0.7)
        mean_counts = control_rows.groupby("bin_index", sort=False).n_control.mean().reindex(range(len(centers)))
        axes[1].plot(centers, mean_counts, color="black", marker="o", markersize=2.5, linewidth=1.2)
        for axis in axes:
            axis.grid(alpha=0.15)
            if program_end_s is not None:
                axis.axvline(program_end_s, color="gray", linestyle=":", linewidth=1)
        axes[0].set_ylim(-3, 103)
        axes[0].set_yticks(np.arange(0, 101, 20))
        axes[0].set_ylabel("Control-active neurons shared (%)")
        axes[0].set_title(f"{side.capitalize()}: significant activity shared with Control in the same bin")
        if labels:
            axes[0].legend(frameon=False, loc="best", ncol=2)
        axes[1].set_ylabel("Control neurons\nper fish")
        axes[1].set_ylim(bottom=0)
        axes[1].set_xlim(edges[0], edges[-1])
        axes[1].set_xlabel("Time after program start (s)")
    return {"figure": figure, "axes": axes}


def plot_control_onset_heatmaps(result, side, figsize=(16, 8), cmap="viridis", font_size=8):
    """Plot ordered onset transitions as equal-fish percentages, one side.

    Rows are control-onset bins; columns are stimulus-onset bins, then no
    detected onset. Gray means no eligible fish for that control bin. White
    boxes mark the same-bin diagonal. Return figure and a label-to-axis map.
    ``result`` comes from ``build_control_onset_comparison``; no recomputation.
    """
    selection = result["side_selections"][side]
    labels = selection["stimulus_labels"]
    edges = result["bin_edges_s"]
    bins = len(edges) - 1
    bin_labels = [f"{a:g}–{b:g}" for a, b in zip(edges[:-1], edges[1:])]
    cols = min(4, len(labels))
    rows = int(np.ceil(len(labels) / cols))
    with plt.rc_context({"font.size": font_size}):
        figure, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False, layout="constrained")
        palette = plt.get_cmap(cmap).copy()
        palette.set_bad("#d9d9d9")
        mapping = {}
        for axis, label in zip(axes.flat, labels):
            table = result["transition_summary"]
            table = table[(table.side == side) & (table.stimulus == label)]
            matrix = table.pivot(index="control_bin", columns="stimulus_bin", values="fraction")
            values = matrix.reindex(index=range(bins), columns=[*range(bins), -1]).to_numpy() * 100
            image = axis.imshow(np.ma.masked_invalid(values), cmap=palette, vmin=0, vmax=100, aspect="auto")
            for index in range(bins):
                axis.add_patch(plt.Rectangle((index - 0.5, index - 0.5), 1, 1,
                    fill=False, edgecolor="white", linewidth=0.65))
            axis.axvline(bins - 0.5, color="white", linewidth=1.5)
            axis.set_xticks(range(bins + 1), [*bin_labels, "No onset"], rotation=90)
            axis.set_yticks(range(bins), bin_labels)
            axis.set_title(label + (" (self)" if label == selection["control_label"] else ""))
            mapping[label] = axis
        for axis in list(axes.flat)[len(labels):]:
            axis.set_visible(False)
        figure.colorbar(image, ax=list(mapping.values()), shrink=0.85, label="Control-bin neurons (%) · equal fish weight")
        figure.supxlabel("Comparison-stimulus onset after program start (s)")
        figure.supylabel(f"{selection['control_label']} onset after program start (s)")
        figure.suptitle(f"{side.capitalize()}: control versus stimulus onset · white boxes = same bin")
    return {"figure": figure, "axes": mapping}


def plot_control_onset_summary(result, side, figsize=(12, 6), font_size=9, show_fish=True):
    """Plot same-bin and any-onset sharing plus reference-population counts.

    Summary lines weight each nonempty fish/bin equally. Optional faint lines
    show individual fish; the bottom panel shows their control-bin counts.
    A response is shared only if it passes the activity rule and has an onset
    in the analysis window. Self-control curves are identities. Return figure
    and axes; values come directly from the prepared tables.
    """
    edges = result["bin_edges_s"]
    centers = (edges[:-1] + edges[1:]) / 2
    selection = result["side_selections"][side]
    labels = selection["stimulus_labels"]
    colors = plt.get_cmap("tab10")(np.arange(len(labels)) % 10)
    with plt.rc_context({"font.size": font_size}):
        figure, axes = plt.subplots(3, 1, sharex=True, figsize=figsize,
            gridspec_kw={"height_ratios": [2, 2, 1]}, layout="constrained")
        for label, color in zip(labels, colors):
            table = result["bin_summary"]
            summary = table[(table.side == side) & (table.stimulus == label)].set_index("control_bin")
            summary = summary.reindex(range(len(centers)))
            for axis, metric in zip(axes[:2], ["fraction_same", "fraction_shared"]):
                if show_fish:
                    raw = result["fish_bins"]
                    raw = raw[(raw.side == side) & (raw.stimulus == label)]
                    for _, data in raw.groupby("fish_id", sort=False):
                        axis.plot(centers, 100 * data.set_index("control_bin")[metric].reindex(
                            range(len(centers))), color=color, alpha=0.12, linewidth=0.6)
                axis.plot(centers, 100 * summary[metric], color=color,
                    marker="o", markersize=3, linewidth=1.5, label=label)
        for axis, title in zip(axes[:2], ["Shared neurons with onset in the same bin",
                                         "Shared neurons with onset anywhere in the analysis window"]):
            axis.set_title(title)
            axis.set_ylabel("Control-bin neurons (%)")
            axis.set_ylim(-3, 103)
            axis.grid(alpha=0.15)
        raw = result["fish_bins"]
        raw = raw[(raw.side == side) & (raw.stimulus == selection["control_label"])]
        for _, data in raw.groupby("fish_id", sort=False):
            axes[2].plot(centers, data.n_control, color="gray", alpha=0.35, linewidth=0.8)
        mean = raw.groupby("control_bin", sort=False).n_control.mean().reindex(range(len(centers)))
        axes[2].plot(centers, mean, color="black", marker="o", markersize=3)
        axes[2].set_ylabel("Control neurons\nper fish")
        axes[2].set_ylim(bottom=0)
        axes[2].set_xticks(centers, [f"{a:g}–{b:g}" for a, b in zip(edges[:-1], edges[1:])], rotation=45)
        axes[2].set_xlabel(f"{selection['control_label']} onset after program start (s)")
        figure.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center",
            ncol=min(4, len(labels)), frameon=False)
        figure.suptitle(f"{side.capitalize()}: onset recruitment relative to {selection['control_label']}")
    return {"figure": figure, "axes": axes}

"""Pooled rasters and per-stimulus mean-trace figures."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pathlib import Path
from src.neuron_ordering import (
    compute_single_sort_order,
)
from src.stimuli_timeline import (
    summarize_durations,
    compute_move_lines_for_flat_matrix,
)
from src.plotting_single_fish import (
    raster_with_stimuli,
)


def plot_stimulus_means(
        mean_traces,
        stimuli_ids,
        stimuli_names,  # list of display names (used to look up styles)
        fps_2p,
        t_post_s,
        t_pre_s,
        stimuli_durations,
        title_prefix="neurons resposive to stimuli",
        show_sem=True,
        # NEW: pass styles in
        stimuli_colors: dict | None = None,  # e.g., {"LLB": (r,g,b), "FLB": ...}
        stimuli_linestyles: dict | None = None,  # e.g., {"FLB": "--", "FRB": "--"}
        # saving controls
        save: bool = False,
        plots_path: Path | str | None = None,
        prefix: str | None = None,
        dpi: int = 600,
        close_after: bool = False,
        kept_cells = None,  # optional: indices of cells to include
        comment="all_stimuli",  # for saving
        figsize=(7, 4.5),
):
    """
    Plots mean ± SEM per stimulus.
    Styles are looked up by stimulus *name* in `stimuli_colors` and `stimuli_linestyles`.
    """
    # defaults if not provided
    if stimuli_colors is None:     stimuli_colors = {}
    if stimuli_linestyles is None: stimuli_linestyles = {}
    pre_frames  = int(round(t_pre_s * fps_2p))
    post_frames = int(round(t_post_s * fps_2p))
    win_lenght = pre_frames + post_frames
    # time axis (s), 0 at static onset
    t = (np.arange(win_lenght) - pre_frames) / float(fps_2p)


    fig, ax = plt.subplots(figsize=figsize)
    color_by_stim = {}

    # optional warnings for missing styles
    missing_colors = [nm for nm in stimuli_names if nm not in stimuli_colors]
    missing_ls = [nm for nm in stimuli_names if nm not in stimuli_linestyles]
    if missing_colors:
        print("[warn] No color for:", missing_colors, "→ using Matplotlib defaults.")
    if missing_ls:
        print("[warn] No linestyle for:", missing_ls, "→ using solid '-'.")

    for i, stim in enumerate(stimuli_ids):
        name = stimuli_names[i]
        M = mean_traces[stim]

        if kept_cells is not None:
            M = M[kept_cells, :] # (n_kept, win_lenght)

        n_sel = M.shape[0]
        trace_mean = np.nanmean(M, axis=0)
        trace_sd = np.nanstd(M, axis=0)
        trace_sem = trace_sd / np.sqrt(max(n_sel, 1))

        color = stimuli_colors.get(name, None)  # None → mpl default
        ls = stimuli_linestyles.get(name, "-")  # default solid

        (line,) = ax.plot(t, trace_mean, label=name, color=color, linestyle=ls)
        color_by_stim[name] = line.get_color()

        if show_sem:
            ax.fill_between(
                t, trace_mean - trace_sem, trace_mean + trace_sem,
                alpha=0.18, color=line.get_color(), linewidth=0
            )

    # visual guides
    summary = summarize_durations(stimuli_durations)
    static_onset = 0
    motion_onset = summary.get("static_before_sec")
    motion_offset =summary.get("total_sec")
    ax.axvline(static_onset, linestyle="--", linewidth=1, color="grey", label="static onset")
    ax.text(static_onset - 0.5, 0.85, "static", rotation=90, va="bottom", ha="center",
            transform=ax.get_xaxis_transform())
    ax.axvline(motion_onset, linewidth=1, color="k", label="motion onset")
    ax.text(motion_onset + 1.5, 0.85, "motion", rotation=90, va="bottom", ha="center",
            transform=ax.get_xaxis_transform())
    ax.axvline(motion_offset, linestyle="--", linewidth=1, color="grey")

    # cosmetics
    ax.set(title=f"{title_prefix}",
           xlabel="Time (s) relative to onset",
           ylabel="ΔF/F")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    plt.subplots_adjust(right=0.78)
    plt.tight_layout()

    out_png = None
    if save:
        if plots_path is None or prefix is None:
            print("⚠️  Save was requested but `plots_path` or `prefix` is missing—skipping save.")
        else:
            plots_path = Path(plots_path)
            plots_path.mkdir(parents=True, exist_ok=True)
            out_png = plots_path / f"{prefix}_dfof_as_func_of_time_{comment}.png"
            fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
            print("✅ Saved:", out_png)

    if close_after:
        plt.close(fig)
    else:
        plt.show()

    return fig, ax, color_by_stim, out_png


def plot_allfish_flat_raster(
    data,
    trial_aligned_traces,
    stim_order,
    stimuli_id_map,
    stimuli_durations,
    stimuli_colors,
    stimuli_linestyles,
    fps_2p,
    t_pre_s,
    combine_mode="concat",
    *,
    sort_mode="kmeans",
    n_clusters=8,
    random_state=0,
    neuron_order=None,
    sort_label=None,
    is_binary=False,
    figsize=(8, 6),
    fish_id="all_fish",
    show_mean_trace=False,
    mean_height_ratio=1.2,
    mean_linewidth=1.5,
    mean_color="black",
    mean_ylabel="Mean activity",
    vmin=None,
    vmax=None,
):
    data = np.asarray(data)
    assert data.ndim == 2, "data must be (n_neurons, n_time)"

    data_for_sort = data.T.copy()

    if is_binary:
        data_for_sort[data_for_sort < 0.5] = 0
        data_for_sort[data_for_sort >= 0.5] = 1

    n_time, n_neurons = data_for_sort.shape

    if neuron_order is not None:
        neuron_order = np.asarray(neuron_order)
        if neuron_order.ndim != 1 or neuron_order.shape[0] != n_neurons:
            raise ValueError(
                f"Custom neuron_order has shape {neuron_order.shape}, "
                f"expected ({n_neurons},)."
            )
        label = sort_label or "custom"
    else:
        neuron_order = compute_single_sort_order(
            data_for_sort,
            sort_mode=sort_mode,
            n_clusters=n_clusters,
            random_state=random_state,
        )
        label = sort_label or sort_mode

    # ------------------------------------------------------------------
    # Axes layout: explicitly reserve a separate colorbar column
    # so the raster and mean panels keep identical widths
    # ------------------------------------------------------------------
    if show_mean_trace:
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(
            2, 2,
            width_ratios=[40, 1.2],
            height_ratios=[6, mean_height_ratio],
            hspace=0.05,
            wspace=0.12,
        )
        ax = fig.add_subplot(gs[0, 0])
        ax_mean = fig.add_subplot(gs[1, 0], sharex=ax)
        cax = fig.add_subplot(gs[0, 1])   # colorbar axis only for top panel
    else:
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(
            1, 2,
            width_ratios=[40, 1.2],
            wspace=0.12,
        )
        ax = fig.add_subplot(gs[0, 0])
        ax_mean = None
        cax = fig.add_subplot(gs[0, 1])

    # --- raster ---
    im = raster_with_stimuli(
        ax=ax,
        data=data_for_sort,
        fps=fps_2p,
        fish_id=fish_id,
        neuron_order=neuron_order,
        is_binary=is_binary,
        title_suffix=f"ordered by stimulus type | sort={label}",
        vmin=vmin,
        vmax=vmax,
    )

    move_starts_s, move_colors, stim_labels = compute_move_lines_for_flat_matrix(
        trial_aligned_traces=trial_aligned_traces,
        stim_order=stim_order,
        stimuli_id_map=stimuli_id_map,
        stimuli_durations=stimuli_durations,
        stimuli_colors=stimuli_colors,
        fps_2p=fps_2p,
        t_pre_s=t_pre_s,
        combine_mode=combine_mode,
    )

    move_styles = [stimuli_linestyles.get(name, "-") for name in stim_labels]

    for pos_s, color, ls in zip(move_starts_s, move_colors, move_styles):
        ax.axvline(pos_s / fps_2p, color=color, linestyle=ls, alpha=0.9, linewidth=1.0)

    # --- mean panel ---
    if show_mean_trace:
        mean_trace = np.nanmean(data, axis=0)
        time_s = np.arange(mean_trace.size) / fps_2p

        ax_mean.plot(time_s, mean_trace, color=mean_color, linewidth=mean_linewidth)

        for pos_s, color, ls in zip(move_starts_s, move_colors, move_styles):
            ax_mean.axvline(pos_s / fps_2p, color=color, linestyle=ls, alpha=0.9, linewidth=1.0)

        ax_mean.set_ylabel(mean_ylabel, fontsize=12)
        ax_mean.set_xlabel("Time (s)", fontsize=12)
        ax_mean.tick_params(labelsize=10)
        ax_mean.spines["top"].set_visible(False)

        # Hide duplicated top x tick labels
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_xlabel("")

    # --- legend ---
    seen = set()
    handles = []
    labels = []

    for name in stim_labels:
        if name in seen:
            continue
        seen.add(name)

        color = stimuli_colors[name]
        ls = stimuli_linestyles.get(name, "-")
        (line,) = ax.plot([], [], color=color, linestyle=ls, label=name, linewidth=2)
        handles.append(line)
        labels.append(name)

    ax.legend(
        handles=handles,
        labels=labels,
        title="Movement onset",
        bbox_to_anchor=(1.18, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=False,
        fontsize=12,
        title_fontsize=14,
    )

    # --- colorbar in dedicated axis ---
    if not is_binary:
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(r"$\Delta F/F$", fontsize=14)
        cbar.ax.tick_params(labelsize=12)
    else:
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Significant activity (0/1)", fontsize=14)
        cbar.ax.tick_params(labelsize=12)

    return fig, ax, im, neuron_order

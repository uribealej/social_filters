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
        ylabel="ΔF/F",
):
    """
    Plots mean ± SEM per stimulus.
    Styles are looked up by stimulus *name* in `stimuli_colors` and `stimuli_linestyles`.
    Set ``ylabel`` to the units of the supplied traces (default: ΔF/F).
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
           ylabel=ylabel)
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


def plot_stimulus_response_panels(display, stimulus_ids=None, layout="combined",
                                  figsize=(12, 5), z_limits=(0, 10), mean_ylim=None,
                                  cmap="Greys", title="", mean_mode="equal_fish",
                                  show_fish_traces=True, proportional_widths=False):
    """Plot independent trial-mean stimulus columns with one shared neuron order.

    Args:
        display (dict): build_stimulus_panel_display result; matrices are
            (neurons, time), fish_means (fish, time), and time_s is seconds.
            Phase-cohort adapters may instead supply phase_trials and alignment
            for native preconditioning/delay/bout annotations.
        stimulus_ids (sequence or None): Nonempty display subset in column order.
        layout (str): raster, mean, or combined (raster above fish mean traces).
        figsize (tuple): Final figure width/height in inches.
        z_limits (tuple): Shared trace color limits, independent of sorting.
        mean_ylim (tuple or None): Shared trace limits; None spans every prepared
            fish/condition, ensuring separate left/right figures are comparable.
        cmap (str): Raster colormap; original trial-mean values remain unchanged.
        title (str): Short figure heading, typically the selected stimulus side.
        mean_mode (str): equal_fish (default) or pooled_neurons. The latter
            averages cached neuron-by-time matrices, giving each neuron one vote.
        show_fish_traces (bool): Include thin individual-fish mean traces.
        proportional_widths (bool): Size columns by their own time spans, with
            independent limits and the same physical seconds scale. Optional
            display['time_bounds_s'] supplies exact requested (start, stop)
            seconds; otherwise use sample edges. Default keeps equal widths
            and shared limits.

    Returns:
        tuple: Figure and a 2D axes array (rows, conditions). No saving/showing.
        Thin colored traces represent fish; black averages fish equally. Static
        and active spans use canonical nominal timing; baseline stays unshaded.
        A phase legend sits in the upper-right figure margin; there is no fish
        legend. Time (s) appears once per figure.
        Font sizes follow the caller's Matplotlib rc_context.
    """
    from matplotlib.ticker import MaxNLocator

    ids = list(display["stimulus_ids"] if stimulus_ids is None else stimulus_ids)
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= set(display["stimulus_ids"]):
        raise ValueError("stimulus_ids must be a nonempty unique subset of the display.")
    if layout not in {"raster", "mean", "combined"}:
        raise ValueError("layout must be raster, mean, or combined.")
    if mean_mode not in {"equal_fish", "pooled_neurons"}:
        raise ValueError("mean_mode must be 'equal_fish' or 'pooled_neurons'.")
    is_raster = display.get("trace_type", "zscore") == "raster"
    color_label = "Significant activity (trial fraction)" if is_raster else "Z-score"
    mean_label = "Mean significant\nactivity (fraction)" if is_raster else "Mean\nz-score"
    if not np.isfinite(z_limits).all() or len(z_limits) != 2 or z_limits[0] >= z_limits[1]:
        raise ValueError("z_limits must be two finite increasing values.")
    has_raster, has_mean = layout != "mean", layout != "raster"
    nrows = 2 if layout == "combined" else 1
    dt = 1 / display["fps_2p"]
    bounds = {sid: display.get("time_bounds_s", {}).get(sid,
        (display["time_s"][sid][0] - dt / 2, display["time_s"][sid][-1] + dt / 2))
        for sid in ids}
    gridspec = {"height_ratios": [3, 1.5]} if nrows == 2 else {}
    if proportional_widths:
        if any(not np.isfinite(b).all() or len(b) != 2 or b[1] <= b[0]
               for b in bounds.values()):
            raise ValueError("Panel time bounds must be finite increasing (start, stop) seconds.")
        gridspec["width_ratios"] = [bounds[sid][1] - bounds[sid][0] for sid in ids]
    fig, axes = plt.subplots(nrows, len(ids), figsize=figsize,
        sharex="col" if proportional_widths else True,
        sharey="row", squeeze=False, gridspec_kw=gridspec)
    fig.subplots_adjust(left=0.10, right=0.90 if has_raster else 0.985,
                        bottom=0.21, top=0.82, wspace=0.12, hspace=0.13)
    if mean_ylim is None:
        mean_values = (display["fish_means"].values() if show_fish_traces or mean_mode == "equal_fish"
                       else (np.nanmean(v, axis=0) for v in display["matrices"].values()))
        finite = np.concatenate([v[np.isfinite(v)] for v in mean_values])
        low, high = (min(0, finite.min()), max(0, finite.max())) if finite.size else (-1, 1)
        pad = max(0.1, (high - low) * 0.06)
        mean_ylim = (low - pad, high + pad)
    colors = plt.get_cmap("tab10")
    phase_trials = display.get("phase_trials")
    im = None
    order = display["neuron_order"]
    for column, sid in enumerate(ids):
        times = display["time_s"][sid]
        if has_raster:
            im = axes[0, column].imshow(display["matrices"][sid][order], aspect="auto",
                interpolation="nearest", origin="upper", cmap=cmap,
                vmin=z_limits[0], vmax=z_limits[1],
                extent=[times[0] - dt / 2, times[-1] + dt / 2, len(order) - 0.5, -0.5])
        if has_mean:
            ax = axes[-1, column]
            values = display["fish_means"][sid]
            if show_fish_traces:
                for index, fid in enumerate(display["fish_ids"]):
                    ax.plot(times, values[index], color=colors(index % 10), alpha=0.65,
                            linewidth=0.4, label=fid)
            mean = (np.nanmean(values, axis=0) if mean_mode == "equal_fish"
                    else np.nanmean(display["matrices"][sid], axis=0))
            ax.plot(times, mean, color="black", linewidth=0.9,
                    label="Equal-fish mean" if mean_mode == "equal_fish" else "Pooled-neuron mean")
            ax.axhline(0, color="gray", linewidth=0.35, zorder=0)
            ax.set_ylim(*mean_ylim)
        for ax in axes[:, column]:
            if phase_trials is not None:
                from src.plotting_single_fish import add_phase_annotations
                trial = phase_trials[sid]
                anchor = trial.onset_s if display["alignment"] == "sequence" else trial.bout_onset_s
                add_phase_annotations(ax, trial, anchor)
                ax.axvline(0, color="#555555", linestyle=":", linewidth=0.45, zorder=3)
                continue
            for phase, start, stop in display["phases"][sid]:
                ax.axvspan(start, stop, facecolor="#808080" if phase == "static" else "#4bb8ae",
                           alpha=0.20 if phase == "static" else 0.10, zorder=2)
                if phase == "active" and start > 0:
                    ax.axvline(start, color="#248f86", linewidth=0.45, zorder=3)
            ax.axvline(0, color="#555555", linestyle=":", linewidth=0.45, zorder=3)
            end = display["phases"][sid][-1][2] if display["phases"][sid] else 0
            ax.axvline(end, color="#555555", linestyle="--", linewidth=0.5, zorder=3)
        axes[0, column].set_title(display['stimulus_labels'][sid], pad=3)
        axes[-1, column].xaxis.set_major_locator(MaxNLocator(nbins=3))
        if proportional_widths:
            axes[0, column].set_xlim(*bounds[sid])
    if not proportional_widths:
        axes[0, 0].set_xlim(min(display["time_s"][sid][0] - dt / 2 for sid in ids),
                            max(display["time_s"][sid][-1] + dt / 2 for sid in ids))
    if has_raster:
        axes[0, 0].set_ylabel("Neurons")
        cax = fig.add_axes([0.92, 0.21 if nrows == 1 else 0.43, 0.012,
                            0.61 if nrows == 1 else 0.39])
        fig.colorbar(im, cax=cax, label=color_label)
    # Keep the phase key together in the corner, clear of the stimulus titles.
    from matplotlib.patches import Patch
    phase_handles = ([Patch(facecolor=color, alpha=0.17, label=label) for label, color in
        [("Preconditioning", "#b59bd6"), ("Blank delay", "#d9dde3"), ("Bout", "#52b9b1")]]
        if phase_trials is not None else
        [Patch(facecolor="#808080", alpha=0.20, label="Static"),
         Patch(facecolor="#4bb8ae", alpha=0.10, label="Motion / flicker")])
    fig.legend(handles=phase_handles,
               loc="upper right", bbox_to_anchor=(0.985, 0.985), ncol=3 if phase_trials is not None else 2,
               frameon=False, borderaxespad=0, handlelength=1.2,
               handletextpad=0.4, columnspacing=1.0,
               fontsize=plt.rcParams["axes.labelsize"])
    if has_mean:
        axes[-1, 0].set_ylabel(mean_label)
        axes[-1, 0].yaxis.set_major_locator(MaxNLocator(nbins=3))
    xlabel = f"From {display['alignment']} onset (s)" if phase_trials is not None else "Time (s)"
    fig.supxlabel(xlabel, y=0.02, fontsize=plt.rcParams["axes.labelsize"])
    if title:
        fig.suptitle(title, x=0.10, y=0.965, ha="left")
    return fig, axes


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

        ax_mean.set_ylabel(mean_ylabel)
        ax_mean.set_xlabel("Time (s)")
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

    compact = figsize[0] <= 7.5
    legend_style = (
        {"loc": "lower center", "ncol": min(7, len(labels))}
        if compact else {"bbox_to_anchor": (1.18, 1), "loc": "upper left"}
    )
    legend_owner = fig if compact else ax
    legend_owner.legend(
        handles=handles,
        labels=labels,
        title="Movement onset",
        borderaxespad=0,
        frameon=False,
        **legend_style,
    )

    # --- colorbar in dedicated axis ---
    if not is_binary:
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(r"$\Delta F/F$")
    else:
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Significant activity (0/1)")

    if compact:
        fig.subplots_adjust(left=0.10, right=0.90, bottom=0.28, top=0.87)

    return fig, ax, im, neuron_order

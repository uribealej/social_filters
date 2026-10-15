"""Motion-delta distributions and pooled active-trace decision figures."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from src.neuron_ordering import (
    _diagnostic_sort_order,
)
from src.plotting_common import (
    _vertical_reference_line,
    _horizontal_reference_line,
    STATIC_FLICKER_STIMULUS_COLORS,
)


def plot_paired_fish_values(table, value_column, condition_order, fish_ids, ax, ylabel):
    """Plot one value per fish/condition and its equally weighted fish mean.

    Args:
        table (DataFrame): fish_id, condition, and numeric value_column; one row
            per fish/condition. Missing values leave gaps rather than zeros.
        value_column (str): Measurement column, with units supplied by ylabel.
        condition_order (sequence): Explicit x-axis order.
        fish_ids (sequence): Stable cohort/color order across mirrored panels.
        ax (Axes): Destination axes.
        ylabel (str): Measurement name and units.

    Returns:
        Axes: The populated axes. Inputs are not mutated; no files are saved.
    """
    x = np.arange(len(condition_order), dtype=float)
    colors = plt.get_cmap("tab10")
    values = []
    offsets = np.linspace(-0.10, 0.10, len(fish_ids)) if len(fish_ids) > 1 else [0.0]
    for index, (fid, offset) in enumerate(zip(fish_ids, offsets)):
        rows = table.loc[table["fish_id"] == fid].set_index("condition")
        y = rows[value_column].reindex(condition_order).to_numpy(dtype=float)
        values.append(y)
        ax.plot(x + offset, y, "o-", color=colors(index % 10), markersize=4,
                linewidth=0.8, alpha=0.65, label=str(fid))
    mean = np.asarray(values, dtype=float)
    valid = np.isfinite(mean)
    totals = np.where(valid, mean, 0).sum(axis=0)
    counts = valid.sum(axis=0)
    average = np.full(len(condition_order), np.nan)
    np.divide(totals, counts, out=average, where=counts > 0)
    ax.plot(x, average, "D-", color="black", markersize=5, linewidth=1.7, label="Fish mean")
    ax.set(xticks=x, xticklabels=condition_order, ylabel=ylabel, xlim=(-0.5, len(x) - 0.5))
    ax.tick_params(axis="x", labelrotation=40)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_fraction_matrix(matrix, ax, title, xlabel, ylabel, cmap="viridis"):
    """Draw an annotated [0, 1] fraction matrix with missing cells shown gray.

    Args:
        matrix (DataFrame): Rows/columns in the desired display order.
        ax (Axes): Destination axes.
        title, xlabel, ylabel (str): Explicit measurement and direction labels.
        cmap (str): Shared fraction color map.

    Returns:
        AxesImage: Image suitable for a shared colorbar. No files are saved.
    """
    values = matrix.to_numpy(dtype=float)
    palette = plt.get_cmap(cmap).with_extremes(bad="0.9")
    image = ax.imshow(np.ma.masked_invalid(values), vmin=0, vmax=1, cmap=palette, aspect="equal")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel,
           xticks=np.arange(matrix.shape[1]), xticklabels=matrix.columns,
           yticks=np.arange(matrix.shape[0]), yticklabels=matrix.index)
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    for i, j in np.ndindex(values.shape):
        value = values[i, j]
        ax.text(j, i, f"{value:.2f}" if np.isfinite(value) else "—", ha="center", va="center",
                fontsize=8, color="black" if not np.isfinite(value) or value > 0.55 else "white")
    return image


def plot_motion_delta_distribution(
    delta_df,
    value_col="delta_integral",
    segment_col="stimulus",
    side_col="side",
    stimuli=None,
    segments=None,
    side="all",
    sides=None,
    figsize=(10, 4),
    point_alpha=0.25,
    point_size=12,
    show_points=True,
    title=None,
    ax=None,
):
    """
    Plot motion-minus-fixed distributions grouped by stimulus label.

    By default this draws one panel with all rows. Pass ``side="left"`` or
    ``side="right"`` only when you explicitly want a side-specific subset.
    """
    missing = {value_col, segment_col} - set(delta_df.columns)
    if side != "all":
        missing |= {side_col} - set(delta_df.columns)
    if missing:
        raise ValueError(f"delta_df is missing required column(s): {sorted(missing)}")

    if side is not None:
        if side not in {"left", "right", "all"}:
            raise ValueError("side must be None, 'left', 'right', or 'all'.")
        sides = (side,)
    elif sides is None:
        sides = ("all",)

    if stimuli is not None:
        segments = stimuli

    if segments is None:
        segments = delta_df[segment_col].dropna().drop_duplicates().tolist()
    else:
        segments = list(segments)

    if not segments:
        raise ValueError(f"No {segment_col} values available to plot.")

    if ax is None:
        fig, axes = plt.subplots(1, len(sides), figsize=figsize, sharey=True)
        if len(sides) == 1:
            axes = np.asarray([axes])
    else:
        axes = np.asarray(ax).ravel()
        fig = axes[0].figure
        if len(axes) != len(sides):
            raise ValueError("Number of axes must match number of sides.")

    for axis, side in zip(axes, sides):
        if side == "all":
            side_df = delta_df
            axis_title = "Selected stimuli"
        else:
            side_df = delta_df[delta_df[side_col] == side]
            axis_title = side.capitalize()
        values_by_segment = [
            side_df.loc[side_df[segment_col] == segment, value_col].dropna().to_numpy()
            for segment in segments
        ]

        positions = np.arange(1, len(segments) + 1)
        nonempty_positions = [pos for pos, vals in zip(positions, values_by_segment) if vals.size > 0]
        nonempty_values = [vals for vals in values_by_segment if vals.size > 0]
        if nonempty_values:
            axis.violinplot(
                nonempty_values,
                positions=nonempty_positions,
                showmeans=False,
                showmedians=True,
                showextrema=False,
            )

            if show_points:
                rng = np.random.default_rng(0)
                for pos, vals in zip(positions, values_by_segment):
                    if vals.size == 0:
                        continue
                    jitter = rng.uniform(-0.08, 0.08, size=vals.size)
                    axis.scatter(
                        np.full(vals.size, pos) + jitter,
                        vals,
                        s=point_size,
                        alpha=point_alpha,
                        linewidths=0,
                    )

        axis.axhline(0, color="0.4", linewidth=1, linestyle="--")
        axis.set_xticks(positions)
        axis.set_xticklabels(segments, rotation=35, ha="right")
        axis.set_title(axis_title)
        axis.set_xlabel(segment_col.replace("_", " "))
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[0].set_ylabel(value_col)
    if title is None:
        title = value_col.replace("_", " ")
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def plot_active_trace_decision_diagnostic(
    diagnostic,
    fps_2p=2.0,
    t_pre_s=5.0,
    stimuli_durations=None,
    sort_mode="decision_then_mean",
    neuron_order=None,
    figsize=(12, 7),
    trace_cmap="Greys",
    decision_cmap="Greys",
    motion_onset_key="static_before_sec",
    motion_line_color="tab:red",
    motion_line_style="--",
    show_active_count_trace=False,
    active_count_threshold=0.5,
    active_count_color="tab:blue",
    active_count_ylabel="# Active neurons",
    active_count_height_ratio=1.2,
    event_markers=None,
    row_separator=None,
    trace_title=None,
    event_marker_label_y=0.98,
    event_marker_label_va="top",
):
    """
    Plot pooled significant-response traces beside strict active decisions.
    """
    trace_matrix = np.asarray(diagnostic["trace_matrix"], dtype=float)
    decision_matrix = np.asarray(diagnostic["decision_matrix"], dtype=float)
    stim_labels = list(diagnostic["stim_labels"])
    block_widths = list(diagnostic["trace_block_widths"])
    block_timepoints = list(diagnostic.get("trace_block_timepoints", block_widths))
    block_reps = list(diagnostic.get("trace_block_reps", [1] * len(block_widths)))
    combine_mode = diagnostic.get("combine_mode", "mean")

    if trace_matrix.ndim != 2 or decision_matrix.ndim != 2:
        raise ValueError("trace_matrix and decision_matrix must be 2D.")
    if trace_matrix.shape[0] != decision_matrix.shape[0]:
        raise ValueError("trace_matrix and decision_matrix must share row count.")

    if neuron_order is None:
        neuron_order = _diagnostic_sort_order(trace_matrix, decision_matrix, sort_mode)
    else:
        neuron_order = np.asarray(neuron_order)
        if neuron_order.ndim != 1 or neuron_order.shape[0] != trace_matrix.shape[0]:
            raise ValueError(
                f"neuron_order has shape {neuron_order.shape}; "
                f"expected ({trace_matrix.shape[0]},)."
            )

    trace_sorted = trace_matrix[neuron_order]
    decision_sorted = decision_matrix[neuron_order]

    fig = plt.figure(figsize=figsize)
    if show_active_count_trace:
        gs = gridspec.GridSpec(
            2,
            4,
            width_ratios=[32, 1.0, 7, 0.8],
            height_ratios=[6, active_count_height_ratio],
            hspace=0.05,
            wspace=0.15,
        )
        ax_trace = fig.add_subplot(gs[0, 0])
        ax_active_count = fig.add_subplot(gs[1, 0], sharex=ax_trace)
        cax_trace = fig.add_subplot(gs[0, 1])
        ax_decision = fig.add_subplot(gs[0, 2], sharey=ax_trace)
        cax_decision = fig.add_subplot(gs[0, 3])
    else:
        gs = gridspec.GridSpec(
            1,
            4,
            width_ratios=[32, 1.0, 7, 0.8],
            wspace=0.15,
        )
        ax_trace = fig.add_subplot(gs[0, 0])
        ax_active_count = None
        cax_trace = fig.add_subplot(gs[0, 1])
        ax_decision = fig.add_subplot(gs[0, 2], sharey=ax_trace)
        cax_decision = fig.add_subplot(gs[0, 3])

    trace_im = ax_trace.imshow(
        trace_sorted,
        aspect="auto",
        interpolation="nearest",
        cmap=trace_cmap,
        vmin=0,
        vmax=1,
    )
    decision_im = ax_decision.imshow(
        decision_sorted,
        aspect="auto",
        interpolation="nearest",
        cmap=decision_cmap,
        vmin=0,
        vmax=1,
    )

    boundaries = np.cumsum(block_widths)
    starts = np.r_[0, boundaries[:-1]]
    centers = starts + np.asarray(block_widths) / 2.0

    for boundary in boundaries[:-1]:
        _vertical_reference_line(ax_trace, boundary - 0.5, color="0.65", linewidth=0.8, alpha=0.8)
        if ax_active_count is not None:
            _vertical_reference_line(ax_active_count, boundary - 0.5, color="0.65", linewidth=0.8, alpha=0.8)

    if row_separator is not None:
        separator = float(row_separator) - 0.5
        if 0 <= separator < trace_matrix.shape[0] - 0.5:
            _horizontal_reference_line(ax_trace, separator, color="black", linewidth=1.0)
            _horizontal_reference_line(ax_decision, separator, color="black", linewidth=1.0)

    if stimuli_durations is not None:
        for start, label, width, n_time, n_reps in zip(
            starts,
            stim_labels,
            block_widths,
            block_timepoints,
            block_reps,
        ):
            duration = stimuli_durations.get(label)
            if duration is None or motion_onset_key not in duration:
                continue
            motion_frame = int(
                np.clip(
                    round((float(t_pre_s) + float(duration[motion_onset_key])) * float(fps_2p)),
                    0,
                    max(int(n_time) - 1, 0),
                )
            )

            if combine_mode == "concat":
                for rep_idx in range(int(n_reps)):
                    xpos = start + rep_idx * int(n_time) + motion_frame
                    if xpos < start + width:
                        _vertical_reference_line(
                            ax_trace,
                            xpos - 0.5,
                            color=motion_line_color,
                            linestyle=motion_line_style,
                            linewidth=1.0,
                            alpha=0.9,
                        )
                        if ax_active_count is not None:
                            _vertical_reference_line(
                                ax_active_count,
                                xpos - 0.5,
                                color=motion_line_color,
                                linestyle=motion_line_style,
                                linewidth=1.0,
                                alpha=0.9,
                            )
            else:
                xpos = start + motion_frame
                if xpos < start + width:
                    _vertical_reference_line(
                        ax_trace,
                        xpos - 0.5,
                        color=motion_line_color,
                        linestyle=motion_line_style,
                        linewidth=1.0,
                        alpha=0.9,
                    )
                    if ax_active_count is not None:
                        _vertical_reference_line(
                            ax_active_count,
                            xpos - 0.5,
                            color=motion_line_color,
                            linestyle=motion_line_style,
                            linewidth=1.0,
                            alpha=0.9,
                        )

    if event_markers:
        marker_colors = STATIC_FLICKER_STIMULUS_COLORS
        for block_index, (start, label, width) in enumerate(zip(starts, stim_labels, block_widths)):
            markers = event_markers.get(label, [])
            if not markers:
                continue
            duration = {} if stimuli_durations is None else stimuli_durations.get(label, {})
            onset_s = float(duration.get(motion_onset_key, 0.0))
            for marker_index, marker in enumerate(markers):
                relative_s = float(marker["time_after_motion_onset_s"])
                xpos = start + (float(t_pre_s) + onset_s + relative_s) * float(fps_2p) - 0.5
                if not (start - 0.5 <= xpos < start + width - 0.5):
                    continue
                color = marker.get("color", marker_colors[marker_index % len(marker_colors)])
                _vertical_reference_line(ax_trace, xpos, color=color, linestyle=(0, (4, 2)), linewidth=1.25, alpha=0.9)
                if ax_active_count is not None:
                    _vertical_reference_line(ax_active_count, xpos, color=color, linestyle=(0, (4, 2)), linewidth=1.25, alpha=0.9)
                marker_label = marker.get(
                    "label",
                    f"{marker.get('flicker_stimulus', 'position')}\n"
                    f"idx {int(marker.get('nearest_bout_position_index', -1))}\n+"
                    f"+{relative_s:.2f} s",
                )
                label_x = marker.get("label_x_axes", xpos)
                label_y = float(marker.get("label_y_axes", event_marker_label_y))
                label_transform = (
                    ax_trace.transAxes
                    if "label_x_axes" in marker
                    else ax_trace.get_xaxis_transform()
                )
                ax_trace.text(
                    label_x,
                    label_y,
                    marker_label,
                    transform=label_transform,
                    rotation=marker.get("label_rotation", 90),
                    va=marker.get("label_va", event_marker_label_va),
                    ha=marker.get("label_ha", "right"),
                    color=color,
                    fontsize=7,
                    clip_on=False,
                )

    ax_trace.set_xticks(centers)
    ax_trace.set_xticklabels(stim_labels, rotation=45, ha="right")
    ax_trace.set_xlabel("Aligned time blocks")
    ax_trace.set_ylabel("Pooled neurons")

    if ax_active_count is not None:
        active_count = np.nansum(trace_matrix >= active_count_threshold, axis=0)
        x = np.arange(trace_matrix.shape[1])
        ax_active_count.plot(x, active_count, color=active_count_color, linewidth=1.5)
        ax_active_count.set_xlim(-0.5, trace_matrix.shape[1] - 0.5)
        ax_active_count.set_xticks(centers)
        ax_active_count.set_xticklabels(stim_labels, rotation=45, ha="right")
        ax_active_count.set_ylabel(active_count_ylabel)
        ax_active_count.set_xlabel("Aligned time blocks")
        finite_counts = active_count[np.isfinite(active_count)]
        count_upper = max(1.0, float(finite_counts.max()) * 1.05) if finite_counts.size else 1.0
        # Motion-marker lines are added before this trace and otherwise lock the
        # subplot to their default 0--1 range, clipping the real cell counts.
        ax_active_count.set_ylim(0, count_upper)
        # Reference lines were created before the count scale was known. Extend
        # their y-data now so block boundaries and motion-onset markers match
        # the temporal raster above across the full count-panel height.
        for line in ax_active_count.lines[:-1]:
            xdata = np.asarray(line.get_xdata(), dtype=float)
            if xdata.size == 2 and np.isclose(xdata[0], xdata[1]):
                line.set_ydata([0.0, count_upper])
        ax_active_count.spines["top"].set_visible(False)
        ax_active_count.spines["right"].set_visible(False)
        plt.setp(ax_trace.get_xticklabels(), visible=False)
        ax_trace.set_xlabel("")

    ax_decision.set_xticks(np.arange(len(stim_labels)))
    ax_decision.set_xticklabels(stim_labels, rotation=45, ha="right")
    ax_decision.set_xlabel("Strict active decision")
    ax_decision.tick_params(labelleft=False)

    if trace_title is not None:
        ax_trace.set_title(str(trace_title))
    elif fps_2p > 0 and trace_matrix.shape[1] > 0:
        seconds = trace_matrix.shape[1] / float(fps_2p)
        ax_trace.set_title(f"Significant raster ({combine_mode}; {seconds:.1f} s shown)")
    else:
        ax_trace.set_title(f"Significant raster ({combine_mode})")
    ax_decision.set_title("Active")

    trace_label = (
        "Fraction of repetitions significant"
        if combine_mode == "mean"
        else "Significant activity (0/1)"
    )
    fig.colorbar(trace_im, cax=cax_trace, label=trace_label)
    fig.colorbar(decision_im, cax=cax_decision, label="Final active (0/1)")

    fig.subplots_adjust(left=0.08, right=0.94, bottom=0.18, top=0.90)
    axes = (ax_trace, ax_decision)
    if ax_active_count is not None:
        axes = (ax_trace, ax_active_count, ax_decision)
    return fig, axes, neuron_order

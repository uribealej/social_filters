"""Bout-referenced flicker-position comparison figures."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.plotting_common import (
    _vertical_reference_line,
    _horizontal_reference_line,
    STATIC_FLICKER_STIMULUS_COLORS,
)
from src.plotting_diagnostics import (
    plot_active_trace_decision_diagnostic,
)


def plot_bout_flicker_position_figure(side_result, figsize=(15, 11)):
    """Plot one hemifield's fixed-order bout/flicker position comparison.

    ``side_result`` is produced by
    ``multifish_analysis.build_bout_flicker_position_analysis``.  It contains
    both the Cell-06-compatible significant activity decisions and the
    descriptive matched-position summary.
    """
    result = dict(side_result)
    labels = list(result["stimulus_labels"])
    if len(labels) != 4:
        raise ValueError("A bout/flicker position figure requires one bout and three flicker panels.")
    traces = result["pooled_traces"]
    time_by_stimulus = result["time_relative_s"]
    n_neurons = int(traces[labels[0]]["zscore"].shape[0])
    if n_neurons == 0:
        raise ValueError("No neurons are available to plot.")

    # A manual layout keeps this many-panel figure responsive even with long
    # vertical marker labels; constrained_layout repeatedly solves a very
    # expensive layout problem here.
    figure = plt.figure(figsize=figsize)
    grid = figure.add_gridspec(4, 4, height_ratios=(4.2, 4.2, 1.7, 2.5))
    z_axes = [figure.add_subplot(grid[0, column]) for column in range(4)]
    # Do not share axes here: inverted image axes plus shared transforms can
    # trigger a Matplotlib transform recursion on Windows when event markers
    # are added.  Every panel gets the same explicit time coordinates below.
    significant_axes = [figure.add_subplot(grid[1, column]) for column in range(4)]
    count_axes = [figure.add_subplot(grid[2, column]) for column in range(4)]
    summary_axes = [figure.add_subplot(grid[3, column]) for column in range(3)]
    figure.add_subplot(grid[3, 3]).set_axis_off()

    all_zscores = np.concatenate([np.ravel(traces[label]["zscore"]) for label in labels])
    finite_zscores = all_zscores[np.isfinite(all_zscores)]
    if finite_zscores.size:
        z_limit = max(2.0, float(np.nanquantile(np.abs(finite_zscores), 0.99)))
    else:
        z_limit = 2.0
    z_image = None
    sig_image = None
    bout_label = result["bout_stimulus"]
    matches = pd.DataFrame(result["position_matches"])
    marker_colors = dict(zip(matches["flicker_stimulus"], STATIC_FLICKER_STIMULUS_COLORS[:len(matches)]))
    x_bounds = []

    for column, label in enumerate(labels):
        time_s = np.asarray(time_by_stimulus[label], dtype=float)
        zscore = np.asarray(traces[label]["zscore"], dtype=float)
        significant = np.asarray(traces[label]["significant"], dtype=float)
        if zscore.shape != significant.shape or zscore.shape[1] != time_s.size:
            raise ValueError(f"Incompatible plot arrays for stimulus {label!r}.")
        extent = [time_s[0] - 0.5 / len(time_s) * (time_s[-1] - time_s[0]), time_s[-1], n_neurons - 0.5, -0.5]
        z_image = z_axes[column].imshow(
            zscore, aspect="auto", cmap="RdBu_r", vmin=-z_limit, vmax=z_limit, extent=extent
        )
        sig_image = significant_axes[column].imshow(
            significant, aspect="auto", cmap="Greys", vmin=0, vmax=1, extent=extent
        )
        active_count = np.sum(significant > 0, axis=0)
        count_axes[column].plot(time_s, active_count, color="#222222", linewidth=1.4)
        count_axes[column].set_ylim(bottom=0)
        z_axes[column].set_title(label, fontweight="bold")
        x_bounds.extend([time_s[0], time_s[-1]])

        timing = result["panel_timing"][label]
        for axis in (z_axes[column], significant_axes[column], count_axes[column]):
            _vertical_reference_line(axis, timing["static_onset_relative_s"], color="#777777", linestyle=":", linewidth=1.0)
            _vertical_reference_line(axis, 0, color="#222222", linestyle="--", linewidth=1.0)
            _vertical_reference_line(axis, timing["analysis_end_relative_s"], color="#777777", linestyle="--", linewidth=0.9)
            if label == bout_label:
                _add_bout_position_markers(axis, matches, marker_colors)

        if 0 < result["n_bout_active"] < n_neurons:
            separator = result["n_bout_active"] - 0.5
            _horizontal_reference_line(z_axes[column], separator, color="#111111", linewidth=0.9)
            _horizontal_reference_line(significant_axes[column], separator, color="#111111", linewidth=0.9)

        z_axes[column].tick_params(labelbottom=False)
        significant_axes[column].tick_params(labelbottom=False)
        count_axes[column].set_xlabel("Time relative to motion/flicker onset (s)")
        count_axes[column].set_xlim(min(x_bounds), max(x_bounds))

    z_axes[0].set_ylabel("Fixed neuron order\n(bout-active → flicker-only)")
    significant_axes[0].set_ylabel("Temporal significant activity")
    count_axes[0].set_ylabel("# significant\nneurons")
    for axis in z_axes[1:] + significant_axes[1:] + count_axes[1:]:
        axis.tick_params(labelleft=False)

    # Keep colour scales stated in the row labels rather than adding colourbar
    # axes: the current Windows Matplotlib build can recurse while rendering
    # colourbar patches beside inverted image axes.
    z_axes[-1].text(1.02, 0.5, "z-score\nblue ↔ red", transform=z_axes[-1].transAxes,
                    rotation=90, va="center", ha="left", fontsize=8)
    significant_axes[-1].text(1.02, 0.5, "significant-trial\nfraction: 0 → 1",
                              transform=significant_axes[-1].transAxes,
                              rotation=90, va="center", ha="left", fontsize=8)
    _plot_bout_flicker_position_summary(summary_axes, result["summary_table"])
    side = str(result["side"]).capitalize()
    figure.suptitle(
        f"{side}: bout-referenced flicker-position responses\n"
        "dotted = static onset; black dashed = motion/flicker onset; grey dashed = Cell 06 analysis-window end",
        fontsize=13,
    )
    figure.subplots_adjust(left=0.06, right=0.90, top=0.89, bottom=0.07, wspace=0.32, hspace=0.40)
    return {"figure": figure, "zscore_axes": z_axes, "significant_axes": significant_axes, "count_axes": count_axes, "summary_axes": summary_axes}


def _add_bout_position_markers(axis, matches, marker_colors):
    for match in matches.itertuples(index=False):
        color = marker_colors[match.flicker_stimulus]
        label = (
            f"{match.flicker_stimulus}\nidx {int(match.nearest_bout_position_index)}\n"
            f"+{float(match.time_after_motion_onset_s):.2f} s"
        )
        _vertical_reference_line(
            axis, float(match.time_after_motion_onset_s), color=color,
            linestyle=(0, (4, 2)), linewidth=1.15, alpha=0.9,
        )
        y0, y1 = axis.get_ylim()
        offset = 0.02 * abs(y1 - y0)
        label_y = y1 + offset if y0 > y1 else y1 - offset
        axis.text(
            float(match.time_after_motion_onset_s), label_y, label, transform=axis.transData,
            rotation=90, va="top", ha="right", fontsize=7, color=color,
        )


def _plot_bout_flicker_position_summary(axes, summary_table):
    summary = pd.DataFrame(summary_table).copy()
    positions = summary[["flicker_stimulus", "nearest_bout_position_index", "time_after_motion_onset_s"]].drop_duplicates()
    positions = positions.sort_values("time_after_motion_onset_s", kind="stable")
    colors = {"bout-active": "#4c78a8", "flicker-only": "#e15759"}
    for axis, position in zip(axes, positions.itertuples(index=False)):
        data = summary.loc[summary["flicker_stimulus"] == position.flicker_stimulus]
        for group, color in colors.items():
            group_data = data.loc[data["group"] == group]
            axis.scatter(
                group_data["bout_mean_zscore"], group_data["flicker_mean_zscore"],
                s=9, alpha=0.32, color=color, linewidths=0, label=group,
            )
        finite = data[["bout_mean_zscore", "flicker_mean_zscore"]].to_numpy(float)
        finite = finite[np.isfinite(finite)]
        limit = max(1.0, float(np.max(np.abs(finite))) if finite.size else 1.0)
        axis.plot([-limit, limit], [-limit, limit], color="#555555", linestyle=":", linewidth=0.9)
        _horizontal_reference_line(axis, 0, color="#bbbbbb", linewidth=0.7)
        _vertical_reference_line(axis, 0, color="#bbbbbb", linewidth=0.7)
        axis.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
        axis.set_title(
            f"{position.flicker_stimulus}: bout idx {int(position.nearest_bout_position_index)}\n"
            f"+{float(position.time_after_motion_onset_s):.2f} s",
            fontsize=9,
        )
        axis.set_xlabel("Bout-window mean z-score")
        if axis is axes[0]:
            axis.set_ylabel("Flicker-window mean z-score")
            axis.legend(frameon=False, fontsize=7, loc="upper left")
    for axis in axes[len(positions):]:
        axis.set_axis_off()


def plot_bout_flicker_position_cell06_style(
    side_result,
    fps_2p=2.0,
    t_pre_s=5.0,
    onset_match_tolerance_s=None,
    figsize=(12, 7),
):
    """Render the Cell 06 significant-raster diagnostic with bout position markers.

    The data already arrive in the fixed order: bout-active rows ranked by
    their bout-control onset, followed by flicker-only rows.  This deliberately
    delegates the visual layout to ``plot_active_trace_decision_diagnostic``
    so the plot remains the same Cell 06 plot rather than a parallel raster.
    """
    result = dict(side_result)
    matches = pd.DataFrame(result["position_matches"])
    colors = STATIC_FLICKER_STIMULUS_COLORS
    markers = []
    for index, match in enumerate(matches.itertuples(index=False)):
        markers.append({
            "flicker_stimulus": match.flicker_stimulus,
            "nearest_bout_position_index": int(match.nearest_bout_position_index),
            "time_after_motion_onset_s": float(match.time_after_motion_onset_s),
            "color": colors[index % len(colors)],
            # Place the three close position labels in a readable strip above
            # the raster; their coloured vertical lines retain the exact time.
            "label": (
                f"{match.flicker_stimulus} | idx {int(match.nearest_bout_position_index)} | "
                f"+{float(match.time_after_motion_onset_s):.2f} s"
            ),
            "label_x_axes": 0.01,
            "label_y_axes": 1.005 + 0.045 * index,
            "label_rotation": 0,
            "label_va": "bottom",
            "label_ha": "left",
        })
    if onset_match_tolerance_s is None:
        onset_match_tolerance_s = 0.5 / float(fps_2p)
    if onset_match_tolerance_s < 0:
        raise ValueError("onset_match_tolerance_s must be >= 0.")

    figure, axes, neuron_order = plot_active_trace_decision_diagnostic(
        diagnostic=result["cell06_style_diagnostic"],
        fps_2p=fps_2p,
        t_pre_s=t_pre_s,
        stimuli_durations=result["stimuli_durations"],
        sort_mode=None,
        neuron_order=np.arange(len(result["order_table"])),
        trace_cmap="Greys",
        show_active_count_trace=False,
        event_markers={result["bout_stimulus"]: markers},
        row_separator=None,
        trace_title="Significant raster",
        event_marker_label_y=1.01,
        event_marker_label_va="bottom",
        figsize=figsize,
    )
    trace_axis = axes[0]
    onset_rows = []
    order = pd.DataFrame(result["order_table"])
    for marker_index, marker in enumerate(markers):
        matching_rows = np.flatnonzero(np.isclose(
            order["bout_response_onset_s"].to_numpy(float),
            marker["time_after_motion_onset_s"],
            atol=float(onset_match_tolerance_s),
            rtol=0,
        ))
        # Rows are ordered by bout onset.  Draw one separator just below the
        # final row in this onset group, rather than one line per neuron.
        # This gives each matched flicker position a single horizontal marker.
        last_matching_row = int(matching_rows[-1]) if matching_rows.size else None
        if last_matching_row is not None:
            _horizontal_reference_line(
                trace_axis,
                last_matching_row + 0.5,
                color=marker["color"],
                linewidth=1.0,
                alpha=0.85,
            )
        onset_rows.append({
            "flicker_stimulus": marker["flicker_stimulus"],
            "matched_rows": matching_rows,
            "last_matching_row": last_matching_row,
        })
    return {
        "figure": figure,
        "axes": axes,
        "neuron_order": neuron_order,
        "markers": markers,
        "onset_coincidence_rows": onset_rows,
    }

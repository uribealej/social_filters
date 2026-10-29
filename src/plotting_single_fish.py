"""Single-fish rasters, aligned stimulus chunks and onset markers."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import src.stimuli_timeline as st
from matplotlib.patches import Patch
from src.neuron_ordering import (
    compute_sort_orders,
    compute_single_sort_order,
)


_PHASE_COLORS = {"Preconditioning": "#b59bd6", "Blank delay": "#d9dde3", "Bout": "#52b9b1"}


def _validate_significant_review_inputs(dfof, centered, raster, fps, neuron_indices):
    arrays = [np.asarray(x) for x in (dfof, centered, raster)]
    if arrays[0].ndim != 2 or not all(x.shape == arrays[0].shape for x in arrays):
        raise ValueError("Original/centered dFoF and raster must share (time, neurons) shape.")
    if not all(arrays[0].shape) or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Review requires nonempty data and a positive finite imaging rate.")
    ids = np.asarray(neuron_indices)
    if (ids.ndim != 1 or len(ids) != arrays[0].shape[1] or len(np.unique(ids)) != len(ids)
            or not np.issubdtype(ids.dtype, np.integer) or (ids < 0).any()):
        raise ValueError("Supply one unique original merged-neuron ID per raster column.")
    return *arrays, ids


def _review_time_limits(time_window_s, n_time, fps):
    if time_window_s is None:
        return (0, n_time / fps)
    window = np.asarray(time_window_s, dtype=float)
    if window.shape != (2,) or not np.isfinite(window).all() or not 0 <= window[0] < window[1] <= n_time / fps:
        raise ValueError("Review time window must lie within the recording, in seconds.")
    return tuple(window)


def plot_significant_trace_comparison(dfof, centered, raster, fps, neuron_indices,
                                      sort_by_peak=False, time_window_s=None,
                                      vmin=0.0, vmax=0.05, figsize=(14, 9), title=""):
    """Compare original/centered dFoF and detected-event rasters before saving.

    Inputs share (time, retained neurons); ``neuron_indices`` maps columns to
    original merged-neuron IDs. Sorting is display-only and shared by all three
    panels. Return figure, axes and the displayed original neuron IDs. No neuron
    is removed, including zero-event neurons. Color/time limits are display-only.
    """
    dfof, centered, raster, ids = _validate_significant_review_inputs(dfof, centered, raster, fps, neuron_indices)
    limits = _review_time_limits(time_window_s, len(dfof), fps)
    order = np.argsort(-np.nanmax(dfof, axis=0), kind="stable") if sort_by_peak else np.arange(len(ids))
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    for ax, values, label, binary in zip(axes, (dfof, centered, raster),
            ("Original Delta F/F", "Noise-centered Delta F/F", "Detected event frames"), (False, False, True)):
        im = raster_with_stimuli(ax, values, fps, fish_id=title, neuron_order=order,
            title_suffix=label, is_binary=binary, vmin=0 if binary else vmin, vmax=1 if binary else vmax)
        ax.set(xlim=limits, ylabel="Neuron rank" if sort_by_peak else "Retained column")
        fig.colorbar(im, ax=ax, label="Event (0/1)" if binary else "Delta F/F")
    axes[-1].set_xlabel("Concatenated imaging time (s)")
    for ax in axes[:-1]:
        ax.set_xlabel("")
    fig.subplots_adjust(left=0.08, right=0.94, bottom=0.08, top=0.96, hspace=0.3)
    fig.canvas.draw()
    return fig, axes, ids[order].copy()


def plot_significant_trace_examples(dfof, centered, raster, fps, neuron_indices,
                                    neuron_ids=None, n_examples=20,
                                    high_activity_fraction=0.1, seed=42,
                                    time_window_s=None, figsize=(16, 14), title=""):
    """Show original/centered traces with every detected frame marked in red.

    Explicit ``neuron_ids`` are original merged-neuron IDs and must be retained.
    Otherwise select reproducible random examples without replacement from the
    highest positive original dFoF peaks, as in the all-neuron review notebook.
    Example selection never changes detector input, reliability or saved arrays.
    Return figure, axes and displayed original IDs (not retained-column indices).
    """
    dfof, centered, raster, ids = _validate_significant_review_inputs(dfof, centered, raster, fps, neuron_indices)
    limits = _review_time_limits(time_window_s, len(dfof), fps)
    if neuron_ids is None:
        if int(n_examples) != n_examples or n_examples < 1 or not 0 < high_activity_fraction <= 1:
            raise ValueError("Use positive integer n_examples and activity fraction in (0, 1].")
        peaks = np.nanmax(dfof, axis=0)
        pool = np.argsort(-peaks, kind="stable")[:max(1, int(np.ceil(high_activity_fraction * len(ids))))]
        pool = pool[peaks[pool] > 0]
        columns = np.random.default_rng(seed).choice(pool, min(n_examples, len(pool)), replace=False)
    else:
        requested = np.asarray(neuron_ids)
        if requested.ndim != 1 or not np.issubdtype(requested.dtype, np.integer) or not np.isin(requested, ids).all():
            raise ValueError("Choose original merged-neuron IDs present in the retained population.")
        columns = np.array([np.flatnonzero(ids == n)[0] for n in requested], dtype=int)
    ncols = min(4, max(1, len(columns)))
    fig, axes = plt.subplots(max(1, int(np.ceil(len(columns) / ncols))), ncols,
                             figsize=figsize, squeeze=False, layout="constrained")
    times = np.arange(len(dfof)) / fps
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, column in zip(axes.flat, columns):
        ax.set_visible(True)
        events = raster[:, column].astype(bool)
        ax.plot(times, dfof[:, column], lw=0.8, label="Original Delta F/F")
        ax.plot(times, centered[:, column], lw=0.8, alpha=0.75, label="Centered Delta F/F")
        ax.scatter(times[events], dfof[events, column], s=7, color="tab:red", label="Detected event frames", zorder=3)
        ax.set(xlim=limits, xlabel="Time (s)", ylabel="Delta F/F", title=f"Merged neuron {ids[column]}")
    if len(columns):
        handles, labels = axes.flat[0].get_legend_handles_labels()
        legend_y = -0.16 if len(columns) == 1 else -0.03
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, legend_y), ncol=3, frameon=False)
    fig.suptitle(title if len(columns) else "No neurons with positive original dFoF peaks")
    fig.canvas.draw()
    return fig, axes, ids[columns].copy()


def _phase_spans(row, anchor):
    edges = [row.onset_s, row.pre_end_s, row.bout_onset_s, row.end_s]
    return [(label, left - anchor, right - anchor)
            for label, left, right in zip(_PHASE_COLORS, edges[:-1], edges[1:]) if right > left]


def add_phase_annotations(ax, trial, anchor_s, offset_s=0.0):
    """Shade explicit phase spans on an existing axis, optionally in a repeat."""
    for label, start, stop in _phase_spans(trial, anchor_s):
        ax.axvspan(start + offset_s, stop + offset_s, color=_PHASE_COLORS[label], alpha=0.17)
        ax.axvline(start + offset_s, color=_PHASE_COLORS[label], linewidth=0.7)


def plot_phase_timing(trials, stimulus_keys, figsize=(12, 6)):
    """Show explicit real-duration phases, excluding external pauses."""
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    for i, key in enumerate(stimulus_keys):
        row = trials.loc[trials.stimulus_key == key].iloc[0]
        for label, start, end in _phase_spans(row, row.onset_s):
            ax.barh(i, end - start, left=start, color=_PHASE_COLORS[label], height=0.65)
    ax.set(yticks=np.arange(len(stimulus_keys)), yticklabels=stimulus_keys,
           xlabel="Seconds from presentation start", title="Stimulus phases (actual trajectory durations)")
    ax.invert_yaxis()
    ax.legend(handles=[Patch(color=c, label=k) for k, c in _PHASE_COLORS.items()],
              loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    return fig, ax


def plot_continuous_phase_raster(data, fps_2p, trials, figsize=(16, 5),
                                  title="Continuous dFoF", vmax=None):
    """Plot (time, neurons) data with phase strips and imaging-block boundaries."""
    data = np.asarray(data)
    if data.ndim != 2 or not data.shape[1]:
        raise ValueError("Continuous raster requires a nonempty neuron axis.")
    fig, (ax, timeline) = plt.subplots(2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [5, 1]}, layout="constrained")
    if vmax is None:
        vmax = max(float(np.nanpercentile(data, 99)), np.finfo(float).eps)
    image = ax.imshow(data.T, aspect="auto", origin="lower", interpolation="nearest",
        extent=[0, len(data) / fps_2p, -0.5, data.shape[1] - 0.5], vmin=0, vmax=vmax, cmap="magma")
    fig.colorbar(image, ax=ax, label="dFoF")
    ax.set(ylabel="Merged neuron index", title=title)
    sides = trials.hemifield.drop_duplicates().tolist() if "hemifield" in trials else ["presentation"]
    for row in trials.itertuples(index=False):
        y = sides.index(getattr(row, "hemifield", "presentation"))
        for label, start, end in _phase_spans(row, 0):
            timeline.broken_barh([(start, end - start)], (y - 0.3, 0.6), facecolors=_PHASE_COLORS[label])
        ax.axvline(row.onset_s, color="white", alpha=0.15, linewidth=0.5)
    for start in trials.block_start_frame.unique():
        for panel in (ax, timeline):
            panel.axvline(start / fps_2p, color="#4085c5", linewidth=1)
    timeline.set(yticks=np.arange(len(sides)), yticklabels=[s.capitalize() for s in sides],
                 xlabel="Concatenated imaging time (s)")
    timeline.legend(handles=[Patch(color=c, label=k) for k, c in _PHASE_COLORS.items()],
                    loc="upper center", bbox_to_anchor=(0.5, -0.55), ncol=3)
    return fig, (ax, timeline)


def prepare_phase_response_display(aligned, stimulus_ids, sort_mode="corravg",
                                    n_clusters=3, random_state=42):
    """Return trial means and one shared existing-method neuron order.

    Means retain every neuron and condition; nonfinite rows are put last by
    correlation ordering. No phase-specific response metric is calculated.
    """
    means = {sid: np.nanmean(aligned["trial_aligned_traces"][sid], axis=2) for sid in stimulus_ids}
    joined = np.concatenate([means[sid] for sid in stimulus_ids], axis=1)
    if joined.shape[0] < 2:
        order = np.arange(joined.shape[0])
    else:
        order = compute_single_sort_order(joined.T, sort_mode, n_clusters, random_state)
        if order is None:
            order = np.arange(joined.shape[0])
    return means, order


def plot_phase_aligned_responses(aligned, stimulus_keys, stimulus_ids, neuron_order,
                                  neuron_indices=None, neuron_labels=None,
                                  figsize=(19, 8), trace_figsize=(19, 10),
                                  trace_count=8, trace_offset=6.0,
                                  vmin=-1.0, vmax=4.0, binary=False, title="",
                                  trace_mode="individual"):
    """Plot trial-mean rasters and individual or population-average traces.

    Conditions follow the supplied order. The same supplied neuron ordering and
    original merged-neuron labels can be reused across alignments/populations.
    Heatmaps contain all selected neurons; line panels explicitly show only the
    first ``trace_count`` rows. Binary means are trial activation fractions.
    ``trace_mode='population'`` instead averages the trial means of exactly all
    raster rows after neuron selection, with equal neuron weights and NaNs
    omitted. Its common y-axis is mean z-score, with no vertical offsets or SEM.
    The default individual mode preserves existing callers.
    No interpolation/time warping is used; axes show nominal 2-Hz sample offsets.
    """
    if trace_mode not in ("individual", "population"):
        raise ValueError("trace_mode must be 'individual' or 'population'.")
    order = np.asarray(neuron_order, dtype=int)
    if neuron_indices is not None:
        order = order[np.isin(order, np.asarray(neuron_indices, dtype=int))]
    if order.size == 0:
        raise ValueError("No neurons selected for phase plots.")
    if len(stimulus_ids) != len(stimulus_keys):
        raise ValueError("Stimulus IDs and labels must have the same length.")
    n_neurons = next(iter(aligned["trial_aligned_traces"].values())).shape[0]
    labels = np.arange(n_neurons) if neuron_labels is None else np.asarray(neuron_labels)
    if len(labels) != n_neurons or len(np.unique(order)) != len(order) or order.min() < 0 or order.max() >= n_neurons:
        raise ValueError("Neuron ordering/labels do not match aligned data.")
    ncols = int(np.ceil(len(stimulus_ids) / 2))
    fig, axes = plt.subplots(2, ncols, figsize=figsize, sharex=True, sharey=True,
                             squeeze=False, layout="constrained")
    line_fig = None
    if (trace_count or trace_mode == "population") and not binary:
        line_fig, line_axes = plt.subplots(2, ncols, figsize=trace_figsize, sharex=True, sharey=True,
                                           squeeze=False, layout="constrained")
    times = aligned["time_s"]
    xmin = min(times[sid][0] for sid in stimulus_ids)
    xmax = max(times[sid][-1] + 1 / aligned["fps_2p"] for sid in stimulus_ids)
    selected_lines = order[:trace_count]
    for i, (sid, key) in enumerate(zip(stimulus_ids, stimulus_keys)):
        ax = axes.flat[i]
        arr = aligned["trial_aligned_traces"][sid]
        mean = np.nanmean(arr, axis=2)
        t = times[sid]
        im = ax.imshow(mean[order], aspect="auto", origin="lower", interpolation="nearest",
            extent=[t[0] - 0.5 / aligned["fps_2p"], t[-1] + 0.5 / aligned["fps_2p"], -0.5, len(order) - 0.5],
            vmin=0 if binary else vmin, vmax=1 if binary else vmax, cmap="magma")
        row = aligned["trials"].loc[aligned["trials"].stimulus_id == sid].iloc[0]
        anchor = row.onset_s if aligned["alignment"] == "sequence" else row.bout_onset_s
        panels = [ax] + ([line_axes.flat[i]] if line_fig is not None else [])
        for panel in panels:
            for label, start, end in _phase_spans(row, anchor):
                panel.axvspan(start, end, color=_PHASE_COLORS[label], alpha=0.17)
                if xmin <= start <= xmax:
                    panel.axvline(start, color=_PHASE_COLORS[label], linewidth=0.8)
            panel.axvline(0, color="#25707b", linewidth=0.7, linestyle="--")
            panel.set(xlim=(xmin, xmax), title=f"{key}\nn={arr.shape[2]} trials")
            if i >= ncols:
                panel.set_xlabel(f"From {'presentation' if aligned['alignment'] == 'sequence' else 'bout'} onset (s)", fontsize=9)
        if i % ncols == 0:
            ax.set_ylabel("Shared neuron order")
        if line_fig is not None:
            line_ax = line_axes.flat[i]
            if trace_mode == "population":
                line_ax.plot(t, np.nanmean(mean[order], axis=0), linewidth=1.3,
                             color="#263c65", label="Population mean")
                line_ax.axhline(0, color="gray", linewidth=0.5, alpha=0.6)
                if i % ncols == 0:
                    line_ax.set_ylabel("Population mean z-score")
            else:
                for j, neuron in enumerate(selected_lines):
                    line_ax.plot(t, mean[neuron] + j * trace_offset, linewidth=0.8, color="#263c65")
                    line_ax.axhline(j * trace_offset, color="gray", linewidth=0.3, alpha=0.4)
                line_ax.set(yticks=np.arange(len(selected_lines)) * trace_offset,
                            yticklabels=labels[selected_lines])
                if i % ncols == 0:
                    line_ax.set_ylabel("Merged neuron ID (offset traces)")
    for unused in range(len(stimulus_ids), 2 * ncols):
        axes.flat[unused].set_visible(False)
        if line_fig is not None:
            line_axes.flat[unused].set_visible(False)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Trial activation fraction" if binary else "Mean z-score", shrink=0.7)
    legend = [Patch(color=c, label=k) for k, c in _PHASE_COLORS.items()]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.055), ncol=3, frameon=False)
    fig.suptitle(f"{title} | {aligned['alignment']}-aligned | {len(order)} neurons")
    if line_fig is not None:
        line_fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.045), ncol=3, frameon=False)
        if trace_mode == "population":
            line_fig.suptitle(f"{title} | {aligned['alignment']}-aligned | mean of all {len(order)} raster neurons")
        else:
            line_fig.suptitle(f"{title} | individual means: first {len(selected_lines)} displayed neurons | spacing {trace_offset:g} z")
        line_fig.canvas.draw()
    fig.canvas.draw()
    return fig, line_fig


def add_stimuli_markers(ax, exp_log, stimuli_durations, stimuli_colors, time_offset=0, trace='movement',
                        stimuli_linestyles=None):
    if stimuli_linestyles is None:
        stimuli_linestyles = {}

    for _, row in exp_log.iterrows():
        if 'stim' in row['event']:
            stim_name = row['event'].split('_')[-1]
            stim_start = row['timestamp'] - exp_log['timestamp'].min() - time_offset
            if stim_name in stimuli_durations:
                if trace == 'movement':
                    dur = stimuli_durations[stim_name]
                    move_start = stim_start + dur['static_before_sec']
                    color = stimuli_colors.get(stim_name, 'black')
                    ls    = stimuli_linestyles.get(stim_name, '-')  # <- dashed here
                    ax.axvline(move_start, color=color, linestyle=ls, alpha=0.9, linewidth=1.8)

    # legend with dummy lines reflecting both color and style
    legend_handles = []
    for stim_name, color in stimuli_colors.items():
        ls = stimuli_linestyles.get(stim_name, '-')
        (line,) = ax.plot([], [], color=color, linestyle=ls, label=stim_name, linewidth=4)
        legend_handles.append(line)
    return legend_handles


def raster_with_stimuli(
    ax,
    data,                 # (frames x neurons) matrix: either ΔF/F or 0/1 significant
    fps,
    fish_id,
    neuron_order=None,
    title_suffix='',
    vmin=None,
    vmax=None,
    perc_for_vmax=99.0,
    is_binary=None,
):
    """
    Plot raster of traces sorted by neuron_order.

    Parameters
    ----------
    ax : matplotlib Axes
    data : array, shape (n_frames, n_neurons)
        ΔF/F or significant traces (0/1 or bool).
    fps : float
        Frames per second (for time axis).
    fish_id : str
        Fish identifier for title.
    neuron_order : array-like, optional
        Indices to sort neurons. If None, use original order.
    title_suffix : str, optional
        Extra text for the title (e.g. "k-means", "significant").
    vmin, vmax : float, optional
        Color limits. If None, chosen automatically from the data.
    perc_for_vmax : float, optional
        Percentile for automatic vmax if not given (for continuous data).
    is_binary : bool, optional
        Force binary handling (0/1). If None, automatically detected.
    add_colorbar : bool, optional
        If True, add colorbar to the same Axes' figure.

    Returns
    -------
    im : AxesImage
    """
    data = np.asarray(data)

    # Expect (frames, neurons); if user passed (neurons, frames), you could auto-detect:
    # if data.shape[0] < data.shape[1]:  # optional heuristic
    #     data = data.T

    n_frames, n_neurons = data.shape

    # If no neuron_order, keep original order
    if neuron_order is None:
        neuron_order = np.arange(n_neurons)
    else:
        neuron_order = np.asarray(neuron_order)
        assert neuron_order.shape[0] == n_neurons, (
            f"neuron_order length {neuron_order.shape[0]} != n_neurons {n_neurons}"
        )

    # Sort data by neuron_order → (neurons, time)
    sorted_data = data[:, neuron_order].T
    time_axis = np.arange(sorted_data.shape[1]) / float(fps)

    # --- Detect whether this is binary (significant traces) ---
    if is_binary is None:
        unique_vals = np.unique(sorted_data[~np.isnan(sorted_data)])  # ignore NaNs
        is_binary = (
            unique_vals.size <= 3 and
            np.all(np.isin(unique_vals, [0, 1]))
        )

    # --- Choose colormap and vmin/vmax ---
    if is_binary:
        # For 0/1 significant traces
        if vmin is None:
            vmin = -0.05
        if vmax is None:
            vmax = 1.05
        interpolation = 'nearest'  # crisp pixels
    else:
        # Continuous ΔF/F
        if vmin is None:
            # Often you want 0 as baseline for dF/F
            vmin = 0.0
        if vmax is None:
            # Use a high percentile to avoid being dominated by a few outliers
            vmax = np.nanpercentile(sorted_data, perc_for_vmax)
        interpolation = 'nearest'

    im = ax.imshow(
        sorted_data,
        aspect='auto',
        cmap='gray_r',
        vmin=vmin,
        vmax=vmax,
        extent=[time_axis[0], time_axis[-1], sorted_data.shape[0], 0],
        interpolation=interpolation,
    )

    ax.set_ylabel("# Neuron")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"{fish_id} - {title_suffix}", wrap=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return im


def save_sorted_rasters_for_all_modes(
    dfof,
    exp_log,
    stimuli_durations,
    stimuli_colors,
    stimuli_linestyles,
    stimuli_ordered,
    fps_2p,
    window_pre,
    window_post,
    plots_path: Path,
    prefix: str,
    n_clusters: int = 5,
    random_state: int = 42,
    average_across_repeats: bool = False,
    folder_name: str | None = None,
    is_binary= False,
):
    """
    Generate experiment-level ΔF/F rasters aligned to stimulus onsets and save one PNG per
    sorting strategy (unsorted, max intensity, PCA, KMeans, hierarchical, correlation).

    If average_across_repeats is:
      - False → use individual trials (your "grouped" version)
      - True  → average across repeats (your "average" version)
    """

    # Ensure output dir exists
    plots_path.mkdir(parents=True, exist_ok=True)

    # ----- 1) Build chunks once --------------------------------------------
    chunked_data, trial_starts, move_starts, move_colors, stim_labels = st.extract_stimulus_chunks(
        deltaF_F=dfof,                 # (frames, neurons)
        exp_log=exp_log,
        stimuli_durations=stimuli_durations,
        stimuli_colors=stimuli_colors,
        fps=fps_2p,
        stimuli_ordered=stimuli_ordered,
        window_pre=window_pre,
        window_post=window_post,
        average_across_repeats=average_across_repeats,
    )
    if chunked_data is None or chunked_data.size == 0:
        raise ValueError("No stimulus chunks were extracted. Check exp_log, stimuli_ordered, and windows.")

    # chunked_data: (neurons, time)
    if not is_binary:
        # Continuous data
        data_for_sort = chunked_data.T  # (time, neurons) or whatever you expect
    else:
        # Binary / thresholded version
        data_for_sort = chunked_data.T
        data_for_sort[data_for_sort < 0.5] = 0
        data_for_sort[data_for_sort >= 0.5] = 1

    # ----- 2) Compute sort orders ------------------------------------------
    # Your compute_sort_orders currently expects data.T in your notebook;
    # keep it consistent here:
    sorters = compute_sort_orders(data_for_sort, n_clusters=n_clusters, random_state=random_state)

    def _validate_order(idx, n_neurons, name):
        if idx is None:
            return None
        if len(idx) != n_neurons:
            raise ValueError(f"{name} length ({len(idx)}) != number of neurons ({n_neurons}).")
        return idx

    # Decide labels depending on average_across_repeats
    if average_across_repeats:
        title_base = "average across trials"
        filename_mid = "average"
    else:
        title_base = "ordered by stimulus type"
        filename_mid = "grouped"

    # ----- 3) Plot once per sort mode --------------------------------------
    saved_files = []

    for mode, idx in sorters.items():
        neuron_order = _validate_order(idx, chunked_data.shape[0], name=mode)

        fig, ax = plt.subplots(figsize=(12, 6))
        im = raster_with_stimuli(
            ax=ax,
            data=chunked_data.T,            # (time, neurons)
            fps=fps_2p,
            fish_id=prefix,                     # or fish_id if you prefer
            neuron_order=neuron_order,          # None = original order
            title_suffix=f"{title_base} | sort={mode}",

        )

        # Movement onset lines with per-stim color + linestyle
        for pos, stim_name in zip(move_starts, stim_labels):
            color = stimuli_colors.get(stim_name, "black")
            ls = stimuli_linestyles.get(stim_name, "-")  # default solid
            ax.axvline(pos / fps_2p, color=color, linestyle=ls, alpha=0.9, linewidth=1.8)

        # Legend (color + linestyle)
        legend_handles = []
        for stim_name in stimuli_ordered:
            if stim_name in stimuli_colors:
                color = stimuli_colors[stim_name]
                ls = stimuli_linestyles.get(stim_name, "-")
                (line,) = ax.plot([], [], color=color, linestyle=ls, label=stim_name, linewidth=1.5)
                legend_handles.append(line)

        ax.legend(
            handles=legend_handles,
            title="Movement onset\nacross stimuli",
            bbox_to_anchor=(1.2, 1),
            loc="upper left",
            borderaxespad=0,
            frameon=False,
        )

        ax.set_xlabel("Chunks aligned to stimulus onset (s)")

        if not is_binary:
            fig.colorbar(im, ax=ax, label=r"$\Delta F/F$")
        else:
            fig.colorbar(im, ax=ax, label=r"Significant activity (0/1)")

        plt.subplots_adjust(right=0.85)

        # Decide where to save
        if folder_name is not None:
            out_dir = plots_path / folder_name
        else:
            out_dir = plots_path

        # Make sure the folder exists
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save
        out_png = out_dir / f"{prefix}_{filename_mid}_dfof_sorted_by_{mode}.png"
        fig.savefig(out_png, dpi=600, bbox_inches="tight")
        plt.close(fig)

        saved_files.append(out_png.name)

    print("✅ Saved:", *[f"- {name}" for name in saved_files], sep="\n")


def plot_sorted_chunks_single_mode(
    dfof,
    exp_log,
    stimuli_durations,
    stimuli_colors,
    stimuli_linestyles,
    stimuli_ordered,
    fps_2p,
    window_pre,
    window_post,
    sort_mode="kmeans",
    n_clusters=3,
    random_state=42,
    average_across_repeats=False,
    figsize=(12, 6),
    fish_id="",
    neuron_order=None,       # 👈 NEW: custom order (optional)
    sort_label=None, # 👈 NEW: label to show in the title
    is_binary =False,
    vmin=None,
    vmax=None,
):
    """
    Build stimulus-aligned chunks and plot a single raster.

    You can either:
      - let the function compute the order via `sort_mode`, or
      - pass your own `neuron_order` (1D index array).

    If `neuron_order` is provided, `sort_mode` is ignored for the ordering.
    """

    # 1) Build chunks
    chunked_data, trial_starts, move_starts, move_colors, stim_labels = st.extract_stimulus_chunks(
        deltaF_F=dfof,
        exp_log=exp_log,
        stimuli_durations=stimuli_durations,
        stimuli_colors=stimuli_colors,
        fps=fps_2p,
        stimuli_ordered=stimuli_ordered,
        window_pre=window_pre,
        window_post=window_post,
        average_across_repeats=average_across_repeats,

    )

    if chunked_data is None or chunked_data.size == 0:
        raise ValueError("No stimulus chunks were extracted. Check exp_log, stimuli_ordered, and windows.")
        # chunked_data: (neurons, time)
    n_neurons = chunked_data.shape[0]

    if not is_binary:
        # Continuous data
        data_for_sort = chunked_data.T  # (time, neurons) or whatever you expect
    else:
        # Binary / thresholded version
        data_for_sort = chunked_data.T
        data_for_sort[data_for_sort < 0.5] = 0
        data_for_sort[data_for_sort >= 0.5] = 1

        # 2) Decide which neuron_order to use
    if neuron_order is not None:
        # Use user-provided order, with sanity check
        neuron_order = np.asarray(neuron_order)
        if neuron_order.ndim != 1 or neuron_order.shape[0] != n_neurons:
            raise ValueError(
                f"Custom neuron_order has shape {neuron_order.shape}, "
                f"expected ({n_neurons},)."
            )
        label = sort_label or "custom"
    else:
        # Compute only the requested sort order
        neuron_order = compute_single_sort_order(
            data_for_sort,
            sort_mode=sort_mode,
            n_clusters=n_clusters,
            random_state=random_state,
        )
        label = sort_label or sort_mode

    # 3) Plot raster
    fig, ax = plt.subplots(figsize=figsize)
    im = raster_with_stimuli(
        ax=ax,
        data=data_for_sort,       # (time, neurons)
        fps=fps_2p,
        fish_id=fish_id,
        neuron_order=neuron_order,
        title_suffix=f"ordered by stimulus type | sort={label}",
        vmax= vmax,
        vmin=vmin,

    )

    # 4) Movement start lines — use style based on each chunk's stimulus label
    move_styles = [stimuli_linestyles.get(name, "-") for name in stim_labels]
    for pos, color, ls in zip(move_starts, move_colors, move_styles):
        ax.axvline(pos / fps_2p, color=color, linestyle=ls, alpha=0.9, linewidth=1.0)

    ax.set_xlabel("Chunks aligned to stimulus onset (s)")

    # 5) Legend that matches color + linestyle (movement onsets)
    legend_handles = []
    for stim_name in stimuli_ordered:
        if stim_name in stimuli_colors:
            color = stimuli_colors[stim_name]
            ls = stimuli_linestyles.get(stim_name, "-")
            (line,) = ax.plot([], [], color=color, linestyle=ls, label=stim_name, linewidth=2)
            legend_handles.append(line)

    # Movement legend on the *figure* (right side)
    mov_legend = fig.legend(
        handles=legend_handles,
        title="Movement onset\nacross stimuli",
        bbox_to_anchor=(0.88, 0.6),  # (x, y) in figure coords
        loc="center left",
        borderaxespad=0,
        frameon=False,
    )

    # 6) Colorbar / binary legend + layout
    if not is_binary:
        # Continuous ΔF/F → colorbar on the axes
        fig.colorbar(im, ax=ax, label=r"$\Delta F/F$",
                     fraction=0.17, pad=0.01)
    else:
        # Binary significance → legend on the axes (does NOT kill the fig legend)
        activity_handles = [
            Patch(facecolor='black', edgecolor='black', label='Sign. (1)'),
            Patch(facecolor='white', edgecolor='black', label='Not sign. (0)'),
        ]
        ax.legend(
            handles=activity_handles,
            title='Activity',
            loc='upper right',
            bbox_to_anchor=(1.2, 1.0),
            frameon=False,
        )

    fig.tight_layout()
    return fig, ax, neuron_order

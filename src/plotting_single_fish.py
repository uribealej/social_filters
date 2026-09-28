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

    ax.set_ylabel("# Neuron", fontsize=16)
    ax.set_xlabel("Time (s)", fontsize=16)
    ax.set_title(f"{fish_id} - {title_suffix}", fontsize=16)
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

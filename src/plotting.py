from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import pdist
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import src.stimuli_timeline as st

# def add_stimuli_markers(ax, exp_log, stimuli_durations, stimuli_colors, time_offset=0, trace='movement'):
#     """
#     Add vertical lines for stimulus movement starts and return legend handles.
#
#     Parameters:
#     - ax: matplotlib Axes object
#     - exp_log: DataFrame with stimulus events and timestamps
#     - stimuli_durations: dict with durations, e.g., {'forward': {...}}
#     - stimuli_colors: dict mapping stimulus names to colors
#     - time_offset: optional offset (e.g., start time) to align timestamps (default=0)
#
#     Returns:
#     - legend_handles: list of matplotlib Line2D objects for legend
#     """
#     for _, row in exp_log.iterrows():
#         if 'stim' in row['event']:
#             stim_name = row['event'].split('_')[-1]
#             stim_start = row['timestamp'] - exp_log['timestamp'].min() - time_offset
#             if stim_name in stimuli_durations:
#                 if trace == 'movement':
#                     dur = stimuli_durations[stim_name]
#                     move_start = stim_start + dur['static_before_sec']
#                     color = stimuli_colors.get(stim_name, 'black')
#                     ax.axvline(move_start, color=color, alpha=0.8, linewidth=1.5)
#
#     # Legend with dummy lines
#     legend_handles = []
#     for stim_name, color in stimuli_colors.items():
#         line, = ax.plot([], [], color=color, label=stim_name, linewidth=4)
#         legend_handles.append(line)
#
#     return legend_handles
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
    ax, deltaF_F, fps, fish_id, neuron_order=None, title_suffix='', min=0, max=0.4):
    """
    Plot raster of ΔF/F traces sorted by neuron_order, with stimulus markers and legend.

    Parameters:
    - ax: matplotlib Axes object
    - deltaF_F: (frames x neurons) ΔF/F matrix
    - fps: frames per second (for time axis)
    - plane_name: string for labeling plot
    - fish_id: string for labeling plot
    - neuron_order: 1D array of neuron indices for sorting (length = neurons)
    - title_suffix: optional extra string for plot title (e.g., clustering method)
    """
    # If no neuron_order, keep original order
    if neuron_order is None:
        neuron_order = np.arange(deltaF_F.shape[1])

    # Sort data by neuron_order
    sorted_data = deltaF_F[:, neuron_order].T  # (neurons, time)
    time_axis = np.arange(sorted_data.shape[1]) / fps

    im = ax.imshow(
        sorted_data,
        aspect='auto',
        cmap='gray_r',
        vmin=min,
        vmax=max,
        extent=[time_axis[0], time_axis[-1], sorted_data.shape[0], 0]
    )

    ax.set_ylabel("# Neuron")
    ax.set_title(f"{fish_id}  DF/F - {title_suffix}")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    return im



def compute_sort_orders(data, n_clusters=3, random_state=42):
    """
    data: array (neurons, time) – e.g. your chunked_data
    returns: dict mode -> index array
    """
    # (a) Max intensity
    max_per_neuron = np.nanmax(data, axis=0)
    maxint_sorted_idx = np.argsort(-max_per_neuron)

    # (b) PCA
    scores = PCA(n_components=n_clusters).fit_transform(data.T)
    pca_order = np.argsort(-scores[:, 0])

    # (c) KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(data.T)
    kmeans_sorted_idx = np.argsort(kmeans.labels_)

    # (d) Hierarchical (Ward)
    Z_hier = linkage(data.T, method='ward')
    clusters = fcluster(Z_hier, t=n_clusters, criterion='maxclust')
    hier_sorted_idx = np.argsort(clusters)

    # (e) Correlation on averaged traces
    dist = pdist(data.T, metric='correlation')
    Z_corr = linkage(dist, method='average')
    corravg_sorted_idx = leaves_list(Z_corr)

    return {
        "unsorted":    None,
        "max_intensity": maxint_sorted_idx,
        "pca":          pca_order,
        "kmeans":       kmeans_sorted_idx,
        "hier":         hier_sorted_idx,
        "corravg":      corravg_sorted_idx,
    }


def compute_single_sort_order(data, sort_mode, n_clusters=3, random_state=42):
    """
    data: array (time, neurons) – e.g. chunked_data.T
    sort_mode: 'unsorted', 'max_intensity', 'pca', 'kmeans', 'hier', 'corravg'
    returns: index array or None (for 'unsorted')
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array (time, neurons), got {data.shape}")

    if sort_mode == "unsorted":
        return None

    # (a) Max intensity per neuron
    if sort_mode == "max_intensity":
        max_per_neuron = np.nanmax(data, axis=0)  # (neurons,)
        return np.argsort(-max_per_neuron)

    # (b) PCA
    if sort_mode == "pca":
        scores = PCA(n_components=n_clusters).fit_transform(data.T)  # (neurons, n_components)
        return np.argsort(-scores[:, 0])

    # (c) KMeans
    if sort_mode == "kmeans":
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(data.T)
        return np.argsort(kmeans.labels_)

    # (d) Hierarchical (Ward)
    if sort_mode == "hier":
        Z_hier = linkage(data.T, method="ward")
        clusters = fcluster(Z_hier, t=n_clusters, criterion="maxclust")
        return np.argsort(clusters)

    # (e) Correlation on averaged traces
    if sort_mode == "corravg":
        dist = pdist(data.T, metric="correlation")
        Z_corr = linkage(dist, method="average")
        return leaves_list(Z_corr)

    raise ValueError(f"Unknown sort_mode {sort_mode!r}. "
                     f"Use one of: 'unsorted', 'max_intensity', 'pca', 'kmeans', 'hier', 'corravg'.")

# and here we assume compute_sort_orders and raster_with_stimuli are in this module


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

    # ----- 2) Compute sort orders ------------------------------------------
    # Your compute_sort_orders currently expects data.T in your notebook;
    # keep it consistent here:
    sorters = compute_sort_orders(chunked_data.T, n_clusters=n_clusters, random_state=random_state)

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
            deltaF_F=chunked_data.T,            # (time, neurons)
            fps=fps_2p,
            fish_id=prefix,                     # or fish_id if you prefer
            neuron_order=neuron_order,          # None = original order
            title_suffix=f"{title_base} | sort={mode}",
            min=0.009,
            max=0.15,
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
        fig.colorbar(im, ax=ax, label=r"$\Delta F/F$")
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
    sort_label=None,         # 👈 NEW: label to show in the title
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
    data_for_sort = chunked_data.T  # (time, neurons)
    n_neurons = chunked_data.shape[0]

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
        deltaF_F=data_for_sort,       # (time, neurons)
        fps=fps_2p,
        fish_id=fish_id,
        neuron_order=neuron_order,
        title_suffix=f"ordered by stimulus type | sort={label}",
        max=1.0,
        min=0.4,
    )

    # 4) Movement start lines — use style based on each chunk's stimulus label
    move_styles = [stimuli_linestyles.get(name, "-") for name in stim_labels]
    for pos, color, ls in zip(move_starts, move_colors, move_styles):
        ax.axvline(pos / fps_2p, color=color, linestyle=ls, alpha=0.9, linewidth=1.0)

    ax.set_xlabel("Chunks aligned to stimulus onset (s)")

    # 5) Legend that matches color + linestyle
    legend_handles = []
    for stim_name, color in stimuli_colors.items():
        ls = stimuli_linestyles.get(stim_name, "-")
        (line,) = ax.plot([], [], color=color, linestyle=ls, label=stim_name, linewidth=2)
        legend_handles.append(line)

    ax.legend(
        handles=legend_handles,
        title="Movement onset\nacross stimuli",
        bbox_to_anchor=(1.15, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=False,
    )

    # 6) Colorbar + layout
    fig.colorbar(im, ax=ax, label=r"$\Delta F/F$")
    fig.tight_layout()

    return fig, ax, neuron_order

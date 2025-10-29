import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
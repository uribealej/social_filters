import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import mode

def get_radius_timing(trajectory_file, framerate=60):
    from pathlib import Path
    import pandas as pd
    import numpy as np

    df = pd.read_csv(Path(trajectory_file))
    n_frames = len(df)
    if n_frames <= 0:
        raise ValueError("Empty trajectory file.")

    radius_cols = [c for c in df.columns if c.endswith("_radius")]
    if not radius_cols:
        raise ValueError("No '*_radius' columns found.")

    starts, ends = [], []
    for c in radius_cols:
        s = df[c].to_numpy()

        # first index where value differs from the initial value
        diff_from_init = (s != s[0])
        start = int(np.argmax(diff_from_init)) if diff_from_init.any() else None

        # last index where value differs from the final value
        diff_from_final = (s != s[-1])
        if diff_from_final.any():
            end = int(len(s) - 1 - np.argmax(diff_from_final[::-1]))
        else:
            end = None

        if start is not None and end is not None:
            starts.append(start)
            ends.append(end)

    if not starts or not ends:
        raise ValueError("No dot shows detectable radius change.")

    # Clamp and ensure end >= start
    motion_start = max(0, min(min(starts), n_frames - 1))
    motion_end   = max(motion_start, min(max(ends),   n_frames - 1))

    # Frame counts
    static_before_frames = motion_start
    motion_frames        = motion_end - motion_start + 1
    static_after_frames  = n_frames - motion_end - 1

    # Convert to seconds (no premature rounding)
    to_sec = lambda f: f / float(framerate)

    return {
        "static_before_sec": to_sec(static_before_frames),
        "motion_sec":        to_sec(motion_frames),
        "static_after_sec":  to_sec(static_after_frames),
        # ✅ fields your downstream code expects:
        "total_sec":         to_sec(n_frames),
        "total_frames":      int(n_frames),
        # extra, consistent with your other function:
        "motion_start_frame": int(motion_start),
        "motion_end_frame":   int(motion_end),
        "start_frame":        int(motion_start),
        "end_frame":          int(motion_end),
    }

#
# def get_radius_timing(trajectory_file, framerate=60):
#     from pathlib import Path
#     import pandas as pd
#     import numpy as np
#
#     trajectory_file = Path(trajectory_file)
#     df = pd.read_csv(trajectory_file)
#     n_frames = len(df)
#
#     radius_cols = [col for col in df.columns if col.endswith('_radius')]
#     dot_ids = [col.rsplit('_', 1)[0] for col in radius_cols]
#
#     start_frames = []
#     end_frames = []
#
#     for dot in dot_ids:
#         r_col = f"{dot}_radius"
#         radius_series = df[r_col]
#
#         # List all unique values to check if any change happened
#         unique_values = radius_series.unique()
#         #print(f"{dot} unique radius values: {unique_values}")
#
#         if len(unique_values) > 1:
#             # Detect frames where radius value changes
#             change_indices = radius_series[radius_series != radius_series.iloc[0]].index
#             start_frame = change_indices.min()
#             end_frame = change_indices.max()
#             print(f"  Change detected: Start at {start_frame}, End at {end_frame}")
#             start_frames.append(start_frame)
#             end_frames.append(end_frame)
#         else:
#             print(f"{dot}: No detectable change")
#
#     if not start_frames or not end_frames:
#         raise ValueError("No dot shows detectable radius change.")
#
#     start_frame = min(start_frames)
#     end_frame = max(end_frames)
#
#     return {
#         'static_before_sec': np.round(start_frame / framerate, 0),
#         'motion_sec': np.round((end_frame - start_frame + 1) / framerate, 0),
#         'static_after_sec': np.round((n_frames - end_frame - 1) / framerate, 0),
#         'start_frame': int(start_frame),
#         'end_frame': int(end_frame),
#     }


def get_stimulus_timing(trajectory_file, framerate=60):
    """
    Analyze stimulus trajectory file and extract movement timing.

    This function assumes:
    - CSV with columns '{dot_id}_x', '{dot_id}_y' for each dot
    - Movement starts when any (x, y) differs from initial value
    - Movement ends when (x, y) reach final values

    The start of movement is taken as the earliest frame at which
    any dot starts moving. The end of movement is taken as the
    latest frame at which any dot stops moving.

    Parameters:
    - trajectory_file (Path or str): CSV file with trajectory data
    - framerate (float): Frames per second

    Returns:
    - dict with:
        'static_before_sec': duration before movement (sec)
        'motion_sec': duration of movement (sec)
        'static_after_sec': duration after movement (sec)
        'start_frame': frame index of first movement onset
        'end_frame': frame index of last movement offset
    """
    trajectory_file = Path(trajectory_file)
    df = pd.read_csv(trajectory_file)
    n_frames = len(df)

    # Automatically find all dot IDs
    dot_cols = [col for col in df.columns if col.endswith(('_x', '_y'))]
    dot_ids = set(col.rsplit('_', 1)[0] for col in dot_cols)

    start_frames = []
    end_frames = []

    for dot in dot_ids:
        x_col = f"{dot}_x"
        y_col = f"{dot}_y"

        initial_x, initial_y = df.loc[0, [x_col, y_col]]
        final_x, final_y = df.loc[n_frames - 1, [x_col, y_col]]

        # Identify when the dot starts moving
        moving_start = (df[x_col] != initial_x) | (df[y_col] != initial_y)
        start_frame = moving_start.idxmax() if moving_start.any() else None

        # Identify when the dot stops moving
        moving_end = (df[x_col] != final_x) | (df[y_col] != final_y)
        end_frame = moving_end[::-1].idxmax() if moving_end.any() else None

        if start_frame is not None and end_frame is not None:
            start_frames.append(start_frame)
            end_frames.append(end_frame)

    if not start_frames or not end_frames:
        raise ValueError("No dot shows detectable movement.")

    # Movement starts at the earliest movement onset across all dots
    # Movement ends at the latest movement offset
    start_frame = min(start_frames)
    end_frame = max(end_frames)

    return {
        'static_before_sec': start_frame / framerate,
        'motion_sec': (end_frame - start_frame + 1) / framerate,
        'static_after_sec': (n_frames - end_frame - 1) / framerate,
        'total_sec': round(n_frames / framerate, 2),
        'motion_start_frame': int(start_frame),
        'motion_end_frame': int(end_frame),
        'total_frames': int(n_frames)
    }

def make_stimulus_traces(log_file, stim_durations, selected_blocks, duration_2p_block_sec, fps_stim=60,
                             traces=('movement', 'appearance')):
    """
    Create stimulus traces using fixed 2p time alignment across selected blocks.

    Parameters:
    - log_file: Path to experiment log CSV.
    - stim_durations: Dictionary with stimulus names and durations.
    - selected_blocks: List of blocks to include (e.g., ['B1', 'B2']).
    - duration_2p_block_sec: Duration of each block in 2p time.
    - fps_stim: Stimulus framerate (Hz).
    - traces: Types of traces to build (e.g., 'movement', 'appearance').

    Returns:
    - result: Dictionary with traces (numpy arrays).
    - adjusted_log: Adjusted log with shifted timestamps.
    """

    log = pd.read_csv(log_file)
    adjusted_log = []

    # Adjust timestamps for selected blocks
    for i, block in enumerate(selected_blocks):
        shift = i * duration_2p_block_sec
        block_mask = log['event'].str.startswith(block)
        block_df = log[block_mask].copy()
        block_df['timestamp'] += shift
        adjusted_log.append(block_df)

    adjusted_log = pd.concat(adjusted_log).sort_values('timestamp').reset_index(drop=True)

    # Create empty trace arrays
    max_time = adjusted_log['timestamp'].max()
    trace_length = int(max_time * fps_stim) + 1
    result = {trace: np.zeros(trace_length) for trace in traces}

    stim_events = adjusted_log[adjusted_log['event'].str.contains('stim', case=False, na=False)].copy()

    for _, row in stim_events.iterrows():
        stim_name = row['event'].split('_')[-1]
        stim_start = row['timestamp']

        if stim_name in stim_durations:
            dur = stim_durations[stim_name]
            static_start1 = stim_start
            static_end1 = stim_start + dur['static_before_sec']
            move_start = static_end1
            move_end = static_end1 + dur['motion_sec']
            static_start2 = move_end
            static_end2 = static_start2 + dur['static_after_sec']

            if 'movement' in traces:
                result['movement'][int(move_start * fps_stim):int(move_end * fps_stim)] = 1

            if 'appearance' in traces:
                result['appearance'][int(static_start1 * fps_stim):int(static_end1 * fps_stim)] = 1
                result['appearance'][int(static_start2 * fps_stim):int(static_end2 * fps_stim)] = 1

    return result, adjusted_log

def make_stimulus_traces_2(log_file, stimuli_durations, selected_blocks, duration_2p_block_sec, fps_stim=60):
    """
    Create a stimulus table and a single numeric stimulus trace using fixed 2p time alignment.

    Parameters:
    - log_file: Path to experiment log CSV.
    - stimuli_durations: Dict of stimulus names -> durations info.
    - selected_blocks: List of blocks to include (e.g., ['B1', 'B2']).
    - duration_2p_block_sec: Duration of each block in 2p time (seconds).
    - fps_stim: Stimulus frame rate (Hz).

    Returns:
    - adjusted_log_dataframe: Adjusted log with shifted timestamps.
    - stimulus_trace: 1D np.ndarray (int16). 0 = no stimulus; 1..K = stimulus type IDs.
    - stimulus_table: DataFrame (one row per trial).
    - stimulus_name_to_id: Dict[str, int] mapping stimulus_name -> stimulus_type_id.
    """

    log = pd.read_csv(log_file)
    adjusted_log = []

    # Adjust timestamps for selected blocks
    for i, block in enumerate(selected_blocks):
        shift = i * duration_2p_block_sec
        block_mask = log['event'].str.startswith(block)
        block_df = log[block_mask].copy()
        block_df['timestamp'] += shift
        adjusted_log.append(block_df)

    adjusted_log = pd.concat(adjusted_log).sort_values('timestamp').reset_index(drop=True)

    # Create empty trace arrays
    max_time = adjusted_log['timestamp'].max()
    trace_length_frames = int(round(max_time * fps_stim, 0))
    stimuli_trace = np.zeros(trace_length_frames, dtype=np.int16)

    # stim_events = adjusted_log[adjusted_log['event'].str.contains('stim', case=False, na=False)].copy()
    # for _, row in stim_events.iterrows():
    #     stim_name = row['event'].split('_')[-1]
    stimuli_events = adjusted_log[adjusted_log['event'].str.contains('stim', case=False, na=False)].copy()
    stimuli_id_map = {name: i + 1 for i, name in enumerate(stimuli_durations.keys())}

    rows = []

    for _, row in stimuli_events.iterrows():
        block, stim_idx, stim_name = row['event'].split('_', 2)
        if stim_name not in stimuli_durations:
            continue

        onset_time = row["timestamp"]
        total_time = stimuli_durations[stim_name]["total_sec"]

        onset_frame = int(round(row['timestamp'] * fps_stim, 0))
        total_frames = stimuli_durations[stim_name]["total_frames"]

        stimuli_trace[onset_frame : onset_frame + total_frames] = stimuli_id_map[stim_name]

        row_dict = {
            "block": int(block.split('B')[-1]),
            "trial": int(stim_idx.split('stim')[1]),       # repetition number within block
            "stimulus_name": stim_name,
            "stimulus_id": stimuli_id_map[stim_name],
            "onset_time": onset_time,
            "offset_time": onset_time + total_time,
            "total_time": total_time,
            "onset_frame": onset_frame,
            "offset_frame": onset_frame + total_frames,
            "total_frames": total_frames}

        rows.append(row_dict)

    stimuli_table = pd.DataFrame(rows)

    # Sort only by columns that actually exist
    sort_cols = [c for c in ["block", "trial"] if c in stimuli_table.columns]

    if sort_cols:
        stimuli_table = stimuli_table.sort_values(sort_cols).reset_index(drop=True)

    return adjusted_log, stimuli_trace, stimuli_table, stimuli_id_map

    # stimuli_table = pd.DataFrame(rows).sort_values(["block", "trial"]).reset_index(drop=True)
    #
    # return adjusted_log, stimuli_trace, stimuli_table, stimuli_id_map


def extract_stimulus_chunks(
    deltaF_F,
    exp_log,
    stimuli_durations,
    stimuli_colors,
    fps,
    stimuli_ordered,
    window_pre=5,
    window_post=2,
    average_across_repeats=False
):
    """
    Extract ΔF/F chunks aligned to stimulus events, optionally averaging repetitions.

    Parameters:
    - deltaF_F: (frames x neurons) ΔF/F matrix
    - exp_log: DataFrame with stimulus events and timestamps
    - stimuli_durations: dict with durations, e.g., {'forward': {...}}
    - stimuli_colors: dict mapping stimulus names to colors
    - fps: sampling rate (frames per second)
    - stimuli_ordered: list of stimulus names to process (in order)
    - window_pre: seconds before stimulus
    - window_post: seconds after stimulus
    - average_across_repeats: bool, if True, averages across repetitions per stimulus

    Returns:
    - chunked_data: (neurons x time) concatenated raster
    - trial_start_positions: list of indices for trial starts (for plotting)
    - movement_start_positions: list of indices for movement starts
    - movement_colors: list of colors for movement starts
    - stim_labels: list of stimulus names (ordered as in raster)
    """
    stim_event_names = np.array([event.split('_')[-1] for event in exp_log['event']])

    stim_chunks = []
    trial_start_positions = []
    movement_start_positions = []
    movement_colors = []
    stim_labels = []

    current_pos = 0

    for stim_name in stimuli_ordered:
        stim_mask = stim_event_names == stim_name
        stim_events = exp_log.loc[stim_mask]

        if stim_events.empty or stim_name not in stimuli_durations:
            continue

        dur = stimuli_durations[stim_name]
        expected_chunk_frames = int(
            (window_pre + dur['static_before_sec'] + dur['motion_sec'] + dur['static_after_sec'] + window_post) * fps)
        move_start_in_chunk = int((window_pre + dur['static_before_sec']) * fps)

        stim_repeats = []

        for _, row in stim_events.iterrows():
            stim_start = row['timestamp']
            stim_end = stim_start + dur['static_before_sec'] + dur['motion_sec'] + dur['static_after_sec']

            frame_start = int((stim_start - window_pre) * fps)
            frame_end = int((stim_end + window_post) * fps)

            if frame_start < 0 or frame_end > deltaF_F.shape[0]:
                continue

            chunk_data = deltaF_F[frame_start:frame_end, :].T  # (neurons x time)

            # Ensure shape (neurons x expected_chunk_frames)
            if chunk_data.shape[1] < expected_chunk_frames:
                pad_width = expected_chunk_frames - chunk_data.shape[1]
                chunk_data = np.pad(chunk_data, ((0, 0), (0, pad_width)), constant_values=np.nan)
            elif chunk_data.shape[1] > expected_chunk_frames:
                chunk_data = chunk_data[:, :expected_chunk_frames]

            stim_repeats.append(chunk_data)

        if average_across_repeats:
            stim_mean = np.mean(np.stack(stim_repeats), axis=0)
            stim_chunks.append(stim_mean)

            trial_start_positions.append(current_pos)
            movement_start_positions.append(current_pos + move_start_in_chunk)
            movement_colors.append(stimuli_colors.get(stim_name, 'black'))
            stim_labels.append(stim_name)

            current_pos += stim_mean.shape[1]
        else:
            for chunk in stim_repeats:
                stim_chunks.append(chunk)

                trial_start_positions.append(current_pos)
                movement_start_positions.append(current_pos + move_start_in_chunk)
                movement_colors.append(stimuli_colors.get(stim_name, 'black'))
                stim_labels.append(stim_name)

                current_pos += chunk.shape[1]

    chunked_data = np.hstack(stim_chunks) if stim_chunks else None

    return chunked_data, trial_start_positions, movement_start_positions, movement_colors, stim_labels
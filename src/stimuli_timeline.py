"""Derive stimulus timing and aligned traces from trajectories and logs."""

import pandas as pd
import numpy as np
from pathlib import Path


def get_angles_from_positions(df_x, df_y, rotation_angle):
    """
    Reverse the applied stimulus rotation and compute angular position in degrees.

    This helper is shared by stimulus inspection code that needs to interpret
    trajectory x/y coordinates consistently with the generator-side convention.
    """
    theta = np.deg2rad(rotation_angle)
    x_original = df_x * np.cos(theta) + df_y * np.sin(theta)
    y_original = -df_x * np.sin(theta) + df_y * np.cos(theta)
    angles_rad = np.arctan2(x_original, y_original)
    return np.rad2deg(angles_rad)


def get_motion_timing_simple(
    trajectory_file,
    framerate=60,
    include_xy=True,
    include_radius=True,
):
    """
    Simple timing extraction:
    - start_frame = earliest frame where any selected column differs from its initial value
    - end_frame   = last frame of the file (n_frames - 1)

    It can look at:
      - x/y columns:   *_x, *_y  (include_xy=True)
      - radius columns: *_radius (include_radius=True)

    Returns a dict with fields compatible with your downstream code.
    """
    trajectory_file = Path(trajectory_file)
    df = pd.read_csv(trajectory_file)
    n_frames = len(df)
    if n_frames <= 0:
        raise ValueError("Empty trajectory file.")

    cols = []
    if include_xy:
        cols += [c for c in df.columns if c.endswith(("_x", "_y"))]
    if include_radius:
        cols += [c for c in df.columns if c.endswith("_radius")]

    # Remove duplicates while preserving order
    cols = list(dict.fromkeys(cols))

    if not cols:
        raise ValueError("No relevant columns found (no *_x/*_y and/or *_radius).")

    start_candidates = []
    for c in cols:
        s = df[c].to_numpy()

        # skip all-NaN columns safely
        if np.all(pd.isna(s)):
            continue

        # first index where value differs from initial value
        diff_from_init = (s != s[0])
        if np.any(diff_from_init):
            start_candidates.append(int(np.argmax(diff_from_init)))

    if not start_candidates:
        # no movement detected anywhere => motion "starts" at frame 0
        start_frame = 0
    else:
        start_frame = min(start_candidates)

    end_frame = n_frames - 1  # per your assumption

    static_before_frames = start_frame
    motion_frames = end_frame - start_frame + 1  # inclusive
    static_after_frames = 0  # because motion ends at last frame

    to_sec = lambda f: f / float(framerate)

    return {
        "static_before_sec": to_sec(static_before_frames),
        "motion_sec": to_sec(motion_frames),
        "static_after_sec": to_sec(static_after_frames),

        "total_sec": to_sec(n_frames),
        "total_frames": int(n_frames),

        "motion_start_frame": int(start_frame),
        "motion_end_frame": int(end_frame),

        # aliases (handy if other code expects these names)
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
    }


def transform_stimuli_duration(stimuli_durations):
    """
    Normalize per-stimulus timing dictionaries for downstream consumers.

    This preserves the repo's current downstream-facing timing contract while
    keeping the normalization logic at the shared timing owner layer.
    """
    out = {}
    for k, v in stimuli_durations.items():
        total_sec = v.get("total_sec", 0)
        static_before = v.get("static_before_sec", 0)
        total_frames = v.get("total_frames", v.get("motion_end_frame"))

        new_v = v.copy()
        new_v["motion_sec"] = round(total_sec - static_before, 3)
        new_v["static_after_sec"] = 0

        if total_frames is not None:
            new_v["motion_end_frame"] = total_frames
            new_v["end_frame"] = total_frames

        out[k] = new_v
    return out


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


def summarize_durations(stimuli_durations):
    '''''
    Given a dict of stimuli durations, summarize common durations.
    If all stimuli have the same value for a field (within tolerance), keep that value.
    Otherwise, compute the mean across stimuli.
    Returns a dict with summarized durations.
    '''''
    fields = ["static_before_sec", "motion_sec", "total_sec"]
    summary = {}

    for field in fields:
        # collect values for this field across all stimuli
        vals = [d[field] for d in stimuli_durations.values() if field in d]

        if not vals:
            continue

        # if all equal (within floating point tolerance), keep that value
        if np.allclose(vals, vals[0]):
            summary[field] = vals[0]
        else:
            # otherwise use the mean
            summary[field] = float(np.mean(vals))

    return summary


def compute_move_lines_for_flat_matrix(
    trial_aligned_traces,
    stim_order,
    stimuli_id_map,
    stimuli_durations,
    stimuli_colors,
    fps_2p,
    t_pre_s,
    combine_mode="concat",
):
    """
    trial_aligned_traces[stim_id] -> (n_neurons, n_time, n_reps)

    Returns
    -------
    move_starts_s : list of floats
        X positions in SECONDS where motion starts (for ax.axvline).
    move_colors   : list of colors
    stim_labels   : list of stimulus names (e.g. 'FL1', 'FR2', ...)
    """
    # invert map: id -> name
    id_to_name = {v: k for k, v in stimuli_id_map.items()}

    move_starts_s = []
    move_colors   = []
    stim_labels   = []

    frame_offset = 0  # global frame index along the flattened time axis

    for stim_id in stim_order:
        stim_name = id_to_name[stim_id]

        # data for this stimulus
        arr = trial_aligned_traces[stim_id]       # (neurons, time, reps)
        _, n_time, n_reps = arr.shape

        # movement onset INSIDE ONE TRIAL (in frames)
        static_before_sec = stimuli_durations[stim_name]["static_before_sec"]
        move_start_time_s = t_pre_s + static_before_sec
        move_start_in_trial = int(round(move_start_time_s * fps_2p))
        move_start_in_trial = np.clip(move_start_in_trial, 0, n_time - 1)

        if combine_mode == "concat":
            # one vertical line per repetition
            for rep in range(n_reps):
                global_frame = frame_offset + rep * n_time + move_start_in_trial
                move_starts_s.append(global_frame )
                move_colors.append(stimuli_colors[stim_name])
                stim_labels.append(stim_name)

            # this block length in frames
            frame_offset += n_time * n_reps

        elif combine_mode == "mean":
            # we averaged across reps, so only ONE trace per stimulus
            global_frame = frame_offset + move_start_in_trial
            move_starts_s.append(global_frame)
            move_colors.append(stimuli_colors[stim_name])
            stim_labels.append(stim_name)

            frame_offset += n_time

        else:
            raise ValueError("combine_mode must be 'concat' or 'mean'")

    return move_starts_s, move_colors, stim_labels

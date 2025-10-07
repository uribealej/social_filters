import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

def generate_circular_rocking_trajectory(
    radius,
    angle_range,             # used for normal mode & angles_deg; pass left range for LR rocking
    speed,
    framerate,
    static_period_sec,
    update_interval_ms,
    dot_size_on,
    flicker_interval_sec=0.3,
    rotation_angle=45,
    flickering=False,
    continuous=False,
    rocking=False,
    rocking_idx_pair=None,   # (i,j) 1-based
    # --- cross left-right rocking (A from left arc, B from right arc) ---
    rocking_lr=False,
    left_angle_range=[-30, -163],
    right_angle_range=[30, 163],
    rocking_lr_indices=None, # (i_left, i_right) 1-based
    ):
    """
    Returns:
        df, angles_deg, total_time
    """

    # ---------- helpers ----------
    def _sample_rotated_xy(range_deg, index_1b):
        # build the *sampling grid* for this arc (same step_size as main)
        start_rad, end_rad = np.deg2rad(range_deg)
        omega = speed / radius
        arc_len = radius * abs(end_rad - start_rad)
        T = round(arc_len / speed, 4)

        step = (1 / framerate) if continuous else (update_interval_ms / 1000.0)
        times = np.round(np.arange(0, T + step, step), 4)

        sign = np.sign(end_rad - start_rad) if end_rad != start_rad else 1.0
        ang = start_rad + sign * omega * times

        x = np.round(radius * np.sin(ang), 3)
        y = np.round(radius * np.cos(ang), 3)

        th = np.deg2rad(rotation_angle)
        xr = np.round(x * np.cos(th) - y * np.sin(th), 3)
        yr = np.round(x * np.sin(th) + y * np.cos(th), 3)

        idx = max(0, min(len(xr) - 1, int(index_1b) - 1))
        return xr[idx], yr[idx], T, np.rad2deg(ang)

    # ---------- main precompute for normal / same-arc modes ----------
    x0, y0, total_time, angles_deg = _sample_rotated_xy(angle_range, 1)
    static_frames = int(round(static_period_sec * framerate))
    n_frames = int(round(framerate * total_time))
    frames_per_update = max(1, int(round((update_interval_ms / 1000.0) * framerate)))

    dot = {'x': [], 'y': [], 'radius': []}

    # ---------- LR ROCKING (left↔right) ----------
    if rocking_lr:
        assert left_angle_range is not None and right_angle_range is not None, "Provide left_angle_range & right_angle_range"
        assert rocking_lr_indices is not None and len(rocking_lr_indices) == 2, "rocking_lr_indices must be [i_left, i_right]"

        iL, iR = rocking_lr_indices
        xA, yA, T_L, angles_deg_L = _sample_rotated_xy(left_angle_range,  iL)
        xB, yB, T_R, _              = _sample_rotated_xy(right_angle_range, iR)

        # Use the shorter of the two to be safe (they should match here)
        total_time = min(T_L, T_R)
        n_frames = int(round(framerate * total_time))
        angles_deg = angles_deg_L  # report left arc angles

        # static (hold LEFT index)
        if static_frames > 0:
            dot['x'] += [xB] * static_frames # began for the second rocking point
            dot['y'] += [yB] * static_frames
            dot['radius'] += [dot_size_on] * static_frames

        # alternate A (left) ↔ B (right)
        filled = 0
        toggle = False
        while filled < n_frames:
            block = min(frames_per_update, n_frames - filled)
            toggle = not toggle
            if toggle:  # A block (left)
                dot['x'] += [xA] * block
                dot['y'] += [yA] * block
            else:       # B block (right)
                dot['x'] += [xB] * block
                dot['y'] += [yB] * block
            dot['radius'] += [dot_size_on] * block
            filled += block

    # ---------- SAME-ARC ROCKING ----------
    elif rocking and rocking_idx_pair is not None:
        i1, i2 = rocking_idx_pair
        # A and B from the *same* arc (angle_range)
        xA, yA, _, _ = _sample_rotated_xy(angle_range, i1)
        xB, yB, _, _ = _sample_rotated_xy(angle_range, i2)

        if static_frames > 0:
            dot['x'] += [xB] * static_frames # began for the second rocking point
            dot['y'] += [yB] * static_frames
            dot['radius'] += [dot_size_on] * static_frames

        filled = 0
        toggle = False
        while filled < n_frames:
            block = min(frames_per_update, n_frames - filled)
            toggle = not toggle
            if toggle:
                dot['x'] += [xA] * block
                dot['y'] += [yA] * block
            else:
                dot['x'] += [xB] * block
                dot['y'] += [yB] * block
            dot['radius'] += [dot_size_on] * block
            filled += block

    # ---------- NORMAL CIRCULAR MODE ----------
    else:
        # rebuild full samples once (same helper, but keep all samples)
        start_rad, end_rad = np.deg2rad(angle_range)
        omega = speed / radius
        arc_len = radius * abs(end_rad - start_rad)
        total_time = round(arc_len / speed, 4)
        step = (1 / framerate) if continuous else (update_interval_ms / 1000.0)
        times = np.round(np.arange(0, total_time + step, step), 4)
        sign = np.sign(end_rad - start_rad) if end_rad != start_rad else 1.0
        ang = start_rad + sign * omega * times
        angles_deg = np.rad2deg(ang)
        x = np.round(radius * np.sin(ang), 3)
        y = np.round(radius * np.cos(ang), 3)
        th = np.deg2rad(rotation_angle)
        xr = np.round(x * np.cos(th) - y * np.sin(th), 3)
        yr = np.round(x * np.sin(th) + y * np.cos(th), 3)

        static_frames = int(round(static_period_sec * framerate))
        n_frames = int(round(framerate * total_time))

        # static at first sample
        if static_frames > 0:
            dot['x'] += [xr[0]] * static_frames
            dot['y'] += [yr[0]] * static_frames
            dot['radius'] += [dot_size_on] * static_frames

        # motion: follow arc / hold last
        for f in range(n_frames):
            t = round(f / framerate, 4)
            if t in times:
                idx = np.where(times == t)[0][0]
                if idx < len(xr) - 1:
                    xv, yv = xr[idx + 1], yr[idx + 1]
                else:
                    xv, yv = xr[-1], yr[-1]
            else:
                xv, yv = dot['x'][-1], dot['y'][-1]
            dot['x'].append(xv)
            dot['y'].append(yv)
            dot['radius'].append(dot_size_on)

    # flicker (optional)
    if flickering:
        flick_frames = max(1, int(round(flicker_interval_sec * framerate)))
        start = static_frames
        on = True
        for i in range(start, len(dot['radius']), flick_frames):
            end = min(i + flick_frames, len(dot['radius']))
            val = dot_size_on if on else 0.0
            dot['radius'][i:end] = [val] * (end - i)
            on = not on

    df = pd.DataFrame(dot)


    return df, angles_deg, total_time


#
# df, angles_deg, total_time = generate_circular_trajectory(
#         radius=1.8,
#         angle_range=[-30, -163],
#         speed=0.497,
#         framerate=60,
#         static_period_sec=3,
#         update_interval_ms=600,
#         rotation_angle=45,
#         dot_size_on=0.2,
#         flickering=False, flicker_interval_sec=0.3,
#         continuous=False,
#         rocking=True, rocking_idx_pair=(7, 10), rocking_lr=True,
#         rocking_lr_indices=[7, 7],
#         save_path=r"C:\Users\suribear\OneDrive - Université de Lausanne\Lab\LR7.csv")
#
# from pathlib import Path   # <-- add this line
#
# # --- Choose where to save ---
# save_dir = Path(r"C:\Users\suribear\OneDrive - Université de Lausanne\Lab")
# save_dir.mkdir(parents=True, exist_ok=True)
#
# filename = "rocking_stimulus.csv"
# save_path = save_dir / filename
#
# df.to_csv(save_path, index=False)
#
# print(f"✅ Saved trajectory to {save_path}")

import json
import pandas as pd
from datetime import datetime
from pathlib import Path

# If your function is in another module, import it like:
# from your_module import generate_circular_rocking_trajectory
# Here we assume it's already defined/imported in scope.

# -----------------------
# Load JSON config (NEW)
# -----------------------
with open("trajectory_rocking.json", "r") as file:
    stimulus_config = json.load(file)

# -----------------------
# Global parameters
# -----------------------
global_params = {
    "radius_cm": 1.8,
    "speed_cm_sec": 0.497,
    "framerate": 60,
    "update_interval_ms": 600,     # also controls rocking flip timing
    "static_period_sec": 8,
    "dot_size_on": 0.2,
    "rotation_angle_deg": 45,
    # kept for compatibility; not used if you're not doing flicker
    "flicker_interval_ms": 300,
    "flicker_interval_sec": 0.3,
    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

# -----------------------
# Repetitions and pauses
# -----------------------
n_repetitions = 2
pause_before_sec = 12.5
pause_after_sec = 12.5

# -----------------------
# Output directory
# -----------------------
output_path = Path(r"Z:\FAC\FBM\CIG\jlarsch\default\D2c\Alejandro\2p\Exp_2_rocking_1\stimuli")
output_path.mkdir(parents=True, exist_ok=True)

# -----------------------
# Book-keeping
# -----------------------
saved_angles = {}   # store angles_deg per stimulus (useful if you later add flicker types)
saved_times  = {}   # store motion time (sec) per stimulus
experiment_params = []

# -----------------------
# Generate stimuli
# -----------------------
for key, params in stimulus_config.items():
    stim_type = params["type"]
    print(f"Generating {key} ({stim_type})")

    n_dots = params.get("n_dots", 1)

    if stim_type == "trajectory":
        # Use provided angle_range if present; default to left arc for normal/rocking
        angle_range = params.get("angle_ranges", [[-30, -163]])[0]

        dfs = []
        for _ in range(n_dots):
            # Branch by mode
            if params.get("rocking_lr", False):
                # LR rocking; function has defaults for left/right ranges,
                # but you can override via JSON with left_angle_range/right_angle_range.
                df_single, angles_deg, total_time = generate_circular_rocking_trajectory(
                    radius=global_params["radius_cm"],
                    angle_range=angle_range,  # still passed for angles bookkeeping
                    speed=global_params["speed_cm_sec"],
                    framerate=global_params["framerate"],
                    static_period_sec=global_params["static_period_sec"],
                    update_interval_ms=global_params["update_interval_ms"],
                    rotation_angle=global_params["rotation_angle_deg"],
                    dot_size_on=global_params["dot_size_on"],
                    # no flickering now
                    flickering=False,
                    # LR mode:
                    rocking_lr=True,
                    rocking_lr_indices=params["rocking_lr_indices"],
                    # Optional overrides if present in JSON:
                    left_angle_range=params.get("left_angle_range"),
                    right_angle_range=params.get("right_angle_range"),
                )

            elif params.get("rocking", False):
                # Same-arc rocking between two sampled indices
                df_single, angles_deg, total_time = generate_circular_rocking_trajectory(
                    radius=global_params["radius_cm"],
                    angle_range=angle_range,
                    speed=global_params["speed_cm_sec"],
                    framerate=global_params["framerate"],
                    static_period_sec=global_params["static_period_sec"],
                    update_interval_ms=global_params["update_interval_ms"],
                    rotation_angle=global_params["rotation_angle_deg"],
                    dot_size_on=global_params["dot_size_on"],
                    flickering=False,
                    rocking=True,
                    rocking_idx_pair=params["rocking_idx_pair"],
                )

            else:
                # Normal circular trajectory (no rocking)
                df_single, angles_deg, total_time = generate_circular_rocking_trajectory(
                    radius=global_params["radius_cm"],
                    angle_range=angle_range,
                    speed=global_params["speed_cm_sec"],
                    framerate=global_params["framerate"],
                    static_period_sec=global_params["static_period_sec"],
                    update_interval_ms=global_params["update_interval_ms"],
                    rotation_angle=global_params["rotation_angle_deg"],
                    dot_size_on=global_params["dot_size_on"],
                    flickering=False,
                )

            dfs.append(df_single)

        # Combine columns if multiple dots
        df = pd.concat(dfs, axis=1)
        df.columns = [f"dot{i}_{coord}" for i in range(n_dots) for coord in ["x", "y", "radius"]]

        # Save for potential later use
        saved_angles[key] = angles_deg
        saved_times[key]  = float(total_time)  # motion only (no flicker added)

    else:
        raise ValueError(f"Unknown stimulus type: {stim_type}")

    # Save each stimulus .csv
    df.to_csv(output_path / f"{key}_trajectory.csv", index=False)

    # Save parameters used (per stimulus)
    combined_params = {
        "stimulus_key": key,
        **params,
        **global_params,
        "total_motion_time_sec": float(total_time),
        "pause_before_sec": pause_before_sec,
        "pause_after_sec": pause_after_sec,
    }
    experiment_params.append(combined_params)

# -----------------------
# Duration computation
# -----------------------
def calculate_experiment_duration(stimulus_config, saved_times,
                                  global_params, n_repetitions,
                                  pause_before_sec, pause_after_sec):
    """
    Sum per-stimulus duration within one repetition, then multiply by repetitions.
    Per-stimulus duration = static_period + motion_time + pauses.
    """
    static_period = float(global_params["static_period_sec"])
    total_time_per_rep = 0.0

    for key, params in stimulus_config.items():
        if params["type"] != "trajectory":
            # You can extend here for other types later if needed
            continue

        motion = float(saved_times.get(key, 0.0))
        duration = static_period + motion + float(pause_before_sec) + float(pause_after_sec)
        print(f"{key}: {duration:.3f} s (static {static_period} + motion {motion:.3f} + pauses {pause_before_sec}+{pause_after_sec})")
        total_time_per_rep += duration

    return total_time_per_rep * int(n_repetitions)

# Calculate total experiment time
total_time_sec = calculate_experiment_duration(
    stimulus_config,
    saved_times,
    global_params,
    n_repetitions,
    pause_before_sec,
    pause_after_sec
)

# -----------------------
# Save metadata
# -----------------------
(output_path / "parameters").mkdir(parents=True, exist_ok=True)
pd.DataFrame(experiment_params).to_csv(output_path / "parameters" / "experiment_parameters.csv", index=False)
pd.DataFrame([{"total_experiment_duration_sec": total_time_sec}]).to_csv(
    output_path / "parameters" / "total_time_sec.csv", index=False
)

print(f"\nTotal experiment time: {total_time_sec:.2f} sec ({total_time_sec/60:.2f} min)")
print("✅ All stimuli generated and saved.")

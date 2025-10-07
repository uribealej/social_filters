import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

def generate_circular_trajectory(radius, angle_range, speed, framerate,
                                 static_period_sec, update_interval_ms, dot_size_on,
                                 flickering=False, flicker_interval_sec=0.3,
                                 continuous=False, rotation_angle=45
):

    start_angle, end_angle = np.deg2rad(angle_range)
    omega = speed / radius
    arc_length = radius * abs(end_angle - start_angle)
    total_time = round(arc_length / speed, 4)
    print('total_time_please', total_time)
    step_size = 1 / framerate if continuous else update_interval_ms / 1000
    frame_times = np.round(np.arange(0, total_time+step_size, step_size), 4)

    angles = start_angle + np.sign(end_angle - start_angle) * omega * frame_times
    angles_deg = np.rad2deg(angles)
    x_values = np.round(radius * np.sin(angles), 3)
    y_values = np.round(radius * np.cos(angles), 3)

    theta = np.deg2rad(rotation_angle)
    x_rotated = np.round(x_values * np.cos(theta) - y_values * np.sin(theta), 3)
    y_rotated = np.round(x_values * np.sin(theta) + y_values * np.cos(theta), 3)

    static_frames = int(static_period_sec * framerate)
    n_frames = int(framerate * total_time)
    dot_dict = {'x': [], 'y': [], 'radius': []}

    dot_dict['x'] += [x_rotated[0]] * static_frames
    dot_dict['y'] += [y_rotated[0]] * static_frames
    dot_dict['radius'] += [dot_size_on] * static_frames

    # Add motion frames
    for f in range(n_frames):
        timepoint = round(f / framerate, 4)
        if timepoint in frame_times:
            idx = np.where(frame_times == timepoint)[0][0]
            x_val = x_rotated[1:][idx]
            y_val = y_rotated[1:][idx]
        else:
            x_val = dot_dict['x'][-1]
            y_val = dot_dict['y'][-1]

        dot_dict['x'].append(x_val)
        dot_dict['y'].append(y_val)
        dot_dict['radius'].append(dot_size_on)  # temp placeholder, will be modified below if flickering
    # for f in range(n_frames+1):
    #     timepoint = round(f / framerate, 4)
    #     if timepoint in frame_times:
    #         idx = np.where(frame_times == timepoint)[0][0]
    #         dot_dict['x'].append(x_rotated[1:][idx])
    #         dot_dict['y'].append(y_rotated[1:][idx])
    #     else:
    #         dot_dict['x'].append(dot_dict['x'][-1])
    #         dot_dict['y'].append(dot_dict['y'][-1])

    # dot_dict['x'] += [x_rotated[-1]] * static_frames
    # dot_dict['y'] += [y_rotated[-1]] * static_frames
    # dot_dict['radius'] = global_params["dot_size_on"]
    # Apply flicker only to movement frames (after static period)
    if flickering:
        
        flicker_interval_frames = int(flicker_interval_sec * framerate)
        start_flicker_idx = static_frames  # flicker begins after static period
        toggle = True
        for i in range(start_flicker_idx, len(dot_dict['radius']), flicker_interval_frames):
            end = min(i + flicker_interval_frames, len(dot_dict['radius']))
            value = dot_size_on if toggle else 0.0
            dot_dict['radius'][i:end] = [value] * (end - i)
            toggle = not toggle

    df = pd.DataFrame(dot_dict)
    return df, angles_deg, total_time


def generate_flickering_dot(radius_cm, angle_deg, framerate, static_period_sec, flicker_duration_sec, flicker_interval_ms, dot_size_on, rotation_angle_deg=45):
    angle_rad = np.deg2rad(angle_deg)
    theta = np.deg2rad(rotation_angle_deg)

    # Base position before rotation
    x = radius_cm * np.sin(angle_rad)
    y = radius_cm * np.cos(angle_rad)

    # Rotate
    x_rot = x * np.cos(theta) - y * np.sin(theta)
    y_rot = x * np.sin(theta) + y * np.cos(theta)

    static_frames = int(static_period_sec * framerate)
    flicker_frames = int(flicker_duration_sec * framerate)
    flicker_interval_frames = int((flicker_interval_ms / 1000) * framerate)

    x_list, y_list, size_list = [], [], []

    # Static ON
    for _ in range(static_frames):
        x_list.append(x_rot)
        y_list.append(y_rot)
        size_list.append(dot_size_on)

    # Flickering ON/OFF
    for i in range(flicker_frames):
        x_list.append(x_rot)
        y_list.append(y_rot)
        if (i // flicker_interval_frames) % 2 == 0:
            size_list.append(dot_size_on)
        else:
            size_list.append(0.0)

    df = pd.DataFrame({'x': x_list, 'y': y_list, 'size': size_list})

    return df


# Load JSON config
with open("package_ale.json", "r") as file:
    stimulus_config = json.load(file)


# Global parameters
global_params = {
    "radius_cm": 1.8,
    "speed_cm_sec": 0.497,
    "framerate": 60,
    "update_interval_ms": 600,
    "static_period_sec": 8,
    "flicker_interval_ms": 300,
    "dot_size_on": 0.2,
    "rotation_angle_deg": 45,
    "flicker_interval_sec":0.3,
    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

# Define repetitions and pause times
n_repetitions = 2
pause_before_sec = 12.5
pause_after_sec = 12.5
# Output directory

output_path = Path(r"Z:\FAC\FBM\CIG\jlarsch\default\D2c\Alejandro\2p\flickering\stimuli")
output_path.mkdir(parents=True, exist_ok=True)

# Save trajectory angles for reuse
saved_angles = {}
saved_times = {}
experiment_params = []

for key, params in stimulus_config.items():
    stim_type = params["type"]
    print(f"Generating {key} ({stim_type})")

    n_dots = params.get("n_dots", 1)

    if stim_type == "trajectory":
        angle_range = params["angle_ranges"][0]

        dfs = []
        for _ in range(n_dots):
            df, angles_deg, total_time= generate_circular_trajectory(
                radius=global_params["radius_cm"],
                angle_range=angle_range,
                speed=global_params["speed_cm_sec"],
                framerate=global_params["framerate"],
                static_period_sec=global_params["static_period_sec"],
                update_interval_ms=global_params["update_interval_ms"],
                rotation_angle=global_params["rotation_angle_deg"],
                dot_size_on = global_params["dot_size_on"],
                flickering = params.get("flickering", False),
                flicker_interval_sec = global_params["flicker_interval_sec"]
            )
            dfs.append(df)

        combined_df = pd.concat(dfs, axis=1)
        combined_df.columns = [f"dot{i}_{coord}" for i in range(n_dots) for coord in ['x', 'y', 'radius']]
        df = combined_df

        saved_angles[key] = angles_deg  # Save for flickering stimuli
        saved_times[key] = total_time+(global_params["flicker_interval_ms"]/1000)

    elif stim_type == "flicker":
        based_on = params["based_on"]
        angle_idx = params["angle_index"]
        angle_deg = saved_angles[based_on][angle_idx]
        flicker_duration_sec = saved_times[based_on]


        dfs = []
        for _ in range(n_dots):
            df_single = generate_flickering_dot(
                radius_cm=global_params["radius_cm"],
                angle_deg=angle_deg,
                framerate=global_params["framerate"],
                static_period_sec=global_params["static_period_sec"]-(global_params["flicker_interval_ms"]/1000),
                flicker_duration_sec=flicker_duration_sec,
                flicker_interval_ms=global_params["flicker_interval_ms"],
                dot_size_on=global_params["dot_size_on"],
                rotation_angle_deg=global_params["rotation_angle_deg"]
            )
            dfs.append(df_single)

        combined_df = pd.concat(dfs, axis=1)
        combined_df.columns = [f"dot{i}_{coord}" for i in range(n_dots) for coord in ['x', 'y', 'radius']]
        df = combined_df


    # Save each stimulus .csv
    df.to_csv(output_path / f"{key}_trajectory.csv", index=False)

    # Save parameters
    combined_params = {
        "stimulus_key": key,
        **params,
        **global_params,
    }
    experiment_params.append(combined_params)

def calculate_experiment_duration_by_type(stimulus_config, saved_times, total_time, global_params,
                                          n_repetitions, pause_before_sec, pause_after_sec):

    static_period = global_params["static_period_sec"]
    flicker_interval_s= global_params["flicker_interval_ms"] / 1000  # Convert ms to sec
    total_time_per_repetition = 0

    for key, params in stimulus_config.items():
        stim_type = params["type"]
        if stim_type == "trajectory":
            duration = static_period + total_time + pause_before_sec + pause_after_sec
        elif stim_type == "flicker":
            flicker_duration = saved_times.get(based_on, 0)
            duration = static_period-flicker_interval_s+ flicker_duration + pause_before_sec + pause_after_sec
        else:
            raise ValueError(f"Unknown stimulus type: {stim_type}")
        print(duration)
        total_time_per_repetition += duration

    total_duration = (total_time_per_repetition * n_repetitions)
    return total_duration

# Calculate total time
total_time_sec = calculate_experiment_duration_by_type(
    stimulus_config,
    saved_times,
    total_time,
    global_params,
    n_repetitions,
    pause_before_sec,
    pause_after_sec
)

# Ensure the 'parameters' subdirectory exists
(output_path / "parameters").mkdir(parents=True, exist_ok=True)

# Save metadata
pd.DataFrame(experiment_params).to_csv(output_path /'parameters'/ "experiment_parameters.csv", index=False)
pd.DataFrame([{"total_experiment_duration_sec": total_time_sec}]).to_csv(
    output_path / 'parameters' / "total_time_sec.csv", index=False
)


print(f"Total experiment time: {total_time_sec:.2f} sec ({total_time_sec/60:.2f} min)")
print("✅ All stimuli generated and saved.")
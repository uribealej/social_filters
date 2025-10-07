import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from ipywidgets import interact

def generate_circular_trajectory(radius, angle_range, speed, framerate, static_period_sec, update_interval_ms, continuous=False, rotation_angle=45):
    start_angle, end_angle = np.deg2rad(angle_range)
    omega = speed / radius
    arc_length = radius * abs(end_angle - start_angle)
    total_time = round(arc_length / speed, 4)

    print('total_time', total_time)
    step_size = 1 / framerate if continuous else update_interval_ms / 1000
    frame_times = np.round(np.arange(0, total_time+step_size, step_size), 4)

    angles = start_angle + np.sign(end_angle - start_angle) * omega * frame_times #
    x_values = np.round(radius * np.sin(angles), 3)
    y_values = np.round(radius * np.cos(angles), 3)

    # Apply rotation of +45 degrees
    theta = np.deg2rad(rotation_angle)
    x_rotated = np.round(x_values * np.cos(theta) - y_values * np.sin(theta), 3)
    y_rotated = np.round(x_values * np.sin(theta) + y_values * np.cos(theta), 3)

    static_frames = int(static_period_sec * framerate)

    n_frames = int(framerate * total_time)
    dot_dict = {'x': [], 'y': []}

    # Add static period at the beginning
    dot_dict['x'] += [x_rotated[0]] * static_frames
    dot_dict['y'] += [y_rotated[0]] * static_frames

    for f in range(n_frames+1):

        timepoint = round(f / framerate, 4)

        if timepoint in frame_times:
            idx = np.where(frame_times == timepoint)[0][0]
            #print('idx', idx)
            dot_dict['x'].append(x_rotated[1:][idx])
            dot_dict['y'].append(y_rotated[1:][idx])
        else:
            dot_dict['x'].append(dot_dict['x'][-1])
            dot_dict['y'].append(dot_dict['y'][-1])

    # Add static period at the end
    dot_dict['x'] += [x_rotated[-1]] * static_frames
    dot_dict['y'] += [y_rotated[-1]] * static_frames

    if set(x_rotated) != set(dot_dict['x']):
        # raise error if x_values and dot_dict['x'] are not equal
        raise ValueError("x_values and dot_dict['x'] are not equal")
    elif set(y_rotated) != set(dot_dict['y']):
        raise ValueError("y_values and dot_dict['y'] are not equal")

    return pd.DataFrame(dot_dict), total_time


def get_angles_from_positions(df_x, df_y, rotation_angle):

    # Calculate the angles in radians using arctan2
    theta = np.deg2rad(rotation_angle)  # Forward rotation angle in radians
    x_original = df_x * np.cos(theta) + df_y * np.sin(theta)
    y_original = -df_x * np.sin(theta) + df_y * np.cos(theta)
    angles_rad = np.arctan2(x_original, y_original)

    # Convert radians to degrees
    angles_deg = np.rad2deg(angles_rad)

    return angles_deg

# Function to plot the selected trajectory
def plot_trajectory(dict_df, key, framerate, radius=1.8):
    df = dict_df[key]['df']  # Select the DataFrame based on the key
    times = np.arange(len(df)) / framerate  # Generate time points

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(7, 6))

    # Plot full reference circle in light gray
    circle_angles = np.linspace(0, 2 * np.pi, 300)
    full_circle_x = radius * np.sin(circle_angles)
    full_circle_y = radius * np.cos(circle_angles)
    ax.plot(full_circle_x, full_circle_y, color="lightgray", linestyle="--", label=None)

    # Color mapping
    norm = mcolors.Normalize(vmin=min(times), vmax=max(times))
    cmap = cm.get_cmap("viridis")
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)

    # Dynamically plot all dots
    num_dots = len(df.columns)//2  # Count dots based on x-coordinates
    for i in range(num_dots):
        ax.scatter(df[f'dot{i}_x'], df[f'dot{i}_y'], c=times, cmap=cmap, edgecolors=None, zorder=3, label=None)

    # Plot center of the circle as a black dot
    ax.scatter(0, 0, color="black", s=50, label=None)

    # Add arrow pointing toward the 0° (which is at -45° in the rotated system)
    arrow_length = radius * 0.3  # Length of the arrow
    # Corrected arrow annotation
    arrow_x = arrow_length * np.cos(np.deg2rad(135))
    arrow_y = arrow_length * np.sin(np.deg2rad(135))

    ax.annotate('', xy=(arrow_x, arrow_y), xytext=(0, 0),
                arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle='->', lw=2))

    # Formatting
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.axis("equal")  # Ensure circle is not distorted
    ax.set_title(
        f"Dot trajectories {key} - {dict_df[key]['angle_ranges']}° \n"
        f" total time: {len(dict_df[key]['df'])/framerate:.2f} s"
    )

    # Add colorbar
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, shrink=0.7)
    cbar.set_label("Time (seconds)")

    plt.legend()
    plt.show()


def plot_angles_over_time(dict_df, framerate, rotation_angle):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Load the PiYG colormap
    cmap = cm.get_cmap("PiYG")

    # Define custom colors from PiYG colormap
    colors = {
        "LLB": cmap(0.9),  # Darker Pink (Bout)
        "LLC": cmap(0.7),  # Lighter Pink (Continuous)
        "RLB": cmap(0.1),  # Darker Green (Bout)
        "RLC": cmap(0.3)  # Lighter Green (Continuous)
    }

    all_angles = []  # Store all angles for y-axis limits

    for key, data in dict_df.items():
        df = data['df']
        times = np.arange(len(df)) / framerate
        n_dots = data['n_dots']
        continuous_flags = data['continuous']

        angles_deg_adjusted = [
            get_angles_from_positions(df[f'dot{i}_x'], df[f'dot{i}_y'], rotation_angle)
            for i in range(n_dots)
        ]

        for i in range(n_dots):
            movement_type = "LL" if "LL" in key else "RL"
            bout_type = "B" if not continuous_flags[i] else "C"
            color = colors[movement_type + bout_type]

            ax.plot(times, angles_deg_adjusted[i], label=data['name'],
                    color=color, linewidth=2)  # Use dynamic colors

            all_angles.extend(angles_deg_adjusted[i])  # Collect angles for y-axis limits

    # Set dynamic y-axis limits
    if max(all_angles) < 0:
        ax.set_ylim(-180, 0)
    else:
        ax.set_ylim(-180, 180)

    # Set x-axis ticks at every 2.5 seconds
    max_time = max(times)
    ax.set_xticks(np.arange(0, max_time + 2.5, 2.5))

    ax.set_title("Dot Angles Over Time for All Stimuli")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Angle (degrees)")
    ax.legend()
    plt.tight_layout()
    plt.show()

# Load JSON dictionary with parameters
with open("stimuli_rad_position.json", "r") as file:
    trajectory_dict = json.load(file)

# Global experiment parameters
global_params = {
    "radius_cm": 1.8,  # cm,
    "speed_cm_sec": 0.497,  # cm/s,
    "framerate": 60, # FPS
    "update_interval_ms": 600, # Update position every 600 ms
    "static_period_sec": 10, # Static period at the beginning and end of the trajectory
    "rotation_angle_deg": 45, # Rotation angle in degrees
    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

# radius_cm = 1.8  # cm
# speed_cm_sec = 0.497  # cm/s
# framerate = 60  # FPS
# update_interval_ms = 600  # Update position every 600 ms
# static_period_sec = 10 # Static period at the beginning and end of the trajectory
# rotation_angle_deg = 45  # Rotation angle in degrees

path = Path(r"C:\Users\zebrafish\code\neuro-group-jfish\stimuli\LR_thalamus_bout_exp01") #TODO find a better name for the experiment
path = path / global_params['date'][:10]
path.mkdir(exist_ok=True)

dict_df = {}

for key, params in trajectory_dict.items():

    dict_df[key] = {}
    dict_df[key]['angle_ranges'] = params["angle_ranges"]
    dict_df[key]['continuous'] = params["continuous"]
    dict_df[key]['n_dots'] = params["n_dots"]
    dict_df[key]['name'] = params["name"]

    dfs = [generate_circular_trajectory(global_params['radius_cm'], ar, global_params['speed_cm_sec'], global_params['framerate'],
                                        global_params['static_period_sec'], global_params['update_interval_ms'], c, global_params['rotation_angle_deg'])
           for ar, c in zip(dict_df[key]['angle_ranges'], dict_df[key]['continuous'])]

    dfs = pd.concat(dfs, axis=1)
    dfs.columns = [f"dot{i}_{coord}" for i in range(dict_df[key]['n_dots']) for coord in ['x', 'y']]
    dict_df[key]['df'] = dfs
    dict_df[key]['time'] = len(dfs) / global_params['framerate']
    dict_df[key]['n_frames'] = len(dfs)
    dfs.to_csv(path / f"{key}_trajectory.csv", index=False)

experiment_params_list = []
for key, params in trajectory_dict.items():
    stimulus_params = {"stimulus_key": key}
    # Include global parameters
    stimulus_params.update(global_params)
    # Include parameters from the JSON file
    stimulus_params.update(params)
    experiment_params_list.append(stimulus_params)

# Create a DataFrame and save as CSV
df_experiment_params = pd.DataFrame(experiment_params_list)
df_experiment_params.to_csv(path / "experiment_parameters.csv", index=False)


# Interactive selection
# plot_trajectory(dict_df, 'LLB+RLB', global_params['framerate'], global_params['radius_cm'])
# to_plot = ['LLB', 'LLC'] #'RLB', 'RLC'
# to_plot_dict = {key: dict_df[key] for key in to_plot if key in dict_df}
# plot_angles_over_time(to_plot_dict, global_params['framerate'], global_params['rotation_angle_deg'])
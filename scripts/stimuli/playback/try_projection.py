import time
from pathlib import Path

import pandas as pd
import pyglet
from psychopy import core, event, monitors, visual

# === Parameters ===
stimuli_path = Path(
    r"\\nasdcsr.unil.ch\RECHERCHE\FAC\FBM\CIG\jlarsch\default\D2c\Alejandro\2p\Exp_6_mapping_positions_retina"
)
screen_index = 0  # Adjust if needed
physical_width_cm = 59.0
viewing_distance_cm = 20.0
pause_between_stimuli_sec = 1.0
write_timing_log = True


def get_stimulus_files(path):
    return sorted(path.glob("*_trajectory.csv"))


def get_screen_resolution(index):
    display = pyglet.canvas.get_display()
    screens = display.get_screens()
    resolution = (screens[index].width, screens[index].height)
    print(f"Screen {index} resolution: {resolution}")
    return resolution


def create_monitor(resolution):
    monitor = monitors.Monitor(name="auto_monitor")
    monitor.setWidth(physical_width_cm)
    monitor.setDistance(viewing_distance_cm)
    monitor.setSizePix(resolution)
    monitor.save()
    return monitor


def create_dots(win, columns):
    # Display wrapper only: infer drawable dot ids from the CSV columns.
    dot_names = sorted({col.split("_")[0] for col in columns if "_" in col})
    return {
        dot_name: visual.Circle(
            win=win,
            radius=1,
            fillColor="black",
            lineColor="black",
            units="cm",
        )
        for dot_name in dot_names
    }


def play_stimulus_file(win, csv_file):
    print(f"Playing: {csv_file.name}")
    df = pd.read_csv(csv_file)
    dots = create_dots(win, df.columns)

    stim_start = time.time()
    print(stim_start)

    for _, row in df.iterrows():
        for dot_name, dot in dots.items():
            x = row[f"{dot_name}_x"]
            y = row[f"{dot_name}_y"]
            radius = row.get(f"{dot_name}_radius", 0.4)

            if radius > 0:
                dot.pos = (x, y)
                dot.radius = radius
                dot.draw()

        win.flip()

        if "escape" in event.getKeys():
            raise KeyboardInterrupt("Playback interrupted by escape key.")

    stim_end = time.time()
    print(stim_end)

    return {
        "stimulus_file": csv_file.name,
        "start_time_unix": stim_start,
        "end_time_unix": stim_end,
        "duration_sec": round(stim_end - stim_start, 3),
    }


def save_timing_log(path, timing_log):
    timing_df = pd.DataFrame(timing_log)
    timing_df.to_csv(path / "stimulus_timing_log.csv", index=False)
    print("Timing log saved.")


def main():
    stimulus_files = get_stimulus_files(stimuli_path)
    resolution = get_screen_resolution(screen_index)
    monitor = create_monitor(resolution)

    win = visual.Window(
        size=resolution,
        units="cm",
        monitor=monitor,
        fullscr=True,
        screen=screen_index,
        color="red",
    )

    timing_log = []
    try:
        for csv_file in stimulus_files:
            timing_log.append(play_stimulus_file(win, csv_file))
            core.wait(pause_between_stimuli_sec)
    finally:
        win.close()

    if write_timing_log:
        save_timing_log(stimuli_path, timing_log)

    core.quit()


if __name__ == "__main__":
    main()

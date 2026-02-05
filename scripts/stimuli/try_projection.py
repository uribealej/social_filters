import pandas as pd
from pathlib import Path
from psychopy import visual, core, event, monitors
import pyglet
import time
# === Parameters ===
stimuli_path = Path(r"Z:\FAC\FBM\CIG\jlarsch\default\D2c\Alejandro\2p\rocking2\stimuli_45")
stimulus_files = sorted(stimuli_path.glob("*_trajectory.csv"))  # Load all CSVs
screen_index = 0# Adjust if needed
physical_width_cm = 59.0
viewing_distance_cm = 20.0

# === Detect screen resolution automatically ===
display = pyglet.canvas.get_display()
screens = display.get_screens()
resolution = (screens[screen_index].width, screens[screen_index].height)
print(f"Screen {screen_index} resolution: {resolution}")

# === Define and register monitor ===
monitor = monitors.Monitor(name='auto_monitor')
monitor.setWidth(physical_width_cm)
monitor.setDistance(viewing_distance_cm)
monitor.setSizePix(resolution)
monitor.save()

# === Open PsychoPy window ===
win = visual.Window(
    size=resolution,
    units="cm",
    monitor=monitor,
    fullscr=True,
    screen=screen_index,
    color="red"
)
# === Timing log ===
timing_log = []

# === Loop through all stimuli files ===
for csv_file in stimulus_files:
    print(f"Playing: {csv_file.name}")
    df = pd.read_csv(csv_file)
    # Timestamp: start
    stim_start = time.time()
    print(stim_start)
    # Extract all dot columns by name pattern
    dot_names = sorted(set(col.split('_')[0] for col in df.columns))
    dots = {}
    for dot_name in dot_names:
        dots[dot_name] = visual.Circle(
            win=win,
            radius=1,  # Will be overridden per frame
            fillColor="black",
            lineColor="black",
            units="cm"
        )

    # === Frame loop ===
    for i, row in df.iterrows():
        for dot_name, dot in dots.items():
            x, y = row[f"{dot_name}_x"], row[f"{dot_name}_y"]
            radius = row.get(f"{dot_name}_radius", 0.4)  # Default radius if missing

            if radius > 0:
                dot.pos = (x, y)
                dot.radius = radius
                dot.draw()
        win.flip()

        # Allow escape
        if "escape" in event.getKeys():
            win.close()
            core.quit()
    # Timestamp: end
    stim_end = time.time()
    print(stim_end)
    # Add to log
    timing_log.append({
        "stimulus_file": csv_file.name,
        "start_time_unix": stim_start,
        "end_time_unix": stim_end,
        "duration_sec": round(stim_end - stim_start, 3)
    })


    core.wait(1)  # Pause between stimuli

win.close()
core.quit()

# === Save timing log ===
timing_df = pd.DataFrame(timing_log)
timing_df.to_csv(stimuli_path / "stimulus_timing_log.csv", index=False)

print("✅ Timing log saved.")
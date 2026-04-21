# Stimulus Authoring Stage Map

Purpose

Provide the ordered flow for generating experiment-specific stimulus assets and handing their timing semantics downstream to calcium-analysis code.

Use this file when

- You need to change trajectory generation or mapping JSONs.
- You need to know which stage writes `*_trajectory.csv` or parameter files.
- You need to decide whether a timing bug belongs in the generator scripts or in `src/stimuli_timeline.py`.

## End-to-end stages
1. Choose or edit experiment-specific mapping JSONs and package files in `scripts/stimuli/`.
2. Generate per-stimulus trajectory tables in the relevant script, including rocking, flicker, radius, and repetition logic.
3. Write parameter tables or package metadata alongside generated trajectories.
4. Inspect generated assets with `plots_stimuli.ipynb` or similar notebook tooling.
5. Convert trajectory and experiment-log semantics into downstream timing traces with `src/stimuli_timeline.py`.
6. Optionally replay generated stimuli or capture timing logs with `try_projection.py`.

## Key outputs by stage
- Stage 2 writes `*_trajectory.csv` files used downstream by calcium-analysis notebooks and timing helpers.
- Stage 3 writes experiment-parameter CSVs and package-style JSON metadata.
- Stage 6 can write `stimulus_timing_log.csv` during playback runs.

## Concept ownership
- `scripts/stimuli/Trayectory_flicker.py`, `Trayectory_rocking_stimuli.py`, and `trayectory_stimuli.py` own experiment-specific trajectory generation.
- Mapping and package JSON files own experiment-specific configuration values.
- `plots_stimuli.ipynb` owns inspection and visualization of generated assets.
- `src/stimuli_timeline.py` owns reusable timing extraction and log-to-trace semantics consumed by analysis code.
- `try_projection.py` owns playback wrapper behavior and timing-log capture.

## Navigation notes
- Read `canonical-outputs.md` for file naming and folder-layout expectations.
- Read `symbol-index.md` before changing reusable timing helpers in `src/stimuli_timeline.py`.
- Read `current-state.md` for repo-specific naming caveats and legacy file surfaces.

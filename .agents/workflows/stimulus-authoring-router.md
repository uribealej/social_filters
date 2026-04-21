# Stimulus Authoring Router

Purpose

Route work related to trajectory generation, rocking and flicker stimulus assets, mapping JSONs, projection playback wrappers, and shared timing semantics for those assets.

Use this file when

- The task targets files under `scripts/stimuli/`.
- The task changes stimulus geometry, flicker cadence, rocking behavior, mapping JSON contents, or saved trajectory CSVs.
- The task mentions timing mismatches between generated stimuli and calcium-analysis alignment.

## Read order
1. `../references/stimulus-authoring-stage-map.md`
2. `../references/canonical-outputs.md` for output names and folder layout
3. `../references/symbol-index.md` for timing owners in `src/stimuli_timeline.py`
4. `../references/refactor-rules.md`
5. `../references/current-state.md` if the task touches legacy naming or downstream timing assumptions
6. `../references/recent-changes-stimulus-authoring.md`

## Task routing table

| Task pattern | Read first | Open next |
| --- | --- | --- |
| Trajectory geometry, rocking pair logic, flicker cadence, repetitions, pauses, or CSV writer behavior | `stimulus-authoring-stage-map.md` | Target script in `scripts/stimuli/` |
| Mapping JSON, package JSON, experiment parameter content, or per-experiment stimulus config | `stimulus-authoring-stage-map.md` | Target JSON file, then generating script |
| Timing mismatch between generated CSVs and calcium-analysis interpretation | `canonical-outputs.md` | `src/stimuli_timeline.py` before touching notebooks or plotting code |
| Projection, PsychoPy playback, or runtime timing-log capture | `stimulus-authoring-stage-map.md` | `scripts/stimuli/try_projection.py` |
| Stimulus plotting or inspection notebook behavior | `stimulus-authoring-stage-map.md` | `scripts/stimuli/plots_stimuli.ipynb` |

## Ownership guidance
- Stimulus scripts own experiment-specific asset generation and packaging.
- JSON files under `scripts/stimuli/` are configuration inputs, not downstream timing authority.
- `src/stimuli_timeline.py` owns reusable timing extraction and log-to-trace semantics consumed by calcium analysis.
- `try_projection.py` owns display wrapper behavior and timing-log capture only; it should not redefine trajectory semantics.
- Smallest practical validation surface: regenerate the smallest affected CSV or parameter set, run timing extraction on that output, and verify that downstream assumptions still match the written trajectory.
- Handoff log: `../references/recent-changes-stimulus-authoring.md`.

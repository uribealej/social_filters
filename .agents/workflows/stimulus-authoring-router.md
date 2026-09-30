# Stimulus Authoring Router

Purpose

Route work related to trajectory generation, rocking and flicker stimulus assets, mapping JSONs, projection playback wrappers, and shared timing semantics for those assets.

Use this file when

- The task targets generators under `scripts/stimuli/generation/`, playback wrappers under `scripts/stimuli/playback/`, configurations under `configs/stimuli/`, or inspection notebooks under `notebooks/stimuli/`.
- The task changes stimulus geometry, flicker cadence, rocking behavior, mapping JSON contents, or saved trajectory CSVs.
- The task mentions timing mismatches between generated stimuli and calcium-analysis alignment.

## Task routing table

Choose one row. A dash means no extra reference is needed before opening the owner.
For everyday stimulus analysis or a new `src/` responsibility, dependency, or public contract decision, read
[the organization policy](../references/src-organization-policy.md) and then
use the matching row below to open the owner.

| Task pattern | Reference if needed | Open next |
| --- | --- | --- |
| Trajectory geometry, rocking pair logic, flicker cadence, repetitions, pauses, or CSV writer behavior | — | Target script in `scripts/stimuli/generation/` |
| Mapping JSON, package JSON, experiment parameter content, or per-experiment stimulus config | — | Target JSON file, then generating script if its behavior matters |
| Timing mismatch between generated CSVs and calcium-analysis interpretation | `canonical-outputs.md` if the file contract is involved | `src/stimuli_timeline.py`, then the generating script if the written trajectory is wrong |
| Repeated generator-only geometry helpers | `refactor-rules.md` if ownership is unclear | Target script in `scripts/stimuli/generation/` |
| Shared angle or timing interpretation, including `get_angles_from_positions` | — | `src/stimuli_timeline.py` |
| Projection, PsychoPy playback, or runtime timing-log capture | — | `scripts/stimuli/playback/try_projection.py` |
| Stimulus plotting or batch inspection behavior | — | `src/stimulus_visualization.py`; notebook only for configuration or reporting |
| Stimulus stage order, writer handoff, or rerun chain | `stimulus-authoring-stage-map.md` | Relevant writer and first consumer |
| Output names, folders, or legacy spelling | `canonical-outputs.md`; `current-state.md` for legacy caveats | Relevant writer |
| Helper name without a known owner, or stable public import path | `symbol-index.md` | Owner identified there |

## Ownership guidance
- Stimulus scripts own experiment-specific asset generation and packaging.
- JSON files under `configs/stimuli/` are configuration inputs, not downstream timing authority.
- `src/stimuli_timeline.py` owns reusable timing extraction and log-to-trace semantics consumed by calcium analysis.
- Shared geometry helpers that are only used by generator and inspection code may still live in `scripts/stimuli/generation/` until they become downstream timing authority.
- `try_projection.py` owns display wrapper behavior and timing-log capture only; it should not redefine trajectory semantics.
- Batch inspection helpers belong in the dedicated `src/stimulus_visualization.py`; the batch notebook contains only configuration and report calls. Spatial coverage uses visible directions, while time traces retain all frames.
- For generator or timing behavior changes, regenerate the smallest affected CSV or parameter set, run timing extraction on that output, and check the downstream interpretation.

# Recent Changes: Stimulus Authoring

Use this log for trajectory generation, mapping JSON changes, timing handoff, and playback-wrapper slices.

## Entry template
- Date and label:
- Slice goal:
- Passes completed in this session:
- What changed:
- What remains broken:
- Remaining in-slice work:
- Next likely breakpoint:
- Rerun implications:

## 2026-04-21 - Initial guidance system seed
- Slice goal: create the routed guidance docs for stimulus-authoring ownership and handoff.
- Passes completed in this session: repo inspection, router creation, stage-map creation, output-contract documentation.
- What changed: added the stimulus-authoring router, stage map, and references that keep reusable timing semantics in `src/stimuli_timeline.py`.
- What remains broken: no stimulus generation code changed in this pass.
- Remaining in-slice work: append future entries when trajectory writers, timing semantics, or playback behavior change.
- Next likely breakpoint: first task that changes rocking, flicker, or downstream timing behavior.
- Rerun implications: none for runtime behavior; docs only.

## 2026-04-22 - Slice 9 wrapper boundary for `try_projection.py`
- Slice goal: keep `scripts/stimuli/try_projection.py` as a playback/display wrapper only.
- Passes completed in this session: routed inspection, wrapper refactor, owner-layer timing smoke, single-file playback smoke.
- What changed: refactored `try_projection.py` into small wrapper helpers plus `main()` without moving timing semantics into the script; preserved playback invocation flow and optional `stimulus_timing_log.csv` writing.
- What remains broken: no generator-side or downstream timing-helper drift addressed in this slice.
- Remaining in-slice work: none.
- Next likely breakpoint: a later slice that extracts any shared stimulus-display utility or addresses separate generator-side drift.
- Rerun implications: playback smoke was validated on one existing trajectory file and rewrote `stimulus_timing_log.csv` in the validation folder.

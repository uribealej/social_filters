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

## 2026-04-22 - Slice 8 config boundary in stimulus mapping JSONs
- Slice goal: keep JSON mapping files as configuration inputs and remove hard-coded experiment run settings from active stimulus generator scripts.
- Passes completed in this session: routed inspection, mapping JSON update, generator-script config loading update, syntax check, primary smoke attempt.
- What changed: added `_experiment` blocks to `trajectory_exp_6_mapping_flickers_and_bouts_2.json` and `trajectory_exp_6_mapping_rocking.json`; updated `Trayectory_flicker.py` and `Trayectory_rocking_stimuli.py` to read experiment-wide settings from JSON with backward-compatible defaults while preserving existing stimulus keys, output names, and generation semantics.
- What remains broken: the primary flicker smoke run could not overwrite `LeB_control_trajectory.csv` in `D:\Alejandro\Data\OneDrive - Université de Lausanne\Lab\Data\stimuli\Exp_6_mapping_positions_retina_2` because the file is locked, so runtime asset regeneration was not fully completed in this session.
- Remaining in-slice work: rerun the flicker script once the target output files are writable and confirm the expected asset set is rewritten successfully.
- Next likely breakpoint: if validation passes after the lock is cleared, the next remaining drift is wrapper-structure normalization in slice 7 rather than additional config-boundary work.
- Rerun implications: rerun `Trayectory_flicker.py` with `scripts/stimuli/trajectory_exp_6_mapping_flickers_and_bouts_2.json` after releasing the output-file lock; both edited scripts passed `py_compile`.

## 2026-04-22 - Slice 6 timing authority in `src/stimuli_timeline.py`
- Slice goal: make `src/stimuli_timeline.py` the single downstream timing authority without changing trajectory CSV interpretation or stimulus-trace shapes.
- Passes completed in this session: routed inspection, owner-helper extraction, first-consumer refactor, owner/consumer timing validation.
- What changed: moved the downstream timing-normalization helper into `src/stimuli_timeline.py`; updated `src/data_loading.py` to call the owner helper instead of rewriting timing fields locally; updated the symbol index to document the owner-level callable surface.
- What remains broken: generator-side duration metadata and local geometry duplication in legacy stimulus scripts still exist, but they remain wrapper-side drift outside this slice.
- Remaining in-slice work: none.
- Next likely breakpoint: a later cleanup slice can decide whether generator-side package-duration helpers or legacy geometry helpers should be consolidated.
- Rerun implications: rerun the timing validation command against one trajectory, one flicker, and one rocking CSV in the existing stimulus output folder when checking future timing changes.

# S14 completion record

S14 validated the supported `src/` workflows on 2026-09-29–30 using the
`social_filters_updated` Conda environment and representative data under
`C:\Users\suribear\OneDrive - Université de Lausanne\Lab`.

## Checkpoints

1. **Foundation:** The full unit and characterization suite passed: 101 tests
   after the S14 fixes. Its source contract imports/compiles `src`, its notebook
   contract compiles every active code cell, and its consumer contract verifies
   active notebook/script imports and indexed symbols.
2. **Preprocessing:** Exp 1 `L433_f02` plane0 extraction reproduced the saved
   `(3340, 442)` dFoF array and retained ROI indices. The NPZ matched both.
   Five planes reconstructed the saved `(3340, 1984)` merged array, ROI vector,
   and map column order. `load_2p_experiment` consumed these outputs and reported
   1670 frames per selected block.
3. **Single fish:** Exp 1 `L433_f02` loaded and aligned 32 real traces across 10
   stimuli and 40 trial windows. Current-mode significant-trace detection
   returned a binary `(3340, 32)` raster. The rendered raster was inspected for
   labels and clipping.
4. **Several-fish analysis:** Exp 1 `L433_f02` and `L433_f04` preserved the
   configured fish order and stimulus order `[4, 1, 2, 3, 8, 5, 6, 7]`. The
   pooled matrix was `(1192, 592)` and the response matrix was `(1192, 8)`;
   neuron-row identity, selected summary rows, and left/right overlap ran.
5. **Figures and LME:** The two-fish raster and mean-trace figures were rendered
   and inspected. A duplicate x-axis label between raster and mean panels was
   removed, and only the two affected figure snapshots and function hash were
   refreshed in `tests/contracts/plotting_s08.json`. Exp 5 `L733_f01` and
   `L733_f06` produced an 11,732-row, 14-stimulus LME response table; a
   448-row mixed-model smoke fit converged.
6. **Stimulus:** Three real Exp 8 trajectory CSVs (`LeBseg1`, `Leflick1`,
   `LeRock`) agreed with timing frame counts and seconds. Standard and polar
   figures were rendered and inspected. The report had the canonical PNG/CSV
   filenames and round-tripped its angle-list CSV schema.
7. **Contracts and closure:** Canonical per-plane, merged, analysis, and
   stimulus output names and schemas were checked at the first consumers.
   No notebook stage order was changed in S14. `symbol-index.md`, relevant
   stage/output references, and the active handoff were updated. The completed
   S12 handoff was preserved in [its step log](src-refactor-s12-step-log.md).

## Defects found and resolved

- The Exp 1 `L433_f02` canonical map had 1,773 rows for a 1,984-column merged
  array. Two timestamped writer fallback maps had 1,984 rows; the newest one
  matched all five per-plane arrays and ROI indices. `src/data_loading.py` now
  selects the canonical map only when its row count matches, otherwise selects
  the newest matching fallback, and raises if no available map matches.
- `src/stimuli_timeline.py` now interprets plain `x,y,radius` trajectory columns
  as well as dot-prefixed columns. Both formats have a known-output timing test.
- `src/plotting_all_fish.py` no longer shows the raster's duplicate x-axis
  label above the mean panel. Focused plotting characterization passed after
  the reviewed contract update.

## Limits and follow-up outside this refactor

- The first selected Exp 5 fish (`L684_f01`) has cloud-only Suite2p metadata in
  this environment; its `stat.npy` read failed. The LME check used two fish
  from the same notebook's configured cohort whose files were local.
- Exp 5 table validation flagged 247 responses outside its suggested
  `(-100, 100)` range. The small model fit converged with singular-covariance
  warnings; its coefficients were not interpreted as scientific results.
- Exp 8 bout/flicker block and post-window settings still need their separate
  scientific comparison, as recorded in `current-state.md`.
- The real-data checks were representative chains, not full publication
  notebook reruns. No external experiment files were rewritten.

The full suite ended with `Ran 101 tests ... OK`; `git diff --check` passed.

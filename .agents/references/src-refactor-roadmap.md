# `src/` Refactor Roadmap

Purpose

Provide a durable, ordered, multi-session plan for reorganizing `src/` without losing scientific behavior, notebook compatibility, output contracts, or figure quality.

Use this file when

- Starting or resuming any repository-wide `src/` cleanup.
- Choosing the next safe refactor slice.
- Recording which slice is active or completed across chats and days.
- Deciding what must be validated before a slice can close.

Read `src-diagnosis.md` first for the factual baseline. Also follow `refactor-loop-policy.md`, the relevant workflow router, and `code-authoring-checklist.md` before editing Python or notebooks.

## How to use this roadmap

- Work in the listed order unless a dependency-free slice is explicitly selected.
- Complete one slice per coherent change. Do not combine unrelated owner modules merely to reduce the number of commits.
- At the start of a session, identify the slice ID and confirm its prerequisites.
- At the end of a session, update the status table below.
- If a slice stops incomplete, record the exact breakpoint in `recent-changes.md` according to `refactor-loop-policy.md`.
- When a slice is complete, remove its temporary handoff from `recent-changes.md`; the durable result belongs in the appropriate reference, stage map, or symbol index.

Allowed status values are `not started`, `in progress`, `blocked`, and `complete`.

## Status tracker

| Slice | Description | Status | Depends on |
| --- | --- | --- | --- |
| S00 | Step 1 - Freeze contracts and create the test foundation | complete | None |
| S01 | Step 2 - Record public API and notebook consumer contracts | complete | S00 |
| S02 | Remove indisputable import/comment debris | complete | S00 |
| S03 | Consolidate timing-duration normalization | complete | S00-S01 |
| S04 | Characterize significant-trace V1 versus V2 | complete | S00-S01 |
| S05 | Consolidate significant-trace implementations behind compatibility modes | complete | S04 |
| S06 | Split `analysis_tools` by scientific responsibility | complete | S00-S01 |
| S07 | Split `multifish_analysis` by scientific domain | not started | S06 |
| S08 | Split plotting by figure family | not started | S00-S01, S06-S07 as applicable |
| S09 | Separate several-fish workflow, reporting, and pure transforms | not started | S06-S08 |
| S10 | Review loaders, extraction modules, and optional dependencies | not started | S00-S03 |
| S11 | Perform evidence-based dead-code retirement | not started | S03-S10 |
| S12 | Migrate consumers and reduce compatibility facades | not started | S03-S11 |
| S13 | Adopt the permanent `src` organization policy | not started | S03-S12 |
| S14 | Run final end-to-end validation and close the refactor | not started | S13 |

## Global invariants for every slice

- Preserve canonical filenames, folder layouts, cache keys, table columns, array dimension order, and notebook stage order unless a slice explicitly defines a migration.
- Preserve documented public imports until consumers have migrated and a compatibility decision has been recorded.
- Keep scientific calculations separate from loading, saving, printing, plotting, and notebook orchestration.
- Do not introduce version-suffixed files such as `_v3`; use explicitly named algorithms, modes, or strategies.
- Do not change scientific behavior during a structural move. Make any intended behavior change a separate reviewed slice with its own before/after evidence.
- Validate the owning helper and its first downstream consumer.
- Render and inspect every new or materially changed figure.
- Keep commits or handoffs scoped to one completed owner slice.

## S00 - Freeze contracts and create the test foundation

Objective

Create the safety net required for all later movement.

Work

1. Add a `tests/` structure organized by owner domain.
2. Add a source compile test and module import smoke tests.
3. Mark the ScanImage dependency as optional in import tests or install/repair it in the supported environment; do not hide unexpected import errors.
4. Add reusable synthetic fixtures with explicit dimension conventions:
   - continuous traces: time x neurons;
   - trial-aligned traces: neurons x time x trials;
   - stimulus logs and timing dictionaries;
   - small response/raster matrices;
   - small DataFrames with stable column order.
5. Add helpers for exact array equality, tolerance-based numerical comparison, DataFrame schema/order checks, and figure smoke validation.
6. Add a notebook source-cell compile check, with the already documented Exp 8 syntax issue handled explicitly rather than silently ignored.

Validation

- The new test suite itself runs in the supported environment.
- Failures identify the module or notebook clearly.
- No production behavior changes in this slice.

Exit criteria

- Later slices can compare old and new implementations using shared fixtures.
- Environment-specific exclusions are documented and narrow.

Completion record - 2026-09-24

- Added a standard-library `unittest` suite under `tests/`; no new package installation is required.
- Added source compilation and isolated module-import checks for every active `src` module.
- Registered `ScanImageTiffReader` as the only accepted optional import failure in the active environment.
- Added deterministic fixtures for continuous traces, trial-aligned traces, stimulus timing/logs, response rasters, and response tables.
- Added reusable exact-array, numerical-tolerance, DataFrame-schema, and figure-structure assertion helpers.
- Added transformed compilation checks for every active notebook code cell.
- Registered the pre-existing syntax failure in code cell 2 of `notebooks/calcium/exp_08_accumulation_ev/02_bout_flicker_position_onset.ipynb`; new failures will fail the suite, and repairing this cell requires removing the registry entry.
- Added a repository-local ignored Matplotlib cache for reproducible test execution.
- Validation command: `python -m unittest discover -s tests -t . -v`.
- Validation result: 9 tests passed as a suite, with 1 explicit environment skip.
- Environment follow-up - 2026-09-25: the `0xc06d007f` Matplotlib failure was traced to threaded MKL hanging in basic BLAS calls. A cloned OpenBLAS environment passed NumPy, SciPy BLAS, PNG rendering, and the full suite; `environment.yml` now pins OpenBLAS and the suite treats BLAS/render hangs as failures.

## S01 - Record public API and notebook consumer contracts

Objective

Turn implicit notebook usage into an explicit compatibility inventory.

Work

1. Extract active `src` imports and called attributes from every active notebook and script.
2. Reconcile those results with `symbol-index.md`.
3. Classify each public helper as:
   - active and stable;
   - active but scheduled to move behind a facade;
   - documented but currently uncalled;
   - private implementation detail;
   - deprecation candidate.
4. Record return shapes, keys, table columns, side effects, and output paths for high-use entrypoints.
5. Add `__all__` only where it clarifies an already agreed public surface; do not use it to remove compatibility accidentally.

Validation

- Notebook import smoke tests cover all active notebooks.
- `symbol-index.md` matches the observed stable surface.

Exit criteria

- Every later move has a known compatibility target.

Completion record - 2026-09-25

- Scanned all 12 active notebooks and all 4 active Python scripts under `scripts/`, excluding `archive/` and history files.
- Recorded the exact imports, used symbols, and registered invalid notebook cells in `tests/contracts/src_api_consumers.json`.
- Added a reusable AST/IPython inventory extractor and contract tests that fail when a consumer changes without review.
- Added a runtime check that every symbol referenced by an active consumer still exists.
- Added a check that every active consumer symbol is documented in `symbol-index.md`.
- Added `.agents/references/src-api-consumer-inventory.md` with classifications, consumer ownership, array shapes, return keys, side effects, output paths, and migration rules.
- Reconciled `symbol-index.md` with active use, including bout-position analysis, static--flicker fish-level statistics, active V2 significant-trace detection, V1 raster plotting, and previously missing plotting helpers.
- Recorded the hidden coupling where `reusable_several_fish.py` accesses `compute_stimulus_selectivity_metrics` through an accidental `multifish_analysis` re-export.
- Did not add `__all__`; the current modules still contain transitional and accidental exports that should not be frozen before facade boundaries are agreed.
- Validation command: `python -m unittest discover -s tests -t . -v`.
- Validation result: 12 tests passed, with 1 explicit skip for the previously recorded Matplotlib native-render limitation.
- No scientific calculations, notebook behavior, or output contracts were changed.

## S02 - Remove indisputable import and comment debris

Objective

Reduce noise without moving scientific behavior.

Work

1. Remove imports proven unused, including the inspected `TelnetServer` import and duplicate local imports where still confirmed.
2. Remove repeated imports later in modules.
3. Remove large commented-out implementations only after checking that Git history preserves them and no explanatory scientific rationale would be lost.
4. Add missing short module docstrings.
5. Do not rename functions or move modules in this slice.

Validation

- Compile and import smoke tests pass.
- Relevant notebook imports still work.
- No numerical or figure output changes are expected.

Exit criteria

- Files are quieter and easier to split, with no API change.

Completion record - 2026-09-25

- Removed four imports proven unused (`TelnetServer`, `PCA`, `Path`, and `scipy.stats.mode`) and repeated imports in `analysis_tools.py`, `plotting.py`, and `stimuli_timeline.py`.
- Removed commented predecessor implementations only where an active replacement was adjacent; commit `23a461a` preserves every removed block, and no scientific rationale or active code was removed.
- Added short module docstrings to the package and the six active modules that lacked them.
- Preserved all function names, signatures, module locations, numerical behavior, and figure behavior.
- Validation command: `python -m unittest discover -s tests -t . -v` using the `social_filters_user` interpreter.
- Validation result before the environment repair: 12 tests passed, with the Matplotlib native-render probe skipped after its documented timeout. After switching the cloned environment to OpenBLAS, all 13 tests passed with no skips, including the new BLAS runtime probe and Matplotlib render probe.

## S03 - Consolidate timing-duration normalization

Objective

Establish one canonical implementation of `transform_stimuli_duration`.

Owner

`src/stimuli_timeline.py`.

Work

1. Add characterization tests covering missing keys, frame aliases, rounding, immutability of inputs, and output dictionary structure.
2. Make `src.data_loading.transform_stimuli_duration` delegate to the timing owner or preserve it as a documented compatibility alias.
3. Update internal consumers to call the canonical owner.
4. Update `symbol-index.md` if the public status changes.

Validation

- Old and canonical results match for representative timing dictionaries.
- `load_2p_experiment` still produces the same timing contract.
- The first alignment consumer still works.

Exit criteria

- Only one implementation contains normalization logic.

Completion record - 2026-09-25

- Kept `src.stimuli_timeline.transform_stimuli_duration` as the sole normalization implementation.
- Preserved `src.data_loading.transform_stimuli_duration` as a compatibility wrapper and updated `load_2p_experiment` to call the canonical timing owner directly.
- Added characterization coverage for missing keys, frame aliases, three-decimal rounding, input immutability, output structure, wrapper equivalence, loader output, and the first alignment consumer.
- Updated `symbol-index.md` and `src-api-consumer-inventory.md` to record the canonical owner and compatibility entrypoint.
- Focused validation command: `python -m unittest tests.timing.test_stimuli_timeline -v`.
- Full validation command: `python -m unittest discover -s tests -t . -v` using the validated `social_filters_openblas` environment.
- Validation result: 20 tests passed with no skips; no numerical timing result, loader return key, or alignment shape changed.

## S04 - Characterize significant-trace V1 versus V2

Objective

Document exactly which detector behaviors are shared and which differ before consolidation.

Work

1. Create deterministic small traces covering finite signals, constant traces, NaNs, low sample counts, and visible transients.
2. Compare every duplicated public stage between V1 and V2.
3. Capture shapes, dtypes, return ordering, warnings, random behavior, and numerical differences.
4. Add a fixed random seed or explicit generator where required for reproducible tests without changing default scientific behavior prematurely.
5. Record which implementation is authoritative for new analysis and which legacy behavior must remain reproducible.

Validation

- Comparison tests expose intentional differences rather than assuming equality.
- The main single-fish notebook's mixed imports are covered.

Exit criteria

- S05 can consolidate implementations without guessing about scientific intent.

Completion record - 2026-09-25

- Added deterministic finite, constant, NaN-containing, low-sample, and visible-transient fixtures under `tests/significant_traces/`.
- Characterized all 13 public stages duplicated by `significant_traces.py` and `significant_traces_v2.py`, including shapes, dtypes, tuple ordering, warnings, exceptions, reproducibility, shared numerical results, and intentional rasterization differences.
- Confirmed that ordinary finite inputs share the early noise-fit, centering, normalization, transition, grid, histogram, and default significance contracts.
- Confirmed that V2 adds controlled random state, diagnostics, finite fallbacks for constant and low-sample traces, NaN-tolerant fitting, safe invalid-sigma handling, bounded neighbor counts, and explicit compatibility/strict modes. V1 raises on several of those inputs and relies on global NumPy random state.
- Selected V2 as the authoritative implementation for new analysis and the S05 consolidation target. Preserve V1 behavior only as an explicit legacy compatibility mode during consolidation; do not maintain two independent detectors.
- Preserved the active single-fish workflow contract: V2 detection followed by V1 `clean_binary_raster_columns` and `plot_dff_and_raster` remains executable until S05 migrates those helpers.
- A warmed seven-repeat synthetic microbenchmark measured median runtimes of 6.3 ms for V1 and 9.6 ms for V2. This small absolute difference does not outweigh V2's reproducibility and robustness, and the synthetic fixtures do not establish biological sensitivity or specificity.
- Focused validation command: `python -m unittest tests.significant_traces.test_v1_v2_characterization -v`.
- Full validation command: `python -m unittest discover -s tests -t . -v` using the validated `social_filters_openblas` environment.
- Validation result: all 8 focused characterization tests and all 28 full-suite tests passed with no skips. The existing mixed-workflow figure rendered successfully; S04 did not change figure construction.

## S05 - Consolidate significant-trace implementations

Objective

Replace duplicated V1/V2 code with one canonical event-detection owner and explicit compatibility behavior.

Work

1. Create a cohesive event-detection implementation with explicitly named legacy/current modes where behavior differs.
2. Move identical shared stages only after S04 proves equivalence.
3. Keep `src.significant_traces` and `src.significant_traces_v2` as thin compatibility facades during migration.
4. Separate diagnostic plotting from detector computation where practical.
5. Preserve the current result tuple or provide named result structures through an additive API before changing consumers.
6. Do not create a `significant_traces_v3.py`.

Validation

- Characterization tests for both compatibility modes pass.
- The main single-fish notebook computes, cleans, and plots the same selected mode.
- Runtime and memory are checked on the smallest representative dataset.
- Any changed figure is rendered and inspected.

Exit criteria

- Shared implementation is no longer duplicated.
- Legacy and current behavior are explicitly selectable and documented.

Completion record - 2026-09-25

- Added `src/significant_trace_detection.py` as the single unversioned owner for Romano-style noise fitting, transition densities, rasterization, raster cleanup, and detector diagnostics.
- Made robust V2 behavior the default `mode="current"` and encoded only the scientifically distinct V1 stages behind explicit `mode="legacy"`; no version-suffixed replacement module was created.
- Reduced `src.significant_traces` and `src.significant_traces_v2` to thin compatibility facades with their historical signatures and no NumPy, SciPy, scikit-image, scikit-learn, or Matplotlib implementation logic.
- Migrated the shared single-fish notebook to import detection, cleanup, and plotting from the canonical owner, and updated the machine-readable consumer contract and routed API documentation.
- Compared both canonical modes against the pre-consolidation sources on the deterministic transient fixture. All eight returned arrays for both current/V2 and legacy/V1 were exactly equal, including dtypes, shapes, tuple order, densities, rasters, and constrained maps.
- Added consolidation tests requiring facade-to-mode equivalence, explicit mode validation, and facade-only dependency boundaries. The S04 characterization suite remains active for finite, constant, NaN, low-sample, random, and intentional-difference behavior.
- Runtime/memory smoke check on the 180-by-4 fixture with 24 bins measured current mode at 0.060 s and 0.20 MiB peak versus legacy mode at 0.078 s and 1.07 MiB peak under `tracemalloc`; both returned the established eight-item tuple.
- Rendered and inspected the canonical legacy/current comparison figure. Titles, labels, axes, and colorbars were readable with no clipping or overlap; blank raster panels correctly reflected zero detected events in the small synthetic fixture.
- Focused validation command: `python -m unittest tests.significant_traces.test_v1_v2_characterization tests.significant_traces.test_consolidation -v`.
- Full validation command: `python -m unittest discover -s tests -t . -v` using `social_filters_openblas`.
- Validation result: all 12 focused tests and all 32 full-suite tests passed with no skips. All active notebook cells retained their registered compilation status.

## S06 - Split `analysis_tools` by scientific responsibility

Objective

Separate pure scientific domains while preserving `src.analysis_tools` as the initial public facade.

Recommended order

1. Alignment and stimulus selection.
2. Response windows and AUC/peak metrics.
3. Motion-delta metrics.
4. Reliability filtering.
5. Response classification and onset ordering.
6. Z-scoring and selectivity helpers.
7. Move diagnostic figures to the plotting owner.

Work rules

- Extract one tightly related function family at a time.
- Keep public names re-exported from `analysis_tools.py` until S12.
- Split long functions into pure calculations plus thin I/O or plotting wrappers.
- Preserve error types, array axes, NaN behavior, thresholds, and return structures.

Validation

- Add known-output tests before each family moves.
- Check the first active notebook consumer for each family.
- Render any moved diagnostic plot.

Exit criteria

- `analysis_tools.py` is a small compatibility facade rather than a mixed implementation module.

Completion record - 2026-09-25

- Replaced the 2,300-line mixed `analysis_tools.py` implementation with a compatibility-only facade that directly re-exports all 24 historical public functions and preserves their signatures.
- Added narrow owners: `trial_alignment.py`, `response_metrics.py`, `motion_metrics.py`, `reliability.py`, `response_classification.py`, `response_selectivity.py`, `response_normalization.py`, and `analysis_io.py`.
- Moved trial alignment/stimulus selection, response windows/AUCs, motion deltas, reliability filtering, active-response classification, selectivity, baseline z-scoring, and file/debug utilities without changing array axes, NaN behavior, thresholds, return structures, DataFrame schemas, or save paths.
- Moved accepted/rejected reliability rasters and the three-stimulus Venn diagnostic to `plotting.py`; reliability computation now delegates its optional figure construction to that owner.
- Added eight S06 tests covering facade ownership, deterministic alignment, response metrics, motion metrics, reliability selection, active-neuron classification, selectivity, normalization, moved figure rendering, and the first loader/multifish consumers.
- Compared the pre-split and post-split implementations directly on representative inputs. Alignment, response metrics, motion metrics, reliability, classification, selectivity, and normalization results matched exactly.
- Rendered and inspected the reliability histogram, threshold curve, accepted/rejected rasters, and three-stimulus Venn figure. Titles, labels, ticks, axes, annotations, and colorbars were clear with no overlap or clipping.
- Focused validation command: `python -m unittest tests.analysis.test_analysis_tools_split -v`.
- Full validation command: `python -m unittest discover -s tests -t . -v` using `social_filters_openblas`.
- Validation result: all 8 focused tests and all 40 full-suite tests passed with no skips; active notebook consumer contracts and source/notebook compilation remained intact.

## S07 - Split `multifish_analysis` by scientific domain

Objective

Separate independent multifish analyses without changing notebook-facing behavior.

Recommended order

1. Core matrix construction and repetition combination.
2. Static-flicker recruitment and fish-level statistics.
3. Bout-flicker position analysis.
4. Stimulus specificity and selectivity.
5. Stimulus-vector similarity and segment permutations.
6. Active-neuron overlap and pooled diagnostics.

Work rules

- Preserve `src.multifish_analysis` as a facade initially.
- Keep shared primitives below experiment-specific analyses in the dependency direction.
- Avoid importing plotting into computation modules.

Validation

- Matrix shape/order and row-metadata equality tests.
- DataFrame column/order and category equality tests.
- Deterministic permutation/statistics tests with explicit random state where applicable.
- First downstream notebook check for every extracted family.

Exit criteria

- Each scientific family has one clear owner and can be tested independently.

## S08 - Split plotting by figure family

Objective

Replace the plotting god module with cohesive figure-family modules while keeping `src.plotting` compatible.

Recommended families

- common styles, labels, and reference-line helpers;
- single-fish rasters and stimulus markers;
- all-fish rasters and mean traces;
- static-flicker and bout-position figures;
- stimulus-specificity and similarity figures;
- motion-delta and active-trace diagnostics;
- LME figures.

Work rules

- Move one figure family per slice.
- Keep figure-specific options explicit at the public call.
- Do not move scientific calculations into figure modules; extract required data preparation upstream first.
- Re-export existing public plotting names from `src.plotting` until S12.

Validation

- Run figure smoke tests with representative synthetic or smallest data-backed inputs.
- Save and inspect screenshots for labels, legends, ticks, annotations, clipping, order, and readability.
- Compare expected axes count, titles, labels, and plotted data where practical.

Exit criteria

- `src.plotting` is a facade and figure families have independent owners.

## S09 - Separate several-fish workflow and reporting

Objective

Untangle `reusable_several_fish.py` into orchestration, reporting/export, pure preparation, and plotting calls.

Recommended order

1. Report serialization and notebook export.
2. Cohort loading and preflight checks.
3. Pure response-selection and keep-mask preparation.
4. Diagnostic data preparation.
5. High-level figure orchestration.

Validation

- Report folder contents and JSON/CSV schema checks.
- Mock or temporary-path tests for export and save behavior.
- Equality checks for keep masks, row order, and prepared matrices.
- First downstream several-fish notebook run.

Exit criteria

- Scientific transformations can run without reporting or plotting dependencies.
- Reporting code does not own scientific calculations.

## S10 - Review loaders, extraction modules, and optional dependencies

Objective

Finish the smaller owner modules after the large-module boundaries are stable.

Work

1. Split `load_2p_experiment` into private lookup/load/assembly stages without changing its public return contract.
2. Add tests for optional merged-map behavior and cache discovery.
3. Confirm `dff_extraction.py` public/private boundaries and add known-output tests for baseline and ROI filtering.
4. Decide and document whether `ScanImageTiffReader` is mandatory for the supported environment or lazily imported only when aux extraction is called.
5. Keep `lme_feature_decomposition.py` cohesive unless tests reveal a clear independent boundary.
6. Leave `stimulus_visualization.py` largely intact unless a real ownership issue appears.

Validation

- Loader fixture tests cover missing optional files and canonical path choices.
- One-plane preprocessing smoke validation where data are available.
- Import smoke checks pass under the documented dependency policy.

Exit criteria

- Smaller modules have clear contracts and no accidental cross-layer dependencies.

## S11 - Evidence-based dead-code retirement

Objective

Remove code only after the new structure and tests can demonstrate that it is unneeded.

Required evidence for each deletion

1. No active Python, notebook, or script reference.
2. No internal call-graph reference.
3. Not required by `symbol-index.md` or another documented contract, or formally deprecated first.
4. Relevant owner tests pass without it.
5. First downstream consumer still passes.
6. Any external-use uncertainty is recorded before removal.

Start with private no-caller candidates, then review public candidates from `src-diagnosis.md`. Treat the aux-trigger API as intentionally public unless its workflow is explicitly retired.

Validation

- Full automated suite.
- Notebook import and source-cell checks.
- Smallest practical data-backed checks for affected owners.

Exit criteria

- No function is removed merely because a static search reported zero hits.

## S12 - Migrate consumers and reduce compatibility facades

Objective

Move notebooks and scripts to the final public owners, then decide which old facades remain supported.

Work

1. Update consumers one workflow at a time.
2. Keep notebook cells short and preserve their stage order and configuration names.
3. Add deprecation warnings only when they are actionable and not disruptive to notebook output.
4. Remove facade re-exports only after every active consumer has migrated and the public compatibility decision is documented.
5. Update workflow routers, stage maps, and `symbol-index.md` alongside public changes.

Validation

- Compile every active notebook code cell.
- Execute the smallest representative notebook chain for preprocessing, single-fish analysis, several-fish analysis, and stimulus visualization.
- Render and inspect every affected figure.

Exit criteria

- Active consumers use the intended public owners.
- Remaining facades are deliberate compatibility surfaces, not accidental leftovers.

## S13 - Adopt the permanent `src` organization policy

Objective

Convert lessons from the completed refactor into enforceable repository guidance.

Deliverable

Create `.agents/references/src-organization-policy.md` and link it from `AGENTS.md` and the workflow routers.

Minimum policy content

- ownership map and dependency direction;
- public API and facade rules;
- one canonical implementation per calculation;
- no version-suffixed implementation files;
- computation/I/O/orchestration/plot separation;
- function and module cohesion expectations;
- array shape, unit, NaN, and ordering documentation;
- testing and characterization requirements;
- dead-code review and deprecation process;
- notebook boundary rules;
- canonical output preservation;
- figure screenshot validation;
- documentation update requirements when public behavior changes.

Validation

- The policy agrees with the final code structure rather than describing an aspirational layout that does not exist.
- Router instructions lead future agents to the smallest relevant owner and reference.

Exit criteria

- Future additions have a clear placement and review policy.

## S14 - Final end-to-end validation and closure

Objective

Demonstrate that the reorganization preserved supported workflows.

Validation matrix

- source compile and import suite;
- full unit and characterization suite;
- preprocessing one-plane smoke run;
- single-fish load, alignment, detector, and raster path;
- several-fish matrix, selection, overlap, and summary path;
- stimulus timing and visualization path;
- LME table/model smoke path where dependencies and data permit;
- notebook source-cell compilation;
- representative rendered figures inspected for layout quality;
- canonical output names and schemas checked.

Closure work

1. Update `symbol-index.md`, stage maps, and canonical-output documentation.
2. Clear completed temporary handoffs from `recent-changes.md`.
3. Move any purely historical completed plan material to `.agents/history/` only after the active policy and owner references are sufficient.
4. Record environment or data limitations honestly.

Exit criteria

- Supported workflows pass their validation surfaces.
- No required work remains hidden in a compatibility wrapper or temporary handoff.
- The permanent organization policy is active and discoverable.

## Recommended next action

S07 is next: split `multifish_analysis` by scientific domain while preserving its public facade. S07 has not started.

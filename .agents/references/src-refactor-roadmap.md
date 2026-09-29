# `src/` Refactor Roadmap

Purpose

Provide a durable, ordered, multi-session plan for reorganizing `src/` without losing scientific behavior, notebook compatibility, output contracts, or figure quality.

Use this file when

- Starting or resuming any repository-wide `src/` cleanup.
- Choosing the next safe refactor slice.
- Recording which slice is active or completed across chats and days.
- Deciding what must be validated before a slice can close.

Start with the status tracker and the relevant unfinished slice. Use `src-diagnosis.md` only for dated audit evidence, `refactor-loop-policy.md` for the structural work loop, and the relevant workflow router to locate an owner. Read `code-authoring-checklist.md` before editing Python or notebooks.

## How to use this roadmap

- Work in the listed order unless a dependency-free slice is explicitly selected.
- Complete one slice per coherent change. Do not combine unrelated owner modules merely to reduce the number of commits.
- At the start of a session, identify the slice ID and confirm its prerequisites.
- At the end of a session, update the status table below. Move completed slice plans and completion records into `.agents/history/`, keeping their status and a history link here.
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
| S07 | Split `multifish_analysis` by scientific domain | complete | S06 |
| S08 | Split plotting by figure family | complete | S00-S01, S06-S07 as applicable |
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

## Completed slices

S00-S08 are complete. Their original plans and completion records are preserved in [S00-S08 history](../history/src-refactor-s00-s08.md), outside normal startup reading.

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

S09 is next: separate several-fish workflow, reporting, and pure transforms. S00-S08 are complete; S09 has not started.

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
| S09 | Separate several-fish workflow, reporting, and pure transforms | complete | S06-S08 |
| S10 | Review loaders, extraction modules, and optional dependencies | complete | S00-S03 |
| S11 | Perform evidence-based dead-code retirement | complete | S03-S10 |
| S12 | Migrate consumers and reduce compatibility facades | complete | S03-S11 |
| S13 | Adopt the permanent `src` organization policy | complete | S03-S12 |
| S14 | Run final end-to-end validation and close the refactor | complete | S13 |

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

S00-S11 are complete. The earlier plans and completion records are preserved in [S00-S08 history](../history/src-refactor-s00-s08.md), the [S09 completion record](../history/src-refactor-s09.md), the [S10 completion record](../history/src-refactor-s10.md), and the [S11 completion record](../history/src-refactor-s11.md), outside normal startup reading.

## S09 - Separate several-fish workflow and reporting

Objective

Untangle `reusable_several_fish.py` into orchestration, reporting/export, pure preparation, and plotting calls.

Recommended order

1. Report serialization and notebook export.
2. Cohort loading and preflight checks.
3. Pure response-selection and keep-mask preparation.
4. Diagnostic data preparation.
5. High-level figure orchestration.

Complete. Reporting, cohort loading, pure selection, pure diagnostic and figure-input preparation, and high-level figure orchestration now belong to `src/several_fish_reporting.py`, `src/several_fish_loading.py`, `src/several_fish_selection.py`, `src/several_fish_diagnostics.py`, and `src/several_fish_figures.py`, respectively. `src/reusable_several_fish.py` remains an import-only compatibility facade for active notebooks. See the [S09 completion record](../history/src-refactor-s09.md) for validation and limits.

Validation

- Report folder contents and JSON/CSV schema checks.
- Mock or temporary-path tests for export and save behavior.
- Equality checks for keep masks, row order, and prepared matrices.
- First downstream several-fish notebook run.

Exit criteria

- Scientific transformations can run without reporting or plotting dependencies.
- Reporting code does not own scientific calculations.

## S10 - Review loaders, extraction modules, and optional dependencies

Complete. Loader internals have private stages and fixture contracts;
per-plane extraction has known-output tests; aux-trigger TIFF loading is lazy;
LME and stimulus visualization remain cohesive owners. See the
[S10 completion record](../history/src-refactor-s10.md) for validation and
the real-plane data limit.

## S11 - Evidence-based dead-code retirement

Complete. The standalone dFoF example runner and two unreferenced,
undocumented timing functions were retired after owner and downstream checks.
Documented, tested, and intentional public helpers remain available for S12's
consumer and facade review. The full suite passed 98 tests, including active
notebook import and code-cell checks. No experimental dataset was rerun. See
the [S11 completion record](../history/src-refactor-s11.md) for candidate
decisions, validation, and external-use uncertainty.

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

Complete. Active consumers use the intended public owners, and retained facades
have a documented compatibility decision. The full test suite and representative
real-data workflows passed. See the [S12 completion record](../history/src-refactor-s12.md)
for validation and limits.

## S13 - Adopt the permanent `src` organization policy

Complete. The [organization policy](src-organization-policy.md) is linked from
`AGENTS.md` and the workflow routers and agrees with the post-S12 owners and
compatibility decision. See the [S13 completion record](../history/src-refactor-s13.md)
for the audit, routing checks, and limits.

## S14 - Final end-to-end validation and closure

Complete. All seven checkpoints passed their available validation surfaces.
See the [S14 completion record](../history/src-refactor-s14.md) for the real-data
checks, resolved defects, figures, and execution limits.

Objective

Demonstrate that the reorganization preserved supported workflows.

Validation checkpoints

Keep S14 as one slice. Complete these checkpoints across sessions if needed;
record the result and exact next breakpoint in `recent-changes.md`. If a check
finds a code defect, fix it in the owning module and rerun the affected check.

1. **Foundation:** Compile and import `src`, run the full unit and characterization
   suite, and compile every active notebook code cell.
2. **Preprocessing:** Run one-plane extraction on representative data; check
   per-plane and merged names, shapes, and the first `data_loading` consumer.
3. **Single fish:** Run loading, alignment, significant-trace detection, and
   raster preparation; render and inspect the representative figure.
4. **Several-fish analysis:** Check matrix construction, selection, active
   overlap, and summaries, including fish, neuron-row, and stimulus order.
5. **Several-fish figures and LME:** Render and inspect representative summary
   figures; build the LME response table and smoke-test the model where its
   dependencies and data permit.
6. **Stimulus:** Check trajectory-to-timing interpretation and visualization;
   render and inspect the stimulus figures.
7. **Contracts and closure:** Check canonical filenames, folders, schemas, and
   stage order; update `symbol-index.md`, stage maps, and output documentation
   where needed. Clear completed handoffs, archive historical plan material
   after active references are sufficient, and record any execution limits.

Exit criteria

- Supported workflows pass their validation surfaces.
- No required work remains hidden in a compatibility wrapper or temporary handoff.
- The permanent organization policy is active and discoverable.

## Recommended next action

The repository-wide `src/` refactor is complete. Keep future owner changes
scoped by the [organization policy](src-organization-policy.md) and their first
downstream consumers.

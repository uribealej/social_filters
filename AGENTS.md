# Agent Entrypoint

This repository uses a routed instruction system under `.agents/` so future agents can read the smallest repo-specific guidance first instead of scanning large notebooks. The goal is to route work by real ownership boundaries: reusable logic in `src/`, notebooks as orchestration and reporting, and stimulus scripts as asset-generation wrappers.

## Required startup order
1. Open `.agents/workflows/social-filters-router.md` and select the matching workflow profile. For a repository-wide `src/` refactor, start with the status tracker and relevant slice in `.agents/references/src-refactor-roadmap.md`.
2. Use one matching row in the top router. For documentation-only or test-only work, open the target file directly. For scientific workflow work, open the selected profile, read its row's reference only when needed, then open the named owner or target file. Consult `.agents/references/recent-changes.md` only when resuming incomplete work.
3. Before creating or modifying Python or notebook code, read `.agents/references/code-authoring-checklist.md`.
4. For reusable behavior, inspect the owning `src/` implementation before the calling notebook or script region. For notebook configuration, reporting, or stimulus asset wrappers, inspect the target file directly. Open only the relevant regions of large files.

## Non-negotiable repo rules
- `src/` owns reusable analysis logic, timing semantics, and plotting helpers.
- Notebooks in `notebooks/calcium/` stay orchestration, exploration, and reporting first.
- Files under `archive/` are historical and out of scope unless the user explicitly asks to inspect or restore them.
- When editing notebook workflows, keep cells short and single-purpose, with minimal success output. Move reusable or verbose loading, analysis, and plotting orchestration into the owning `src/` helper as part of the relevant owner slice. Existing preprocessing writer cells remain the batch notebook's output stage until explicitly migrated.
- Stimulus generation scripts in `scripts/stimuli/generation/` are wrappers around experiment-specific asset generation, not downstream timing authority.
- Fix scientific or file-contract semantics at the narrowest owner layer instead of patching downstream notebooks.
- Preserve canonical output names, folder layouts, and stage order unless a task explicitly includes a migration.

## Validation by change
- Documentation-only: inspect the diff and verify affected paths, links, and factual claims against the repository.
- Code behavior or scientific calculation: run a focused known-output test or the smallest practical smoke or data-backed check; do not claim scientific success from static reasoning alone.
- Public helper, writer, or file contract: also check the first downstream consumer and relevant names, shapes, and ordering.
- Notebook behavior: compile affected code cells and run the smallest affected data-backed chain when data are available; report any execution limitation.
- New or materially changed figure: render a screenshot and inspect overlapping labels, legends, ticks, annotations, clipped content, and readability. Iterate until clear; record any environment limitation that prevents inspection.

## Reference files

This is a lookup list, not a required reading list. Read only the references needed for the current task.
- `.agents/workflows/social-filters-router.md` - top-level dispatcher for all repo work.
- `.agents/workflows/calcium-preprocessing-router.md` - dFoF extraction, parameter exploration, merge outputs, and file-ops utilities.
- `.agents/workflows/calcium-analysis-router.md` - experiment loading, alignment, response analysis, and plotting.
- `.agents/workflows/stimulus-authoring-router.md` - trajectory generation, mapping JSONs, timing handoff, and playback wrappers.
- `.agents/references/symbol-index.md` - notebook-callable and script-callable public surface in `src/`.
- `.agents/references/canonical-outputs.md` - authoritative writer stages, file names, and output folders.
- `.agents/references/current-state.md` - mixed-state caveats, legacy entrypoints, and practical warnings.
- `.agents/references/code-authoring-checklist.md` - mandatory compact rules for Python and notebook changes.
- `.agents/references/refactor-rules.md` - ownership and edit-scope rules for this repo.
- `.agents/references/refactor-loop-policy.md` - default slice size, keep-going rules, and handoff expectations.
- `.agents/references/src-diagnosis.md` - dated factual baseline for the repository-wide `src/` architecture audit.
- `.agents/references/src-refactor-roadmap.md` - ordered multi-session `src/` refactor slices, validation gates, and status tracker.
- `.agents/references/src-api-consumer-inventory.md` - active notebook/script consumers and public compatibility contracts for `src`.
- `.agents/references/recent-changes.md` - active incomplete-work handoff only.

## Cross-experiment notebook copies

When creating an analysis notebook for a new experiment from an existing notebook, keep the shared `src/` helper workflow and discover the new experiment's fish IDs, stimulus IDs/order, timing, controls, and paths from its data or configuration—never copy these values blindly. Keep figure-size controls beside their individual plotting calls, preserve the configured stimulus order in summary plots, and validate every code cell plus the smallest practical data-backed run. Do not alter the source experiment notebook.
## Scope note

Keep this file short. The full rationale and publication guidance live in `docs/reproducibility_policy.md`; agents read it only for data-policy, architecture, or publication-level decisions. Operational coding rules, workflow routing, stage maps, output semantics, and current-state warnings live under `.agents/`. Completed plans and logs live under `.agents/history/` and are not part of normal startup.

# Agent Entrypoint

This repository uses a routed instruction system under `.agents/` so future agents can read the smallest repo-specific guidance first instead of scanning large notebooks. The goal is to route work by real ownership boundaries: reusable logic in `src/`, notebooks as orchestration and reporting, and stimulus scripts as asset-generation wrappers.

## Required startup order
1. Open `.agents/workflows/social-filters-router.md`.
2. Follow its dispatch table to the matching workflow profile router.
3. Read the smallest relevant reference file under `.agents/references/`.
4. Open the stage map or `symbol-index.md` only if the smaller reference file is not enough.
5. Open the owning `src/` module before large notebooks or scripts.
6. Open large notebook or script regions only when owner-module context is still insufficient.

## Non-negotiable repo rules
- `src/` owns reusable analysis logic, timing semantics, and plotting helpers.
- Notebooks in `scripts/calcium_analysis/` stay orchestration, exploration, and reporting first.
- Keep notebook cells short and single-purpose, with minimal success output. Move reusable or verbose loading, analysis, and plotting orchestration into the owning `src/` helper; notebook cells should show only the configuration, a brief explanation, and the call.
- Stimulus generation scripts in `scripts/stimuli/` are wrappers around experiment-specific asset generation, not downstream timing authority.
- Fix scientific or file-contract semantics at the narrowest owner layer instead of patching downstream notebooks.
- Preserve canonical output names, folder layouts, and stage order unless a task explicitly includes a migration.
- Validate changes with the smallest practical rerun or smoke check; do not claim success from static reasoning alone.
- For every new or materially changed figure, render a screenshot during validation and inspect it for overlapping labels, legends, ticks, annotations, clipped content, or unreadable layout. Iterate on the figure until the screenshot is clear; record any environment limitation that prevents this check.

## Reference files
- `.agents/workflows/social-filters-router.md` - top-level dispatcher for all repo work.
- `.agents/workflows/calcium-preprocessing-router.md` - dFoF extraction, sweeps, merge outputs, and file-ops utilities.
- `.agents/workflows/calcium-analysis-router.md` - experiment loading, alignment, response analysis, and plotting.
- `.agents/workflows/stimulus-authoring-router.md` - trajectory generation, mapping JSONs, timing handoff, and playback wrappers.
- `.agents/references/symbol-index.md` - notebook-callable and script-callable public surface in `src/`.
- `.agents/references/canonical-outputs.md` - authoritative writer stages, file names, and output folders.
- `.agents/references/current-state.md` - mixed-state caveats, legacy entrypoints, and practical warnings.
- `.agents/references/refactor-rules.md` - ownership and edit-scope rules for this repo.
- `.agents/references/refactor-loop-policy.md` - default slice size, keep-going rules, and handoff expectations.
- `.agents/references/recent-changes.md` - index for workflow-specific handoff logs.

## Cross-experiment notebook copies

When creating an analysis notebook for a new experiment from an existing notebook, keep the shared `src/` helper workflow and discover the new experiment's fish IDs, stimulus IDs/order, timing, controls, and paths from its data or configuration—never copy these values blindly. Keep figure-size controls beside their individual plotting calls, preserve the configured stimulus order in summary plots, and validate every code cell plus the smallest practical data-backed run. Do not alter the source experiment notebook.
## Scope note

Keep this file short. Workflow-specific routing, stage maps, output semantics, current-state warnings, and handoff logs live under `.agents/`.

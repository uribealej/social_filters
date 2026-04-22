# Recent Changes: Calcium Analysis

Use this log for experiment loading, alignment, filtering, classification, plotting, and multi-fish analysis slices.

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
- Slice goal: create the routed guidance docs for calcium-analysis ownership and handoff.
- Passes completed in this session: repo inspection, router creation, stage-map creation, symbol indexing, and current-state documentation.
- What changed: added the calcium-analysis router and reference docs that point agents to `src/` owners before large notebooks.
- What remains broken: known mixed-state issues are documented in `current-state.md`; no analysis code changed here.
- Remaining in-slice work: append future entries when analysis helpers, loader contracts, or notebook handoffs change.
- Next likely breakpoint: first loader, alignment, or plotting task that changes public behavior.
- Rerun implications: none for runtime behavior; docs only.

## 2026-04-22 - Slice 1 merged-output loader hardening
- Slice goal: make merged dFoF loading less brittle around merged-file discovery and missing or compatibility map CSVs.
- Passes completed in this session: routed workflow read, owner-module update in `src/data_loading.py`, canonical and compatibility-path validation, and doc refresh.
- What changed: `src/data_loading.py` now resolves canonical merged filenames by `{prefix}` first, falls back to compatibility merged files when needed, and keeps base experiment loading working when the merged map CSV is missing while skipping map-dependent plane metadata.
- What remains broken: experiments that truly require plane metadata still need a valid merged map CSV with a `plane` column; rebuilding or normalizing merge outputs remains upstream work.
- Remaining in-slice work: none after validation.
- Next likely breakpoint: slice 2 if merge-output generation or extraction-side contracts need refactoring.
- Rerun implications: rerun the smallest `load_2p_experiment` smoke checks on one canonical dataset and one compatibility or missing-map case after future loader changes.

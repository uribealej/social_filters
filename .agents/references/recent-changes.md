# Active Handoff

Purpose

Record only unfinished work that a future agent must understand before continuing. Completed plans and logs are preserved under `../history/` and are not part of normal startup.

Use this file when

- You are resuming partially completed work.
- A change stopped mid-slice with remaining breakage or an unresolved user decision.

## Current handoff

- The repository-organization changes are not yet committed.
- `notebooks/calcium/exp_08_accumulation_ev/01_all_fish_raster.ipynb` has an unresolved local-versus-index difference: the staged organization version preserves the complete 17-cell committed notebook, while the local working notebook has 14 cells and retained outputs. Do not overwrite either version until the user decides how to reconcile them.
- No `src/` refactoring plan is active. Define it with the user before changing analysis modules.

## Maintenance rule

- Add an entry only when work stops incomplete or a user decision remains unresolved.
- Remove the entry when the issue is resolved.
- Move durable completed context to `../history/` only when it will help later work.

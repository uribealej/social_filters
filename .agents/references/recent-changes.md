# Active Handoff

Purpose

Record only unfinished work that a future agent must understand before continuing. Completed plans and logs are preserved under `../history/` and are not part of normal startup.

## Current handoff

None. The repository-wide `src/` refactor is complete through S14; see the [S14 completion record](../history/src-refactor-s14.md). The earlier S12 step log is preserved in [history](../history/src-refactor-s12-step-log.md).

## Maintenance rule

- Add an entry only when work stops incomplete or a user decision remains unresolved.
- Remove the entry when the issue is resolved.
- Move durable completed context to `../history/` only when it will help later work.

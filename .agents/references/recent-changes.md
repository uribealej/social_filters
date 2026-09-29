# Active Handoff

Purpose

Record only unfinished work that a future agent must understand before continuing. Completed plans and logs are preserved under `../history/` and are not part of normal startup.

Use this file when

- You are resuming partially completed work.
- A change stopped mid-slice with remaining breakage or an unresolved user decision.

## Current handoff

S12 is in progress. Step 0 checkpointed completed S09-S11 work in commit
`d35d0d6`. The consumer baseline is
[`src-api-consumer-inventory.md`](src-api-consumer-inventory.md) and its exact
machine contract is `tests/contracts/src_api_consumers.json` (11 active
`src`-using notebooks and 4 scanned scripts). No S12 consumer import has yet
been changed. Start with the shared single-fish notebook migration, then
validate its owning helpers and first downstream notebook chain.

The S11 completion record reports 98 passing tests in
`social_filters_openblas`. This shell's `python` resolves to an unusable Windows
app alias, so the suite could not be rerun at the S12 checkpoint. The staged
checkpoint passed `git diff --cached --check`.

## Maintenance rule

- Add an entry only when work stops incomplete or a user decision remains unresolved.
- Remove the entry when the issue is resolved.
- Move durable completed context to `../history/` only when it will help later work.

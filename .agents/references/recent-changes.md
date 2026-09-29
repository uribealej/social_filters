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
`src`-using notebooks and 4 scanned scripts).

Step 1 migrated `General_analysis_single_fish_reusable.ipynb` from the
`analysis_tools` and `plotting` facades to the narrow analysis and plotting
owners. Its configuration, cell order, call arguments, and output paths are
unchanged. The consumer JSON, inventory, symbol index, analysis stage map, and
router were updated. All 16 notebook code cells compiled after removing the
three IPython magic lines; the exact consumer contract, indexed symbols, owner
definitions, and 11 facade-to-owner re-exports matched. `git diff --check`
passed.

Step 2 migrated the shared several-fish notebook and the Exp 1, 5, and 8
all-fish raster notebooks to their scientific, plotting, cohort, figure, and
reporting owners. The 42 affected code cells compiled, and each notebook's
imports and symbols matched the reviewed machine contract and symbol index.
Comparison with the checkpoint confirmed that all non-setup cells changed only
the owner-qualified call names; settings, call arguments, stage order, saved
outputs, and notebook outputs are unchanged. No facade export was removed.
Across all 16 active notebooks and scripts, 111 source units compiled and the
full consumer inventory matched the contract; the pre-existing raw cell 2
syntax failure in the Exp 8 bout/flicker notebook remains registered.
Continue with the specialized analysis notebooks (S12 step 3).

The S11 completion record reports 98 passing tests in
`social_filters_openblas`. This shell's `python` resolves to an unusable Windows
app alias. The accessible Inkscape Python lacks `IPython`, `pandas`, and
`matplotlib`, so the repository test suite and notebook execution could not
be run here. The configured `D:` experiment and stimulus data paths are also
unavailable. Representative single-fish and several-fish data runs and figure
review remain for an environment with those inputs; no figure implementation
changed in steps 1-2.

## Maintenance rule

- Add an entry only when work stops incomplete or a user decision remains unresolved.
- Remove the entry when the issue is resolved.
- Move durable completed context to `../history/` only when it will help later work.

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
full consumer inventory matched the contract. The Exp 8 bout/flicker syntax
failure was still registered at the end of step 2.

Step 3 migrated the Exp 1 and Exp 8 bout/flicker, Exp 1 static/flicker
recruitment, and Exp 5 LME notebooks to their narrow owners. Exp 8
bout/flicker raw cell 2's extra comma and duplicate load were removed; Cell 03
remains the load stage. Its fish IDs and stimulus order match the maintained
Exp 8 raster notebook, but its configured blocks (`B1`-`B5`) and post window
(32 s) differ from that notebook (`B1`-`B4`, 30 s). Those scientific settings
were preserved pending a data-backed check. The notebook syntax registry and
consumer contract now expect no failures. Across all 16 active notebooks and
scripts, 112 source units compiled, all 69 active symbols matched the index
and owner definitions, and no active consumer imported `analysis_tools`,
`multifish_analysis`, `plotting`, or `reusable_several_fish`. No facade export
was removed. Synthetic notebook-cell tests were updated to call the migrated
owners while retaining separate facade compatibility assertions; all test
modules compiled.

Step 4 audited the remaining preprocessing and stimulus consumers. Their
existing `src` imports already point to `dff_extraction` and
`stimulus_visualization`, and the file-ops, generator, and playback wrappers
have no `src` dependency. The dFoF batch notebook's obsolete external-repo
fallback guidance was removed without changing its import or writer behavior.
All 112 active source units compiled, the exact consumer contract matched, and
the edited notebook kept its cell metadata and outputs. The output names are
unchanged. `current-state.md` now records the corrected Exp 8 syntax and the
outstanding block/timing comparison.
Continue with the S12 compatibility decision (step 5).

The S11 completion record reports 98 passing tests in
`social_filters_openblas`. This shell's `python` resolves to an unusable Windows
app alias. The accessible Inkscape Python lacks `IPython`, `pandas`, and
`matplotlib`, so the repository test suite and notebook execution could not
be run here. The configured `D:` experiment and stimulus data paths are also
unavailable. Representative single-fish, several-fish, and specialized data
runs and figure review remain for an environment with those inputs; no figure
implementation changed in steps 1-4.

## Maintenance rule

- Add an entry only when work stops incomplete or a user decision remains unresolved.
- Remove the entry when the issue is resolved.
- Move durable completed context to `../history/` only when it will help later work.

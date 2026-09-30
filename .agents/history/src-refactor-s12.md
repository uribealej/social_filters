# S12 completion record

S12 migrated active notebook consumers to narrow public owners and documented
the historical compatibility surfaces in
[`src-compatibility-decision.md`](../references/src-compatibility-decision.md).
The five steps were recorded in commits `2a80d1d`, `0fa5b5c`, and `7c7a826`
and the earlier S12 history. No public facade export was removed.

Validation on 2026-09-29 used the `social_filters_updated` Conda environment
and the available data under `C:\Users\suribear\OneDrive - Université de Lausanne\Lab\Data\2p`:

- `python -m unittest discover -s tests -t . -q`: 98 tests passed.
- Real Exp 1 suite2p fluorescence produced finite dFoF with 449 unique ROIs.
- Real Exp 1 single-fish and two-fish cohort loading, alignment, ordered
  several-fish trace preparation, and figure rendering succeeded. The figure
  was inspected for labels, legend, timing markers, and clipping.
- Real Exp 8 single-fish loading and stimulus trajectory inspection succeeded;
  two trajectory figures were rendered and inspected.

These are representative data-backed chains, not full end-to-end notebook
runs. The Exp 8 bout/flicker notebook has settings that require its own
scientific review. Exp 1 and Exp 8 have different experiment designs, so
matching their block counts or stimulus names is not an S12 validation gate.
The final validation is recorded in the [S14 completion record](src-refactor-s14.md).

## Step chronology

1. Migrated the shared single-fish notebook from broad analysis and plotting
   facades to narrow owners without changing its settings or stage order.
2. Migrated the shared several-fish and Exp 1, 5, and 8 raster notebooks;
   compiled their cells and checked the consumer contract.
3. Migrated the specialized bout/flicker, recruitment, and LME notebooks.
   Corrected the pre-existing Exp 8 bout/flicker setup syntax; its block and
   post-window settings remain a separate scientific review item.
4. Audited preprocessing and stimulus consumers. Their existing imports were
   already on their narrow owners; removed obsolete dFoF notebook guidance.
5. Recorded supported historical facade paths and switched remaining internal
   facade calls to direct owners. Kept all public exports and behavior.

This completion record, the cited commits, and the archived
[step log](src-refactor-s12-step-log.md) are the durable handoff for S12.

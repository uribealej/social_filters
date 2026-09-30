# Current State

Purpose

Capture current caveats that can affect active workflows.

Use this file when

- A notebook and a `src/` module seem to disagree.
- A loader or notebook failure suggests a current contract or configuration mismatch.

## Current notes
- `src/data_loading.py` treats the merged map CSV as optional when absent. When
  maps exist, it requires a readable map with a `plane` column and one row per
  merged neuron, preferring a matching canonical map and accepting a matching
  timestamped writer fallback when the canonical copy is stale.
- `src/auxtrigger_extraction.py` imports without `ScanImageTiffReader`; only calls to `extract_aux_trigger_frames` require the package. The full `environment.yml` declares it for aux-trigger workflows.
- `notebooks/calcium/preprocessing/2P_Experiment_FileOps.ipynb` is a utility and migration surface. It is not the authority for scientific extraction semantics or canonical output contracts.
- Stimulus files use existing repo spellings such as `Trayectory_*`. Preserve current file names unless a task explicitly includes a naming migration.
- `notebooks/calcium/exp_08_accumulation_ev/02_bout_flicker_position_onset.ipynb` now compiles after its `stim_order` comma was corrected. Its configured blocks (`B1`-`B5`) and post window (32 s) still differ from the maintained Exp 8 raster notebook (`B1`-`B4`, 30 s); compare these settings against experiment data before treating its scientific output as validated.

## Practical reading rule
- When a notebook-local helper and a `src/` helper overlap, start with the `src/` owner and treat the notebook copy as drift until proven otherwise.
- Ignore `archive/` during normal work. Inspect or restore archived files only when the user explicitly requests it.

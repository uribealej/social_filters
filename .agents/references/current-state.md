# Current State

Purpose

Capture active mixed-state caveats and legacy surfaces so they do not get mistaken for authoritative architecture.

Use this file when

- A notebook and a `src/` module seem to disagree.
- A task touches legacy imports or missing modules.
- A loader or notebook failure suggests partial migration state.

## Current mixed-state notes
- `scripts/calcium_analysis/stimuli_base_analysis.ipynb` contains copied helper implementations that now also exist in `src/`; treat the `src/` modules as the authority unless a task explicitly revives the notebook-local copy.
- `scripts/calcium_analysis/LR_bout_analysis_onefish.ipynb` references missing `src.two_p.*` and `src.utils` modules. Treat this notebook as legacy until the repo restores those modules or migrates the notebook.
- `scripts/calcium_analysis/Exp_1_flickering_analysis.ipynb` still contains early bare-module imports, while the current shared-helper path is `src.*`.
- `src/data_loading.py` now treats the merged map CSV as optional for base experiment loading: core dFoF loading still works without it, while plane metadata is skipped unless a readable merged map with a `plane` column is available.
- `scripts/calcium_analysis/2P_Experiment_FileOps.ipynb` is a utility and migration surface. It is not the authority for scientific extraction semantics or canonical output contracts.
- Stimulus files use existing repo spellings such as `Trayectory_*`. Preserve current file names unless a task explicitly includes a naming migration.

## Practical reading rule
- When a notebook-local helper and a `src/` helper overlap, start with the `src/` owner and treat the notebook copy as drift until proven otherwise.

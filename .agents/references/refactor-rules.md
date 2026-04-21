# Refactor Rules

Purpose

Define the repo-specific ownership and edit-scope rules that keep reusable logic in the correct layer.

Use this file when

- A notebook is growing helper code.
- A task could be fixed in either a notebook or a `src/` module.
- You are deciding where new logic should live.

## Ownership rules
- New reusable analysis logic belongs in `src/`.
- Plot construction and reusable raster or mean-trace helpers belong in `src/plotting.py`.
- Timing extraction and log-to-trace semantics belong in `src/stimuli_timeline.py`.
- Experiment bundle assembly and cache discovery belong in `src/data_loading.py`.
- Response filtering, classification, and derived metrics belong in `src/analysis_tools.py`.
- The significant-trace and rasterization model belongs in `src/significant_traces.py`.

## Notebook and script rules
- Analysis notebooks remain orchestration, exploration, and reporting layers.
- Do not add new notebook-local helpers when the logic should be callable across notebooks.
- Stimulus generation scripts may keep experiment-specific wrapper logic, but reusable timing semantics still belong upstream in `src/stimuli_timeline.py`.
- Utility notebooks such as `2P_Experiment_FileOps.ipynb` may manage paths and copy operations, but they do not own scientific semantics.

## Fix-at-owner rules
- Fix output semantics at the writer stage, not in a downstream notebook that loads the output.
- Fix timing semantics at `src/stimuli_timeline.py` or the generator script that wrote the asset, not in plotting code.
- Do not patch downstream consumers just to compensate for an upstream naming or contract bug.

## Preservation rules
- Preserve canonical output filenames, folder names, and stage order unless the task explicitly includes a migration.
- Preserve notebook stage flow and established config variable names unless the task is intentionally reorganizing the notebook.
- If a public helper or output contract changes, update the relevant stage map, `symbol-index.md`, and output docs in the same change.

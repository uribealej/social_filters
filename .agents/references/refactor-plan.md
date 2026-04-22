# Prioritized Small-Slice Refactor Plan for Calcium Preprocessing and Stimulus Authoring

## Summary

This repo is already trying to enforce a cleaner split: reusable scientific logic in `src/`, notebooks as orchestration/reporting, and stimulus scripts as experiment-specific wrappers. The safest refactor path is to work in small slices that move duplicated or fragile behavior to the current owner layer without changing canonical filenames, folder layout, or timing semantics.

Priority order:
1. Fix owner-layer fragility that can break downstream workflows.
2. Remove duplicated scientific logic from notebooks by extracting or consolidating into `src/`.
3. Standardize wrapper/orchestration code while preserving current outputs and names.
4. Leave naming migrations and larger notebook redesigns for a later explicit migration project.

## Calcium Preprocessing

### Slice 1: Harden merged-output loading contracts
- Goal: make downstream preprocessing/analysis less brittle around merged output discovery, especially map CSV presence and file lookup.
- Owner layer: `src/` loader and writer contract boundary.
- Involved files or folders: `src/data_loading.py`, `scripts/calcium_analysis/DeltaFF_batch_pipeline.ipynb`, `.agents/references/canonical-outputs.md`.
- Must remain stable: merged folder `03_analysis/functional/suite2P/merged_dFoF`; filenames `{prefix}_dFoF_merged.npy`, `{prefix}_dFoF_merged_filtered_roi_indices.npy`, `{prefix}_dFoF_merged_map.csv`.
- Smallest validation step: run the smallest load path on one experiment with complete merged outputs and one with missing or older map variants; confirm canonical files still load when present and failures are clearer when absent.
- Risk level: High.

### Slice 2: Consolidate per-plane extraction semantics in `src/dff_extraction.py`
- Goal: keep all reusable dF/F math, filtering, and ROI-retention logic in one owner module and prevent notebooks from drifting.
- Owner layer: `src/dff_extraction.py`.
- Involved files or folders: `src/dff_extraction.py`, `scripts/calcium_analysis/DeltaFF_batch_pipeline.ipynb`, `scripts/calcium_analysis/Baseline_Evaluation_single_plane.ipynb`, `scripts/calcium_analysis/STD_Zcore_threshold_analyis.ipynb`.
- Must remain stable: array orientation conventions, retained ROI index meaning, per-plane output names, baseline/tau/percentile parameter behavior unless explicitly changed.
- Smallest validation step: rerun one plane through `process_suite2p_fluorescence` and compare output shapes and retained-index behavior against the current notebook path.
- Risk level: Medium-High.

### Slice 3: Extract sweep-only repeated setup from preprocessing notebooks
- Goal: reduce notebook duplication in baseline/threshold sweeps without moving experiment-specific reporting out of notebooks.
- Owner layer: notebook orchestration plus small reusable helper layer in `src/` only if reused across multiple notebooks.
- Involved files or folders: `scripts/calcium_analysis/Baseline_Evaluation_single_plane.ipynb`, `scripts/calcium_analysis/STD_Zcore_threshold_analyis.ipynb`, possibly `src/dff_extraction.py`.
- Must remain stable: sweep parameter grids, summary CSV/figure content, notebook-driven reporting flow.
- Smallest validation step: rerun a single parameter combination in each notebook and verify the same summary row/diagnostic figure is produced.
- Risk level: Medium.

### Slice 4: Isolate file-ops utilities from scientific semantics
- Goal: keep `2P_Experiment_FileOps.ipynb` as a migration/copy utility only, not a second place where preprocessing contracts are defined.
- Owner layer: utility notebook only.
- Involved files or folders: `scripts/calcium_analysis/2P_Experiment_FileOps.ipynb`, `.agents/references/current-state.md`.
- Must remain stable: experiment tree copy behavior and existing storage layout assumptions.
- Smallest validation step: dry-run or inspect one experiment-tree copy path and confirm no scientific output naming rules were moved into the utility notebook.
- Risk level: Low-Medium.

### Slice 5: Clarify auxiliary trigger ownership
- Goal: ensure TIFF metadata trigger parsing stays in `src/auxtrigger_extraction.py` and is not reimplemented downstream.
- Owner layer: `src/auxtrigger_extraction.py`.
- Involved files or folders: `src/auxtrigger_extraction.py`, any preprocessing notebook cells that manually parse TIFF metadata.
- Must remain stable: trigger frame extraction semantics and TIFF metadata assumptions.
- Smallest validation step: run aux-trigger extraction on one known TIFF and compare extracted trigger frames with current behavior.
- Risk level: Medium.

## Stimulus Authoring

### Slice 6: Make `src/stimuli_timeline.py` the single timing authority
- Goal: centralize downstream timing semantics so notebooks and playback wrappers do not redefine motion onset, duration, or stimulus trace logic.
- Owner layer: `src/stimuli_timeline.py`.
- Involved files or folders: `src/stimuli_timeline.py`, `scripts/stimuli/*.py`, `scripts/stimuli/plots_stimuli.ipynb`, `.agents/references/canonical-outputs.md`.
- Must remain stable: interpretation of existing trajectory CSVs, downstream stimulus-trace shapes, current file naming.
- Smallest validation step: run timing extraction on one existing `*_trajectory.csv` from each stimulus family and confirm the same motion-start and total-duration values currently expected by downstream analysis.
- Risk level: High.

### Slice 7: Normalize trajectory-writer structure across stimulus scripts
- Goal: make flicker, rocking, and generic trajectory scripts follow the same internal structure while staying experiment-specific wrappers.
- Owner layer: `scripts/stimuli/` wrapper scripts.
- Involved files or folders: `scripts/stimuli/Trayectory_flicker.py`, `scripts/stimuli/Trayectory_rocking_stimuli.py`, `scripts/stimuli/trayectory_stimuli.py`.
- Must remain stable: generated `*_trajectory.csv` contents, adjacent parameter files, current spelling-based filenames like `Trayectory_*`.
- Smallest validation step: regenerate one small output CSV per script and compare columns, row counts, and timing-relevant values with the current version.
- Risk level: Medium-High.

### Slice 8: Separate config data from generation logic
- Goal: keep JSON mapping/package files as configuration inputs and reduce hard-coded experiment settings in stimulus scripts.
- Owner layer: script config boundary in `scripts/stimuli/`.
- Involved files or folders: `scripts/stimuli/*.json`, `scripts/stimuli/*.py`.
- Must remain stable: JSON filenames, key names already consumed by scripts, package/mapping layout.
- Smallest validation step: run one script using an existing JSON config and confirm it writes the same asset set as before.
- Risk level: Medium.

### Slice 9: Keep playback wrapper separate from trajectory semantics
- Goal: ensure `try_projection.py` only handles display/playback and timing-log capture, not motion-definition logic.
- Owner layer: `scripts/stimuli/try_projection.py`.
- Involved files or folders: `scripts/stimuli/try_projection.py`, `src/stimuli_timeline.py`.
- Must remain stable: playback invocation path and optional `stimulus_timing_log.csv` output behavior.
- Smallest validation step: run the smallest playback/timing-log path available without changing generated trajectory assets; verify timing log shape and column presence remain unchanged.
- Risk level: Medium.

### Slice 10: Reduce inspection-notebook drift
- Goal: keep `plots_stimuli.ipynb` focused on visualization/inspection, not timing authority or CSV rewriting logic.
- Owner layer: notebook orchestration.
- Involved files or folders: `scripts/stimuli/plots_stimuli.ipynb`, `src/stimuli_timeline.py`.
- Must remain stable: current inspection figures and manual exploratory workflow.
- Smallest validation step: rerun one plotting path on an existing trajectory CSV and confirm the same visual interpretation is available without notebook-local timing rules.
- Risk level: Low-Medium.

## Cross-Workflow Ownership Slices

### Slice 11: Document and enforce current public helper surface
- Goal: align refactors with the existing routed guidance so future notebook edits start from the right owner module.
- Owner layer: repo guidance docs.
- Involved files or folders: `.agents/references/symbol-index.md`, `.agents/references/canonical-outputs.md`, relevant workflow router docs.
- Must remain stable: current public helper names unless a refactor explicitly changes them and updates the docs in the same slice.
- Smallest validation step: for each refactored slice, confirm the owning helper and canonical output are still correctly described in the guidance docs.
- Risk level: Low.

### Slice 12: Identify repeated notebook helpers for later extraction backlog
- Goal: create a bounded backlog of repeated notebook logic without doing a large migration all at once.
- Owner layer: workflow-level planning and future `src/` extraction candidates.
- Involved files or folders: preprocessing notebooks in `scripts/calcium_analysis/` and stimulus inspection notebooks/scripts in `scripts/stimuli/`.
- Must remain stable: notebook execution order, reporting cells, and experiment-specific configuration style.
- Smallest validation step: produce a duplicate-helper inventory grouped by “extract now”, “keep notebook-local”, and “legacy only”.
- Risk level: Low.

## Validation Strategy

- Prefer one experiment or one plane for preprocessing validation.
- Prefer one generated trajectory CSV per stimulus family for stimulus validation.
- After any writer-stage or timing change, verify the first downstream consumer immediately:
  `src/data_loading.py` for merged outputs, `src/stimuli_timeline.py` for stimulus timing.
- Do not rename canonical files or folders unless a later migration slice explicitly covers it.

## Assumptions and Defaults

- Default refactor direction: move reusable logic toward `src/`, keep notebooks and stimulus scripts as orchestration/wrappers.
- Existing misspellings and filenames such as `Trayectory_*`, `count_nuerons.ipynb`, and current canonical output names stay unchanged in these slices.
- Mixed-state and legacy surfaces remain readable but not authoritative unless a slice explicitly revives them.
- Slice sizing target: each slice should be implementable and validated independently in a single focused pass.

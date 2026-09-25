# `src/` Architecture Diagnosis

Purpose

Preserve the read-only architecture audit of `src/` performed on 2026-09-24 so future refactor sessions can work from the same factual baseline without repeating the full inspection.

Use this file when

- Planning or resuming a repository-wide `src/` refactor.
- Deciding which module owns a function or which large module should be split next.
- Reviewing duplicate, legacy, or apparently unused functions.
- Checking the baseline risks that must be protected by tests before moving code.

Pair this diagnosis with `src-refactor-roadmap.md`. The diagnosis records what was observed; the roadmap defines the ordered work.

## Scope and method

The audit covered active Python modules under `src/` and their references from active notebooks and scripts. Files under `archive/` and `.agents/history/` were intentionally excluded.

The inspection included:

- module and function inventory;
- line counts and rough function-size/branching checks;
- internal `src` dependency mapping;
- references from Python files and notebook source cells;
- duplicate top-level function names and implementation similarity;
- compile and module-import smoke checks;
- unused-import and no-reference candidate checks.

Static reference results are evidence for review, not proof that a function is safe to delete. Dynamic notebook use, external consumers, and scientific compatibility must be checked before removal.

## Baseline summary

- `src/` contains 12 substantive Python modules plus an empty `src/__init__.py`.
- Total size at inspection time was approximately 12,967 lines.
- There was no active `tests/` directory.
- Every source file compiled successfully.
- Eleven of the twelve substantive modules imported successfully in the active `social_filters_user` environment.
- `src.auxtrigger_extraction` failed to import because `ScanImageTiffReader` was not installed in that environment, although it is declared under the pip dependencies in `environment.yml`.
- Notebook consumers mostly import whole modules through aliases such as `at`, `mfa`, `plott`, and `rsf`. Preserving those module entrypoints during the refactor will substantially reduce migration risk.

## Module inventory

| Module | Approximate lines | Top-level functions | Diagnosis |
| --- | ---: | ---: | --- |
| `analysis_tools.py` | 2,400 | 33 | Alignment, response metrics, classification, reliability, inspection utilities, and plotting are mixed together. |
| `auxtrigger_extraction.py` | 46 | 1 | Small and cohesive, but its optional runtime dependency is not importable in the current environment. |
| `data_loading.py` | 413 | 7 | Reasonably bounded, but contains a duplicate timing-normalization implementation and a long high-level loader. |
| `dff_extraction.py` | 359 | 11 | Mostly cohesive preprocessing owner; public/private boundaries and focused tests are still needed. |
| `lme_feature_decomposition.py` | 473 | 8 | Cohesive feature/model workflow with one long validation function. |
| `multifish_analysis.py` | 2,396 | 46 | Contains several independent scientific domains: bout-position, static-flicker recruitment, matrix construction, specificity, similarity, permutations, and overlap. |
| `plotting.py` | 3,038 | 44 | Largest module; combines many unrelated figure families, sorting helpers, stimulus markers, and substantial commented legacy code. |
| `reusable_several_fish.py` | 1,268 | 26 | Mixes cohort loading, reporting/export, pure transformations, analysis orchestration, and plotting wrappers. |
| `significant_traces.py` | 738 | 14 | V1 Romano-style event detector plus diagnostic plotting; still supplies helpers used by the main single-fish notebook. |
| `significant_traces_v2.py` | 1,080 | 18 | Modified detector that duplicates much of V1 and also contains version-comparison tooling. |
| `stimuli_timeline.py` | 558 | 8 | Correct timing owner, but includes older apparently unused timing functions and some local imports. |
| `stimulus_visualization.py` | 198 | 7 | Small, cohesive, and already close to the desired structure. |

## Dependency direction observed

The active internal imports were relatively shallow:

```text
stimuli_timeline ──► data_loading ──► reusable_several_fish
analysis_tools   ──► data_loading ──► reusable_several_fish
analysis_tools   ──► multifish_analysis ──► reusable_several_fish
stimuli_timeline ──► plotting ──► reusable_several_fish
stimuli_timeline ──► stimulus_visualization
```

This is a useful starting point. The refactor should preserve a one-way flow from pure scientific calculation toward I/O, orchestration, and figures, and should avoid introducing circular imports.

## Confirmed duplication and mixed-version behavior

### Timing normalization

`transform_stimuli_duration` exists in both:

- `src/data_loading.py`;
- `src/stimuli_timeline.py`.

The implementations perform the same downstream normalization. `src/stimuli_timeline.py` is the documented owner, so `data_loading.py` should eventually delegate to it while temporarily preserving its old entrypoint for compatibility.

### Significant-trace detector

`significant_traces.py` and `significant_traces_v2.py` share 13 public function names. Four inspected implementations were textually identical and several others were highly similar, while the main pipeline and rasterization functions contained meaningful changes.

The main single-fish notebook currently imports:

- `clean_binary_raster_columns` and `plot_dff_and_raster` from V1;
- `compute_noise_model_romano_fast_modular` from V2.

Therefore neither file is presently safe to delete. The two versions require characterized comparison data and an explicit compatibility strategy before consolidation.

## Responsibility problems

### `analysis_tools.py`

The module currently combines:

- trial alignment;
- response-window resolution;
- AUC and peak metrics;
- response and selectivity classification;
- reliability filtering with file output and plots;
- motion-delta analysis;
- debugging/inspection utilities;
- Venn and raster diagnostic plotting.

Several functions are longer than 130 lines. `compute_static_flicker_trial_metrics`, `filter_neurons_by_trial_reliability`, `classify_responses_from_raster`, and `zscore_dfof_from_prestim_baseline` are especially important characterization targets.

### `multifish_analysis.py`

The module contains multiple domains that can change independently:

- bout-flicker position analysis;
- static-flicker recruitment and statistics;
- per-fish and pooled matrix construction;
- stimulus specificity and selectivity;
- stimulus-vector similarity;
- segment-selectivity permutations;
- active-neuron overlap and trace diagnostics.

This module should be split by scientific concept rather than merely by file length.

### `plotting.py`

The module contains:

- bout-position figures;
- static-flicker figures;
- stimulus style discovery;
- similarity figures;
- specificity figures;
- LME figures;
- motion-delta figures;
- diagnostic rasters;
- single-fish sorted chunks and means;
- all-fish flat rasters.

It also contains more than 600 comment-only lines, including large commented implementations, an unused `TelnetServer` import, and a repeated `gridspec` import. Plotting families should be separated only after figure characterization and screenshot validation exist.

### `reusable_several_fish.py`

The module mixes four layers:

- cohort loading and preflight checks;
- report serialization and notebook export;
- pure data preparation and filtering;
- high-level plotting orchestration.

Report/export code and scientific data preparation should not remain in the same owner indefinitely.

## Candidate unused or legacy code

The following had no active external code references at inspection time. They must not be removed solely from this list.

Higher-confidence private candidates with no observed callers:

- `plotting._natural_sort_key`;
- `multifish_analysis._exact_sign_flip_pvalue`.

Public or potentially interactive candidates requiring stronger confirmation:

- `analysis_tools.inspect_obj`;
- `analysis_tools.classify_responses_from_raster`;
- `analysis_tools.compute_left_right_index`;
- `analysis_tools.build_neuron_order_groupwise_onset`;
- `analysis_tools.plot_venn_3stim`;
- `multifish_analysis.classify_stimulus_specificity_summary_table`;
- `multifish_analysis.build_stimulus_specificity_neuron_order`;
- `multifish_analysis.build_segment_selectivity_permutation_summary`;
- `plotting.plot_bout_flicker_position_figure`;
- `plotting.plot_static_flicker_category_proportions`;
- `plotting.plot_stimulus_specificity_summary`;
- `plotting.save_sorted_rasters_for_all_modes`;
- `significant_traces_v2.compare_significant_trace_versions`;
- `stimuli_timeline.get_radius_timing`;
- `stimuli_timeline.get_stimulus_timing`.

`auxtrigger_extraction.extract_aux_trigger_frames` also had no active repository caller, but it is a documented preprocessing API and should be treated as intentionally available unless its workflow is formally retired.

## Documentation and API observations

- Function docstring coverage is generally good.
- Most public functions do not have complete type annotations; the repository policy prefers documenting types, shapes, and units in Google-style docstrings rather than requiring inline annotations.
- Several modules lack the required module-level purpose docstring.
- `src/__init__.py` is empty, so public API boundaries are currently expressed mainly through notebook usage and `.agents/references/symbol-index.md`.
- Some functions listed as public in `symbol-index.md` have no current active code caller. Public documentation is still part of the compatibility contract and must be considered before removal.

## Main refactor risks

1. **Scientific drift:** Small changes to alignment, window selection, detector thresholds, axis order, or NaN handling could alter results without raising an error.
2. **Notebook-only API use:** Conventional static analyzers can miss calls embedded in notebook JSON or interactive use.
3. **Output-contract drift:** Renaming cache files, folders, keys, columns, or return tuple positions could break downstream notebooks.
4. **V1/V2 ambiguity:** Consolidating the significant-trace implementations without named compatibility modes could silently select different scientific behavior.
5. **Figure regressions:** Moving plotting helpers can change axes, ordering, colors, labels, or save behavior even if code still runs.
6. **Environment mismatch:** Optional preprocessing dependencies can make an otherwise unrelated import smoke test fail.
7. **Over-large changes:** Moving multiple scientific domains in one change would make failures difficult to localize or review.

## Safety conclusions

- Do not begin with a directory-wide physical reorganization.
- First add characterization tests around current behavior and output contracts.
- Refactor one owner slice at a time, following `refactor-loop-policy.md`.
- Preserve current module entrypoints as compatibility facades while internals move.
- Do not delete public or apparently unused scientific helpers until reference search, characterization, and downstream validation all agree.
- Do not mix scientific behavior changes with structural moves. If behavior must change, isolate and document it as a separate slice.
- Use `src-refactor-roadmap.md` as the ordered execution and handoff record.

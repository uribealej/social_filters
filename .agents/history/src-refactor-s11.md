# S11 - Evidence-based dead-code retirement

Started 2026-09-29 with an inventory and classification, before making deletion
decisions. It supersedes the candidate-use observations in
`src-diagnosis.md` where the S06-S10 splits changed ownership or tests.

## Evidence checked

- Searched literal symbol references in active `src/`, `notebooks/`, `scripts/`,
  and `tests/`; excluded `archive/` under the repository scope rule.
- Checked the current owner/facade imports, `symbol-index.md`,
  `src-api-consumer-inventory.md`, and the S07/S08 characterization contracts.
- A repository search cannot establish whether external notebooks or interactive
  sessions import a public name. Record that uncertainty before any removal.
- Contract tests exercise some functions without an active notebook call.
  Their snapshots are evidence of behavior and compatibility, not proof that
  the function is needed in the current experimental workflow.

## Candidate classification

| Candidate from diagnosis | Current evidence | Step 1 disposition |
| --- | --- | --- |
| `plotting._natural_sort_key` | Canonical `plotting_common._natural_sort_key` is called twice within `plotting_common`; `plotting` re-exports it and the S08 contract records that export. | Keep implementation. Review facade alias in S12. |
| `multifish_analysis._exact_sign_flip_pvalue` | Defined in `static_flicker_analysis`, re-exported by the facade, and called by `test_multifish_split`; the S07 contract records the facade export. No owner call was found. | Compatibility-bound private-looking name; review with the static-flicker test/contract, not as an unguarded private deletion. |
| `analysis_tools.inspect_obj` | Defined in `analysis_io` and explicitly exported by `analysis_tools.__all__`; no active notebook/script or internal caller found. | Public facade review; external interactive use is unknown. |
| `analysis_tools.classify_responses_from_raster` | Defined in `response_classification`, listed in `symbol-index.md`, and explicitly exported by `analysis_tools.__all__`; no active notebook/script or internal caller found. | Preserve documented API pending compatibility decision. |
| `analysis_tools.compute_left_right_index` | Same documented facade/owner status as the preceding row; no active notebook/script or internal caller found. | Preserve documented API pending compatibility decision. |
| `analysis_tools.build_neuron_order_groupwise_onset` | Same documented facade/owner status as the preceding row; no active notebook/script or internal caller found. | Preserve documented API pending compatibility decision. |
| `analysis_tools.plot_venn_3stim` | Implemented in `plotting_reliability`, exported through `plotting` and `analysis_tools`, and exercised by S08 figure characterization and the S06 facade test. | Keep while reviewing the figure contract and public facade decision. |
| `multifish_analysis.classify_stimulus_specificity_summary_table` | Defined in `stimulus_specificity`, listed in `symbol-index.md`, and exercised by the S07 characterization fixture; no active notebook/script call found. | Preserve documented and tested API pending compatibility decision. |
| `multifish_analysis.build_stimulus_specificity_neuron_order` | Same owner, documentation, and S07 fixture status as the preceding row. | Preserve documented and tested API pending compatibility decision. |
| `multifish_analysis.build_segment_selectivity_permutation_summary` | Defined in `stimulus_similarity`, listed in `symbol-index.md`, and exercised by S07 characterization plus seeded tests; no active notebook/script call found. | Preserve documented and tested API pending compatibility decision. |
| `plotting.plot_bout_flicker_position_figure` | Defined in `plotting_bout_position`; S08 characterization and empty-input tests call it, although active notebooks use the separate `plot_bout_flicker_position_cell06_style`. | Figure-contract review before any retirement. |
| `plotting.plot_static_flicker_category_proportions` | Defined in `plotting_static_flicker`, listed in `symbol-index.md`, and exercised by S08 characterization; no active notebook/script call found. | Preserve documented and tested API pending compatibility decision. |
| `plotting.plot_stimulus_specificity_summary` | Defined in `plotting_specificity`, listed in `symbol-index.md`, and exercised by S08 characterization; no active notebook/script call found. | Preserve documented and tested API pending compatibility decision. |
| `plotting.save_sorted_rasters_for_all_modes` | Defined in `plotting_single_fish`; S08 contract and save-behavior test cover it, although the active single-fish notebook uses `plot_sorted_chunks_single_mode`. | Figure/output-contract review before any retirement. |
| `significant_traces_v2.compare_significant_trace_versions` | V2 facade wrapper delegates to the documented canonical `significant_trace_detection.compare_significant_trace_versions`; no active notebook/script call found. | Keep canonical helper; decide facade exposure with S12 compatibility review. |
| `stimuli_timeline.get_radius_timing` | Publicly named function with no found `src/`, active notebook/script, or test caller; absent from `symbol-index.md`. | Retired in Step 2 after timing-owner and loader-consumer checks. External use remains unknown. |
| `stimuli_timeline.get_stimulus_timing` | Same no-caller and undocumented status as `get_radius_timing`; it implemented a distinct trajectory interpretation. | Retired in Step 2 after timing-owner and loader-consumer checks. External use remains unknown. |
| `auxtrigger_extraction.extract_aux_trigger_frames` | Documented intentional preprocessing API; its own `__main__` and S10 tests call it. | Keep unless the aux-trigger workflow is explicitly retired. |

## Additional S10 handoffs and boundary cases

| Candidate | Current evidence | Step 1 disposition |
| --- | --- | --- |
| `dff_extraction.py` standalone `__main__` | Hard-coded historical data path; wrote unprefixed example outputs, while `DeltaFF_batch_pipeline.ipynb` is the canonical per-plane writer. No active repository workflow invoked the standalone runner. | Retired in Step 2; the canonical writer and output names are unchanged. |
| `dff_extraction` loading, baseline, and ROI-filter helpers | Called by `process_suite2p_fluorescence` and/or directly by S10 known-output tests; `symbol-index.md` also records their callable compatibility. | Keep implementations; review public naming only through compatibility policy. |
| `data_loading.transform_stimuli_duration` | Delegating compatibility wrapper; canonical timing implementation is used by `load_2p_experiment`, and S03 tests compare the two paths. | S12 facade decision, not an S11 calculation deletion. |
| `significant_traces` and `significant_traces_v2` modules | S05 compatibility facades with documented legacy/current behavior. | S12 facade decision. |

## Step 2 result and remaining gates

- Removed the standalone dFoF example runner and its now-unused `json` and
  `time` imports. The `process_suite2p_fluorescence` gateway and the batch
  notebook's prefixed writer were unchanged. Updated `canonical-outputs.md`.
  Focused extraction, source import, notebook compilation, and active consumer
  tests passed: 10 tests in `social_filters_openblas`.
- Removed `get_radius_timing` and `get_stimulus_timing` as a separate timing-owner
  slice. They used movement-end interpretations distinct from the active
  `get_motion_timing_simple` loader path. Timing and loader baseline tests
  passed before removal (10 tests); timing, loader, source import, notebook
  compilation, and active consumer tests passed afterward (16 tests).
- No real Suite2p plane or experimental calcium dataset was rerun in this
  pass. The focused extraction test uses a synthetic plane;
  the timing and loader tests use synthetic trajectory/data fixtures. No
  canonical file contract or scientific result in an active path changed.
- These two timing names were public-looking but undocumented. External use
  cannot be ruled out by repository search, so removal can break an external
  import. Their prior implementations remain recoverable from version history.

## Step 3 public review and closure

- Retain `analysis_tools.inspect_obj` and `plot_venn_3stim`. The first is an
  explicit `analysis_tools.__all__` export with uncertain interactive use; the
  second is an S06/S08 tested export. Neither has sufficient evidence for
  public retirement in S11.
- Retain the three documented `analysis_tools` response-classification helpers,
  the three documented `multifish_analysis` specificity/segment helpers, and
  the documented plotting summary helpers. Their public and/or
  characterization contracts remain supported even without active notebook
  calls.
- Retain `plot_bout_flicker_position_figure` and
  `save_sorted_rasters_for_all_modes` because S08 figure and save contracts
  exercise them. Retain `multifish_analysis._exact_sign_flip_pvalue` as a
  tested historical facade export. Keep the internally called
  `plotting_common._natural_sort_key`; its facade alias is an S12 decision.
- Retain the canonical significant-trace comparison, the intentional
  aux-trigger API, and the lower-level dFoF helpers. S12 will decide whether
  old comparison, timing, plotting, analysis, and multifish facade exports
  remain after consumer migration. No S07/S08 snapshot was changed.
- Full validation in `social_filters_openblas`:
  `python -m unittest discover -s tests -t . -q` passed all 98 tests.
  The suite includes source imports and compilation, active notebook import
  contracts, notebook source-cell compilation, focused owner fixtures, and
  first loader/preprocessing consumers. `git diff --check` found no whitespace
  errors. Existing seaborn pending-deprecation warnings did not fail tests.
- Validation was fixture-backed; no real Suite2p plane or experimental calcium
  dataset was rerun. No figure implementation changed in S11, so no new figure
  screenshot was required. External use of the two retired, undocumented
  timing names remains unknown, as recorded above.

S11 is complete. S12 is the next roadmap slice.

# `src` API and Consumer Inventory

Purpose

Record the active notebook/script consumers of `src` and the compatibility contracts that later refactor slices must preserve.

Use this file when

- Moving or renaming a public helper.
- Deciding whether a compatibility facade is required.
- Checking whether a function is active, internal, documented-but-uncalled, or a retirement candidate.
- Selecting the first downstream consumer for validation.

This inventory was established on 2026-09-25 during S01 of the [completed refactor roadmap](../history/src-refactor-roadmap.md). The exact machine-readable contract is `tests/contracts/src_api_consumers.json`, and `tests/contracts/test_src_api_consumers.py` fails when an active consumer or referenced symbol changes without review.

## Classification language

- **External-active:** referenced by an active notebook or script. Preserve its import path and observable contract until consumers migrate.
- **Internal-active:** called by another `src` module or by another helper in its owner module. It may move internally, but callers must migrate in the same slice.
- **Active compatibility facade:** external-active import path whose implementation has moved to a narrower owner. Preserve the import until consumers migrate.
- **Retained compatibility facade:** historical public import path with no active notebook/script caller; keep its documented exports until the compatibility decision is recorded.
- **Active facade candidate:** external-active mixed-responsibility module whose implementation is scheduled to move.
- **Documented, currently uncalled:** listed in `symbol-index.md` but not called by an active notebook/script. It is not dead merely because it is currently uncalled.
- **Deprecation review candidate:** duplicated, accidentally re-exported, or unreferenced code requiring per-symbol evidence and a documented compatibility decision before retirement.
- **Private implementation:** underscore-prefixed helper or a helper deliberately excluded from the stable notebook API.

## Audit coverage

- 19 active notebooks are covered by the consumer inventory, including Exp 8 notebooks 03 (control-referenced onset comparisons) and 04 (population trajectories).
- 4 active Python scripts under `scripts/` were scanned with the Python AST.
- 18 notebooks currently import or call `src`.
- `notebooks/calcium/preprocessing/2P_Experiment_FileOps.ipynb` currently has no `src` import.
- None of the 4 stimulus generation/playback scripts currently imports `src`.
- All active notebook code cells now compile after S12 step 3 corrected the pre-existing syntax error in raw cell 2 of `notebooks/calcium/exp_08_accumulation_ev/02_bout_flicker_position_onset.ipynb`.
- Files under `archive/` and `.agents/history/` were excluded.

S12 step 4 checked the remaining preprocessing and stimulus consumers against
their owner and output stages. The dFoF batch notebook already calls
`src.dff_extraction.process_suite2p_fluorescence`; its import guidance now
points only to this repository. The file-ops notebook has no reusable `src`
call. The stimulus inspection notebook calls `src.stimulus_visualization`
directly. The three trajectory generators and playback script intentionally
remain experiment-specific asset wrappers without `src` imports; downstream
timing interpretation belongs to `src.stimuli_timeline`. No consumer import or
machine-contract symbol changed in this audit.

## Consumer overview

`notebooks/calcium/shared/General_analysis_single_fish_reusable.ipynb` resolves
its Lab root with `src.data_loading.resolve_lab_root()` in Cell 02 and derives
both `Data/2p` and `Data/stimuli` from it, supporting either workstation without
editing absolute paths.

Exp 8 `04_population_trajectories.ipynb` independently loads the maintained cohort
and four acquisition blocks. `build_population_trajectory_analysis` retains its
historical per-fish PCA default plus pooled/both options. Pooled neuron features
use 1/sqrt(number_of_fish * neurons_in_fish) weights to balance neuron counts,
not response energy. `build_joint_population_pca` intersects original finite
neuron IDs across sides within fish, audits exclusions and fits one actual PCA
across side/condition/time rows; independently rotated PCs are never overlaid.
Default fit: six conditions per side in [0, 8) s from the common 8 s program
anchor, 192 joint rows, projected through 0–14 s. Default display: Control,
Seg 2, Seg 1, Seg 1/3/5 and one-position flicker, with rocking selectable.
`DISPLAY_STIMULI` drives one mean-only overlay per side and optional left/right
overlays. Individual sides retain their per-side basis by default; combined
plots always use the joint fit. `PCA_BASIS='joint'` selects a common basis for
every layout. `PCA_LAYOUTS` selects left/right/both,
and solid/dashed lines distinguish sides on a joint plot. There is no separate
0–4 s view. Full-space measurements remain per fish and separate from PCA.
Every configured comparison is stimulus versus same-side Control (10 total);
there are no segment-pair tests. `test_population_separation` corrects across
all calculation-window times and those 10 pairs, independently of displayed
subsets. `plot_control_distance_overlay` superposes selected curves with equal-fish
means/SEM and returns first displayed corrected evidence times as a table, without
significance markers or evidence annotations on the plot. This is not a biological
change-point, proof of earlier equivalence, or a test between non-control curves.
Optional combined distance axes use solid/dashed side styles; optional symlog
preserves negative estimates. `check_population_subsampling` retains 50 default
equal-count draws; quantiles describe neuron-sampling sensitivity, not fish CIs.
No cross-fish trials are fabricated; no Lab analysis caches are changed.
Each figure has an independent local SVG switch (default True) at the top of its
plotting cell: six `SAVE_PCA_{LEFT|RIGHT|BOTH}_{2|3}D_SVG` flags plus
`SAVE_DISTANCE_SVG` and `SAVE_SIMILARITY_SVG`. The same cell draws and optionally
calls `plotting_common.save_publication_figure`; no global gate or batch-export
cell remains. Portable `SVG_DIR` resolves under
`Lab/Results/Paper/figures/main/fig03/assets/generated`. Existing
`exp08_population_` filenames and editable text are preserved. False skips only
that figure's writer call; other figures' choices are independent.
`first_trajectory_difference` remains an optional saved-asset timing audit.

Notebook 04's combined 3D plotting cell has independent `PCA_BOTH_3D_*` controls
for figure size, axes/title font, legend font, columns, location and optional
axes-coordinate anchor. `plot_pooled_population_trajectories` accepts optional
`legend_font_size`, `legend_loc` and `legend_bbox_to_anchor`; omitted values
preserve its historical defaults. These affect cached figure styling only.
The distance overlay x-axis reads "Time from stimulus motion onset (s)" using
the unchanged common 8 s motion-program anchor, not each condition's first
coordinate change.

Notebook 04 section 5 calls `build_control_pattern_similarity` and
`plot_control_similarity_curves`. `stimulus_similarity.compute_population_pattern_correlation`
owns finite matching-time Pearson calculations across neurons; constant vectors
yield NaN. For each fish, original selected z-scores feed a 1D correlation curve,
not PCA coordinates. Default `split_half` averages all three 2+2 block partitions
and both cross-half directions (six curves), so stimulus/Control
and Control repeatability use matched two-repetition means. This reference is
descriptive, not a noise ceiling or equivalence threshold. `trial_mean` uses
all-four-repetition means and omits the unmatched reference. Return fish-level
curve and repeatability tables, SEM and valid-fish counts; undefined samples
remain NaN. Display-window endpoints include native sample centers. Selected
stimuli reuse `DISPLAY_STIMULI`, with [-1,1] correlation limits. Existing distance
tests and PCA preparation remain separate. Cross-time calculations and their
plotting API were removed at the user's request.

Exp 8 `03_control_onset_recruitment.ipynb` calls
`active_neuron_analysis.build_control_onset_comparison` and both figure helpers
in `plotting_onset_comparison` directly. The comparison keeps original neuron
IDs and retained rows, requires finite binary (neurons, time, trials) rasters
and complete onset-window coverage, and preserves configured side/stimulus order.
Control-active neurons with a sustained consensus onset in the half-open bin
span define each reference population. Comparison-inactive/no-onset neurons
remain in the no-onset column. Counts and fractions use all control-bin neurons
within each fish; summaries weight nonempty fish bins equally and return valid
fish counts. Empty bins remain NaN, not zero. Self-control identity is not an
independent-trial reliability measurement. Its 8 s shared program anchor and
default [0, 9) s onset window are distinct from the longer Cell 06 activity
window. The public onset primitive in `response_classification` also backs the
historical bout helper through its unchanged private wrapper. Cohort preflight
accepts optional `settings['lab_root']`, resolved by `data_loading.resolve_lab_root`,
before the historical `data_base` lookup; its returned bundle is unchanged.
Exp 5 notebook 01 uses `data_loading.resolve_lab_root()` in its settings to
discover the local Lab folder on either workstation, including the
`SOCIAL_FILTERS_LAB_ROOT` override.

Notebook 03 section 6 independently calls `build_control_bin_overlap` and
`plot_control_bin_overlap` for ongoing activity at matching times, without
changing the onset comparison above. Finite binary retained-row rasters are
classified inside each half-open bin using the canonical per-trial fraction and
consecutive-epoch rule, then a minimum number of passing trials. Defaults are
1 s epochs in at least two of four trials, 10% significant frames, 1 s bins in
[0, 14) s relative to the shared 8 s program anchor. Epochs must fit wholly in
each bin/trial. There is no global activity/onset eligibility gate, and neurons
may contribute in multiple bins. Missing/nonbinary inputs, incomplete temporal
coverage, mismatched retained rows/repetitions and bins too short for the epoch
raise. Each fish/bin fraction divides the Control/stimulus intersection by that
bin's Control-active count; stimulus-only neurons are audited separately. Empty
Control bins stay NaN. Means weight eligible fish equally, SEM requires at least
two fish, and valid-fish counts are retained. `bin_activity[side][fish][label]`
is (retained neurons, bins), aligned to `neuron_ids[fish]`; `bin_windows` records
native samples and `fish_bins` preserves raw counts. Ordered Seg 2/Seg 1/Seg
1-3-5/one-position-flicker curves reuse notebook 04's colors, with Control support
below and a 9 s program-end marker. Self-Control is omitted from the plot and
remains identity in the tables; no independent repeatability or separation-time
test is implied. All notebook results remain in memory; no Lab files are written.

Both Exp 9 notebooks use `data_loading.resolve_lab_root()` to discover the local
OneDrive Lab folder from environment variables, home, or `D:/Alejandro/Data`.
An explicit Lab root or `SOCIAL_FILTERS_LAB_ROOT` overrides discovery. Reviewed
phase caches may move between data roots when their source path matches saved
configuration provenance and the complete relative fish/stage/filename agrees;
session, timing, settings, shape and neuron-identity validation remains required.

Exp 9 `02_population_analysis.ipynb` loads reviewed caches through the existing
loading owners and preserves original retained-neuron IDs. Its phase adapters
reuse `multifish_matrices` repetition/pooling, `neuron_ordering`, existing AUC
and peak/mean helpers, the canonical fraction/epoch classifier and the existing
paired-fish plots. Explicit native-time window selection supports different
preconditioning/delay/bout phases without an implicit response tail. Invalid
activity trials keep the selected denominator and yield missing fractions with
valid counts. Scalar metrics are summarized repetitions -> neurons -> fish;
across-fish means give each fish equal weight.
Single-ready-fish output is explicitly a preview. The machine contract lists
the notebook's exact public calls.

The shared `build_phase_population_display` accepts `sort_scope="within_fish"`
(default) or `"pooled"`, and `repetitions=None` (all) or unique one-based trial
numbers. Selection applies to displayed z-score/raster matrices and equal-fish
means only, without mutating the cohort or changing later metrics. Sorting uses
all z-score trials so subsets and alignments share row order. Optional
`sort_stimulus_keys`, `sort_alignment` and `sort_window_s` restrict the reference
to selected conditions and nominal display samples in [start, stop) seconds;
defaults preserve full sequence sorting. Exp 9 now uses left/right C1 only,
bout-aligned 0-3.6 s, averaged over all four repetitions, without changing metrics.
`displayed_row_metadata` preserves original fish/neuron identities for pooled
ordering. `plot_phase_cohort_rasters` defaults to `raster_cmap="Greys"`, labels
original trial numbers and omits fish boundaries/ticks for pooled sorting.
Its optional `zscore_cmap` retains the magma default; the notebook selects Greys
for z-score heatmaps as well, preserving shared z_limits.

Cell 06b calls `build_phase_recruitment_analysis` with `analysis['trial_metrics']`,
explicit C1/S1-S5 side lists, repetitions 2-4 and a 2/3 passing-trial rule.
Existing native-window per-trial classifications are reused; C2 is excluded.
Shared-C1 means both same-side C1 and condition bout-active; new-after-pre means
condition bout-active and inactive in its own preconditioning. C1 preconditioning
is not applicable. Missing required decisions yield missing counts/fractions with
valid counts. A side's unique union counts each new-after-pre neuron once.
The preconditioning display averages S1-S5/repetitions equally, while membership
classification remains per condition. Neuron identities and grouped row order
are retained for export. `plot_phase_recruitment_rasters` shows one preconditioning
panel and six bouts; `plot_phase_recruitment_counts` shows equal-fish mean count
bars and individual fish points. The additional `plot_phase_recruitment_proportions`
uses two rows (axes shape `(2, number of sides)`): existing `fraction_c1_retained`,
shared C1/condition neurons divided by
same-side C1-active neurons, calculated per fish before averaging defined fish
equally. Stimulus-only neurons do not contribute. Empty C1 pools and missing
required decisions remain NaN (N/A); a defined C1 self-overlap is 100%.
The second row uses existing `fraction_new_after_pre`, bout-active neurons
inactive during their own preconditioning divided by all retained neurons in
that fish, regardless of C1 membership. C1's second-row entry is N/A because
it has no preconditioning. Its denominator stays fixed even if C1 has no active
neurons. Defined fish fractions are averaged equally for both rows.
Cell 05's native measurements keep all four trials.
The display's stable preconditioning pool now counts native passing trials
across S1-S5 and selected repetitions: `pre_active_fraction=0.75` requires 12/15
(ceil). Its exported `preconditioning_pool` table retains passing/valid/expected
counts; missing trials give missing decisions. This pooled display criterion
does not replace own-condition preconditioning in the recruitment counts.
Display rows place the stable preconditioning pool first; the added
`plot_phase_recruitment_decisions` reuses the Exp 7 primitive in
`plotting_diagnostics` to show mean traces beside an actual 0/1 decision matrix
(preconditioning 12/15, bouts 2/3); missing decisions use gray.

Exp 9 notebook 02 no longer has the old Cells 06-10 (bout AUC/peak plots,
preconditioning/early-late/pre-bout metrics, paired comparisons, memberships/
correlations and the report export). Their only consumers were removed with
them: `plot_phase_fish_summary`, `plot_phase_population_comparisons`,
`plot_phase_population_relationships`, the `comparison_pairs` argument and the
`phase_relationships`/`correlations`/`phase_memberships`/`paired_differences`,
early/late, pre-bout level and delta outputs of `build_phase_population_analysis`
(now `build_phase_population_analysis(cohort, settings)`). Measurement windows
that exceed the aligned samples raise ValueError (`select_event_window(...,
sample_period_s=...)`); `several_fish_loading.align_phase_cohort` builds a
longer-recovery cohort (e.g. 15 s) from the loaded traces without reloading.
Cell 06c prepares these once on a `recovery_s=15` cohort copy; Cells 06d (bars),
06e (rasters), 07a/07b (C2 versus C1) and 08 (three-row summary) only redraw.
Cell 07c calls `build_phase_control_shared_nopre_response` on Cell 06c's 0-10 s
`c2_counts` (C1- and C2-bout-active, C2-pre-inactive neurons; per-fish and
equal-fish mean z and time courses) and Cell 07d draws it per side with
`plot_phase_control_shared_nopre_response`; the opt-in `baseline_window='pre_bout'` (Cell 07c `SHARED_NOPRE_BASELINE`, measured by Cell 06c over -2 to 0 s from bout onset) corrects for C2 preconditioning carry-over and the plot labels it. Cell 02's `ANALYSIS_REPETITIONS`/`ANALYSIS_MIN_ACTIVE_REPS` (applied in Cell 06c; also Cell 04d's `PER_REP_ANALYSIS_REPETITIONS`) set one repetition rule for Cells 06d-08 and build the 07a z-score full-sequence rows with `build_phase_stimulus_panel_displays(repetitions=...)` plus `build_phase_sequence_panel_displays`, independently of Cell 04a. Cell 06e re-sorts `recruitment_15s` with `reorder_phase_recruitment_rows` (default: preconditioning pool first, then all other neurons by C1 bout z-score AUC over 0-15 s; `'groups'` keeps 06b order) and hides the membership strips, legend and dotted group lines via `plot_phase_recruitment_rasters(show_membership_strips=..., show_group_lines=...)` and `plot_phase_recruitment_decisions(show_group_lines=...)`; defaults keep 06b/07a unchanged.
These cells reuse `build_phase_recruitment_analysis` with `bout_10s` (bars) and
`bout_15s` (rasters/decisions), the existing plotters' backward-compatible
style keywords (anonymous sage/slate-blue bars, hidden row labels, legend and
font controls, `compact_layout` for fixed-size mm exports, `window_label` titles), `build_phase_control_comparison` with
`plot_phase_control_comparison_summary` for C2-versus-C1, `select_phase_panel_rows`
for row-restricted full-sequence z-score panels (07a passes the rows reversed so
the z-score panel matches the rasters' bottom-to-top group order), and
`build_phase_recruitment_strength_inputs` to feed the Exp 8 Cell 6c
`plot_recruitment_strength_summary` (whose new `ylabels`/legend/title/`title_suffix`
keywords keep Exp 8 output unchanged). Figure exports use `save_publication_figure` with
`exp09_` names under the Lab Fig. 07 directory; nothing else is written.

The Exp 9 individual-fish notebook is an opt-in consumer of explicit phase loading,
event-based trial alignment, phase-aware plots, existing baseline normalization,
whole-sequence reliability and the canonical significant detector. It does not
use generic motion inference or silently match logged-name prefixes. Its arrays
retain `(neurons, time, trials)`; sequence lengths may differ by condition, while
original neuron and trial identities remain available. Existing loader/alignment
signatures and defaults are unchanged. See the machine contract for exact imports.

Its significant-trace stage now adds display-only dFoF/centered/raster comparisons
and original-ID trace inspection before an opt-in final save. The writer's optional
`significant_run` and `review_figures` arguments protect current run provenance and
export reviewed diagnostics without changing canonical cache names or columns.

Exp 9 notebook 02 Cell 04 now calls `build_phase_stimulus_panel_displays`,
returning trace-type -> side -> cached stimulus displays. Selected trials are
averaged per neuron; significant values are trial-active fractions. All retained
original fish/neuron identities stay aligned. `cell06` requires matching cached
recruitment identities and reuses its exact independent side orders; other modes
use a common all-trial z-score reference. The signed left/right index compares
mean C1 z-scores in the explicit reference window, with undefined values last.
The plotting primitive accepts optional `phase_trials` and `alignment` metadata
to reuse `add_phase_annotations` rather than Exp 8 timing. Matrices/fish means
retain the full comparison set for common left/right scales; stimulus_ids picks
each side's columns. Cell 04a prepares the existing analysis and recruitment once;
Cells 05/06b reuse them. Cell 04b exposes pooled-neuron/equal-fish weighting,
overlays, trace types, alignment, layout and optional Fig. 7 SVG export.
Cell 04c calls `build_phase_sequence_panel_displays(cohort, panel_displays,
t_pre_s=5.0, recovery_s=15.0)` to rebuild full sequence windows from saved
continuous traces while preserving the cached side orders, original IDs and
selected repetitions. Its new matrices and fish means retain the same shapes
and weighting conventions with longer time axes; inputs are not mutated.
Recovery is measured from bout offset. Imaging-block crossings fail explicitly.
The companion has independent style/export controls and distinct figure names.
It enables `proportional_widths=True` in the plotting gateway/primitive, using
the helper's `time_bounds_s` to end every raster/trace column at its own bout
offset + 15 s. Column widths track durations with a common physical seconds
scale; other callers keep equal widths and shared limits by default.
Cell 04d calls `build_phase_sequence_repetition_displays(cohort,
phase_panel_displays['sequence'], PER_REP_REPETITIONS, t_pre_s=5.0,
recovery_s=15.0)` (same window and cached rows as 04c, one single-trial
display per repetition) and draws each side/type with
`plotting_all_fish.plot_stimulus_response_panel_grid` (rows = repetitions,
columns = conditions). The notebook imports `src.plotting_all_fish` for this
grid; 04b/04c output is unchanged because both plotters share the same
per-panel drawing routines.

Exp 9 sequence/bout line panels use `trace_mode="population"`: average each
neuron over trials, then average exactly the selected raster rows with equal
neuron weights, omitting NaNs. All-neuron and reliability-filtered panels use
their respective raster populations. The stacked individual-trace helper mode
remains the backward-compatible default for other callers.

| Consumer | Active `src` imports | Compatibility role |
| --- | --- | --- |
| Exp 1 all-fish raster | `plotting_common`, `several_fish_figures`, `several_fish_loading`, `several_fish_reporting` | Shared cohort loading, raster/mean figures, diagnostics, report saving |
| Exp 1 bout/flicker position | `bout_flicker_analysis`, `plotting_bout_position`, `several_fish_loading` | Bout-position analysis and Cell 06-style figure |
| Exp 1 static/flicker recruitment | `plotting_static_flicker`, `response_metrics`, `several_fish_loading`, `several_fish_reporting`, `static_flicker_analysis` | Recruitment calculation, validation, statistics, and figures |
| Exp 5 all-fish raster | `active_neuron_analysis`, `multifish_matrices`, `plotting_common`, `plotting_diagnostics`, `several_fish_figures`, `several_fish_loading`, `several_fish_reporting`, `trial_alignment` | Response matrices, active decisions, pooled diagnostics, standard reports |
| Exp 5 LME decomposition | `data_loading`, `lme_feature_decomposition`, `multifish_matrices`, `plotting_lme`, `trial_alignment` | Loader/alignment, response matrices, model fitting, and figures |
| Exp 7 all-fish raster | `plotting_common`, `several_fish_figures`, `several_fish_loading`, `several_fish_reporting` | Exp 7 cohort/stimulus validation, shared raster/mean figures, activity diagnostics, sparseness, report saving |
| Exp 7 response overlap/selectivity, stages 1–4 | `several_fish_diagnostics`, `several_fish_figures`, `several_fish_loading`, `several_fish_reporting` | Mirrored recruitment/activity, containment, similarity, breadth/combinations, signed response profiles, and within-family positive-response lifetime sparseness |
| Exp 8 all-fish raster | `data_loading`, `multifish_matrices`, `plotting_common`, `plotting_diagnostics`, `several_fish_diagnostics`, `several_fish_figures`, `several_fish_loading`, `several_fish_reporting`, `trial_alignment` | Response matrices, independent stimulus panels, shared-control AUC, combined recruitment/strength summaries, diagnostics, reports, Fig. 7 pooled-panel SVGs and Fig. 3 summary SVGs |
| Exp 8 bout/flicker position | `bout_flicker_analysis`, `plotting_bout_position`, `several_fish_loading` | Same public bout-position surface as Exp 1 |
| dFoF batch preprocessing | `dff_extraction` | Per-plane Suite2p-to-dFoF extraction entrypoint |
| Shared single-fish analysis | `data_loading`, `trial_alignment`, `response_normalization`, `response_metrics`, `reliability`, `plotting_common`, `plotting_single_fish`, `plotting_all_fish`, `significant_trace_detection` | Load, align, filter, detect significant traces, and plot through the owning modules |
| Shared several-fish analysis | `active_neuron_analysis`, `data_loading`, `motion_metrics`, `multifish_matrices`, `plotting_all_fish`, `plotting_common`, `plotting_diagnostics`, `plotting_specificity`, `several_fish_diagnostics`, `several_fish_reporting`, `several_fish_selection`, `stimulus_similarity` | Reusable pooled matrices, selection, overlap, diagnostics, and figures |
| Stimulus batch plots | `stimulus_visualization` | Load trajectories, summarize, plot, and save reports |

The exact per-consumer function list lives in the machine contract rather than being duplicated here.

Exp 5 and Exp 8 notebook 01 section 4a call
`several_fish_diagnostics.build_left_right_active_overlap_diagnostics` once.
The resulting left/right decision-then-mean orders are reused by Cell 06's
appearance-only redraw and section 4b's `PANEL_SORT_MODE='cell06'` default.
`plot_all_fish_stimulus_figures(..., side_order_sources=overlap_results)`
checks fish/retained/source neuron identities and each complete permutation,
then returns additional `side_displays` with the per-side order and displayed
  row metadata. Trace matrices and fish means remain unchanged. Without this
  optional argument, the existing return keys and shared-order behavior remain.

Exp 8 notebook 01 section 4b also calls `build_stimulus_panel_display` with
`trace_type='raster'` (default remains `'zscore'`). Significant arrays are
already in retained-neuron order and must have exactly the retained row count;
source indices are not applied again. Averaging repetitions gives trial-active
fractions, with no new classification/threshold. Both types return `trace_type`
and `pooled_means` alongside the existing matrix/mean/identity fields. Other
sorting modes use the z-score order for both types; Cell 06 reuses exact side
orders for both. `plot_all_fish_stimulus_figures` adds optional `layouts`,
`mean_mode` (`equal_fish` or `pooled_neurons`) and `show_fish_traces`, preserving
old defaults. This notebook selects combined layouts, pooled-neuron weighting
and no fish overlays by default. Its off-by-default SVG gate now targets Fig. 7
and `exp08_pooled_{zscore|raster}_{side}_{layout}.svg`; summary exports remain Fig. 3.
The historical `plot_left_right_active_overlap_diagnostics` still prepares
and plots in one call, now delegating to the same preparation/redraw owners.

Exp 5 section 4b uses the same five sorting modes and compact raster/mean/
combined layouts as Exp 8, with Exp 5's own cohort, four repetitions, loaded
timing and stimulus map. Its appearance cell selects/reorders columns through
`PANEL_STIMULI_BY_SIDE`, chooses `PANEL_SIDE`, and controls font, sizes, colors
and limits without reclassification or sorting. Cell 06 retains its original
activity defaults and redraws the section 4a cache. Existing Exp 5 plots remain.
Each Exp 5 plotting cell has an independent SVG switch (default False) and
calls `plotting_common.save_publication_figure` for its own figure handles.
The shared `SVG_OUTPUT_DIR` is Lab-relative
`Results/Paper/figures/main/fig04/assets/generated`; names identify cell and
side/layout, with panel sorting mode retained in section 4b filenames.

Exp 8 notebook 01 sections 6b/6c opt into `motion_onset_s=None` and
`tau_s=SHARED_AUC_DECAY_S / 2` for summary preparation. The shared window owner
then uses each condition's `static_before_sec` through motion end plus 5 s
(22 s for the current Exp 8 trajectories). Recruitment and shared-neuron
selection are recalculated within this same window; retained source identities
and fish/stimulus order are preserved. Earlier section 6 plots retain their
configured numeric onset and tail. Numeric-onset callers, including Exp 7,
keep their existing behavior.

## External-active surface by import module

Canonical owner sections list symbols directly referenced by active notebooks. The facade sections record compatibility paths retained after consumer migration; they have no active notebook or script import.

### `src.analysis_tools`

Classification: supported compatibility facade. S06 moved implementations to responsibility-specific owners while preserving historical `src.analysis_tools.<name>` imports and signatures. No active notebook or script imports this module after S12 step 3. Its explicit `__all__` callable surface remains supported under the [S12 compatibility decision](src-compatibility-decision.md).

### Single-fish analysis owners

- `src.trial_alignment.build_trial_aligned_traces`, `resolve_selected_stimuli`
- `src.response_normalization.zscore_dfof_from_prestim_baseline`
- `src.response_metrics.compute_trial_mean_response_metrics`
- `src.reliability.filter_neurons_by_trial_reliability`

Classification: canonical owner imports used by the shared single-fish notebook after its S12 migration. The function arguments, return values, and notebook stage order are unchanged.

### `src.data_loading`

- `load_2p_experiment`
- `load_and_align_2p_experiment`

Classification: stable public gateways. S10 separated loader internals into private stages; their input/output contracts remain public.

### `src.dff_extraction`

- `process_suite2p_fluorescence`

Classification: stable preprocessing gateway.

### `src.lme_feature_decomposition`

- `build_lme_response_table`
- `fit_lme_models`
- `summarize_lme_model_results`
- `validate_lme_response_table`

Classification: stable public scientific workflow.

### `src.multifish_analysis`

Classification: supported compatibility facade. S07 moved implementations to scientific-domain owners. No active notebook or script imports this module after S12 step 3. Its documented public functions and characterized selectivity re-export remain supported under the [S12 compatibility decision](src-compatibility-decision.md).

### Several-fish scientific owners

- `src.motion_metrics` - motion-minus-fixed peak and integral calculations in the shared several-fish notebook.
- `src.multifish_matrices` - pooled matrices and per-fish or pooled z-score response matrices.
- `src.active_neuron_analysis` - active-neuron matrices and pooled trace diagnostics.
- `src.stimulus_similarity` - selected stimulus-vector similarity.
- `src.trial_alignment` - response selection and response-window frames in the cohort notebooks.

Classification: canonical owner imports used by the shared several-fish and all-fish raster notebooks after their S12 migration. The response-row, fish, and stimulus ordering contracts above remain in force.

### Specialized analysis owners

- `src.bout_flicker_analysis.build_bout_flicker_position_analysis` - pooled bout-position matching and ordered responses for Exp 1 and Exp 8.
- `src.static_flicker_analysis.build_static_flicker_recruitment_analysis`, `compute_static_flicker_fish_level_statistics` - Exp 1 recruitment summaries and fish-level inference.
- `src.response_metrics.validate_static_flicker_recruitment_result` - per-fish result validation.
- `src.multifish_matrices.build_zscore_response_matrices_all_fish` and `src.trial_alignment` - Exp 5 LME input preparation.

Classification: canonical owner imports used by the specialized notebooks after S12 step 3. Exp 8 bout/flicker cell 2's pre-existing syntax error and duplicate load were corrected without changing its block or timing settings; those settings still need a data-backed comparison with the maintained Exp 8 raster workflow.

### `src.plotting`

Classification: supported compatibility facade. S08 moved implementations to figure-family owners. No active notebook or script imports this module after S12 step 3. Its documented public figure functions remain supported under the [S12 compatibility decision](src-compatibility-decision.md).

### Single-fish plotting owners

- `src.plotting_common.list_stimulus_names`, `build_stimulus_style_maps`
- `src.plotting_single_fish.raster_with_stimuli`, `add_stimuli_markers`, `plot_sorted_chunks_single_mode`
- `src.plotting_all_fish.plot_stimulus_means`

Classification: canonical owner imports used by the shared single-fish notebook after its S12 migration. The plotted data, figure arguments, and optional save behavior are unchanged.

### Several-fish plotting owners

- `src.plotting_common` - stimulus style maps.
- `src.plotting_all_fish` - flat pooled rasters and stimulus means.
- `src.plotting_diagnostics` - active-trace decisions and motion-delta figures.
- `src.plotting_specificity` - similarity and stimulus-specificity figures.

Classification: canonical owner imports used by the shared several-fish and all-fish raster notebooks after their S12 migration. Figure arguments, row ordering, and optional save behavior are unchanged.

### Specialized plotting owners

- `src.plotting_bout_position.plot_bout_flicker_position_cell06_style` - Exp 1 and Exp 8 bout-position figures.
- `src.plotting_static_flicker` - Exp 1 category, shared AUC, and recruitment figures.
- `src.plotting_lme.plot_lme_model_outputs` - Exp 5 model figures.

Classification: canonical owner imports used by the specialized notebooks after S12 step 3. Figure call arguments and configured save behavior are unchanged.

### `src.reusable_several_fish`

Classification: supported import-only compatibility facade after S09. No active notebook or script imports this module after S12 step 3. Its documented several-fish callables remain supported under the [S12 compatibility decision](src-compatibility-decision.md).

The migrated notebooks now import `src.several_fish_loading`, `src.several_fish_selection`, `src.several_fish_diagnostics`, `src.several_fish_figures`, and `src.several_fish_reporting` as their calls require. The canonical file, table, and figure contracts below are unchanged.

S09 step 1: `save_analysis_report_run` and `export_notebook_report` now belong to `src.several_fish_reporting`; `src.reusable_several_fish` re-exports both for historical notebook callers. The report folder and return contracts below are unchanged.

S09 step 2: `load_and_preflight_fish_raster_inputs` now belongs to `src.several_fish_loading`; `src.reusable_several_fish` re-exports it for historical notebook callers. The returned bundle, path choices, fish/stimulus order, and all-or-error contract below are unchanged.

S09 step 3: pure stimulus/control selection, response summary and keep-mask preparation, trace filtering, and active-matrix row/column selection now belong to `src.several_fish_selection`. `src.reusable_several_fish` re-exports the notebook-facing names. Pooled response-row ordering and summary return keys are unchanged.

S09 step 4: response-window tables, high-sparseness raster inputs, mean traces, and overlap-diagnostic preparation now belong to `src.several_fish_diagnostics`. `src.reusable_several_fish` re-exports the notebook-facing names and former diagnostic subhelpers. Plotting and report dependencies are absent from the diagnostic owner; prepared arrays and row order are unchanged.

S09 step 5: pooled raster inputs, lifetime-sparseness summaries, and motion active-count tables now also belong to `src.several_fish_diagnostics`. The five high-level figure functions belong to `src.several_fish_figures`; `src.reusable_several_fish` re-exports them. The historical public signatures, returned structures, figure data, and ordering are unchanged. No active notebook import was migrated.

### Significant traces

- `src.significant_trace_detection.clean_binary_raster_columns`
- `src.significant_trace_detection.compute_noise_model_romano_fast_modular`
- `src.significant_trace_detection.plot_dff_and_raster`

Classification: stable canonical surface. The shared single-fish notebook uses the current-mode detector, cleanup, and plotting helpers from one owner. Historical `src.significant_traces` and `src.significant_traces_v2` paths remain compatibility facades but have no active notebook/script consumer.

### `src.stimulus_visualization`

- `plot_stimulus_phase_timing`
- `load_stimulus_trajectories`
- `plot_stimulus_positions`
- `plot_stimulus_traces`
- `summarize_stimuli`

Classification: stable public stimulus-inspection workflow.

The batch notebook targets Experiment 9, reads resolved settings and the timing summary, and saves only spatial/timing previews; standard traces and angle summaries remain notebook displays. The generic `save_stimulus_report` helper remains available for other callers. The phase helper preserves timing-table condition order and rejects mismatched stimuli/frame counts; it does not infer bout onset from motion. Experiment 9 generation remains a script-owned asset workflow reusing the unified generator's geometry.

## Important internal-active surface

The following are not necessarily direct notebook calls, but active `src` code depends on them:

- Timing: `get_angles_from_positions`, `get_motion_timing_simple`, `make_stimulus_traces_2`, and `extract_stimulus_chunks`.
- Analysis primitives retain their `analysis_tools` facade paths but are implemented by `analysis_io`, `response_selectivity`, `response_metrics`, and `response_classification`. Internal owner-to-owner dependencies now use the narrow modules rather than the facade.
- `data_loading`, `reliability`, and `several_fish_figures` also import their analysis and plotting owners directly after S12 step 5; `reusable_several_fish` alone keeps facade module aliases for historical imports.
- Multifish composition: `combine_reps_one_stim`, `build_matrix_for_fish`, `build_neuron_stimulus_summary_table`, `add_selectivity_metrics_to_summary_table`, `resolve_segment_labels`, `compute_active_neuron_jaccard_overlap`, and `build_active_neuron_overlap_matrices_all_fish`.
- Several-fish reporting: `export_notebook_report` and `resolve_response_control_columns`.
- Significant-trace comparison: `significant_trace_detection.compare_significant_trace_versions` calls the canonical detector in explicit legacy and current modes; the V2 facade delegates to that comparison helper.

Internal-active helpers can move only with their immediate callers and focused tests.

## Documented but currently uncalled public helpers

These are listed in `symbol-index.md` but have no active notebook/script call and no confirmed active public workflow call at the time of this audit:

- `analysis_tools.classify_responses_from_raster`
- `analysis_tools.compute_left_right_index`
- `analysis_tools.build_neuron_order_groupwise_onset`
- `multifish_analysis.classify_stimulus_specificity_summary_table`
- `multifish_analysis.build_stimulus_specificity_neuron_order`
- `multifish_analysis.build_segment_selectivity_permutation_summary`
- `plotting.plot_stimulus_specificity_summary`
- `plotting.plot_static_flicker_category_proportions`
- `stimuli_timeline.transform_stimuli_duration` as a direct notebook/script caller surface; it is now the canonical implementation used internally by `data_loading`.
- `auxtrigger_extraction.extract_aux_trigger_frames`; this remains an intentional documented preprocessing API despite having no active repository caller.

This classification does not authorize deletion. S11 retained these documented
helpers after its evidence review; any later change requires a new
compatibility decision. See the [S11 completion record](../history/src-refactor-s11.md).

## Compatibility boundary after S12

- `data_loading.transform_stimuli_duration` remains a supported historical wrapper around the canonical timing owner.
- `multifish_analysis.compute_stimulus_selectivity_metrics` remains a supported characterized re-export from `response_selectivity`. The `mfa` alias remains on `reusable_several_fish.py` for historical compatibility.
- `analysis_tools.inspect_obj` and `analysis_tools.plot_venn_3stim` remain supported explicit `__all__` exports, along with the other documented callables.
- `src.significant_traces` and `src.significant_traces_v2` remain supported mode-specific compatibility facades without independent scientific array logic.
- Private helpers and third-party aliases historically exposed by the facades remain importable but are not new stable API. The [S12 compatibility decision](src-compatibility-decision.md) records the removal criteria and warning policy.

## High-value compatibility contracts

### Array conventions

- Continuous dFoF and raster arrays normally use `(time, neurons)`.
- Trial-aligned arrays use `(neurons, time, trials)`.
- Pooled flat matrices use `(pooled neurons, concatenated stimulus time)`.
- Neuron order must remain aligned across traces, rasters, response matrices, decisions, metadata, and figures.

### `build_trial_aligned_traces`

Inputs are `(T, N)` continuous traces, a 60 Hz stimulus-ID trace, imaging FPS, and pre/post windows. It returns a dictionary containing:

- `cell_traces`: `(N, T)`;
- `stimuli_trace`: imaging-rate stimulus IDs;
- ordered `stimuli_ids` and `stimuli_names`;
- `trial_aligned_traces`: `stimulus_id -> (N, window, trials)`;
- `onsets_by_id`, `pre_frames`, `post_frames`, and `win_length`.

### Experiment loaders

`load_2p_experiment` returns a dictionary whose established keys include:

- base data: `dfof`, `fps_2p`, `frames_per_block`, `duration_2p_block_sec`;
- timing/log data: `stimuli_durations`, `adjusted_log`, `stimuli_trace_60`, `stimuli_table`, `stimuli_id_map`;
- locations: `paths`;
- optional caches/metadata: `dFoF_merged_map`, `z_traces`, `raster`, `deltaF_center`, `kept_neuron_indices`, `filtered_roi_indices`, `plane_ids`, `mean_imgs`, and `stat_per_plane`.

`load_and_align_2p_experiment` requires dFoF, raster, centered dFoF, and z-score caches. It returns the four aligned trace families plus stimulus identity, onsets, raw arrays, kept indices, durations, map data, and a nested `meta` dictionary. Several-fish helpers rely on these exact keys.

### Preprocessing gateway

`process_suite2p_fluorescence` reads Suite2p plane inputs and returns:

- dFoF with shape `(T, retained neurons)`;
- retained ROI indices relative to the full Suite2p ROI list.

Its scientific thresholds remain explicit parameters. Output filenames are owned by the calling preprocessing notebook, not this function.

### Multifish matrices

- `build_matrix_all_fish` returns `(sum selected neurons, sum stimulus-block time)` and must preserve fish order, neuron order, and configured stimulus order.
- `build_zscore_response_matrices_all_fish` returns `response_matrices`, `pooled_response_matrix`, `row_metadata`, `selected_stimulus_ids`, and `selected_stimulus_labels`. `row_metadata` columns are `fish_id`, `neuron_id`, `source_neuron_id`, and `global_neuron_id`.
- `build_active_neuron_matrices_all_fish` returns `fish_id -> neuron-by-stimulus active matrix`; with `return_counts=True`, it returns `(active_matrices, active_counts)`.
- `build_pooled_active_trace_diagnostic` returns `trace_matrix`, `decision_matrix`, `row_metadata`, `stim_order`, `stim_labels`, block width/time/repetition metadata, and `combine_mode`.

### Several-fish workflow gateways

Exp 7 notebook 02 omits original sections 6, 9–11 and 15–20, retaining original numbering. It displays recruitment/activity, control/all-rocking containment, family breadth, and exact combinations. Shared population/profile preparation remains required by the existing breadth helper. Its optional report excludes pairwise overlap/similarity, continuous-profile, and sparseness tables; publication export contains only sections 5, 7, 13 and 14. Notebook 01 and all shared helpers remain unchanged by this cleanup.

`build_pooled_response_profiles` remains available for other consumers, retaining per-neuron fish/source/global identities and identical left/right row order. Joint ordering concatenates both sides' six non-control response features; control remains displayed. `build_response_profile_sort_order` supports optional Pearson-distance or Euclidean-distance average-linkage ordering, retaining its historical `preferred` default. Pearson requires complete nonconstant rows; Euclidean requires complete rows; undefined profiles follow in source order without imputation. Zero/one clusterable-row inputs remain valid. `plot_pooled_response_profiles` returns left/right figure/axis pairs with matched raw signed values and shared symmetric color limits.

Exp 7 notebook 02 sections 13–14 select `denominator="family_responsive"`: each fish/side/family uses the union of that family's variant-responsive neurons, excludes zero-active categories, and contributes equally to the mean when the denominator is nonzero. `stimulus_specificity.build_family_responsive_membership_fractions` owns the calculation. `build_response_breadth_profiles` adds `response_breadth_family_responsive` and `response_combinations_family_responsive` tables with `n_family_responsive_neurons` and `fraction_family_responsive`. Existing all-kept tables, counts, profiles, and helper defaults remain intact. Report serialization includes both new tables and records the plotted denominator separately from the base-table denominator.

The summary figure helpers accept additive `separate=False` and `individual_style="identified"` options. Defaults retain `(Figure, 2-by-2 axes)` grids and identified fish lines. Separate mode returns four ordered `panel_key: (Figure, Axes)` entries, with `figsize` applying to each panel and matching left/right scales. Anonymous-point style retains fish data and means while hiding identity and individual connecting lines, including CDF evaluation dots. Exp 7 notebook 02 uses the recruitment, containment, and breadth helpers in sections 5, 7, 13 and 14 with 45 × 27.5 mm panels and 6 pt Arial, exporting each panel independently. Separate exact-combination panels use binary tick codes in the family's labeled variant order, preserving category order and membership values. Sparseness figure helpers remain available to other consumers.

Exp 7 notebooks 01/02 declare final-size publication styles and local millimetre figure sizes. `plotting_common.save_publication_figure` optionally saves PDF/SVG/PNG without tight cropping, preserving the configured canvas size and explicit raster DPI. `plot_response_profile_panels(..., fish_ids=None)` retains whole-cohort behavior by default; a nonempty ordered subset splits profiles into compact blocks without recalculating or rescaling responses. `plot_left_right_active_overlap_diagnostics` additionally retains `overlap_figure` and `diagnostic_figure` in each side's result when those figures are drawn, enabling exports after notebook display.

`build_family_lifetime_sparseness(breadth_profiles, cdf_grid)` adds stage 4 without modifying any preceding API or result. It consumes prepared signed normalized-AUC profiles and calls the unchanged canonical `response_selectivity.compute_stimulus_selectivity_metrics` separately for each family, fish, side, and existing population. Control is excluded; negative values are clipped to zero for this metric only. Every family response must be finite; missing values are not imputed as zero. Complete profiles without positive responses have NaN sparseness; roundoff is bounded to [0, 1]. Signed/positive response values, source identities, activity counts, and defined/incomplete/nonpositive status remain in the neuron table. No extra activity filter is applied; high concentration of very small positive responses is descriptive, not independently validated preference.

The returned `tables` are `family_sparseness_neurons`, `family_sparseness_per_fish`, `family_sparseness_summary`, `family_sparseness_cdf`, and `family_sparseness_mean_cdf`. Per-fish summaries retain population, defined, incomplete, nonpositive, negative-value, and active-family counts, plus mean/median sparseness among defined neurons. Cohort means average defined fish means equally, rather than pooling neurons; per-fish CDFs use defined neurons and are averaged equally at the caller's strictly increasing 0–1 grid, retaining `valid_fish`. Empty distributions stay missing. Two mirrored family CDF figures compare all-kept and all-rocking populations; a third shows fish means with mirrored sides and population rows. Saving remains disabled by default, and independent-trial preference validation remains deferred.

`build_response_breadth_profiles(analysis, similarity)` adds stage-3 family membership and signed profiles from matching prepared bundles, without loading, classification, or normalization. It checks cached response values and source-neuron identities. The result contains fish/condition/population order, family labels, exact combination order, `profiles[fish][side][population]`, `profile_color_limit`, and report tables. Memberships use all kept neurons, independently within each fish, side, and family. All breadth and exact-combination categories are present, including zero counts; empty denominators produce NaN. Summary fractions are equal-fish arithmetic means with `valid_fish`, while summed counts/denominators are retained separately and are not used as pooled fraction estimates.

The stage-3 tables are `response_memberships`, `response_breadth`, `response_combinations`, their two `*_summary` tables, and `response_profile_rows`. Profiles retain signed normalized AUC in canonical condition order. Each panel retains display-row identities and is sorted by largest signed weaker-variant response, descending largest value, then original row order; rows with no finite weaker-variant values go last. Sorting is descriptive and does not validate preference. Both population figures use one full-range symmetric color bound, with a bound of 1 only for all-zero/all-missing data; missing values are gray and empty cores are annotated. Left/right core selections and display rows are independent. Notebook 02 reports membership/breadth/combination tables, with saving disabled by default; response-profile rows are omitted from its report. The shared stage-4 sparseness workflow remains available to other consumers.

`build_response_similarity_comparison(analysis)` is an additive stage-2 gateway consuming the unchanged stage-1 result. It uses identity-aligned signed `normalized_auc` rows, never reloads data or repeats activity classification, and delegates Pearson/cosine to `stimulus_similarity.build_stimulus_vector_similarity`. It compares `all_kept` and `all_rocking` populations separately within each fish and side. The latter is selected by intersecting that side's rocking activity columns, with one fixed neuron selection across all stimulus comparisons; rocking-versus-rocking Jaccard is therefore one by construction when the core is nonempty.

The stage-2 result includes `fish_ids`, `condition_labels`, `populations`, `population_labels`, `per_fish_matrices`, `mean_matrices`, `response_vectors`, `row_metadata`, and `tables`. Its report tables have distinct `response_similarity_` prefixes: `pairs`, `summary`, `population_counts`, and `population_summary`. Pair rows retain population sizes and pairwise finite-response neuron counts; summaries contain arithmetic equal-fish means and metric-specific valid-fish counts. All-kept Jaccard reuses stage-1 matrices. Constants, zero norms, missing responses, and empty cores preserve existing undefined-value semantics rather than becoming zero similarity. All stage-1 objects and existing helper bodies/defaults remain unchanged. Figures mirror side columns and population rows, with Jaccard limits 0–1 and Pearson/cosine limits -1–1; undefined cells are gray.

`build_recruitment_containment_analysis` is an additive stage-1 gateway. It calls the existing active-decision, AUC, and timing owners, requiring filtered raster rows in the same kept-neuron order used to select continuous responses. Time/trial axes and row counts must match; failures are explicit. All existing function bodies, defaults, and notebook 01 behavior are unchanged.

Its result contains `fish_ids`, `condition_labels`, `side_selections`, `active_matrices`, `responses` (including source-neuron metadata), `per_fish_matrices`, `mean_matrices`, and `tables`. Tables include `neuron_responses`, `recruitment`, `response_windows`, `window_summary`, `pairwise`, `matrix_summary`, and `containment_summary`. Raw signed AUC is in z-score seconds; normalized AUC divides by the actual trapezoidal integration timespan (one frame period for a single sample), not the requested window duration. Requested/observed bounds and clipping remain visible.

Containment row A/column B is `|A intersection B| / |A|`; Jaccard is intersection over union. Matrix summaries weight fish equally, skip undefined values, and record valid-fish counts. Empty sources remain NaN. Control and all-rocking summary rows retain contained/non-contained counts and fractions; all-rocking means the intersection, never the union. Figures always mirror left/right using the same functional order, scales, and fish colors. Reporting uses the existing optional writer and remains disabled by default.

`load_and_preflight_fish_raster_inputs` returns settings, timing, fish/stimulus order, resolved paths, `all_fish_data`, reference-fish identity/data, aligned z-score traces, the stimulus map, durations, and names. It raises if any configured fish or stimulus is unavailable rather than silently producing a partial cohort.

`settings['data_base']=None` enables automatic path discovery in this order: `OneDriveCommercial`, `OneDrive`, university OneDrive folders under the current user's home, then those under `D:\Alejandro\Data`. Candidates are deduplicated and must contain both `Lab/Data/2p` and `Lab/Analysis`; the two returned roots always come from the same candidate. Explicit `data_base` values remain parent-folder overrides with the previous lookup and failure behavior, without automatic fallback. Exp 7 opts into discovery and displays the resolved roots after loading; existing experiment notebooks keep their explicit settings.

`build_all_fish_raster_figure` returns a dictionary containing the flat matrix, optional left-right index, neuron order, figure, axes, and image.

Exp 7 notebook 01 section 4 overrides its panel's sort mode via `RASTER_SORT_MODE`
(Corravg by default; Rastermap, left/right index and unsorted remain selectable).
`RASTER_SIDE` selects both/left/right stimulus conditions in configured order,
retaining every neuron. Corravg/Rastermap fit displayed conditions; left/right
index still uses both controls. Local selection does not change later analyses.
`compute_single_sort_order(..., "rastermap")` lazily imports Rastermap >=1.0,<2,
fits finite nonconstant traces, and appends other rows in source order. It uses
up to 200 PCs/100 clusters, passes the existing random seed, and returns a full
permutation without replacing displayed data with Rastermap's normalized/binned
matrix. Fewer than three usable traces retain their source order. Missing
Rastermap raises an installation hint; other modes do not require the package.
The optional report records the actual panel sort, side, and stimulus order.

Exp 7 notebook 01 section 6c adds `plot_recruitment_strength_summary`, returning
`side: (Figure, length-3 Axes array)` for population activity, recruitment and
shared-neuron normalized AUC differences. It consumes the existing recruitment
and shared-AUC bundles for the same fish/controls, aligns conditions by stimulus
ID, preserves missing control self-comparisons, and uses shared bottom ticks and
one Fish/Mean legend. Notebook-local colors, dimensions and `SAVE_SUMMARY_SVG`
control redraw/export without rerunning preparation. Original windows and
population definitions remain unchanged. These figures also join the optional
publication export.

Exp 5 notebook 01 section 6c uses the same summary layout with its own controls,
four position conditions, flicker and rocking, and the existing five-fish cohort.
Its two preparation cells use the loaded motion onset through motion end plus
5 s for activity, recruitment and pairwise shared-neuron AUC, with the existing
section 4a activity rules. Population rows use all kept neurons; shared strength
uses each control/comparison intersection. The helper's historical `rocking`
group holds the position conditions and rocking; only its population/recruitment
rows feed this figure. Earlier Cell 06 windows and decisions stay separate.
Optional `plot_recruitment_strength_summary(..., side_stimuli=...)` accepts a
complete stimulus-ID permutation per side, reordering all three rows/labels
without mutating prepared results; default calls preserve their original order.
Exp 5 passes its configured order, retaining flicker before rocking.
`SAVE_SUMMARY_SVG` independently saves the two figures to Fig. 4 using the common
`SVG_OUTPUT_DIR`; appearance changes do not rerun preparation.

Exp 7 notebook 01 section 6b adds shared-neuron response-strength comparisons.
Section 6 exposes its unchanged defaults through `SECTION6_ACTIVE_SETTINGS`;
`build_shared_control_auc_analysis` consumes the resulting unsorted binary
decision matrices and row identities, validating current fish/kept rows and
continuous/raster time/trial shapes. It does not rerun activity classification.
Each control/comparison pair uses its own intersection. Measurement starts at
that condition's loaded static-before boundary and ends at motion end plus
`SHARED_AUC_DECAY_S` (default 5 s), independently of section 6's selection window.
`compute_motion_decay_auc` delegates integration/window selection to existing
owners, retains signed values, rejects nonfinite neuron/trials, and averages
fully finite trials. It returns raw AUC (z-score seconds), normalized AUC divided
by the actual sample span (one frame period for one sample), valid-trial counts,
and requested/observed bounds with clipping flags. Shared-neuron source indices,
pairwise deltas and finite/shared counts remain in `shared_neuron_auc`,
`shared_auc_per_fish`, `shared_auc_summary`, and `shared_auc_windows`. Empty
intersections/missing pairs remain NaN. Cohort summaries average finite per-fish
paired means equally. `plot_shared_control_auc` defaults to normalized AUC;
`raw_auc` remains selectable. Separate side figures use anonymous fish dots and
a zero reference; positive means stronger control. Tables/measurement settings
join the optional report, and `shared_control_auc_*` joins publication export.

Exp 7 notebook 01 section 4b adds `build_all_fish_stimulus_figures` without
altering section 4. Each requested side gets standalone rasters, per-fish traces
with an equal-fish mean, and rasters above the same traces. Its diagnostic owner
`build_stimulus_panel_display` averages repetitions per neuron, preserves source
row identities, and returns condition-keyed neuron-by-time matrices and
fish-by-time means. A supplied original-raster order is reused; otherwise the
configured sorter runs once across all configured conditions. Left/right panels
share that order and common color/trace limits. Time axes restart relative to
stimulus appearance for each column. `stimulus_phase_intervals` validates the
existing static-before/active/static-after duration fields; it does not infer
new event timings. The plotting primitive adds nominal phase shading and
stimulus-on/off markers. New figures are included in optional publication
export under `stimulus_panels_*`, leaving existing export names unchanged.

Section 4b now separates `build_stimulus_panel_display` (one preparation/sorting
cell) from `plot_all_fish_stimulus_figures` (appearance-only redraws). The latter
accepts the prepared display and returns the same figure/axes bundle without
reloading traces, rebuilding trial means, sorting, or mutating the cached result. The original
`build_all_fish_stimulus_figures` remains a compatible prepare-and-draw wrapper.
The plotting cell owns `PANEL_FONT_PT` (default 6 for all text), side selection,
limits, colors, and halved physical dimensions: raster 150 × 45 mm, means
150 × 42.5 mm, combined 150 × 65 mm. Titles contain only stimulus names, fish
traces have no name legend, and the phase key and Time (s) label appear once.

`save_analysis_report_run` is intentionally stateful. When enabled it creates a timestamped report folder containing settings JSON, report-settings JSON, run metadata JSON, comments Markdown, optional CSV tables, and an optional notebook export. It returns `run_id`, `run_dir`, saved table paths, and export status; when saving is disabled it returns `None`.

### Several-fish selection

`build_selected_neuron_summary` returns `neuron_summary_table`, `neuron_summary_all_stimuli_table`, `left_right_index`, `plot_neuron_keep_mask`, `plot_neuron_keep_indices`, `resolved_preference_controls`, `selected_auc_response_matrix`, and `plot_filter_table`. The index, keep mask, selected responses, and filter table follow pooled response-row order; the summary-table owner sorts its rows by `global_neuron_id`. `build_fish_keep_masks` splits the pooled mask by fish without reordering rows; trace and active-matrix filtering then apply each fish's rows in their original order.

### Significant-trace detector

The canonical current-mode detector accepts `(T, N)` dFoF and returns an eight-item tuple:

1. significance map;
2. centered dFoF;
3. data density;
4. noise density;
5. x grid;
6. y grid;
7. `(T, N)` binary raster;
8. joint constrained significance map.

With `return_diagnostics=True`, a ninth diagnostics dictionary is appended. Tuple order is a compatibility contract until an additive named-result API is introduced.

### Stimulus visualization

- `load_stimulus_trajectories` returns a tidy table with `stimulus_name`, `dot_id`, `time_s`, `angle_deg`, `dot_size`, and `visible`.
- `summarize_stimuli` returns `stimulus_name`, `unique_angles_deg`, `n_unique_angles`, and `n_dots` in stable column order.
- Plot helpers return Matplotlib figures without saving.
- `save_stimulus_report` overwrites `stimuli_standard.png`, `stimuli_polar.png`, and `stimuli_summary.csv` under the supplied output directory and returns that directory.

### Plotting contracts

Active plotting helpers must preserve configured stimulus order, neuron row alignment, return object structure, and existing optional save behavior. Moving a plotting family requires figure rendering and screenshot inspection. S08 is complete; `tests/README.md` documents the historical Matplotlib rendering failure and the current render probe.

## Change protocol

When an active consumer changes:

1. Update `tests/contracts/src_api_consumers.json` only after reviewing the change.
2. Update this reference if classification, owner, shape, keys, side effects, or output paths changed.
3. Update `symbol-index.md` when the stable public surface changed.
4. Validate the owner and the first downstream consumer.
5. Preserve an old import through a facade until the planned migration slice closes.

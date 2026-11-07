# Exp 9 analysis: review handoff

Continue reviewing `02_population_analysis.ipynb` with the user. Refer to its
stable **Cell 01-08** labels (with 04a-d, 06b-e and 07a-d), not kernel execution counters such as `In[57]`.
The notebook is implemented; the next task is to inspect and discuss its results
and fix specific issues the user identifies.

## Notebook cell map

| Cell | Purpose |
| --- | --- |
| 01 | Imports and shared `src` helpers |
| 02 | Fish selection, paths, phase windows, activity thresholds and display settings; `ANALYSIS_REPETITIONS` / `ANALYSIS_MIN_ACTIVE_REPS` (the one repetition switch for 04d labels and 06c-08) |
| 03 | Readiness table, included fish and saved detector settings |
| 04a | Prepare shared native measurements/recruitment and pooled left/right orders for both alignments |
| 04b | Pooled z-score/significant panels with optional traces below and optional Fig. 07 SVG export |
| 04c | Companion full-sequence panels: keep preconditioning and the complete blank delay, then 15 s after bout offset; reuse 04a's exact rows/repetitions |
| 04d | Same 04c window and rows without trial averaging: one row of panels per repetition (`PER_REP_REPETITIONS`, default all four), one column per condition, per side and trace type; significant panels are single-trial 0/1 traces; repetitions outside `PER_REP_ANALYSIS_REPETITIONS` (default Cell 02's `ANALYSIS_REPETITIONS`, trials 2-4) are labelled excluded from analyses; optional pooled-neuron mean below each row; optional SVG |
| 05 | Per-trial/per-neuron/per-fish measurements and native sample coverage |
| 06b | Significant recruitment from trials 2-4: one mean preconditioning panel, C1/S1-S5 bouts, counts and percentages for C1 overlap and new after own preconditioning; C2 excluded (unchanged, 0-3.6 s) |
| 06c | Prepare once: Cell 02's `ANALYSIS_REPETITIONS` (default `(2, 3, 4)`, `(1, 2, 3, 4)` = all trials) with `ANALYSIS_MIN_ACTIVE_REPS=2` sets the rule for 06d-08; 15 s-recovery copy of the cohort (`align_phase_cohort`), `bout_10s`/`bout_15s`/`pre_bout` (-2 to 0 s) measurements, recruitment at 0-10 s (bars) and 0-15 s (rasters), C2-versus-C1 comparison, the 07a z-score full-sequence rows (own repetitions, not Cell 04a's) and the three-row summary inputs; 04a-06b results are unchanged |
| 06d | 0-10 s shared-with-C1 / new-after-own-preconditioning counts and percentages: slate-blue equal-fish bars, sage fish dots, no lines or fish identities; optional SVG |
| 06e | 0-15 s significant rasters and decision diagnostics (bouts 0-15 s, full 8.1 s preconditioning); rows default to preconditioning pool first, then all other neurons by C1 bout z-score AUC (0-15 s, high to low), or 06b grouping (`RASTER15_SORT='groups'`); `RASTER15_ROWS_TOP_DOWN=True` (default) puts row 1 (the pool) at the top, `False` keeps 06b's bottom-up order; no row labels, membership strips, legend or group lines by default; optional SVG |
| 07a | C2 versus C1 per side (no S stimuli): rows active in the C2 bout, grouped by own (distant) preconditioning and C1-bout activity; 06b-style raster and decisions (0-15 s) and 04c-style z-score full sequence (5 s before onset to 15 s after bout offset) restricted to those rows, with their mean; all three share one vertical orientation (first group at the bottom); optional SVG |
| 07b | C2 versus C1 quantification at 0-10 s: C2-active counts, % also active in C2 pre and in the C1 bout, and the Cell 08 three-row summary for C1/C2; optional SVG |
| 07c | Prepare: neurons active in both the C1 and C2 bouts but not in C2's own preconditioning (07a group "C1 bout shared (no C2 pre)"; 06c's 0-10 s `c2_counts`, `ANALYSIS_REPETITIONS`, missing decisions excluded); optional `SHARED_NOPRE_BASELINE='pre_bout'` subtracts each trial's -2 to 0 s pre-bout mean z (C2 preconditioning carry-over); per-neuron 0-10 s mean z for C1 and C2, per-fish means (fish without neurons missing), equal-fish means and bout-aligned time courses; tables of neurons per fish |
| 07d | Per side: C1/C2 mean z-score time course (-5 to +15 s around bout onset, optional SEM band, 0-10 s window and bout shaded) above paired per-fish 0-10 s mean z (slate-blue equal-fish bars, sage fish dots, optional lines) with the C1 − C2 difference; baseline-corrected values are labelled and their SVGs end in `_baseline`; optional SVG |
| 08 | Exp 8 Cell 6c-style population activity, recruitment and shared-with-C1 strength (0-10 s, `ANALYSIS_REPETITIONS`, default trials 2-4, >= 2/3), C1/C2/S1-S5; optional SVG |

All figure cells have size (mm), font, legend and optional SVG controls. SVGs go to
`<LAB_ROOT>/Results/Paper/figures/main/fig07/assets/generated` with `exp09_` names;
`SAVE_..._SVG=False` by default. The old Cells 06-10 (AUC/peak plots,
preconditioning metrics, paired comparisons, memberships/correlations and the
report export) were removed together with their unused `src` helpers.

Cell 06b's stable preconditioning pool requires at least 75% of the selected
15 presentations to pass the existing native-window activity rule: **12/15**.
The added Exp 7-style diagnostic shows mean significant traces beside strict
0/1 phase decisions (preconditioning 12/15, each bout 2/3), with the same rows.
This selection changes the displayed preconditioning pool; per-condition
new-after-preconditioning counts still use each stimulus's own preconditioning.

## Scientific context and agreed choices

- Experiment: `Exp_9_RF_memory_inhibition`; fish `L978_f01` through `L978_f08`.
- Each imaging block and its projection share a start trigger. Imaging is 2 Hz;
  trajectories are 60 Hz. Use each fish's recorded block-log timestamps, with
  B1-B4 concatenated in imaging order. Exclude resting B0 and interblock waiting.
  Stimulus timestamps do not identify or correct dropped imaging frames.
- Both sides have C1 bout-only control, C2 distant preconditioning, and S1-S5
  overlapping preconditioning. Actual preconditioning lasts 8.1 seconds;
  blanks are 0.3, 0.6, 2, 4 and 8 seconds; bouts last 3.6 seconds. C2 has a
  0.3-second blank. C1 has neither preconditioning nor an internal blank.
- Logged short names such as `LeC1` and `RiS5` map explicitly to asset names
  such as `LeC1_Bout` and `RiS5_Overlap`. Never silently skip unmatched names.
- Use saved whole-sequence reliability-selected neurons in every condition.
  This selection is distinct from classifying activity within a bout window.
- Display sequence and bout alignments, trial averages and concatenated trials.
  Averaged significant rasters are trial activation fractions, not binary traces.
- Cell 04 pools all retained neurons, with separate left/right stimulus figures.
  Both z-scores and significant trial-active fractions use grayscale and the
  same row identities. `DISPLAY_REPETITIONS=(2, 3, 4)` selects the last three
  trials; `None` uses all four. `PANEL_SORT_MODE='cell06'` reuses Cell 06b's
  exact independent side orders. Corravg, optional Rastermap, left/right C1
  index and unsorted are also available. The first two use the configured
  all-trial z-score reference; the index compares signed mean left/right C1
  responses within that window. Undefined indices go last.
- Cell 04a prepares the existing native measurements and recruitment once for
  Cells 05/06b as well as these displays. Cell 04b changes appearance without
  repeating preparation: choose sequence/bout alignment, trace type, side,
  `PANEL_SHOW_TRACES`, `PANEL_MEAN_MODE` (pooled neurons or equal fish), fish
  overlays, font and dimensions. Z-scores keep `Z_LIMITS`; significant activity
  is the fraction of displayed repetitions active, not a new detector result.
- `SAVE_PANEL_SVG=False` displays only. True writes the displayed panels to
  `<LAB_ROOT>/Results/Paper/figures/main/fig07/assets/generated` as
  `exp09_pooled_panel_<alignment>_<type>_<side>_<layout>.svg`. All retained
  neurons and native measurements remain fixed.
- Cell 04c independently redraws the complete sequence with 5 s before sequence
  onset and `DELAY_PANEL_POST_S=15.0` seconds after bout **offset**. It extracts
  the longer display from saved continuous traces and reuses the exact cached
  side orders/repetitions without sorting or changing measurements. Time zero
  is sequence onset; the full condition-specific blank remains visible. All
  columns share a physical seconds scale, with widths proportional to their own
  duration and exact limits at bout offset + recovery (no padding to the longest
  condition). `DELAY_PANEL_PROPORTIONAL_WIDTHS=True` enables this in 04c.
  `SAVE_DELAY_PANEL_SVG=False` defaults off; True exports
  separate `exp09_sequence_delay_panel_15s_<type>_<side>_<layout>.svg` files to
  the same Fig. 07 directory. Block-crossing windows fail explicitly.
- Cell 04d draws the same full-sequence window and cached rows once per
  repetition (no averaging): rows are repetitions, columns conditions, with
  04c's common seconds scale. Significant panels are each trial's saved 0/1
  trace. `SAVE_PER_REP_SVG=False` defaults off; True exports
  `exp09_sequence_delay_panel_15s_per_rep_<type>_<side>.svg` to Fig. 07.
- Cells 04a-06b measure bouts in [0, 3.6) and [0, 8.6) seconds from logged bout
  onset; Cells 06c-08 use [0, 10) (bars, summary) and [0, 15) (rasters) on a
  15 s-recovery cohort. Windows beyond the aligned samples fail explicitly. Use actual event-relative sample times, without interpolation, and
  report selected sample counts and coverage.
- AUC is signed z-score seconds. Calculate metrics per neuron/repetition, then
  average repetitions within neurons and neurons within fish. Across-fish means
  give each fish equal weight regardless of neuron count.
- A passing trial has at least 10% significant frames and a consecutive active
  epoch of at least 1 second. A neuron is active when at least 2/4 trials pass.
  Cell 06b uses repetitions 2-4 and requires at least 2/3 passing trials; Cells
  06c-08 follow Cell 02's `ANALYSIS_REPETITIONS` (same default; `(1, 2, 3, 4)`
  uses all trials with at least 2/4), and their titles show the repetitions used.
  Settings remain configurable. Keep the same selected-neuron denominator;
  incomplete classifications produce missing fractions with valid counts.
- Preconditioning is measured only during its explicit 8.1 s phase. C1's
  absent phase is not applicable, not zero. Bout-window activity is not
  necessarily caused by the bout; control differences remain descriptive.
- Keep left and right separate. Conditioned stimuli are compared with same-side
  C1 (Cells 06b-06e, 08); Cell 07 compares C2 with C1 without S stimuli.
- Cell 06b retains the counts and adds the percentage of C1-responsive neurons
  also responsive to each stimulus: shared C1/stimulus count divided by the
  same-side C1 count, per fish. Stimulus-only neurons are excluded. Bars average
  defined fish fractions equally; points show individual fish. Empty C1 pools
  and missing required classifications are N/A; defined C1 self-overlap is 100%.
  The percentage plot's second row shows bout-active neurons inactive during
  their own preconditioning divided by all retained neurons in the fish,
  regardless of C1 membership, again averaged equally across defined fish.
  C1 has no preconditioning, so its second-row value is N/A.

## Files and readiness: check before assuming a fish needs rerunning

Both notebooks discover the local OneDrive Lab folder with
`data_loading.resolve_lab_root()`: OneDrive environment variables first, then
university OneDrive folders under the current user's home or `D:/Alejandro/Data`.
Set `SOCIAL_FILTERS_LAB_ROOT` to override discovery with a specific Lab folder.
The data root is `<LAB_ROOT>/Data/2p`; for example, on the D: workstation:

```text
D:\Alejandro\Data\OneDrive - Université de Lausanne\Lab\Data\2p
```

Reviewed caches can be synced between these roots without resaving them. The
loader checks saved source provenance and the relative experiment path, plus
session, timing, settings and retained-neuron identities.

For each fish, `01_individual_fish_analysis.ipynb` reviews normalization,
reliability, significant-trace parameters and matching dFoF/raster diagnostics.
Its final save cell must be enabled to write reviewed results. Running analysis
cells alone can leave results only in kernel memory. Detector settings may
differ between fish; the population notebook reads them rather than recomputing
or imposing one common detector configuration.

Under `<fish>_Exp_9_RF_memory_inhibition/03_analysis/functional/plots/`, Cell 03
requires:

- `z_core/<fish>_zcore.npz`
- `filtered_neurons_by_stimuli/<fish>_kept_neuron_indices.npy`
- `significant_traces/<fish>_significant_traces.npz`
- `phase_alignment/<fish>_trial_timing.csv`
- `phase_alignment/<fish>_analysis_configuration.json`

The latest Cell 04 validation found all eight fish ready in these canonical
locations, with 2,453 retained neurons in total. Recheck the current readiness
table, individual-notebook save settings and any redirected `OUTPUT_ROOT`
before concluding that a fish needs recomputing.

The population notebook reports every unavailable/inconsistent fish. One ready
fish is labeled a single-fish preview; it does not establish across-fish variation.

## Ownership, saving and validation

Follow the root AGENTS instructions and routed guidance. The user explicitly
requires reuse of existing functions and an orderly `src`: loading, timing,
response metrics/classification, pooling, plotting and reporting remain in their
existing owners. Extend compatible helpers rather than copying calculations
into notebooks or creating duplicate experiment-specific implementations.

The population notebook writes only optional figure SVGs (Fig. 07 directory
above) and never overwrites individual-fish caches. The former Cell 10 report
export was removed.

Before the Cells 06c-08 rewrite, the Cell 01-05 chain was executed on all eight reviewed fish. Checks
matched every panel's selected-trial means, original identities, exact Cell 06b
order and pooled mean; both sequence/bout alignments and the raster-only switch
were exercised. False/True SVG saving was checked with writes redirected to
`tmp/exp09_fig07_validation`, where rendered previews also live. Focused tests
cover unequal neuron counts, weighting, native windows, activity decisions,
missing phases, retained-row reuse and cache consistency. The unchanged later
scientific plots were previously exercised on f01; this panel check does not
claim a fresh full-notebook run.

An earlier stale plotting-definition problem in the individual notebook was
resolved by recompiling the current plotting source in its imports cell.
The population notebook's Cell 01 also refreshes the timing, loading, ordering,
panel, diagnostics and figure helpers from current source. If an existing kernel cannot find a newly added helper,
rerun Cell 01, then rerun the affected analysis cell; loaded data can be reused.
If a helper reports an unexpected keyword, inspect the loaded module path and
signature before assuming that the current source lacks that parameter.

## Suggested opening message for the next chat

> Continue reviewing Exp 9's `02_population_analysis.ipynb`. Read the README in
> its folder and the repository's routed instructions. Use the numbered cells
> when discussing results. First check Cell 03: I have run other fish, so verify
> their saved output paths and save settings before requesting recomputation.
> Reuse existing `src` functions and preserve individual-fish caches. We will
> review the recruitment rasters/bars, the C2-versus-C1 cells and the summary together.

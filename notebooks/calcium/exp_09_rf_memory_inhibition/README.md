# Exp 9 analysis: review handoff

Continue reviewing `02_population_analysis.ipynb` with the user. Refer to its
stable **Cell 01-10** labels, not kernel execution counters such as `In[57]`.
The notebook is implemented; the next task is to inspect and discuss its results
and fix specific issues the user identifies.

## Notebook cell map

| Cell | Purpose |
| --- | --- |
| 01 | Imports and shared `src` helpers |
| 02 | Fish selection, paths, phase windows, activity thresholds and display settings |
| 03 | Readiness table, included fish and saved detector settings |
| 04a | Prepare shared native measurements/recruitment and pooled left/right orders for both alignments |
| 04b | Pooled z-score/significant panels with optional traces below and optional Fig. 07 SVG export |
| 04c | Companion full-sequence panels: keep preconditioning and the complete blank delay, then 15 s after bout offset; reuse 04a's exact rows/repetitions |
| 05 | Per-trial/per-neuron/per-fish measurements and native sample coverage |
| 06 | Bout AUC, peak and active-neuron fractions in both windows |
| 06b | Significant recruitment from trials 2-4: one mean preconditioning panel, C1/S1-S5 bouts, counts and percentages for C1 overlap and new after own preconditioning; C2 excluded |
| 07 | Preconditioning, early/late change, pre-bout level, corrected AUC and repetition consistency |
| 08 | Paired control comparisons and delay curves |
| 09 | Preconditioning/bout memberships, neuron scatterplots and descriptive correlations |
| 10 | Optional report export; `SAVE_OUTPUTS = False` by default |

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
  `exp09_pooled_panel_<alignment>_<type>_<side>_<layout>.svg`, independently of
  Cell 10's report. All retained neurons and native measurements remain fixed.
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
- Bout measurement windows are [0, 3.6) and [0, 8.6) seconds from logged bout
  onset. Use actual event-relative sample times, without interpolation, and
  report selected sample counts and coverage.
- AUC is signed z-score seconds. Calculate metrics per neuron/repetition, then
  average repetitions within neurons and neurons within fish. Across-fish means
  give each fish equal weight regardless of neuron count.
- A passing trial has at least 10% significant frames and a consecutive active
  epoch of at least 1 second. A neuron is active when at least 2/4 trials pass.
  Settings remain configurable. Keep the same selected-neuron denominator;
  incomplete classifications produce missing fractions with valid counts.
- Preconditioning is measured only during its explicit phase, including an
  early/late comparison using its two halves. C1's absent phase is not applicable,
  not zero. The two-second pre-bout level describes ongoing activity, which can
  include conditioning activity. Bout-window activity is not necessarily caused
  by the bout; correlations and control differences remain descriptive.
- Keep left and right separate. Compare conditioned stimuli against same-side
  C1, C2 against S1, and S1-S5 against their actual unequal blank delays.
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

Cell 10 reuses the report writer. When enabled, it writes configuration,
readiness, metric tables, original-neuron display mappings and figures under:

```text
Lab/Results/Paper/2p_derived/Exp_9_RF_memory_inhibition/reports/<timestamp>_population_analysis/
```

The population notebook never overwrites individual-fish caches. Validation
reports are local under `.test_cache/exp9/population_run_final/`; cache hashes
were unchanged after the test save.

The updated Cell 01-05 chain was executed on all eight reviewed fish. Checks
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
The population notebook's Cell 01 also refreshes diagnostics and figure helpers
from current source. If an existing kernel cannot find a newly added helper,
rerun Cell 01, then rerun the affected analysis cell; loaded data can be reused.
If a helper reports an unexpected keyword, inspect the loaded module path and
signature before assuming that the current source lacks that parameter.

## Suggested opening message for the next chat

> Continue reviewing Exp 9's `02_population_analysis.ipynb`. Read the README in
> its folder and the repository's routed instructions. Use the numbered cells
> when discussing results. First check Cell 03: I have run other fish, so verify
> their saved output paths and save settings before requesting recomputation.
> Reuse existing `src` functions and preserve individual-fish caches. We will
> review the rasters, bout-window metrics and preconditioning comparisons together.

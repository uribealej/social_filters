# Exp5 Stimulus Specificity Plan

Purpose

Plan the slice-by-slice implementation of neuron-level stimulus specificity analysis for
`scripts/calcium_analysis/Exp5_map_positions_several.ipynb`.

## Target Analysis

Build a configurable neuron-level stimulus specificity analysis that combines:

- mean z-score response strength from `trial_aligned_traces_z_core`
- binary active/inactive stimulus identity from `active_matrices`
- selectivity metrics
- neuron classification
- summary tables
- summary plots
- a re-sorted Cell 18 raster visualization

The analysis must support any selected subset of stimuli and a user-defined
`analysis_label`, such as `Sparseness_Right`, `Sparseness_Left`, or
`Sparseness_All`.

## Recommended Owner Layout

Use a combination of shared helpers and notebook orchestration:

- `src/analysis_tools.py`: reusable response-window and per-stimulus z-score AUC primitives.
- `src/multifish_analysis.py`: all-fish neuron summary table builder that joins z-score responses with `active_matrices`.
- `src/plotting.py`: reusable summary plots and selectivity-sorted raster helper.
- `Exp5_map_positions_several.ipynb`: configuration, selected stimuli, analysis label, display, and narrative.

This follows the repository rule that reusable scientific logic belongs in `src/`,
while notebooks remain orchestration, exploration, and reporting layers.

## Existing Objects To Reuse

- `all_fish_data[fid]["trial_aligned_traces_z_core"]`
- `active_matrices`
- `stimuli_id_map`
- `stimuli_durations`
- `kept_neuron_indices`
- `build_active_neuron_matrices_all_fish`
- `build_pooled_active_trace_diagnostic`
- `plott.plot_active_trace_decision_diagnostic`

## Expected Data Shapes

`trial_aligned_traces_z_core`

- Stored per fish under `all_fish_data[fid]["trial_aligned_traces_z_core"]`.
- Dict mapping `stim_id -> array`.
- Keys are usually integer stimulus IDs, for example `1..14`.
- Arrays are shaped `(n_neurons_full, n_time, n_reps)`.
- In the current notebook config, `t_pre_s=5`, `t_post_s=25`, and `fps_2p=2`, so `n_time=60`.
- Z-score traces are full-neuron arrays.

`active_matrices`

- Dict mapping `fish_id -> pandas.DataFrame`.
- Rows match the filtered/raster neuron set, not the full z-score neuron count.
- Columns are selected stimulus IDs, for example `[7, 3, 4, 5, 6]`.
- Values are binary `0/1`.
- Current pooled Cell 18 diagnostic output is `(1665, 5)`, matching kept neurons across fish.

Important alignment rule:

- Subset z-score arrays with `kept_neuron_indices` before joining to `active_matrices`.

## Naming And Identity Notes

Stimuli are user-facing as `Bcontrol`, `B1`, `B2`, `B3`, and `B4`, but this
notebook uses side-prefixed stimulus names:

- left: `LeBcontrol`, `LeB1`, `LeB2`, `LeB3`, `LeB4`
- right: `RiBcontrol`, `RiB1`, `RiB2`, `RiB3`, `RiB4`

The implementation should accept selected stimuli as IDs or names and preserve
the selected order.

Recommended neuron identity columns:

- `neuron_id`: filtered within-fish row ID, matching raster and Cell 18 order.
- `source_neuron_id`: original z-score/dFoF neuron index from `kept_neuron_indices`.
- `global_neuron_id`: pooled row index across fish.

## Response Window Rule

Use the same response-window logic for every neuron and every selected stimulus.

- Motion begins at `motion_onset_s`, default `8.0` seconds after trial onset.
- Aligned trace time should be computed as:
  `time_s = np.arange(n_time) / fps_2p - t_pre_s`
- Response starts at `motion_onset_s`.
- Response ends at `motion_onset_s + motion_duration_s + tau_s * 2`.
- `motion_duration_s` should come from `stimuli_durations` when available, using `motion_sec`.
- Clip the end to the available trace length.
- Convert seconds to frame masks or frame indices with `fps_2p`.
- Raise a clear error if the clipped response window contains no frames.

## Z-Score Response Extraction

For each fish, neuron, and selected stimulus:

1. Load the selected stimulus array from `trial_aligned_traces_z_core`.
2. Apply `kept_neuron_indices` so rows match `active_matrices`.
3. Restrict to the response window.
4. Compute area under the z-score curve for each repetition.
5. Average across repetitions.

This gives one response value per filtered neuron per selected stimulus.

## Negative Z-Score Handling

Preserve raw AUC responses in output columns such as `response_LeB1`.

For selectivity calculations only:

- Use `response_for_selectivity = max(raw_response, 0)`.
- If all clipped responses are zero, set `simple_selectivity` and
  `lifetime_sparseness` to `NaN`.
- Do not silently convert zero-response cases into apparently valid low-selectivity neurons.

Recommended preference fields:

- `preferred_stimulus`, `max_response`, and `mean_response` should use raw response values for transparency.
- Classification should use positive/clipped responses for weak/strong response thresholding.

## Selectivity Metrics

For each neuron:

- `preferred_stimulus`: selected stimulus with largest raw response.
- `max_response`: maximum raw response.
- `mean_response`: mean raw response across selected stimuli.
- `simple_selectivity`: preferred positive response divided by total positive response.
- `lifetime_sparseness`:

```text
S = (1 - ((mean(R) ** 2) / mean(R ** 2))) / (1 - (1 / N))
```

where `R` is the clipped non-negative response vector for selected stimuli.

Edge cases:

- `N` must be greater than 1.
- If all clipped responses are zero or `mean(R ** 2) == 0`, return `NaN`.
- Divide-by-zero must be explicit and checked.

Binary breadth from `active_matrices`:

- `n_active_stimuli`: number of selected stimuli where the neuron is active.
- `response_breadth`: `n_active_stimuli / N`.

## Proposed Classification Thresholds

Expose all thresholds in the notebook config:

```python
classification_thresholds = {
    "high_lifetime_sparseness": 0.70,
    "intermediate_lifetime_sparseness": 0.40,
    "broad_breadth_threshold": 0.80,
    "strong_response_quantile": 0.75,
    "weak_response_quantile": 0.25,
}
```

Use positive/clipped `max_response` values to derive strong and weak thresholds
within the current `analysis_label`, because AUC units depend on window length
and selected stimuli.

Suggested class order:

1. `Weak/unclear`: low max positive response, all-zero selectivity responses, or NaN sparseness.
2. `Strong broad responder`: response breadth >= broad threshold and max response >= strong threshold.
3. `Stimulus-specific neuron`: `n_active_stimuli == 1` and sparseness >= high threshold.
4. `Subset-selective neuron`: active for more than one but fewer than broad threshold, and sparseness >= intermediate threshold.
5. `Broadly active neuron`: response breadth >= broad threshold and sparseness < intermediate threshold.
6. Otherwise `Weak/unclear`.

The classification must adapt to the number of selected stimuli and avoid
hardcoded five-stimulus assumptions.

## Summary Table Columns

Create one output table with one row per filtered neuron:

- `fish_id`
- `neuron_id`
- `source_neuron_id`
- `global_neuron_id`
- `preferred_stimulus`
- `max_response`
- `mean_response`
- `simple_selectivity`
- `lifetime_sparseness`
- `n_active_stimuli`
- `response_breadth`
- `neuron_class`
- `analysis_label`
- `selected_stimuli`
- per-stimulus response columns, for example `response_LeB1`
- per-stimulus active columns, for example `active_LeB1`

## Plots

Plot 1: lifetime sparseness vs response strength

- x-axis: `lifetime_sparseness`
- y-axis: `max_response`
- color: `preferred_stimulus`
- title includes `analysis_label`

Plot 2: active-stimuli histogram

- x-axis: `n_active_stimuli`
- y-axis: number of neurons
- title includes `analysis_label`

Plot 3: preferred stimulus distribution

- bar plot of neuron counts per selected preferred stimulus
- title includes `analysis_label`

Plot 4: re-sorted Cell 18 raster

Sort neurons by:

1. `preferred_stimulus` in selected stimulus order
2. descending `lifetime_sparseness`
3. descending `max_response`

Then replot using the selected stimulus order.

## Slice Plan

Slice 1

- Add selected-stimuli config and response-window helper.
- Validate frame index calculations on existing notebook objects.
- No behavior changes elsewhere.

Slice 2

- Build the response matrix from `trial_aligned_traces_z_core`.
- Produce neurons x selected-stimuli responses per fish and concatenated across fish.

Slice 3

- Join the response matrix with `active_matrices`.
- Create the neuron summary table with identity columns and per-stimulus columns.

Slice 4

- Add selectivity metrics:
  - `preferred_stimulus`
  - `max_response`
  - `mean_response`
  - `simple_selectivity`
  - `lifetime_sparseness`
  - `n_active_stimuli`
  - `response_breadth`

Slice 5

- Add neuron classification with configurable thresholds.

Slice 6

- Add summary plots:
  - sparseness vs max response scatter
  - active-stimuli histogram
  - preferred-stimulus bar plot

Slice 7

- Re-sort Cell 18 raster by preferred stimulus, lifetime sparseness, and max response.

## Validation Checklist

Before finalizing implementation:

- selected stimuli exist in both `trial_aligned_traces_z_core` and `active_matrices`
- response-window frame indices are valid and clipped explicitly
- kept z-score neuron count matches active-matrix row count per fish
- no silent divide-by-zero in selectivity metrics
- output table has one row per filtered neuron
- plots work for all selected stimuli and subsets
- Cell 18 re-sorted raster row count matches `summary_table.global_neuron_id` order

# `src/` Organization Policy

Purpose

Keep reusable scientific behavior in an identifiable owner while preserving
the repository's supported imports, data contracts, and notebook workflows.
Use the [top router](../workflows/social-filters-router.md) to select a workflow
before opening an owner. The [symbol index](symbol-index.md) locates a named
public helper; the [refactor rules](refactor-rules.md) cover routine edit scope.
This policy describes the present flat `src/` package. It does not require a
directory move or a change to scientific behavior.

## Ownership and dependency direction

| Responsibility | Current owner |
| --- | --- |
| Suite2p fluorescence and aux-trigger extraction | `dff_extraction.py`, `auxtrigger_extraction.py` |
| Experiment bundle loading and output discovery | `data_loading.py`; file lookup helpers in `analysis_io.py` |
| Shared trajectory timing, log-to-trace conversion, and angle interpretation | `stimuli_timeline.py` |
| Trial alignment, response metrics, normalization, classification, selectivity, reliability, and motion metrics | `trial_alignment.py`, `response_*.py`, `reliability.py`, `motion_metrics.py` |
| Significant-trace noise model and rasterization | `significant_trace_detection.py` |
| Several-fish matrices, active-neuron analysis, stimulus specificity/similarity, static/flicker and bout/flicker analyses, and LME features | `multifish_matrices.py`, `active_neuron_analysis.py`, `stimulus_specificity.py`, `stimulus_similarity.py`, `static_flicker_analysis.py`, `bout_flicker_analysis.py`, `lme_feature_decomposition.py` |
| Several-fish cohort loading, selection, diagnostic data, figure orchestration, and reports | Corresponding `several_fish_*.py` modules |
| Reusable figures and display preparation | Relevant `plotting_*.py` module; `neuron_ordering.py` for ordering; `stimulus_visualization.py` for stimulus inspection |

New notebooks and scripts call the narrow owner. A narrow owner may call
another narrow owner for a lower-level calculation. Figure modules may use
analysis and timing results; analysis modules should not depend on a broad
compatibility facade. Keep experiment-specific stimulus generators and
playback wrappers under `scripts/stimuli/`; downstream timing interpretation
belongs in `stimuli_timeline.py`. The preprocessing batch notebook remains the
per-plane and merged dFoF writer stage. See [canonical outputs](canonical-outputs.md)
for the writer and first consumer of each artifact.

Keep a function and its module focused on a clear responsibility. Put a new
calculation in its scientific owner, data discovery or serialization in an I/O
owner, and reusable figure construction in the relevant plotting family.
Keep scientific calculations callable without a notebook. Pass thresholds,
stimulus choices, and other experiment settings as arguments; avoid hidden
global state. Prefer a calculation that returns data over one that also saves,
prints, or plots. Split a larger public workflow into internal calculation
and side-effect helpers when changing it, while preserving its public contract.

Existing public gateways are not all pure calculations. `reliability.py` can
plot and save selected indices, `stimulus_visualization.py` loads trajectories
and saves an inspection report, and `several_fish_figures.py` imports
`slugify_label` from `several_fish_reporting.py`. These are current
boundaries, not patterns to copy into new scientific functions. Changing them
requires a separate owner-scoped change with consumer validation.

## Canonical implementations and public imports

- Maintain one canonical implementation per scientific calculation. Delegate
  historical signatures to that implementation instead of copying equations.
  `significant_trace_detection.py` is the canonical detector; its legacy and
  current modes preserve characterized differences.
- Do not add version-suffixed implementation modules such as `_v3.py`. Use a
  named algorithm, strategy, or explicit mode in the canonical owner. The
  existing `significant_traces.py` and `significant_traces_v2.py` files are
  supported compatibility wrappers, not precedents for new implementations.
- New code imports canonical owners directly. The supported historical paths
  are `analysis_tools`, `multifish_analysis`, `plotting`,
  `reusable_several_fish`, both significant-trace wrappers, and
  `data_loading.transform_stimuli_duration`; their exact callable boundaries
  are in the [S12 compatibility decision](src-compatibility-decision.md).
  Keep those paths working without routine runtime deprecation warnings.
- Do not add scientific or plotting implementations to a compatibility facade.
  Its existing private helper and dependency aliases remain importable for
  compatibility but are not new public API. Do not make new code depend on them.
- A public function's import path, signature, shape, units, ordering, NaN
  treatment, saved output, and observable failure behavior are part of its
  contract. Review a proposed change against the [consumer inventory](src-api-consumer-inventory.md)
  and first downstream consumer before changing any of them.

## Scientific data contracts

- Document input and output shapes and dimension order at the function that
  owns the calculation. Continuous dFoF and raster traces are generally
  `(time, neurons)`; trial-aligned traces are generally
  `(neurons, time, trials)`. State every intentional transpose or different
  convention explicitly.
- Document physical units and conversions: seconds versus frames, sampling
  rates, angles, and integrated-response units. Pass rates and windows
  explicitly rather than relying on experiment-specific constants.
- State how missing values, non-finite values, empty inputs, and insufficient
  trials are handled. Preserve established NaN behavior during structural
  moves; an intentional change needs known-output evidence and consumer review.
- Preserve the configured stimulus order, fish and neuron row order, trial
  order, and table column meaning through pooled matrices, summaries, figures,
  and saved artifacts. Describe any sorting or filtering and return enough
  metadata for a caller to align rows again.
- Preserve canonical filenames, folders, cache keys, table schemas, and
  notebook stage order unless the task explicitly defines a migration. Change
  file-contract semantics at the writer or loader owner, then check the first
  consumer. Do not repair an upstream contract by compensating in a notebook.

## Notebook and script boundary

Notebooks select experiments and data, declare settings, call `src/`, explain
results, and present figures. Keep cells short and ordered. Move repeated
calculation or verbose reusable orchestration into its owner. Keep figure
size, colors, and other figure-specific controls beside each plotting call.
Preprocessing writer cells and experiment-specific stimulus asset scripts keep
their established stages and output names until an explicit migration. See
the [code authoring checklist](code-authoring-checklist.md) for coding and
notebook details, and [reproducibility policy](../../docs/reproducibility_policy.md)
for data and publication policy.

## Evidence, retirement, and documentation

- For a structural move, characterize behavior before changing it and compare
  the owning helper plus its first consumer afterward. Do not combine a
  scientific behavior change with a structural move. For important scientific
  calculations, use a small known-output test or the smallest practical
  data-backed check; static inspection alone does not establish scientific
  equivalence.
- For notebook behavior, compile affected code cells and run the smallest
  affected data-backed chain when inputs are available. For a new or materially
  changed figure, render it and inspect labels, legends, ticks, annotations,
  clipping, and readability. Record unavailable data or dependencies rather
  than claiming a run occurred.
- Do not retire a helper from zero static call counts alone. Check active
  notebook/script consumers, internal calls, documented imports, tests, and
  plausible external use. Removing or warning on a supported facade path
  requires a separate compatibility decision, migration route, and checks of
  signatures, scientific outputs, and first consumers.
- When a public helper changes, update `symbol-index.md` and the relevant
  router. Update `src-api-consumer-inventory.md` and its machine contract when
  consumer imports or callable contracts change. Update the relevant stage map
  or `canonical-outputs.md` when stage order or file contracts change. Record
  an incomplete slice in `recent-changes.md`; remove its handoff when complete.

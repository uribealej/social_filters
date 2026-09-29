# S12 Compatibility Decision

Purpose

Record which historical `src` imports remain supported after active notebook
and script consumers moved to their implementation owners.

## Supported paths

| Historical path | Supported callable surface | Canonical owner |
| --- | --- | --- |
| `src.analysis_tools` | Its explicit `__all__` functions, including documented analysis and reliability helpers | `trial_alignment`, `response_*`, `motion_metrics`, `reliability`, `analysis_io`, and `plotting_reliability` |
| `src.multifish_analysis` | Documented public analysis functions and the characterized `compute_stimulus_selectivity_metrics` re-export | `multifish_matrices`, `static_flicker_analysis`, `bout_flicker_analysis`, `stimulus_specificity`, `stimulus_similarity`, `active_neuron_analysis`, and `response_selectivity` |
| `src.plotting` | Documented public plotting functions | The relevant `plotting_*` module, `neuron_ordering`, or `stimuli_timeline` |
| `src.reusable_several_fish` | Documented several-fish loading, selection, diagnostics, figure, and reporting functions | `several_fish_loading`, `several_fish_selection`, `several_fish_diagnostics`, `several_fish_figures`, and `several_fish_reporting` |
| `src.significant_traces` | Historical V1 function signatures and legacy-mode behavior | `significant_trace_detection` |
| `src.significant_traces_v2` | Historical V2 function signatures and current-mode behavior | `significant_trace_detection` |
| `src.data_loading.transform_stimuli_duration` | Historical timing-normalization callable | `stimuli_timeline.transform_stimuli_duration` |

The broad facades' documented callable names are in `symbol-index.md`;
`analysis_tools.__all__` and the mode-specific facade modules make their other
supported boundaries explicit. The owners and characterization tests define
signatures and behavior. Keep these paths working without runtime deprecation warnings. A
warning would add routine notebook output, and no removal date or external
consumer migration has been established.

## Boundary

- New notebooks, scripts, and reusable `src` implementations import the
  canonical owner directly. The active repository notebooks and scripts have
  no imports of the six broad or versioned facades.
- The broad facades remain thin compatibility imports, with no independent
  scientific or figure implementation. `src.reusable_several_fish` still
  exposes its historical `at`, `mfa`, and `plott` module aliases for old code;
  those aliases are compatibility links, not new owner dependencies.
- Historical underscore-prefixed helpers and third-party dependency aliases
  currently exposed by the facades remain importable to avoid an unmeasured
  break. They are not promoted to new stable callable API by this decision.
  Do not add new work through those aliases.
- A later removal or warning needs a separate review of external use, a
  documented migration path for each affected symbol, and checks of its
  characterized signatures, scientific outputs, and first consumers. Zero
  active repository notebook calls alone is insufficient evidence.

S12 step 5 moved the remaining direct facade calls in `analysis_tools`,
`data_loading`, `reliability`, and `several_fish_figures` to their owner modules. No facade
export, function signature, output name, or notebook call was removed.

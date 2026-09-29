# Project Reproducibility and Coding Structure

## Goal

Keep the project easy to understand, reproducible, and publication-ready without making the working analysis unnecessarily complicated.

Apply this policy when writing, editing, or reviewing project code. The coding guidance incorporates the supplied rules distilled from *The Good Research Code Handbook* (goodresearch.dev) and Larsch lab conventions.

For routine Python and notebook changes, agents use the compact [code-authoring checklist](../.agents/references/code-authoring-checklist.md). This document is read in full when changing data policy, repository architecture, publication workflows, or the checklist itself.

The main principle is:

```text
processed calcium data
        ↓
shared analysis code in src/
        ↓
figure/experiment-specific settings
        ↓
notebooks
        ↓
publication figures
```

We keep only the essential processed data and recalculate lightweight analysis outputs when needed.

---

## Repository structure

The layouts below are illustrative. Follow the current ownership and canonical file contracts in [AGENTS.md](../AGENTS.md) and its routed references; this policy does not require moving existing files or renaming outputs.

```text
project/
├── README.md
├── environment.yml
│
├── src/
│   └── thalamus_analysis/
│       ├── alignment.py
│       ├── responses.py
│       ├── activity.py
│       ├── selectivity.py
│       └── plotting.py
│
├── scripts/
│   └── prepare_data.py
│
├── experiments/
│   ├── experiment_01/
│   │   ├── config.py
│   │   └── figures/
│   │       ├── figure_1.ipynb
│   │       └── figure_2.ipynb
│   │
│   └── experiment_02/
│       ├── config.py
│       └── figures/
│
└── docs/
```

Large processed datasets stay outside Git.

Example:

```text
published_data/
└── experiment_01/
    ├── fish_01/
    │   ├── merged_dfof.npy
    │   ├── significant_raster.npy
    │   ├── neuron_map.npy
    │   └── metadata.json
    │
    ├── fish_02/
    └── stimuli/
        ├── stimulus_timing.csv
        └── stimulus_definitions.csv
```

---

## Data policy

Save the essential processed data from which the analysis can be reproduced:

- merged ΔF/F traces
- significant traces / significant raster
- neuron map and neuron IDs
- stimulus definitions
- stimulus timing
- basic metadata

Do **not** permanently save lightweight derived values such as:

- AUC
- peak response
- active-neuron classifications
- lifetime sparseness
- left-right index
- selectivity index

These should be recalculated from the processed data using the shared `src/` functions.


---

## `src/` policy

`src/` owns the scientific calculations.

Examples:

- `align_trials()`
- `compute_auc()`
- `compute_peak()`
- `detect_active_neurons()`
- `compute_lifetime_sparseness()`
- `compute_left_right_index()`

Rules:

1. One function = one clear job. Keep functions short, roughly one screen (about 40 lines), splitting by responsibility rather than an arbitrary line limit.
2. Use descriptive function names.
3. Inputs and outputs must be explicit. Prefer pure functions; pass state as arguments or encapsulate it in a class rather than relying on globals. Choose either in-place mutation or returning a new result; do not combine both contracts in one function.
4. Document expected array shapes and units.
5. Computation functions should return data rather than automatically save files. Concentrate file reads/writes, network calls, and printing in dedicated I/O helpers, separate from scientific calculations.
6. Scientific thresholds must be function parameters, not hidden constants.
7. Do not include experiment names, fish names, or stimulus names inside shared functions.
8. Use one canonical function for each calculation.
9. Important scientific functions should have small tests with known expected outputs.
10. Keep dimension conventions consistent across the project.

Example convention:

```text
continuous traces: time × neurons
trial-aligned data: neurons × time × trials
```

Example:

```python
def compute_auc(traces, time_s, response_window):
    """Compute response AUC.

    Args:
        traces (numpy.ndarray): Traces with shape (neurons, time, trials).
        time_s (numpy.ndarray): Sample times in seconds, shape (time,).
        response_window (tuple[float, float]): Start and end times in seconds.

    Returns:
        numpy.ndarray: AUC with shape (neurons, trials), in trace units × seconds.
    """
```

---

## Coding style and documentation

### Naming and readable implementation

- Use `snake_case` for variables, functions, and modules, `CamelCase` for classes, and `UPPER_CASE` for constants.
- Follow PEP 8 with four-space indentation. Keep lines readable without mechanically wrapping at 80 columns.
- Use descriptive names instead of single-letter or cryptic names (for example, `word`, `line`, and `key`).
- Keep imports at the top of modules, outside functions and `if __name__ == "__main__":` blocks.
- Use f-strings and named constants instead of unexplained numbers or strings. Experiment-dependent scientific thresholds remain explicit parameters, as required by the `src/` policy.
- Prefer standard-library tools over hand-written equivalents. Favor clear loops over dense expressions when performance is not a bottleneck; vectorize operations over full tracking videos or imaging traces.
- Reduce nesting with guard clauses and early `continue` or `return` statements.
- Use a named `def` for reused or nontrivial logic rather than a lambda. Short, one-off sort keys such as `key=lambda item: item.name` are acceptable.
- Assign multistep computations to a descriptive result variable before returning it, so the result is easy to inspect during debugging.

### Documentation and failures

- Start every Python module with a short docstring explaining its purpose, before the imports.
- Give every function a Google-style docstring with a summary, argument types, and return type; include shapes and units where relevant, as in the example above.
- Document types in docstrings rather than adding inline signature annotations such as `def function(name: str) -> int:`.
- Add brief inline comments for non-obvious operations: explain a tricky slice, the reason for a constant, or what a loop iteration represents.
- Report invalid configuration and programmer errors explicitly with descriptive exceptions. Use appropriate errors such as `ValueError` or `NotImplementedError`; reserve assertions for internal invariants rather than user-input validation.
- At the outer batch boundary, warn and skip missing or invalid per-fish/per-file data when items can be processed independently. Include the item identifier and reason, summarize skipped items, and never silently claim that an incomplete batch succeeded. Do not suppress configuration errors or unexpected programming failures.

---

## Notebook policy

Notebooks should be thin and easy to read.

They should mainly:

1. define experiment/figure settings,
2. load processed data,
3. call functions from `src/`,
4. select neurons/stimuli,
5. create plots.

Avoid putting long scientific calculations directly inside notebooks.

Exploratory calculations may start in a cell, but the finished workflow should call reusable functions in `src/`. Any calculation used in more than one notebook must have a shared owner there.

- Use short, single-purpose code cells, each preceded by one or two sentences of Markdown explaining its purpose without restating the code.
- Put imports and configuration constants near the top. Name configuration constants in `UPPER_CASE`, distinct from ordinary working variables.
- Keep execution dependencies explicit and ordered. A step should consume the preceding step's result or clearly declared setup/configuration, not stale variables from exploratory or out-of-order execution.
- Keep figure-specific controls (axes, colors, thresholds, and figure size) beside the plotting call so they remain easy to adjust interactively. Short exploratory plotting may remain in cells; reusable or verbose plot construction belongs in the appropriate `src/` helper.

Before publication, every notebook should:

```text
Restart kernel
→ Run all cells
→ Finish without hidden manual steps
→ Reproduce the expected figure
```

---

## Experiment-specific configuration

Stimulus generation has shared configuration files under `configs/stimuli/`.
Analysis settings that vary between experiments or figures stay close to the corresponding experiment.

Examples:

```python
STIMULI = ["R300Le", "F300Le", "LeBcontrol"]
SIDE = "left"
MIN_ACTIVE_REPS = 0.5
LIFETIME_SPARSENESS_THRESHOLD = 0.75
```

The distinction is:

```text
src/      = how the calculation is performed
config    = what this experiment/figure should analyze
notebook  = calls the analysis and creates the figure
```

---

## Git policy

Keep in Git:

```text
src/
scripts/
experiments/
notebooks
documentation
environment files
small metadata
```

Keep outside Git:

```text
large calcium datasets
merged ΔF/F arrays
significant rasters
large neuron maps
raw imaging data
```

Commit early and often, with each commit representing one coherent change and a meaningful message describing what actually changed.

When the user explicitly delegates committing to an agent:

- Run `git status` and inspect `git diff`; show the actual changes intended for staging before committing.
- Stage the intended paths or hunks explicitly rather than blindly using `git add -A`.
- Split unrelated changes into separate commits, or clarify which change is intended when the scope is ambiguous. Preserve unrelated user work.

An agent must not run `git push` without an explicit request to push. Likewise, `git rebase`, `git commit --amend`, `git push --force`, and other history-rewriting operations each require an explicit request for that specific action; authorization to commit alone does not authorize them.

---

## Reproducibility rule

Someone should eventually be able to:

```text
1. Clone the repository
2. Download the processed dataset
3. Install the environment
4. Open/run the relevant notebook
5. Reproduce the publication figure
```

The analysis logic should remain centralized in `src/` so every experiment uses the same scientific calculations.

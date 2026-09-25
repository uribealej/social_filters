# Code Authoring Checklist

Purpose

Apply the practical coding rules from `docs/reproducibility_policy.md` without rereading the full policy for every edit. Read this file before creating or modifying Python or notebook code, then follow the relevant workflow router and owner-specific reference.

The full reproducibility policy remains the source of truth. Read it when changing repository architecture, data/provenance policy, publication workflows, or these rules.

## Shared ownership

- Put reusable scientific calculations, timing semantics, and reusable plotting in the narrowest owning `src/` module.
- Keep experiment, fish, stimulus, threshold, color, and figure settings outside shared functions.
- Preserve canonical output names, layouts, dimension conventions, and stage order unless the task explicitly includes a migration.
- Fix scientific and file-contract behavior at its owner rather than compensating in a downstream notebook.

## When writing `src/` code

- Give each function one clear responsibility and a descriptive name; split long functions by responsibility.
- Make inputs and returned outputs explicit. Prefer pure functions and avoid hidden global state.
- Choose either in-place mutation or returning a new result; do not mix both contracts.
- Document array shapes, dimension order, units, arguments, and return values in the docstring.
- Pass scientific thresholds and experiment-dependent choices as parameters; do not hide them as constants.
- Keep experiment names, fish IDs, and stimulus names out of reusable functions.
- Separate scientific calculations from file I/O, network access, saving, and printing.
- Maintain one canonical implementation for each calculation instead of copying logic.
- Raise descriptive exceptions for invalid configuration or inputs. Never silently report a partial batch as complete.
- Add the smallest known-output test or focused validation practical for important scientific behavior.

## When writing notebooks

- Use notebooks to declare settings, load processed data, call `src/`, select data, and present results.
- Keep imports and `UPPER_CASE` configuration values near the top.
- Use short, single-purpose code cells with a brief Markdown explanation before each analysis step.
- Keep execution ordered and explicit; do not depend on stale variables or out-of-order execution.
- Move calculations reused by multiple notebooks, or verbose orchestration, into the owning `src/` module.
- Keep figure-specific controls beside the plotting call; move reusable plot construction into `src/`.
- Keep routine output minimal and make skipped fish, files, or incomplete batches visible.
- For publication validation, restart the kernel, run all cells without manual intervention, and reproduce the expected figure.

## Validation before handoff

- Run the smallest practical smoke check or data-backed execution for the changed owner.
- When a public helper or writer changes, also check its first downstream consumer.
- Render and inspect every new or materially changed figure for clipping, overlaps, unreadable labels, and misleading layout.
- State what was validated and what still requires data-backed or publication-level validation.

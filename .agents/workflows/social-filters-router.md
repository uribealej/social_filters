# social_filters Router

Purpose

Dispatch future work to the smallest workflow profile and reference doc that matches the real owner in this notebook-heavy calcium imaging and stimulus-analysis repo.

For everyday scientific analysis and new `src/` work, use the
[organization policy](../references/src-organization-policy.md) for current
ownership and contract rules. The workflow profile then points to the
concrete owner and first consumer.

Use this file when

- You have not yet classified the task.
- The task mentions a notebook, output artifact, or scientific behavior and you need the correct owner layer first.
- You need to decide whether the work belongs to calcium preprocessing, calcium analysis, or stimulus authoring.

## Dispatch
Use this table to choose a workflow profile. Its task table selects any needed reference and the first file to inspect. The listed documents are choices, not a reading sequence. Do not open `archive/` or `.agents/history/` unless the user explicitly requests historical inspection or restoration.

## Workflow profile dispatch

| Primary target or query signal | Open this router or reference | Typical owner layer |
| --- | --- | --- |
| Documentation-only, test-only, or repository guidance change without a scientific workflow change | Open the target document or test directly; use a workflow profile only if an owner or output contract needs checking | Target file and the code or contract it describes |
| `src/` owner placement, dependency direction, public API, compatibility, retirement, or validation policy | [src-organization-policy.md](../references/src-organization-policy.md); then the matching workflow router for the concrete owner | Narrow `src/` owner and its first consumer |
| Historical S00–S14 rationale or validation evidence | [Completed roadmap](../history/src-refactor-roadmap.md) and its linked completion records; consult only for the specific historical question | History, then the current owner and policy for new work |
| `src/dff_extraction.py`, `src/auxtrigger_extraction.py`, `DeltaFF_batch_pipeline.ipynb`, `2P_Experiment_FileOps.ipynb`, merged dFoF writer behavior | `calcium-preprocessing-router.md` | `src/` extraction code plus preprocessing notebooks |
| `src/data_loading.py`, `src/reusable_several_fish.py`, narrow analysis or plotting modules and their compatibility facades, `src/significant_trace_detection.py`, notebooks under `notebooks/calcium/`, response classification, alignment, plotting, all-fish summaries | `calcium-analysis-router.md` | `src/` analysis modules plus analysis notebooks |
| `scripts/stimuli/generation/*.py`, `scripts/stimuli/playback/*.py`, `configs/stimuli/*.json`, `notebooks/stimuli/*.ipynb`, trajectory generation, flicker/rocking behavior, stimulus timing handoff | `stimulus-authoring-router.md` | Stimulus scripts plus shared timing owner in `src/stimuli_timeline.py` |

## Compact scaling rule
- Add a new notebook or script to an existing workflow profile when it shares the same owner modules, output semantics, and validation surface.
- Create a new workflow profile only if the repo grows a genuinely different stage map, handoff log, and owner stack.

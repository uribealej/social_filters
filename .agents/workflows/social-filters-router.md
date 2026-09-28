# social_filters Router

Purpose

Dispatch future work to the smallest workflow profile and reference doc that matches the real owner in this notebook-heavy calcium imaging and stimulus-analysis repo.

Use this file when

- You have not yet classified the task.
- The task mentions a notebook, output artifact, or scientific behavior and you need the correct owner layer first.
- You need to decide whether the work belongs to calcium preprocessing, calcium analysis, or stimulus authoring.

## Read this first
- Follow the startup order and repository rules in `AGENTS.md`; use this table to select task-specific guidance.
- Do not open `archive/` or `.agents/history/` unless the user explicitly requests historical inspection or restoration.

## Workflow profile dispatch

| Primary target or query signal | Open this router | Read this reference first | Typical owner layer |
| --- | --- | --- | --- |
| Repository-wide `src/` diagnosis, organization, API migration, dead-code review, or multi-session refactor | Relevant owner router after the cross-cutting references | `src-refactor-roadmap.md` status tracker and relevant unfinished slice; `src-diagnosis.md` only for dated audit evidence; `src-api-consumer-inventory.md` before moving a public helper | Multiple `src/` owners, one roadmap slice at a time |
| `src/dff_extraction.py`, `src/auxtrigger_extraction.py`, `DeltaFF_batch_pipeline.ipynb`, `Baseline_Evaluation_single_plane.ipynb`, `STD_Zcore_threshold_analyis.ipynb`, `2P_Experiment_FileOps.ipynb`, merged dFoF writer behavior | `calcium-preprocessing-router.md` | `canonical-outputs.md` for writer/file-contract issues, otherwise `calcium-preprocessing-stage-map.md` | `src/` extraction code plus preprocessing notebooks |
| `src/data_loading.py`, `src/analysis_tools.py` or its narrow S06 owners, `src/plotting.py`, `src/significant_trace_detection.py`, notebooks under `notebooks/calcium/`, response classification, alignment, plotting, all-fish summaries | `calcium-analysis-router.md` | `current-state.md` for mixed-state notebook/path issues, otherwise `calcium-analysis-stage-map.md` | `src/` analysis modules plus analysis notebooks |
| `scripts/stimuli/generation/*.py`, `scripts/stimuli/playback/*.py`, `configs/stimuli/*.json`, `notebooks/stimuli/*.ipynb`, trajectory generation, flicker/rocking behavior, stimulus timing handoff | `stimulus-authoring-router.md` | `stimulus-authoring-stage-map.md` or `canonical-outputs.md` for file-layout questions | Stimulus scripts plus shared timing owner in `src/stimuli_timeline.py` |

## Cross-workflow invariants
- For a repository-wide `src/` refactor, follow `src-refactor-roadmap.md` in ordered ownership slices and update its status tracker at each completed slice.
- Treat wrappers and migration utilities as wrapper owners only; they do not become business-logic authority.
- Timing semantics belong at the writer or shared timing layer, especially `src/stimuli_timeline.py`, not in downstream plotting code.
- After changing a writer stage or public helper, verify the first downstream consumer as well as the edited owner.

## Compact scaling rule
- Add a new notebook or script to an existing workflow profile when it shares the same owner modules, output semantics, and validation surface.
- Create a new workflow profile only if the repo grows a genuinely different stage map, handoff log, and owner stack.

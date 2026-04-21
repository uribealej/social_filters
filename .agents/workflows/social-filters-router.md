# social_filters Router

Purpose

Dispatch future work to the smallest workflow profile and reference doc that matches the real owner in this notebook-heavy calcium imaging and stimulus-analysis repo.

Use this file when

- You have not yet classified the task.
- The task mentions a notebook, output artifact, or scientific behavior and you need the correct owner layer first.
- You need to decide whether the work belongs to calcium preprocessing, calcium analysis, or stimulus authoring.

## Read this first
- Start at `AGENTS.md`, then use this dispatch table.
- After choosing a workflow router, read the smallest relevant reference doc before opening source files.
- Open `src/` owners before large notebooks whenever the notebook is calling shared logic.

## Workflow profile dispatch

| Primary target or query signal | Open this router | Read this reference first | Typical owner layer |
| --- | --- | --- | --- |
| `src/dff_extraction.py`, `src/auxtrigger_extraction.py`, `DeltaFF_batch_pipeline.ipynb`, `Baseline_Evaluation_single_plane.ipynb`, `STD_Zcore_threshold_analyis.ipynb`, `2P_Experiment_FileOps.ipynb`, merged dFoF writer behavior | `calcium-preprocessing-router.md` | `canonical-outputs.md` for writer/file-contract issues, otherwise `calcium-preprocessing-stage-map.md` | `src/` extraction code plus preprocessing notebooks |
| `src/data_loading.py`, `src/analysis_tools.py`, `src/plotting.py`, `src/significant_traces.py`, single-fish notebooks, several-fish notebooks, `count_nuerons.ipynb`, response classification, alignment, plotting, all-fish summaries | `calcium-analysis-router.md` | `current-state.md` for mixed-state notebook/path issues, otherwise `calcium-analysis-stage-map.md` | `src/` analysis modules plus analysis notebooks |
| `scripts/stimuli/*.py`, `scripts/stimuli/*.json`, `plots_stimuli.ipynb`, `try_projection.py`, trajectory generation, flicker/rocking behavior, stimulus timing handoff | `stimulus-authoring-router.md` | `stimulus-authoring-stage-map.md` or `canonical-outputs.md` for file-layout questions | Stimulus scripts plus shared timing owner in `src/stimuli_timeline.py` |

## Cross-workflow invariants
- Prefer edits in `src/` over edits in notebooks when the behavior is reusable or already has a module owner.
- Preserve canonical outputs and stage semantics written by upstream stages; do not redefine them in downstream notebooks.
- Treat wrappers and migration utilities as wrapper owners only; they do not become business-logic authority.
- Timing semantics belong at the writer or shared timing layer, especially `src/stimuli_timeline.py`, not in downstream plotting code.
- After changing a writer stage or public helper, verify the first downstream consumer as well as the edited owner.

## Compact scaling rule
- Add a new notebook or script to an existing workflow profile when it shares the same owner modules, output semantics, and validation surface.
- Create a new workflow profile only if the repo grows a genuinely different stage map, handoff log, and owner stack.

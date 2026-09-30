# S13 completion record

S13 adopted the permanent [organization policy](../references/src-organization-policy.md)
after S12 closed. The policy is linked from `AGENTS.md`, the top router, and
all three workflow routers. It states current `src/` ownership, dependency
direction, public and compatibility API boundaries, scientific data contracts,
notebook boundaries, validation, retirement, and documentation rules.

## Work and findings

1. Audited the flat `src/` layout. Extraction, loading, timing, narrow
   single-fish and several-fish analysis, figure families, and reports have
   named owners. The broad analysis, multifish, plotting, and several-fish
   modules and both significant-trace wrappers remain compatibility surfaces.
2. Described existing mixed public gateways accurately: `reliability` can
   optionally plot and save indices, `stimulus_visualization` loads and saves
   inspection reports, and `several_fish_figures` uses a report naming helper.
3. Drafted and linked the policy. The workflow routers still send concrete
   tasks to the smallest owner or writer stage.
4. Corrected stale Markdown in the Exp 5 LME notebook to name
   `src.multifish_matrices` as its response-matrix owner. All 12 code cell
   sources and the notebook's 21-cell structure remained unchanged.

## Final validation

- The [S12 completion record](src-refactor-s12.md) documents 98 passing tests
  and representative real-data and figure checks. S14 retains the final
  end-to-end validation.
- All 30 owner modules sampled from the policy exist. Import inspection found
  no broad-facade imports inside `src/` except the documented historical
  `at`, `mfa`, and `plott` aliases in `reusable_several_fish`.
- The policy links from `AGENTS.md` and the four routers resolve. Checked routes
  for merged dFoF, trial response metrics, facade compatibility, stimulus
  timing, and several-fish reports against their owner modules. The merged
  dFoF route also identifies the first downstream loader.
- Reviewed documentation diffs, the policy's relative links, notebook JSON,
  and whitespace. No scientific function, writer contract, or notebook code
  cell changed in S13.

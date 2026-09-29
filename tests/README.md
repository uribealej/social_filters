# Test suite

Purpose

Protect the current scientific and file-contract behavior while `src/` is reorganized in small refactor slices.

Activate an environment created from `environment.yml` (default name: `social_filters`), then run the complete suite from the repository root:

```powershell
python -m unittest discover -s tests -t . -v
```

## Current foundation

- `tests/smoke/test_source_contracts.py` compiles every `src/*.py` file and imports every required module in an isolated subprocess.
- `src.auxtrigger_extraction` now passes the required module-import check without `ScanImageTiffReader`; extraction itself requires the reader and reports how to install it.
- `tests/smoke/test_notebook_contracts.py` transforms IPython syntax and compiles every active notebook code cell.
- The known syntax failure in code cell 2 of `notebooks/calcium/exp_08_accumulation_ev/02_bout_flicker_position_onset.ipynb` is registered explicitly. Any new failure fails the suite; fixing the registered failure also requires removing it from the registry.
- `tests/support/fixtures.py` provides deterministic synthetic arrays, stimulus metadata, rasters, and tables for later characterization slices.
- `tests/support/assertions.py` provides reusable array, DataFrame, and figure-contract checks.
- `tests/contracts/src_api_consumers.json` records every active notebook/script `src` import and referenced symbol, including consumers with no current `src` dependency.
- `tests/contracts/test_src_api_consumers.py` detects consumer drift, missing referenced symbols, and active symbols missing from `.agents/references/symbol-index.md`.
- `tests/significant_traces/test_v1_v2_characterization.py` compares all 13 historical V1/V2 stages, edge-case behavior, random-state contracts, and legacy mixed-workflow reproducibility.
- `tests/significant_traces/test_consolidation.py` requires one canonical detector owner and checks both compatibility facades against explicit current and legacy modes.
- `tests/analysis/test_analysis_tools_split.py` protects the S06 owner boundaries, public facade, known numerical outputs, first downstream consumers, and moved reliability diagnostics.
- `tests/analysis/test_multifish_split.py` protects the S07 scientific owners and the `multifish_analysis` compatibility facade.
- `tests/plotting/test_plotting_split.py` and `tests/plotting/test_plotting_consumers.py` protect the S08 figure owners, public facade, and active consumer calls.
- `tests/analysis/test_several_fish_reporting.py` checks the S09 report owner, historical notebook import, temporary report files, export fallback, and the Exp 1 notebook report cell.
- `tests/analysis/test_several_fish_loading.py` checks the S09 cohort-loading owner, historical notebook import, ordered bundle, failure paths, and the Exp 1 notebook loading cell.
- `tests/analysis/test_several_fish_selection.py` checks the S09 pure selection owner, historical imports, known summary and keep-mask outputs, row filtering, and the shared notebook summary call.
- `tests/analysis/test_several_fish_diagnostics.py` checks the S09 pure diagnostic owner, response windows, prepared array order, historical imports, and unchanged shared notebook high-sparseness, mean-trace, and overlap calls.
- `tests/plotting/test_several_fish_figures.py` checks S09 figure ownership, pure raster/sparseness/count inputs, unchanged notebook figure calls, and representative rendered figures.
- `tests/analysis/test_data_loading.py` checks S10 loader path and cache discovery, optional merged maps, and the first alignment consumer.
- `tests/preprocessing/test_dff_extraction.py` checks S10 known-output baseline and ROI filtering plus a synthetic one-plane Suite2p extraction call.
- `tests/preprocessing/test_auxtrigger_extraction.py` checks S10 lazy TIFF-reader loading, missing-dependency errors, and known trigger events.

## Known environment limitations

- The Windows `social_filters_user` environment previously hung in threaded MKL during both small BLAS operations and Matplotlib rendering. `environment.yml` now pins OpenBLAS, and the test suite probes a small BLAS dot product before requiring an isolated Matplotlib Agg render to pass.

Future tests should be grouped by owner domain and should use the shared fixtures and assertions where practical.

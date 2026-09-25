# Test suite

Purpose

Protect the current scientific and file-contract behavior while `src/` is reorganized in small refactor slices.

Run the complete suite from the repository root with the supported environment:

```powershell
& 'C:\Users\larsch-lab\.conda\envs\social_filters_openblas\python.exe' -m unittest discover -s tests -t . -v
```

## Current foundation

- `tests/smoke/test_source_contracts.py` compiles every `src/*.py` file and imports every required module in an isolated subprocess.
- `src.auxtrigger_extraction` is an explicit optional-import check because the active environment does not currently provide `ScanImageTiffReader`.
- `tests/smoke/test_notebook_contracts.py` transforms IPython syntax and compiles every active notebook code cell.
- The known syntax failure in code cell 2 of `notebooks/calcium/exp_08_accumulation_ev/02_bout_flicker_position_onset.ipynb` is registered explicitly. Any new failure fails the suite; fixing the registered failure also requires removing it from the registry.
- `tests/support/fixtures.py` provides deterministic synthetic arrays, stimulus metadata, rasters, and tables for later characterization slices.
- `tests/support/assertions.py` provides reusable array, DataFrame, and figure-contract checks.
- `tests/contracts/src_api_consumers.json` records every active notebook/script `src` import and referenced symbol, including consumers with no current `src` dependency.
- `tests/contracts/test_src_api_consumers.py` detects consumer drift, missing referenced symbols, and active symbols missing from `.agents/references/symbol-index.md`.
- `tests/significant_traces/test_v1_v2_characterization.py` compares all 13 historical V1/V2 stages, edge-case behavior, random-state contracts, and legacy mixed-workflow reproducibility.
- `tests/significant_traces/test_consolidation.py` requires one canonical detector owner and checks both compatibility facades against explicit current and legacy modes.
- `tests/analysis/test_analysis_tools_split.py` protects the S06 owner boundaries, public facade, known numerical outputs, first downstream consumers, and moved reliability diagnostics.

## Known environment limitations

- The Windows `social_filters_user` environment previously hung in threaded MKL during both small BLAS operations and Matplotlib rendering. `environment.yml` now pins OpenBLAS, and the test suite probes a small BLAS dot product before requiring an isolated Matplotlib Agg render to pass.

Future tests should be grouped by owner domain and should use the shared fixtures and assertions where practical.

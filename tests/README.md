# Test suite

Purpose

Protect the current scientific and file-contract behavior while `src/` is reorganized in small refactor slices.

Run the complete suite from the repository root with the supported environment:

```powershell
& 'C:\Users\larsch-lab\.conda\envs\social_filters_user\python.exe' -m unittest discover -s tests -t . -v
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

## Known environment limitations

- In the active `social_filters_user` environment, a minimal Matplotlib Agg canvas draw exits with native Windows exception `0xc06d007f`. The render capability is probed in an isolated subprocess and reported as a skipped test so it cannot crash the suite. Figure-producing refactor slices still require screenshot validation in a repaired rendering environment.

Future tests should be grouped by owner domain and should use the shared fixtures and assertions where practical.

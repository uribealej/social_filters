"""Smoke tests for source compilation and module imports."""

from __future__ import annotations

import os
import json
import subprocess
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
MATPLOTLIB_CONFIG_ROOT = REPOSITORY_ROOT / ".test_cache" / "matplotlib"


def _source_modules() -> list[str]:
    """Return importable module names for top-level Python files in `src`."""

    return [
        f"src.{path.stem}"
        for path in sorted(SOURCE_ROOT.glob("*.py"))
        if path.name != "__init__.py"
    ]


def _import_many_in_subprocess(module_names: list[str]) -> subprocess.CompletedProcess[str]:
    """Import modules together while retaining per-module failure details."""

    MATPLOTLIB_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(MATPLOTLIB_CONFIG_ROOT)
    code = (
        "import importlib, json; "
        f"modules={module_names!r}; "
        "failures={}; "
        "exec('for module_name in modules:\\n"
        "    try:\\n"
        "        importlib.import_module(module_name)\\n"
        "    except Exception as exc:\\n"
        "        failures[module_name] = f\"{type(exc).__name__}: {exc}\"'); "
        "print('SRC_IMPORT_FAILURES=' + json.dumps(failures)); "
        "raise SystemExit(bool(failures))"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


class SourceContractTests(unittest.TestCase):
    """Protect basic source readability and importability contracts."""

    def test_every_source_file_compiles(self) -> None:
        """Compile every active source module without writing bytecode."""

        failures = []
        for path in sorted(SOURCE_ROOT.glob("*.py")):
            try:
                compile(path.read_text(encoding="utf-8"), str(path), "exec")
            except Exception as exc:  # pragma: no cover - reports future failures
                failures.append(f"{path.relative_to(REPOSITORY_ROOT)}: {exc}")

        self.assertEqual([], failures, "Source compilation failures:\n" + "\n".join(failures))

    def test_required_modules_import(self) -> None:
        """Import every source module, including optional-workflow modules."""

        result = _import_many_in_subprocess(_source_modules())
        marker_lines = [
            line
            for line in result.stdout.splitlines()
            if line.startswith("SRC_IMPORT_FAILURES=")
        ]
        self.assertTrue(marker_lines, result.stderr.strip() or result.stdout.strip())
        failures = json.loads(marker_lines[-1].split("=", 1)[1])
        self.assertEqual({}, failures, "Required module import failures")


if __name__ == "__main__":
    unittest.main()

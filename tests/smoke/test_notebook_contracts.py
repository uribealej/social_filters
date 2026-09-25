"""Compile active notebook code cells after transforming IPython syntax."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from IPython.core.inputtransformer2 import TransformerManager


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOT = REPOSITORY_ROOT / "notebooks"

# Values are one-based code-cell numbers, not raw notebook cell indexes.
EXPECTED_SYNTAX_FAILURES = {
    "notebooks/calcium/exp_08_accumulation_ev/02_bout_flicker_position_onset.ipynb": {2},
}


class NotebookContractTests(unittest.TestCase):
    """Protect notebook code cells from new syntax regressions."""

    def test_code_cells_have_only_registered_syntax_failures(self) -> None:
        """Compile transformed cells and compare failures with the registry."""

        transformer = TransformerManager()
        actual_failures: dict[str, set[int]] = {}
        failure_details = []

        for notebook_path in sorted(NOTEBOOK_ROOT.rglob("*.ipynb")):
            relative_path = notebook_path.relative_to(REPOSITORY_ROOT).as_posix()
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
            code_cell_number = 0

            for raw_cell_index, cell in enumerate(notebook.get("cells", [])):
                if cell.get("cell_type") != "code":
                    continue

                code_cell_number += 1
                source = "".join(cell.get("source", []))
                try:
                    transformed = transformer.transform_cell(source)
                    compile(
                        transformed,
                        f"{relative_path}::code_cell_{code_cell_number}",
                        "exec",
                    )
                except SyntaxError as exc:
                    actual_failures.setdefault(relative_path, set()).add(code_cell_number)
                    failure_details.append(
                        f"{relative_path}: code cell {code_cell_number} "
                        f"(raw cell {raw_cell_index}, line {exc.lineno}): {exc.msg}"
                    )

        self.assertEqual(
            EXPECTED_SYNTAX_FAILURES,
            actual_failures,
            "Notebook syntax-failure registry changed. Fix new failures or remove "
            "repaired failures from EXPECTED_SYNTAX_FAILURES.\n"
            + "\n".join(failure_details),
        )


if __name__ == "__main__":
    unittest.main()

"""Extract active `src` imports and symbol use from notebooks and scripts."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from IPython.core.inputtransformer2 import TransformerManager


def discover_active_consumers(repository_root: Path) -> list[Path]:
    """Return active notebooks and Python scripts, excluding archive content."""

    notebooks = sorted((repository_root / "notebooks").rglob("*.ipynb"))
    scripts = sorted((repository_root / "scripts").rglob("*.py"))
    return notebooks + scripts


def _consumer_units(path: Path) -> tuple[list[tuple[str, int]], bool]:
    """Return source units and whether IPython transformation is required."""

    if path.suffix != ".ipynb":
        return [(path.read_text(encoding="utf-8"), 0)], False

    notebook = json.loads(path.read_text(encoding="utf-8"))
    units = [
        ("".join(cell.get("source", [])), raw_cell_index)
        for raw_cell_index, cell in enumerate(notebook.get("cells", []))
        if cell.get("cell_type") == "code"
    ]
    return units, True


def extract_src_consumer_contract(path: Path) -> dict[str, list]:
    """Extract imported `src` paths, used symbols, and invalid raw cells."""

    units, is_notebook = _consumer_units(path)
    transformer = TransformerManager()
    module_aliases: dict[str, str] = {}
    symbol_aliases: dict[str, str] = {}
    imported_paths: set[str] = set()
    used_symbols: set[str] = set()
    syntax_failure_raw_cells: list[int] = []

    for source, raw_cell_index in units:
        try:
            transformed = transformer.transform_cell(source) if is_notebook else source
            tree = ast.parse(transformed)
        except SyntaxError:
            syntax_failure_raw_cells.append(raw_cell_index)
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not alias.name.startswith("src."):
                        continue
                    bound_name = alias.asname or alias.name.split(".")[0]
                    module_aliases[bound_name] = alias.name
                    imported_paths.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("src."):
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    full_name = f"{node.module}.{alias.name}"
                    symbol_aliases[alias.asname or alias.name] = full_name
                    imported_paths.add(full_name)

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in module_aliases
            ):
                used_symbols.add(f"{module_aliases[node.value.id]}.{node.attr}")
            elif (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id in symbol_aliases
            ):
                used_symbols.add(symbol_aliases[node.id])

    return {
        "imports": sorted(imported_paths),
        "symbols": sorted(used_symbols),
        "syntax_failure_raw_cells": syntax_failure_raw_cells,
    }


def build_src_consumer_inventory(repository_root: Path) -> dict[str, dict[str, list]]:
    """Build a path-keyed inventory for every active notebook and script."""

    return {
        path.relative_to(repository_root).as_posix(): extract_src_consumer_contract(path)
        for path in discover_active_consumers(repository_root)
    }

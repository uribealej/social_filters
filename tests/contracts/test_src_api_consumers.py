"""Keep the recorded notebook/script `src` API inventory synchronized."""

from __future__ import annotations

import importlib
import json
import re
import unittest
from pathlib import Path

from tests.support.consumer_inventory import build_src_consumer_inventory


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPOSITORY_ROOT / "tests" / "contracts" / "src_api_consumers.json"
SYMBOL_INDEX_PATH = REPOSITORY_ROOT / ".agents" / "references" / "symbol-index.md"


class SourceApiConsumerContractTests(unittest.TestCase):
    """Protect the discovered `src` API used by active notebooks and scripts."""

    def test_inventory_matches_active_consumers(self) -> None:
        """Fail when a consumer or its `src` usage changes without review."""

        expected = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        actual = build_src_consumer_inventory(REPOSITORY_ROOT)
        self.assertEqual(expected, actual)

    def test_every_used_symbol_exists(self) -> None:
        """Import each referenced owner module and resolve every used symbol."""

        inventory = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        symbols = sorted(
            {
                symbol
                for contract in inventory.values()
                for symbol in contract["symbols"]
            }
        )
        imported_modules = {}
        missing = []
        for symbol in symbols:
            module_name, attribute_name = symbol.rsplit(".", 1)
            try:
                module = imported_modules.setdefault(
                    module_name,
                    importlib.import_module(module_name),
                )
                getattr(module, attribute_name)
            except Exception as exc:  # pragma: no cover - reports future drift
                missing.append(f"{symbol}: {exc}")

        self.assertEqual([], missing, "Missing consumer symbols:\n" + "\n".join(missing))

    def test_every_used_symbol_is_in_symbol_index(self) -> None:
        """Require the human public index to include every active symbol."""

        inventory = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        active_symbols = {
            symbol
            for contract in inventory.values()
            for symbol in contract["symbols"]
        }

        documented_symbols = set()
        current_module = None
        for line in SYMBOL_INDEX_PATH.read_text(encoding="utf-8").splitlines():
            heading = re.match(r"^### `src/([A-Za-z0-9_]+)\.py`$", line)
            if heading:
                current_module = f"src.{heading.group(1)}"
                continue
            symbol = re.match(r"^- `([A-Za-z_][A-Za-z0-9_]*)`", line)
            if current_module and symbol:
                documented_symbols.add(f"{current_module}.{symbol.group(1)}")

        missing = sorted(active_symbols - documented_symbols)
        self.assertEqual([], missing, "Active symbols missing from symbol-index.md")


if __name__ == "__main__":
    unittest.main()

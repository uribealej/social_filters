"""Protect the S05 canonical detector and compatibility facades."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src import significant_trace_detection as canonical
from src import significant_traces as legacy_facade
from src import significant_traces_v2 as current_facade
from tests.support.fixtures import significant_trace_cases


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class SignificantTraceConsolidationTests(unittest.TestCase):
    """Require one scientific owner with stable V1 and V2 entrypoints."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.traces = significant_trace_cases()["finite_with_transients"]
        cls.kwargs = {
            "n_bins": 24,
            "k_neighbors": 5,
            "confCutOff": 95,
            "fps": 2.0,
            "tauDecay": 6.0,
        }

    def test_current_facade_matches_canonical_default_mode(self) -> None:
        """Keep the former V2 entrypoint identical to canonical current mode."""

        with mock.patch.object(canonical, "print", create=True):
            facade = current_facade.compute_noise_model_romano_fast_modular(
                self.traces,
                random_state=83,
                return_diagnostics=True,
                **self.kwargs,
            )
            direct = canonical.compute_noise_model_romano_fast_modular(
                self.traces,
                random_state=83,
                return_diagnostics=True,
                **self.kwargs,
            )
        self.assertEqual(9, len(facade))
        self.assertEqual("current", direct[8]["params"]["mode"])
        for facade_value, direct_value in zip(facade[:8], direct[:8]):
            np.testing.assert_array_equal(facade_value, direct_value)
        facade_diagnostics = dict(facade[8])
        direct_diagnostics = dict(direct[8])
        facade_diagnostics.pop("runtime_sec")
        direct_diagnostics.pop("runtime_sec")
        self.assertEqual(facade_diagnostics.keys(), direct_diagnostics.keys())
        for key in facade_diagnostics:
            left = facade_diagnostics[key]
            right = direct_diagnostics[key]
            if isinstance(left, np.ndarray):
                np.testing.assert_array_equal(left, right)
            else:
                self.assertEqual(left, right)

    def test_legacy_facade_matches_canonical_legacy_mode(self) -> None:
        """Keep the former V1 entrypoint reproducible through one legacy mode."""

        np.random.seed(97)
        with mock.patch.object(canonical, "print", create=True):
            facade = legacy_facade.compute_noise_model_romano_fast_modular(
                self.traces,
                **self.kwargs,
            )
        np.random.seed(97)
        with mock.patch.object(canonical, "print", create=True):
            direct = canonical.compute_noise_model_romano_fast_modular(
                self.traces,
                mode="legacy",
                **self.kwargs,
            )
        self.assertEqual(8, len(facade))
        for facade_value, direct_value in zip(facade, direct):
            np.testing.assert_array_equal(facade_value, direct_value)

    def test_facades_contain_no_scientific_dependency_or_array_logic(self) -> None:
        """Prevent either compatibility module from becoming a second owner."""

        forbidden_import_roots = {
            "matplotlib",
            "numpy",
            "scipy",
            "skimage",
            "sklearn",
        }
        for relative_path in (
            Path("src/significant_traces.py"),
            Path("src/significant_traces_v2.py"),
        ):
            with self.subTest(path=str(relative_path)):
                source = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
                tree = ast.parse(source)
                imported_roots = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imported_roots.update(
                            alias.name.split(".", 1)[0] for alias in node.names
                        )
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imported_roots.add(node.module.split(".", 1)[0])
                self.assertTrue(forbidden_import_roots.isdisjoint(imported_roots))
                self.assertLess(len(source.splitlines()), 260)

    def test_invalid_canonical_mode_is_rejected(self) -> None:
        """Require explicit current or legacy compatibility semantics."""

        with self.assertRaisesRegex(ValueError, "current.*legacy"):
            canonical.gaussfit_neg(self.traces[:, 0], mode="v3")


if __name__ == "__main__":
    unittest.main()

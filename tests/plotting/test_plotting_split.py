"""S08 regression contracts captured before any figure family was extracted."""

import ast
import contextlib
import hashlib
import importlib
import inspect
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import plotting
from tests.plotting.figure_contracts import figure_contract, value_contract, compact_figure_contract
from tests.plotting.plotting_cases import FAMILIES, inputs, figure_cases


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "tests/contracts/plotting_s08.json"
RENDERS = ROOT / ".test_cache/s08"


def capture_family(api, fixture, family, render_dir):
    """Render every case and capture exact plotted-data and return contracts."""
    render_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, call in figure_cases(api, fixture, family, render_dir).items():
        plt.close("all")
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout), mock.patch.object(plt, "show"):
                returned = call()
            figures = [plt.figure(number) for number in plt.get_fignums()]
            snapshots = []
            for index, fig in enumerate(figures):
                fig.canvas.draw()
                snapshots.append(compact_figure_contract(figure_contract(fig)))
                fig.savefig(render_dir / f"{family}_{name}_{index}.png", dpi=80, bbox_inches="tight")
            result[name] = dict(returned=value_contract(returned), figures=snapshots,
                stdout=stdout.getvalue().replace(str(render_dir), "<output>"))
        finally:
            plt.close("all")
    # Normalize NumPy scalar values and tuple/list differences through JSON once.
    return json.loads(json.dumps(result))


class PlottingCharacterizationTests(unittest.TestCase):
    """Check each family independently against its untouched implementation."""

    @classmethod
    def setUpClass(cls):
        cls.fixture = inputs()
        cls.baseline = json.loads(CONTRACT.read_text(encoding="utf-8"))

    def check_family(self, family):
        actual = capture_family(plotting, self.fixture, family, RENDERS / "after")
        for name, expected in self.baseline["families"][family].items():
            with self.subTest(family=family, case=name):
                self.assertEqual(expected, actual[name])

    def test_common(self):
        self.check_family("common")

    def test_single_fish(self):
        self.check_family("single_fish")

    def test_all_fish(self):
        self.check_family("all_fish")

    def test_static_bout(self):
        self.check_family("static_bout")

    def test_specificity(self):
        self.check_family("specificity")

    def test_diagnostics(self):
        self.check_family("diagnostics")

    def test_lme(self):
        self.check_family("lme")

    def test_reliability(self):
        self.check_family("reliability")

    def test_signatures_exports_and_function_bodies(self):
        """Keep every historical function body, default and dependency export."""
        for name, signature in self.baseline["signatures"].items():
            function = getattr(plotting, name)
            with self.subTest(name=name):
                self.assertEqual(signature, str(inspect.signature(function)))
                tree = ast.parse(inspect.getsource(function))
                digest = hashlib.sha256(ast.dump(tree).encode()).hexdigest()
                self.assertEqual(self.baseline["function_ast_sha256"][name], digest)
        for name in self.baseline["exports"]:
            self.assertTrue(hasattr(plotting, name), name)

    def test_sorting_timing_and_exceptions(self):
        """Compare pure preparation and invalid-input contracts captured before moves."""
        self.assertEqual(self.baseline["preparation"], preparation_contract(plotting, self.fixture))

    def test_save_contracts(self):
        """Keep filenames, output directories, requested DPI and close behavior."""
        self.assertEqual(self.baseline["saving"], saving_contract(plotting, self.fixture))

    def test_owner_boundaries_and_facade(self):
        """Require direct re-exports and independent figure/scientific owners."""
        facade = ast.parse((ROOT / "src/plotting.py").read_text(encoding="utf-8"))
        self.assertTrue(all(isinstance(node, (ast.Expr, ast.Import, ast.ImportFrom))
                            for node in facade.body))
        owners = {
            "plotting_common", "plotting_single_fish", "plotting_all_fish",
            "plotting_static_flicker", "plotting_bout_position", "plotting_specificity",
            "plotting_diagnostics", "plotting_lme", "plotting_reliability",
            "neuron_ordering", "stimuli_timeline",
        }
        for name in self.baseline["signatures"]:
            function = getattr(plotting, name)
            self.assertIn(function.__module__.removeprefix("src."), owners)
            self.assertIs(function, getattr(importlib.import_module(function.__module__), name))
        for owner in sorted(owners):
            forbidden = ["src.plotting", "src.analysis_tools", "src.multifish_analysis",
                         "src.reusable_several_fish"]
            if owner in {"neuron_ordering", "stimuli_timeline"}:
                forbidden.append("matplotlib")
            code = (f"import src.{owner}, sys; "
                    f"assert not set({forbidden!r}) & sys.modules.keys()")
            result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(0, result.returncode, f"{owner}: {result.stderr}")


def preparation_contract(api, fixture):
    """Exercise every sorter, timing mode and representative invalid configuration."""
    values = fixture["single"]["dfof"]
    result = {"sort_orders": value_contract(api.compute_sort_orders(values, n_clusters=3))}
    for mode in ["unsorted", "max_intensity", "pca", "kmeans", "hier", "corravg"]:
        result[mode] = value_contract(api.compute_single_sort_order(values, mode))
    ref = fixture["data"]["fish_b"]
    for mode in ["mean", "concat"]:
        result["markers_" + mode] = value_contract(api.compute_move_lines_for_flat_matrix(
            ref["trial_aligned_traces_z_core"], [3, 7], ref["stimuli_id_map"],
            ref["stimuli_durations"], fixture["colors"], 1., 1., mode))
    result["durations"] = api.summarize_durations({"a": {"static_before_sec": 1., "total_sec": 4.},
                                                "b": {"static_before_sec": 3., "total_sec": 4.}})
    errors = [
        lambda: api.build_stimulus_style_maps(),
        lambda: api.compute_single_sort_order(np.zeros((2, 3, 4)), "pca"),
        lambda: api.compute_single_sort_order(values, "bad"),
        lambda: api.plot_similarity_by_distance(pd.DataFrame()),
        lambda: api.plot_stimulus_specificity_sparseness(pd.DataFrame()),
        lambda: api.plot_stimulus_specificity_selectivity_index(pd.DataFrame()),
        lambda: api.plot_active_stimuli_histogram(pd.DataFrame()),
        lambda: api.plot_preferred_stimulus_distribution(pd.DataFrame(), []),
        lambda: api.plot_motion_delta_distribution(pd.DataFrame()),
        lambda: api._diagnostic_sort_order(np.zeros((2, 3)), np.zeros((2, 2)), "bad"),
        lambda: api.plot_bout_flicker_position_figure({"stimulus_labels": []}),
        lambda: api.plot_sorted_chunks_single_mode(**fixture["single"], neuron_order=[1]),
    ]
    result["errors"] = []
    for call in errors:
        try:
            call()
        except Exception as exc:
            result["errors"].append([type(exc).__name__, str(exc)])
        else:
            raise AssertionError("An invalid-input characterization unexpectedly succeeded")
    return json.loads(json.dumps(result))


def saving_contract(api, fixture):
    """Write actual small PNGs while recording the historical 600-DPI requests."""
    from matplotlib.figure import Figure

    folder = RENDERS / "saving"
    folder.mkdir(parents=True, exist_ok=True)
    saved = []
    real_save = Figure.savefig

    def save_small(figure, path, **kwargs):
        saved.append([str(Path(path).relative_to(folder)), kwargs.copy()])
        kwargs["dpi"] = 40
        return real_save(figure, path, **kwargs)

    with mock.patch.object(Figure, "savefig", save_small), contextlib.redirect_stdout(io.StringIO()):
        result = api.save_sorted_rasters_for_all_modes(**fixture["single"], plots_path=folder,
            prefix="synthetic", n_clusters=3, folder_name="rasters")
    assert result is None
    assert not plt.get_fignums()
    for filename, _ in saved:
        assert (folder / filename).is_file()
    return saved


if __name__ == "__main__":
    if sys.argv[1:] == ["--capture-baseline"]:
        if CONTRACT.exists():
            raise FileExistsError("The pre-extraction baseline must not be overwritten")
        fixture = inputs()
        functions = {name: function for name, function in vars(plotting).items()
                     if inspect.isfunction(function) and function.__module__ == plotting.__name__}
        baseline = dict(source_commit="093bbfa", signatures={n: str(inspect.signature(f)) for n, f in functions.items()},
            exports=[n for n in vars(plotting) if not n.startswith("__")],
            function_ast_sha256={n: hashlib.sha256(ast.dump(ast.parse(inspect.getsource(f))).encode()).hexdigest()
                                for n, f in functions.items()}, families={})
        for family in FAMILIES:
            baseline["families"][family] = capture_family(plotting, fixture, family, RENDERS / "before")
            print(f"Captured {family}", flush=True)
        baseline["preparation"] = preparation_contract(plotting, fixture)
        baseline["saving"] = saving_contract(plotting, fixture)
        CONTRACT.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    else:
        unittest.main()

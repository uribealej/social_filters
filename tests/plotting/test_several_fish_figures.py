"""S09 figure orchestration and first notebook-consumer checks."""

import ast
import json
from pathlib import Path
import unittest
from unittest import mock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import reusable_several_fish as rsf
from src import several_fish_diagnostics as diagnostics
from src import several_fish_figures as figures
from tests.analysis.multifish_cases import FISH_IDS, STIM_IDS, STIM_LABELS, TIMING
from tests.plotting.plotting_cases import inputs


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks/calcium/exp_01_flickering/01_all_fish_raster.ipynb"


def run_notebook_assignment(name, namespace):
    """Execute one unchanged Exp 1 figure-wrapper assignment."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code" or f"{name} = rsf." not in "".join(cell["source"]):
            continue
        for node in ast.walk(ast.parse("".join(cell["source"]))):
            if (isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == name
                            for target in node.targets)):
                exec(compile(ast.Module(body=[node], type_ignores=[]),
                             str(NOTEBOOK), "exec"), namespace)
                return namespace[name]
    raise AssertionError(f"Notebook assignment {name!r} was not found")


class SeveralFishFigureTests(unittest.TestCase):
    """Check the figure facade, pure inputs, and representative renders."""

    @classmethod
    def setUpClass(cls):
        """Build one reproducible synthetic cohort for all five notebook calls."""
        cls.fixture = inputs()

    def setUp(self):
        """Render with Agg without showing interactive windows."""
        self.show = mock.patch.object(plt, "show")
        self.show.start()
        plt.close("all")

    def tearDown(self):
        """Close only figures created during this test."""
        plt.close("all")
        self.show.stop()

    def test_compatibility_facade_is_import_only(self):
        """Keep historical names as direct aliases to the five figure owners."""
        for name in (
            "build_all_fish_raster_figure", "plot_lifetime_sparseness_analysis",
            "build_plot_all_fish_mean_zscore_traces",
            "plot_left_right_active_overlap_diagnostics",
            "plot_motion_active_neuron_counts",
        ):
            self.assertIs(getattr(rsf, name), getattr(figures, name))
        facade = ast.parse((ROOT / "src/reusable_several_fish.py").read_text(encoding="utf-8"))
        self.assertTrue(all(isinstance(node, (ast.Expr, ast.Import, ast.ImportFrom))
                            for node in facade.body))

    def test_pure_raster_summary_and_active_count_inputs(self):
        """Check that calculations are available without calling the figure owner."""
        data = self.fixture["data"]
        reference = data["fish_b"]
        settings = {"combine_mode": "mean", "trace_type": "zscore",
                    "raster_sort_mode": "left_right_index", "left_control": 7,
                    "right_control": 3}
        raster = diagnostics.build_all_fish_raster_inputs(
            data, FISH_IDS, STIM_IDS, reference, TIMING, settings)
        self.assertEqual(raster["flat_matrix_all_fish"].shape[0], 8)
        self.assertEqual(raster["raster_left_right_index"].shape, (8,))
        self.assertEqual(sorted(raster["raster_neuron_order"].tolist()), list(range(8)))
        self.assertEqual(raster["raster_sort_label"], "left-right index")

        sparse = diagnostics.build_lifetime_sparseness_summary(
            data, FISH_IDS, reference, TIMING, STIM_IDS)
        self.assertEqual(sparse["selection"]["stimulus_labels"], STIM_LABELS)
        self.assertEqual(sparse["summary_table"]["fish_id"].tolist(),
                         ["fish_b"] * 4 + ["fish_a"] * 4)
        self.assertIn("lifetime_sparseness", sparse["summary_table"].columns)

        active = {
            "fish_b": pd.DataFrame({7: [1, 0], 3: [0, 1]}),
            "fish_a": pd.DataFrame({7: [1, 1], 3: [0, 0]}),
        }
        counts = diagnostics.build_motion_active_counts(
            active, FISH_IDS, [3, 7], ["FRB", "FLB"])
        self.assertEqual(counts.index.tolist(), FISH_IDS)
        self.assertEqual(counts.columns.tolist(), ["FRB", "FLB"])
        self.assertEqual(counts.values.tolist(), [[1, 1], [0, 2]])

    def test_exp1_figure_calls_render_and_preserve_outputs(self):
        """Run all five notebook wrapper calls and save representative PNGs."""
        fixture = self.fixture
        reference = fixture["data"]["fish_b"]
        namespace = {
            "rsf": rsf, "all_fish_data": fixture["data"],
            "fish_ids": FISH_IDS, "stim_order": STIM_IDS,
            "reference_fish": reference, "timing": dict(TIMING, t_post_s=11.),
            "stimuli_colors": fixture["colors"],
            "stimuli_linestyles": fixture["styles"],
            "settings": {"combine_mode": "mean", "trace_type": "zscore",
                         "raster_sort_mode": "left_right_index",
                         "left_control": 7, "right_control": 3,
                         "analysis_label": "synthetic"},
            "FIGSIZE": (8, 8),
            "OVERLAP_STIMULI": {"left": [7, 11], "right": [3, 21]},
            "OVERLAP_FIGSIZE": (5, 4), "RASTER_FIGSIZE": (12, 7),
            "SPARSENESS_STIMULI": STIM_IDS,
            "HIGH_LIFETIME_SPARSENESS": -1.0,
            "HIGH_SPARSENESS_COMBINE_MODE": "mean",
            "SPARSENESS_FIGSIZE": (5, 4),
        }
        raster = run_notebook_assignment("raster_result", namespace)
        self.assertEqual(raster["flat_matrix_all_fish"].shape[0], 8)
        self.assertEqual(raster["neuron_order"].shape, (8,))

        namespace["FIGSIZE"] = (5.5, 3)
        means = run_notebook_assignment("mean_trace_result", namespace)
        self.assertEqual(means["stimulus_labels"], STIM_LABELS)
        self.assertEqual(list(means["mean_traces"]), STIM_IDS)

        before_overlap = set(plt.get_fignums())
        overlap = run_notebook_assignment("overlap_results", namespace)
        overlap_figures = [plt.figure(number) for number in plt.get_fignums()
                           if number not in before_overlap]
        self.assertEqual(list(overlap), ["left", "right"])
        self.assertGreaterEqual(len(overlap_figures), 2)

        namespace["FIGSIZE"] = (7, 2.5)
        motion = run_notebook_assignment("motion_active_result", namespace)
        self.assertEqual(motion["counts"].columns.tolist(), STIM_LABELS)
        self.assertEqual(motion["counts"].index.tolist(), FISH_IDS)

        namespace["RASTER_FIGSIZE"] = (12, 8)
        sparse = run_notebook_assignment("sparseness_result", namespace)
        self.assertEqual(sparse["summary_table"].shape[0], 8)
        self.assertIsNotNone(sparse["high_sparseness_figure"])

        before_active_overlap = set(plt.get_fignums())
        rsf.plot_left_right_active_overlap_diagnostics(
            fixture["data"], FISH_IDS, reference, namespace["timing"],
            namespace["OVERLAP_STIMULI"], show_significant_raster=False,
            motion_onset_s=3., tau_s=0., overlap_figsize=(5, 4),
        )
        active_overlap_figures = [plt.figure(number) for number in plt.get_fignums()
                                  if number not in before_active_overlap]
        self.assertEqual(len(active_overlap_figures), 2)
        active_motion = rsf.plot_motion_active_neuron_counts(
            fixture["data"], FISH_IDS, reference, namespace["timing"],
            STIM_IDS, fixture["colors"], motion_onset_s=3., tau_s=0.,
            figsize=(7, 2.5),
        )
        self.assertGreater(active_motion["counts"].to_numpy().sum(), 0)

        output = ROOT / ".test_cache/s09/figures"
        output.mkdir(parents=True, exist_ok=True)
        examples = {
            "raster": raster["figure"], "mean_traces": means["figure"],
            "overlap_heatmap": overlap_figures[0],
            "overlap_heatmap_active": active_overlap_figures[0],
            "motion_active": motion["figure"],
            "motion_active_nonzero": active_motion["figure"],
            "sparseness": sparse["sparseness_figure"],
            "high_sparseness_raster": sparse["high_sparseness_figure"],
        }
        for name, figure in examples.items():
            figure.canvas.draw()
            figure.savefig(output / f"{name}.png", dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    unittest.main()

"""Run the unchanged first notebook/wrapper callers with synthetic inputs."""

import ast
import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest import mock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import analysis_tools as at, multifish_analysis as mfa, plotting as plott
from src import reusable_several_fish as rsf
from tests.plotting.plotting_cases import inputs, lme_inputs
from tests.analysis.multifish_cases import FISH_IDS, STIM_IDS, STIM_LABELS, TIMING


ROOT = Path(__file__).resolve().parents[2]


def cell(path, index):
    """Read the exact consumer cell; never modify notebook configuration."""
    document = json.loads((ROOT / "notebooks/calcium" / path).read_text(encoding="utf-8"))
    return "".join(document["cells"][index]["source"])


class PlottingConsumerTests(unittest.TestCase):
    """Exercise source consumers instead of only testing re-export availability."""

    @classmethod
    def setUpClass(cls):
        cls.fixture = inputs()

    def setUp(self):
        self.show = mock.patch.object(plt, "show")
        self.show.start()
        self.stdout = contextlib.redirect_stdout(io.StringIO())
        self.stdout.__enter__()
        plt.close("all")

    def tearDown(self):
        for index, number in enumerate(plt.get_fignums()):
            figure = plt.figure(number)
            figure.canvas.draw()
            directory = ROOT / ".test_cache/s08/consumers"
            directory.mkdir(parents=True, exist_ok=True)
            figure.savefig(directory / f"{self._testMethodName}_{index}.png", dpi=80, bbox_inches="tight")
        plt.close("all")
        self.stdout.__exit__(None, None, None)
        self.show.stop()

    def test_single_fish_style_raster_chunks_and_means(self):
        folder = ROOT / ".test_cache/s08/consumer_stimuli"
        folder.mkdir(parents=True, exist_ok=True)
        for label in ["FL2", "FR1"]:
            (folder / f"{label}_trajectory.csv").write_text("x,y\n0,0\n", encoding="utf-8")
        ns = dict(plott=plott, at=at, plt=plt, np=np, paths=dict(stimuli_path=folder,
            plots_path=folder, prefix="synthetic"), stimuli_id_map={"FL2": 1, "FR1": 2},
            stimuli_order_ids=[2, 1], experiment_name="synthetic", fish_id="synthetic",
            sort_mode="max_intensity", sort_n_clusters=3, random_state=42,
            average_across_repeats=False, z_traces=self.fixture["single"]["dfof"],
            **self.fixture["single"])
        path = "shared/General_analysis_single_fish_reusable.ipynb"
        for index in [5, 6, 7]:
            exec(compile(cell(path, index), path, "exec"), ns)
        np.testing.assert_array_equal(ns["im"].get_array(), ns["dfof"].T)
        ns.update(trial_aligned_traces={s: np.arange(120.).reshape(5, 12, 2) / 50 for s in [2, 1]},
                  stimuli_ids=[2, 1], stimuli_names=["FR1", "FL2"], trial_t_pre_s=1.,
                  trial_t_post_s=5., mean_trace_dpi=72, save_mean_trace_plots=False,
                  mean_trace_comment="synthetic")
        exec(compile(cell(path, 9), path, "exec"), ns)
        self.assertIsNone(ns["out_path_all"])
        self.assertEqual(["FR1", "FL2"], list(ns["used_colors_all"]))

    def test_allfish_wrappers(self):
        fixture = self.fixture
        reference = fixture["data"]["fish_b"]
        result = rsf.build_all_fish_raster_figure(fixture["data"], FISH_IDS, STIM_IDS, reference,
            TIMING, dict(combine_mode="mean", trace_type="zscore", raster_sort_mode="max_intensity"),
            fixture["colors"], fixture["styles"], figsize=(12, 8))
        np.testing.assert_array_equal(result["image"].get_array(),
            fixture["outputs"]["matrix_mean_zscore"][result["neuron_order"]])
        means = rsf.build_plot_all_fish_mean_zscore_traces(fixture["data"], FISH_IDS, STIM_IDS,
            reference, dict(TIMING, t_post_s=11.), fixture["colors"], fixture["styles"], figsize=(10, 5))
        self.assertEqual(STIM_LABELS, means["stimulus_labels"])
        self.assertEqual(STIM_LABELS, [line.get_label() for line in means["axes"].lines[:4]])

    def test_bout_notebook_cell(self):
        ns = dict(plott=plott, plt=plt, timing=TIMING,
            position_result=self.fixture["outputs"]["bout_False"],
            BOUT_FLICKER_SETTINGS=dict(onset_match_tolerance_s=.5, save_figures=False),
            display=lambda figure: figure.canvas.draw())
        path = "exp_01_flickering/02_bout_flicker_position_onset.ipynb"
        exec(compile(cell(path, 5), path, "exec"), ns)
        self.assertEqual(["right", "left"], list(ns["figures"]))
        for result in ns["figures"].values():
            np.testing.assert_array_equal(result["neuron_order"], np.arange(len(result["neuron_order"])))

    def test_static_notebook_plot_calls(self):
        # Execute the exact plot expressions while supplying synthetic scientific
        # outputs upstream of the notebook's experiment-specific sampling settings.
        static = self.fixture["outputs"]["static"]
        ns = dict(plott=plott, recruitment_results=static, raster_plot_results=static,
            auc_plot_results=static, cell07_stimuli=STIM_IDS, cell09_stimuli=STIM_IDS,
            CELL06_WINDOWS={"static": {"label": "Static"}, "comparison": {"label": "Motion"}},
            cell08_statistics=mfa.compute_static_flicker_fish_level_statistics(static["fish_median_delta_auc"]))
        path = "exp_01_flickering/02_static_flicker_recruitment.ipynb"
        count = 0
        for index in [6, 7, 8, 9]:
            tree = ast.parse(cell(path, index))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "plott":
                    result = eval(compile(ast.Expression(node), path, "eval"), ns)
                    self.assertEqual(["left", "right"], list(result))
                    count += 1
        self.assertEqual(4, count)

    def test_specificity_similarity_notebook_cell(self):
        summary = self.fixture["outputs"]["metrics"].copy()
        summary["plot_neuron_keep"] = True
        ns = dict(plott=plott, mfa=mfa, plt=plt, outputs=dict(similarity=True, stimulus_specificity=True),
            selected_auc_response_matrix=self.fixture["outputs"]["responses"]["pooled_response_matrix"],
            plot_stimulus_labels=STIM_LABELS, neuron_summary_table=summary,
            settings={"analysis_label": "synthetic"}, display=lambda *args: None)
        path = "shared/several_fish_reusable_analysis.ipynb"
        exec(compile(cell(path, 4), path, "exec"), ns)
        self.assertEqual(3, len(plt.get_fignums()))

    def test_lme_notebook_cell(self):
        results, table = lme_inputs()
        ns = dict(plott=plott, plt=plt, model_results=results, response_table=table)
        path = "exp_05_map_positions/02_lme_feature_decomposition.ipynb"
        exec(compile(cell(path, 20), path, "exec"), ns)
        self.assertEqual(["fixed_effects", "model_comparison", "observed_vs_fitted", "response_distribution"], list(ns["figures"]))

    def test_motion_and_active_diagnostic_consumers(self):
        fixture = self.fixture
        reference = fixture["data"]["fish_b"]
        results = rsf.plot_left_right_active_overlap_diagnostics(fixture["data"], FISH_IDS,
            reference, TIMING, {"left": [7, 11], "right": [3, 21]},
            show_overlap=False, show_significant_raster=True, motion_onset_s=3., tau_s=0.,
            raster_figsize=(14, 8))
        self.assertEqual(["left", "right"], list(results))
        self.assertEqual(2, len(plt.get_fignums()))
        ns = dict(plott=plott, rsf=rsf, at=at, plt=plt, all_fish_data=fixture["data"],
            fish_ids=FISH_IDS, reference_fish=reference, timing=TIMING,
            plot_options=dict(apply_neuron_filter_to_cell06=False, motion_delta_fish_id=None),
            outputs=dict(mean_traces=False, motion_delta=True),
            stimulus_sets={"motion_delta": STIM_LABELS, "mean_traces": [], "main_plots": STIM_LABELS})
        path = "shared/several_fish_reusable_analysis.ipynb"
        exec(compile(cell(path, 6), path, "exec"), ns)
        self.assertEqual(4, len(plt.get_fignums()))

    def test_reliability_first_consumer(self):
        traces = np.random.default_rng(15).normal(size=(5, 12, 4))
        traces[0] = np.arange(12.)[:, None]
        result = at.filter_neurons_by_trial_reliability(self.fixture["single"]["dfof"],
            {1: traces}, [1], 2., ROOT / ".test_cache/s08", make_plots=True, save_indices=False,
            custom_threshold=.5, hist_bins=8)
        self.assertIn(0, result["kept_neuron_indices"])
        self.assertEqual(3, len(plt.get_fignums()))


if __name__ == "__main__":
    unittest.main()

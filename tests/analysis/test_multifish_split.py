"""S07 characterization against outputs captured before extracting any code."""

import ast
import contextlib
import inspect
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from src import multifish_analysis as mfa
from src import reusable_several_fish as rsf
from src import bout_flicker_analysis as bout_analysis
from src import multifish_matrices
from src import plotting_specificity as plot_specificity
from src import static_flicker_analysis as static_analysis
from src import stimulus_similarity
from src.response_metrics import validate_static_flicker_recruitment_result
from tests.analysis.multifish_cases import (
    FISH_IDS, SIDES, STATIC_SETTINGS, STIM_IDS, STIM_LABELS, TIMING,
    cases, cohort, encode, trajectory_folder, write_trajectories,
)


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "tests/contracts/multifish_s07.json"


class MultifishCharacterizationTests(unittest.TestCase):
    """Protect exact scientific outputs independently of the extraction layout."""

    @classmethod
    def setUpClass(cls):
        """Run the compact fixture once and read the pre-extraction baseline."""
        cls.baseline = json.loads(CONTRACT.read_text(encoding="utf-8"))
        with trajectory_folder() as folder:
            cls.outputs = cases(mfa, folder)

    def test_exact_pre_extraction_outputs(self):
        """Check values, nested return types, table axes/dtypes and ordered keys."""
        self.assertEqual(list(self.outputs), list(self.baseline["outputs"]))
        for name, value in self.outputs.items():
            with self.subTest(case=name):
                self.assertEqual(encode(value), self.baseline["outputs"][name])

    def test_historical_signatures_and_imports(self):
        """Retain all locally defined functions and previously imported helpers."""
        for name, signature in self.baseline["signatures"].items():
            with self.subTest(name=name):
                self.assertEqual(str(inspect.signature(getattr(mfa, name))), signature)
        for name in self.baseline["exports"]:
            self.assertTrue(hasattr(mfa, name), name)
        self.assertIs(rsf.mfa.compute_stimulus_selectivity_metrics,
                      mfa.compute_stimulus_selectivity_metrics)

    def test_narrow_owners_and_facade_boundaries(self):
        """Keep implementation out of the facade and plotting out of computation."""
        import importlib

        owners = {
            "multifish_matrices": "build_matrix_all_fish",
            "static_flicker_analysis": "build_static_flicker_recruitment_analysis",
            "bout_flicker_analysis": "build_bout_flicker_position_analysis",
            "stimulus_specificity": "build_neuron_stimulus_summary_table",
            "stimulus_similarity": "build_segment_selectivity_permutation_summary",
            "active_neuron_analysis": "build_active_neuron_overlap_matrices_all_fish",
        }
        facade = ast.parse((ROOT / "src/multifish_analysis.py").read_text(encoding="utf-8"))
        self.assertTrue(all(isinstance(node, (ast.Expr, ast.Import, ast.ImportFrom))
                            for node in facade.body))
        for owner, name in owners.items():
            module = importlib.import_module(f"src.{owner}")
            self.assertIs(getattr(mfa, name), getattr(module, name))
        for name in self.baseline["signatures"]:
            function = getattr(mfa, name)
            self.assertIn(function.__module__.removeprefix("src."), owners)
            self.assertIs(function, getattr(importlib.import_module(function.__module__), name))
        code = "import importlib, sys\n"
        code += f"for name in {list(owners)!r}: importlib.import_module('src.' + name)\n"
        code += "assert not {'src.plotting', 'src.analysis_tools', 'matplotlib'} & sys.modules.keys()"
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(0, result.returncode, result.stderr)

    def test_known_order_and_categories(self):
        """Assert independently interpretable row, stimulus and category contracts."""
        response = self.outputs["responses"]
        self.assertEqual(STIM_LABELS, response["pooled_response_matrix"].columns.tolist())
        self.assertEqual(["fish_b"] * 4 + ["fish_a"] * 4,
                         response["row_metadata"].fish_id.tolist())
        self.assertEqual([3, 0, 4, 1] * 2, response["row_metadata"].source_neuron_id.tolist())
        self.assertEqual(list(range(8)), response["row_metadata"].global_neuron_id.tolist())
        self.assertEqual({"non-responsive", "static-only", "shared", "newly recruited"},
                         set(self.outputs["static"]["neuron_stimulus_metrics"].category))
        for result in self.outputs["static"]["per_fish"].values():
            validate_static_flicker_recruitment_result(result, fps_2p=1.0)
        raw = np.arange(12).reshape(2, 3, 2)
        np.testing.assert_array_equal(mfa.combine_reps_one_stim(raw),
                                      [[0, 2, 4, 1, 3, 5], [6, 8, 10, 7, 9, 11]])

    def test_seeded_statistics_and_permutations(self):
        """Require repeatable draws and preserve invalid-trial skip semantics."""
        args = (cohort(), FISH_IDS, STIM_IDS, STIM_LABELS, ["flickL", "flickR"])
        repeated = mfa.build_segment_selectivity_permutation_summary(
            *args, n_permutations=13, random_seed=19, **TIMING)
        self.assertEqual(encode(repeated), encode(self.outputs["permutation"]))
        changed = mfa.build_segment_selectivity_permutation_summary(
            *args, n_permutations=13, random_seed=20, **TIMING)
        self.assertFalse(np.array_equal(repeated["si_shuffle"], changed["si_shuffle"]))
        self.assertEqual(1.0, mfa._wilcoxon_signed_rank_pvalue([0, 0, np.nan]))
        self.assertEqual(0.5, mfa._exact_sign_flip_pvalue([1, 2]))
        self.assertEqual(mfa._bootstrap_median_ci([1, 2, 4], seed=7),
                         mfa._bootstrap_median_ci([1, 2, 4], seed=7))
        result = mfa._run_segment_selectivity_permutations(
            {"a": [np.nan], "b": [1]}, 5, np.random.default_rng(19))
        self.assertEqual("missing_or_empty_trials", result[3])
        self.assertTrue(np.isnan(result[2]).all())

    def test_first_downstream_summary_and_overlap_consumers(self):
        """Execute the several-fish notebook's real preparation wrappers."""
        response = self.outputs["responses"]
        active = self.outputs["active_counts"][0]
        summary = rsf.build_selected_neuron_summary(
            response["response_matrices"], response["pooled_response_matrix"],
            response["row_metadata"], active, STIM_IDS, STIM_LABELS,
            STIM_IDS, STIM_LABELS, analysis_label="characterization")
        pd.testing.assert_frame_equal(summary["neuron_summary_table"].drop(
            columns=["left_right_index", "plot_neuron_keep"]), self.outputs["metrics"], check_exact=True)
        result = rsf.build_overlap_diagnostic_data(
            cohort(), active, FISH_IDS, {"left": [7, 11], "right": [3, 21]},
            {"left": ["flickL", "boutL"], "right": ["flickR", "boutR"]},
            response["row_metadata"])
        self.assertIsInstance(result, dict)

    def test_first_notebook_analysis_cells(self):
        """Execute actual Exp 1 and LME analysis cells with small synthetic data."""
        with trajectory_folder() as folder:
            write_trajectories(folder)
            namespace = dict(bout_analysis=bout_analysis, np=np, pd=pd, all_fish_data=cohort(),
                fish_ids=FISH_IDS, timing=TIMING, stimuli_path=folder,
                display=lambda *args: None,
                BOUT_FLICKER_SETTINGS=dict(side_conditions=SIDES,
                    active_fraction_threshold=0.1, min_epoch_s=1., min_active_reps=2,
                    expected_reps=4, require_expected_reps=True, motion_onset_s=3., tau_s=0.,
                    onset_persistence_frames=2, onset_min_significant_trial_fraction=None,
                    require_bout_and_flicker_active=False, matching_method="euclidean",
                    coordinate_dot_prefix=None, position_window_offsets_s=(-1., 1.),
                    flicker_window_offsets_s=(0., 2.)))
            path = ROOT / "notebooks/calcium/exp_01_flickering/02_bout_flicker_position_onset.ipynb"
            source = "".join(json.loads(path.read_text(encoding="utf-8"))["cells"][4]["source"])
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(source, str(path), "exec"), namespace)
            self.assertEqual(encode(namespace["position_result"]), encode(self.outputs["bout_False"]))
        namespace = dict(multifish_matrices=multifish_matrices, np=np, pd=pd, all_fish_data=cohort(), fish_ids=FISH_IDS,
                         selected_stimuli=STIM_LABELS, display=lambda *args: None,
                         motion_duration_key="motion_sec", **TIMING)
        path = ROOT / "notebooks/calcium/exp_05_map_positions/02_lme_feature_decomposition.ipynb"
        source = "".join(json.loads(path.read_text(encoding="utf-8"))["cells"][7]["source"])
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source, str(path), "exec"), namespace)
        self.assertEqual(encode(namespace["response_outputs"]), encode(self.outputs["responses"]))

    def test_static_and_specificity_notebook_consumers(self):
        """Run unchanged static-recruitment and shared summary/figure call cells."""
        import matplotlib.pyplot as plt

        path = ROOT / "notebooks/calcium/exp_01_flickering/02_static_flicker_recruitment.ipynb"
        source = "".join(json.loads(path.read_text(encoding="utf-8"))["cells"][4]["source"])
        namespace = dict(static_analysis=static_analysis, all_fish_data=cohort(), fish_ids=FISH_IDS,
            stimuli_id_map={"FR1": 3, "FL1": 7}, timing=TIMING,
            STATIC_FLICKER_SETTINGS=dict(STATIC_SETTINGS, min_consecutive_active_frames=2,
                                         min_active_trial_fraction=0.5))
        exec(compile(source, str(path), "exec"), namespace)
        # The notebook constructs left then right; compare to the same old facade contract.
        expected = mfa.build_static_flicker_recruitment_analysis(
            cohort(), FISH_IDS, {"left": [7], "right": [3]}, **STATIC_SETTINGS)
        self.assertEqual(encode(expected), encode(namespace["recruitment_results"]))
        summary = self.outputs["metrics"].copy()
        summary["plot_neuron_keep"] = True
        namespace = dict(stimulus_similarity=stimulus_similarity, plot_specificity=plot_specificity, plt=plt,
            outputs={"similarity": True, "stimulus_specificity": True},
            selected_auc_response_matrix=self.outputs["responses"]["pooled_response_matrix"],
            plot_stimulus_labels=STIM_LABELS, neuron_summary_table=summary,
            settings={"analysis_label": "characterization"}, display=lambda *args: None)
        path = ROOT / "notebooks/calcium/shared/several_fish_reusable_analysis.ipynb"
        source = "".join(json.loads(path.read_text(encoding="utf-8"))["cells"][4]["source"])
        try:
            with mock.patch.object(plt, "show"):
                exec(compile(source, str(path), "exec"), namespace)
            self.assertEqual(6, len(namespace["segment_similarity_df"]))
            for figure_id in plt.get_fignums():
                plt.figure(figure_id).canvas.draw()
        finally:
            plt.close("all")

    def test_errors_and_edge_cases(self):
        """Preserve existing exception types and messages, including legacy quirks."""
        for name, args, kwargs, exception, message in error_cases():
            with self.subTest(function=name, message=message):
                with self.assertRaises(exception) as caught:
                    getattr(mfa, name)(*args, **kwargs)
                self.assertEqual(message, str(caught.exception))


def error_cases():
    """Return invalid configurations characterized against the old module."""
    return [
        ("combine_reps_one_stim", [np.zeros((2, 3))], {}, AssertionError,
         "Expected 3D array (neurons, time, reps)"),
        ("combine_reps_one_stim", [np.zeros((2, 3, 4))], {"mode": "bad"}, ValueError,
         "Unknown mode 'bad', use 'concat' or 'mean'"),
        ("build_matrix_all_fish", [cohort(), STIM_IDS], {"trace_type": "bad"}, ValueError,
         "Unknown trace_type='bad'. Use 'dfof', 'raster', or 'zscore'."),
        ("build_pooled_active_trace_diagnostic", [{}, {}, []], {"combine_mode": "bad"}, ValueError,
         "combine_mode must be 'mean' or 'concat'."),
        ("build_bout_flicker_position_analysis", [{}, [], SIDES, "."], {}, ValueError,
         "fish_ids must contain at least one fish."),
        ("build_bout_flicker_position_metadata", [".", {}, SIDES], {"matching_method": "bad"},
         ValueError, "matching_method must be 'euclidean'."),
        ("build_stimulus_vector_similarity", [pd.DataFrame()], {}, ValueError,
         "selected_stimuli must contain at least one stimulus."),
        ("resolve_segment_labels", [["L"], ["flickL", "boutL"]], {}, ValueError,
         "Segment 'L' matched multiple labels: ['flickL', 'boutL']. Use an exact label."),
        ("resolve_segment_labels", [["a", "a"], ["a"]], {}, ValueError,
         "Resolved segment labels are not unique: ['a', 'a']"),
        ("add_selectivity_metrics_to_summary_table", [pd.DataFrame(), ["a"]], {}, ValueError,
         "At least two selected stimuli are required."),
    ]

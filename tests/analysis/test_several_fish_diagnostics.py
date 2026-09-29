"""S09 diagnostic preparation and downstream notebook contracts."""

import ast
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from src import reusable_several_fish as rsf
from src import several_fish_diagnostics as diagnostics
from tests.analysis.multifish_cases import FISH_IDS, STIM_IDS, STIM_LABELS, TIMING, cohort


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks/calcium/shared/several_fish_reusable_analysis.ipynb"


def run_notebook_assignment(name, namespace):
    """Execute one unchanged assignment from the shared several-fish notebook."""
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


class SeveralFishDiagnosticTests(unittest.TestCase):
    """Check pure ownership and prepared diagnostic values and ordering."""

    def test_pure_owner_and_historical_imports(self):
        """Keep one implementation without a plotting or reporting import."""
        for name in (
            "build_response_window_validation", "build_high_sparseness_raster_data",
            "build_pooled_mean_trace_by_stimulus", "build_overlap_diagnostic_data",
            "build_diagnostic_trace_source_data", "apply_diagnostic_keep_mask",
            "build_decision_then_mean_sort_order", "build_metric_sort_order",
        ):
            self.assertIs(getattr(rsf, name), getattr(diagnostics, name))
        code = ("import sys; import src.several_fish_diagnostics; "
                "assert not {'matplotlib', 'seaborn', 'src.several_fish_reporting', "
                "'src.reusable_several_fish'} & sys.modules.keys()")
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_response_window_table_preserves_fish_and_stimulus_order(self):
        """Compare known clipped frame windows across two fish and stimuli."""
        result = rsf.build_response_window_validation(
            cohort(), FISH_IDS, STIM_IDS[:2], STIM_LABELS[:2], **TIMING)
        self.assertEqual(result[["fish_id", "stimulus_id"]].values.tolist(),
                         [["fish_b", 7], ["fish_b", 11], ["fish_a", 7], ["fish_a", 11]])
        self.assertEqual(result["start_frame"].tolist(), [4] * 4)
        self.assertEqual(result["stop_frame"].tolist(), [8] * 4)
        self.assertEqual(result["n_response_frames"].tolist(), [4] * 4)
        self.assertEqual(result["end_s"].tolist(), [7.0] * 4)

    def test_high_sparseness_notebook_call_keeps_rank_and_matrix_rows(self):
        """Execute the notebook call and check selected neuron sorting."""
        summary = pd.DataFrame({
            "global_neuron_id": [0, 1, 2, 3],
            "lifetime_sparseness": [0.9, 0.6, 0.8, np.nan],
            "preferred_stimulus": ["FLB", "FRB", "FLB", "FRB"],
            "max_response": [2., 5., 3., 1.],
        })
        matrix = np.arange(4 * 6).reshape(4, 6)
        namespace = {
            "rsf": rsf, "all_fish_data": {}, "fish_ids": ["fish_b", "fish_a"],
            "neuron_summary_table": summary, "plot_stimulus_ids": [4, 8],
            "plot_stimulus_labels": ["FLB", "FRB"],
            "plot_neuron_keep_mask": np.ones(4, dtype=bool),
            "plot_options": {"lifetime_sparseness_threshold": 0.65,
                             "combine_mode": "mean"},
        }
        with mock.patch.object(diagnostics, "build_matrix_all_fish", return_value=matrix):
            result = run_notebook_assignment("high_sparseness", namespace)
        np.testing.assert_array_equal(result["global_neuron_order"], [2, 0])
        np.testing.assert_array_equal(result["matrix"], matrix[[2, 0]])
        np.testing.assert_array_equal(result["plot_neuron_order"], [0, 1])
        self.assertEqual(result["summary_table"]["preferred_stimulus"].tolist(),
                         ["FLB", "FLB"])

    def test_mean_trace_notebook_call_preserves_filtered_fish_rows(self):
        """Average trials, then kept and pooled-filtered neurons in fish order."""
        fish_b = np.array([[[1., 3.], [2., 4.]],
                           [[5., 7.], [6., 8.]],
                           [[9., 11.], [10., 12.]]])
        fish_a = np.array([[[19., 21.], [20., 22.]],
                           [[29., 31.], [30., 32.]]])
        all_fish_data = {
            "fish_b": {"kept_neuron_indices": [2, 0],
                       "trial_aligned_traces_z_core": {7: fish_b}},
            "fish_a": {"kept_neuron_indices": [1, 0],
                       "trial_aligned_traces_z_core": {7: fish_a}},
        }
        namespace = {
            "rsf": rsf, "all_fish_data": all_fish_data,
            "fish_ids": ["fish_b", "fish_a"], "mean_stimulus_ids": [7],
            "response_row_metadata": pd.DataFrame(
                {"fish_id": ["fish_b", "fish_b", "fish_a", "fish_a"]}),
            "cell06_keep_mask": [True, False, False, True],
        }
        result = run_notebook_assignment("mean_trace_selection", namespace)
        self.assertEqual(list(result), [7])
        np.testing.assert_array_equal(result[7], [[10., 11.], [20., 21.]])

    def test_overlap_notebook_call_preserves_diagnostic_rows(self):
        """Execute the shared notebook call through z-score alignment and filtering."""
        active = {
            fid: pd.DataFrame({7: [1, 0, 1, 0], 11: [0, 1, 1, 0],
                               3: [0, 1, 0, 1], 21: [1, 0, 0, 1]})
            for fid in FISH_IDS
        }
        metadata = pd.DataFrame({
            "fish_id": ["fish_b"] * 4 + ["fish_a"] * 4,
            "neuron_id": list(range(4)) * 2,
        })
        namespace = {
            "rsf": rsf, "all_fish_data": cohort(), "active_matrices": active,
            "fish_ids": FISH_IDS, "response_row_metadata": metadata,
            "plot_neuron_keep_mask": [True, False, True, False,
                                      False, True, False, True],
            "side_stimulus_ids": {"left": [7, 11], "right": [3, 21]},
            "side_stimulus_labels": {"left": ["flickL", "boutL"],
                                     "right": ["flickR", "boutR"]},
            "overlap_config": {"side": "left", "aggregation": "pooled",
                               "plot": "z_score", "combine_mode": "mean",
                               "sort_mode": "decision_then_mean"},
            "neuron_summary_table": pd.DataFrame(),
        }
        result = run_notebook_assignment("overlap_data", namespace)
        self.assertEqual(result["diagnostic_trial_key"],
                         "trial_aligned_traces_z_core")
        self.assertEqual(result["diagnostic_stim_order"], [7, 11])
        diagnostic = result["active_trace_diagnostic"]
        self.assertEqual(diagnostic["trace_matrix"].shape, (4, 24))
        self.assertEqual(diagnostic["decision_matrix"].shape, (4, 2))
        self.assertEqual(diagnostic["row_metadata"]["fish_id"].tolist(),
                         ["fish_b", "fish_b", "fish_a", "fish_a"])
        self.assertEqual(sorted(result["diagnostic_neuron_order"].tolist()),
                         [0, 1, 2, 3])
        pd.testing.assert_frame_equal(
            result["matrix_to_plot"], result["overlap_results"]["pooled"]["left"])


if __name__ == "__main__":
    unittest.main()

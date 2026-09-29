"""S09 pure selection, summary, and pooled-row characterization."""

import ast
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
import pandas as pd

from src import reusable_several_fish as rsf
from src import several_fish_selection as selection


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks/calcium/shared/several_fish_reusable_analysis.ipynb"


class SeveralFishSelectionTests(unittest.TestCase):
    """Keep the notebook-facing selection and row-order contracts intact."""

    def test_pure_owner_and_historical_imports(self):
        """Ensure one implementation and no plotting/reporting dependency on import."""
        for name in (
            "resolve_stimulus_set", "resolve_response_control_columns",
            "build_selected_neuron_summary", "build_fish_keep_masks",
            "build_filtered_trial_aligned_traces_for_fish", "subset_active_matrices",
            "filter_active_matrices_by_keep_mask",
        ):
            self.assertIs(getattr(rsf, name), getattr(selection, name))
        code = ("import sys; import src.several_fish_selection; "
                "assert not {'matplotlib', 'seaborn', 'src.several_fish_reporting', "
                "'src.reusable_several_fish'} & sys.modules.keys()")
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_stimulus_and_control_selection_preserve_requested_order(self):
        """Check ID/name resolution, fallback, and control-column lookup."""
        reference = {
            "stimuli_names": ["FRB", "FLB"],
            "stimuli_id_map": {"FLB": 4, "FRB": 8},
            "trial_aligned_traces_z_core": {4: np.zeros((1, 2, 1)),
                                            8: np.zeros((1, 2, 1))},
        }
        self.assertEqual(rsf.resolve_stimulus_set([8, "FLB"], reference)["stimulus_ids"],
                         [8, 4])
        self.assertEqual(rsf.resolve_stimulus_set(None, reference)["stimulus_labels"],
                         ["FRB", "FLB"])
        self.assertEqual(rsf.resolve_stimulus_set(None, reference, ["FLB"])["stimulus_ids"],
                         [4])
        self.assertEqual(rsf.resolve_response_control_columns(
            [4, "FRB"], [8, 4], ["FRB", "FLB"], ["FRB", "FLB"]), ("FLB", "FRB"))
        with self.assertRaisesRegex(KeyError, "could not be resolved"):
            rsf.resolve_response_control_columns(
                [4, 99], [8, 4], ["FRB", "FLB"], ["FRB", "FLB"])

    def test_summary_and_keep_mask_have_known_rows_and_values(self):
        """Protect preference signs, selected rows, column order, and all tables."""
        response_matrices = {
            "fish_b": pd.DataFrame({"FLB": [4., 1.], "FRB": [1., 4.]}),
            "fish_a": pd.DataFrame({"FLB": [2.], "FRB": [2.]}),
        }
        pooled = pd.concat(response_matrices.values(), ignore_index=True)
        metadata = pd.DataFrame({
            "fish_id": ["fish_b", "fish_b", "fish_a"],
            "neuron_id": [0, 1, 0], "source_neuron_id": [3, 1, 2],
            "global_neuron_id": [0, 1, 2],
        })
        active = {
            "fish_b": pd.DataFrame({4: [1, 0], 8: [0, 1]}),
            "fish_a": pd.DataFrame({4: [1], 8: [1]}),
        }
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        summary_cell = next("".join(cell["source"]) for cell in notebook["cells"]
                            if cell["cell_type"] == "code"
                            and "summary_results = rsf.build_selected_neuron_summary("
                            in "".join(cell["source"]))
        assignment = next(node for node in ast.parse(summary_cell).body
                          if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name)
                                  and target.id == "summary_results"
                                  for target in node.targets))
        namespace = {
            "rsf": rsf, "response_matrices_by_fish": response_matrices,
            "pooled_response_matrix": pooled, "response_row_metadata": metadata,
            "active_matrices": active, "response_stimulus_ids": [4, 8],
            "response_stimulus_labels": ["FLB", "FRB"],
            "plot_stimulus_ids": [8, 4], "plot_stimulus_labels": ["FRB", "FLB"],
            "settings": {"analysis_label": "synthetic"},
            "filter_settings": {
                "left_right_controls": [4, 8], "left_right_filter_mode": "left",
                "left_right_threshold": 0.5, "left_right_range": (-0.3, 0.3),
            },
        }
        exec(compile(ast.Module(body=[assignment], type_ignores=[]),
                     str(NOTEBOOK), "exec"), namespace)
        result = namespace["summary_results"]
        np.testing.assert_allclose(result["left_right_index"], [0.6, -0.6, 0.])
        np.testing.assert_array_equal(result["plot_neuron_keep_mask"],
                                      [True, False, False])
        np.testing.assert_array_equal(result["plot_neuron_keep_indices"], [0])
        self.assertEqual(result["resolved_preference_controls"], ("FLB", "FRB"))
        self.assertEqual(result["selected_auc_response_matrix"].columns.tolist(),
                         ["FRB", "FLB"])
        self.assertEqual(result["selected_auc_response_matrix"].values.tolist(),
                         [[1.0, 4.0]])
        self.assertEqual(result["plot_filter_table"]["fish_id"].tolist(),
                         ["fish_b", "fish_b", "fish_a"])
        self.assertEqual(result["neuron_summary_table"]["global_neuron_id"].tolist(),
                         [0, 1, 2])
        self.assertEqual(result["neuron_summary_all_stimuli_table"].shape[0], 3)

    def test_keep_masks_traces_and_active_rows_preserve_fish_order(self):
        """Map pooled flags to fish rows before filtering traces and active data."""
        metadata = pd.DataFrame({"fish_id": ["fish_b", "fish_a", "fish_b", "fish_a"]})
        mask = np.array([True, False, False, True])
        fish_masks = rsf.build_fish_keep_masks(metadata, mask, ["fish_a", "fish_b"])
        self.assertEqual(list(fish_masks), ["fish_a", "fish_b"])
        np.testing.assert_array_equal(fish_masks["fish_a"], [False, True])
        np.testing.assert_array_equal(fish_masks["fish_b"], [True, False])

        active = {
            "fish_a": pd.DataFrame({8: [10, 11], 4: [20, 21]}),
            "fish_b": pd.DataFrame({8: [30, 31], 4: [40, 41]}),
        }
        subset = rsf.subset_active_matrices(active, ["fish_a", "fish_b"], [4, 8])
        self.assertEqual(subset["fish_a"].columns.tolist(), [4, 8])
        filtered = rsf.filter_active_matrices_by_keep_mask(
            subset, metadata, mask, ["fish_a", "fish_b"])
        self.assertEqual(filtered["fish_a"].values.tolist(), [[21, 11]])
        self.assertEqual(filtered["fish_b"].values.tolist(), [[40, 30]])

        traces = np.arange(4 * 2 * 1).reshape(4, 2, 1)
        fish_data = {"kept_neuron_indices": [3, 1, 0],
                     "trial_aligned_traces_z_core": {"8": traces}}
        selected = rsf.build_filtered_trial_aligned_traces_for_fish(
            fish_data, [8], fish_keep_mask=[True, False, True])
        np.testing.assert_array_equal(selected[8], traces[[3, 0]])
        with self.assertRaisesRegex(ValueError, "do not match the filter rows"):
            rsf.build_filtered_trial_aligned_traces_for_fish(
                fish_data, [8], fish_keep_mask=[True, False])
        with self.assertRaisesRegex(ValueError, "length does not match"):
            rsf.build_fish_keep_masks(metadata, mask[:-1], ["fish_a", "fish_b"])

    def test_shared_notebook_calls_remain_available(self):
        """Check the first consumer still calls the preserved public functions."""
        source = "\n".join("".join(cell.get("source", []))
                           for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"])
        for name in ("resolve_stimulus_set", "build_selected_neuron_summary",
                     "build_fish_keep_masks", "build_filtered_trial_aligned_traces_for_fish"):
            self.assertIn(f"rsf.{name}(", source)


if __name__ == "__main__":
    unittest.main()

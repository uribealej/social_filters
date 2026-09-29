"""S09 report ownership and file-contract checks."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from src import reusable_several_fish as rsf
from src import several_fish_reporting as reporting


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks/calcium/exp_01_flickering/01_all_fish_raster.ipynb"


class SeveralFishReportingTests(unittest.TestCase):
    """Protect the existing notebook-facing report and export contracts."""

    def test_notebook_facade_saves_the_expected_run_files(self):
        """Exercise the first notebook's call shape through its historical import."""
        self.assertIs(rsf.save_analysis_report_run, reporting.save_analysis_report_run)
        self.assertIs(rsf.export_notebook_report, reporting.export_notebook_report)
        self.assertIs(rsf._slugify_label, reporting._slugify_label)
        settings = {"experiment_name": "Exp 1", "fish_ids": ["fish_a"],
                    "data_path": Path("somewhere"), "combine_mode": "mean",
                    "trace_type": "zscore", "raster_sort_mode": "corravg"}
        report_settings = {"save_report": True, "run_label": "checked run",
                           "report_format": "html", "comments": "Reviewed."}
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        report_cell = next("".join(cell["source"]) for cell in notebook["cells"]
                           if cell["cell_type"] == "code"
                           and "report_result = rsf.save_analysis_report_run(" in "".join(cell["source"]))
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(reporting, "export_notebook_report",
                                   return_value={"ok": True, "message": "exported", "paths": []}) as export:
                with contextlib.redirect_stdout(io.StringIO()):
                    namespace = {"rsf": rsf, "pd": pd, "repo_root": ROOT,
                                 "settings": settings, "REPORT_SETTINGS": report_settings,
                                 "analysis_path": folder, "stim_order": np.array([1, 2]),
                                 "flat_matrix_all_fish": np.zeros((3, 8)),
                                 "all_fish_data": {"fish_a": {
                                     "kept_neuron_indices": [0, 1, 2],
                                     "stimuli_ids": [1, 2],
                                 }}}
                    exec(compile(report_cell, str(NOTEBOOK), "exec"), namespace)
                    result = namespace["report_result"]
            run_dir = result["run_dir"]
            self.assertEqual(run_dir.parent, Path(folder) / "Exp 1" / "reports")
            self.assertTrue(result["run_id"].endswith("_checked-run"))
            self.assertEqual({path.name for path in run_dir.iterdir()},
                             {"settings.json", "report_settings.json", "run_metadata.json",
                              "comments.md", "tables"})
            self.assertEqual(json.loads((run_dir / "settings.json").read_text(encoding="utf-8")),
                             {**settings, "data_path": "somewhere"})
            metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["all_fish_matrix_shape"], [3, 8])
            self.assertEqual(metadata["stim_order"], [1, 2])
            self.assertEqual(metadata["notebook_path"], str(NOTEBOOK))
            self.assertEqual((run_dir / "comments.md").read_text(encoding="utf-8"),
                             "# Comments\n\nReviewed.\n")
            pd.testing.assert_frame_equal(
                pd.read_csv(run_dir / "tables" / "run_summary.csv"),
                pd.DataFrame({"fish_id": ["fish_a"], "kept_neurons": [3],
                              "detected_stimuli": [2]}))
            self.assertEqual(result["tables"],
                             {"run_summary": run_dir / "tables" / "run_summary.csv"})
            self.assertEqual(result["export"]["ok"], True)
            self.assertEqual(export.call_args.kwargs["report_name"],
                             "01_all_fish_raster_report")
            self.assertEqual(export.call_args.kwargs["output_dir"], run_dir)

    def test_export_formats_and_pdf_fallback(self):
        """Check nbconvert arguments, expected files, and historical fallback status."""
        with tempfile.TemporaryDirectory() as folder:
            output_dir = Path(folder) / "report"
            calls = []

            def fake_run(command, **kwargs):
                calls.append(command)
                fmt = command[command.index("--to") + 1]
                if fmt == "webpdf":
                    return mock.Mock(returncode=1, stderr="browser unavailable", stdout="")
                suffix = ".html" if fmt == "html" else f".{fmt}"
                (output_dir / f"example{suffix}").write_text("export", encoding="utf-8")
                return mock.Mock(returncode=0, stderr="", stdout="")

            with mock.patch.object(reporting.subprocess, "run", side_effect=fake_run):
                result = rsf.export_notebook_report(NOTEBOOK, output_dir, "example")
            self.assertEqual([call[call.index("--to") + 1] for call in calls],
                             ["webpdf", "html"])
            self.assertEqual(result["paths"], [output_dir / "example.html"])
            self.assertFalse(result["ok"])
            self.assertIn("browser unavailable", result["message"])

    def test_disabled_report_has_no_output(self):
        """Keep opt-in saving inert for the notebook's default settings."""
        with tempfile.TemporaryDirectory() as folder:
            with contextlib.redirect_stdout(io.StringIO()):
                result = rsf.save_analysis_report_run(
                    {"experiment_name": "Exp 1"}, {"save_report": False}, folder)
            self.assertIsNone(result)
            self.assertEqual(list(Path(folder).iterdir()), [])


if __name__ == "__main__":
    unittest.main()

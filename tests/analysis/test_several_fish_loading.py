"""S09 cohort-loading and first-consumer contract checks."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from src import reusable_several_fish as rsf
from src import several_fish_loading as loading
from src import plotting_common as plot_common


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks/calcium/exp_01_flickering/01_all_fish_raster.ipynb"


def fish_bundle(stimuli=(4, 8)):
    """Build the loader keys consumed by the cohort preflight and notebook cell."""
    return {
        "trial_aligned_traces_z_core": {
            stimulus: np.zeros((2, 3, 1)) for stimulus in stimuli},
        "stimuli_id_map": {"FLB": 4, "FRB": 8},
        "stimuli_durations": {4: 5.0, 8: 5.0},
        "stimuli_names": ["FLB", "FRB"],
    }


class SeveralFishLoadingTests(unittest.TestCase):
    """Protect cohort order, path choice, errors, and the notebook-facing bundle."""

    def setUp(self):
        """Create a small canonical Lab root without experimental data."""
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.data_base = Path(self.folder.name)
        self.onedrive = self.data_base / "OneDrive - Universite de Lausanne"
        self.onedrive.mkdir()
        self.settings = {
            "experiment_name": "Exp_1_flickering",
            "data_base": self.data_base,
            "fish_ids": ["fish_b", "fish_a"],
            "selected_blocks": ["B1", "B2"],
            "verbose_loading": False,
            "timing": {"fps_2p": 2.0, "t_pre_s": 5.0, "t_post_s": 32.0},
            "stim_order": [8, 4],
        }

    def test_exp1_notebook_load_cell_preserves_bundle_and_order(self):
        """Execute the migrated first consumer with two ordered synthetic fish."""
        self.assertIs(rsf.load_and_preflight_fish_raster_inputs,
                      loading.load_and_preflight_fish_raster_inputs)
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        load_cell = next("".join(cell["source"]) for cell in notebook["cells"]
                         if cell["cell_type"] == "code"
                         and "raster_inputs = several_loading.load_and_preflight_fish_raster_inputs("
                         in "".join(cell["source"]))
        bundles = {fish_id: fish_bundle() for fish_id in self.settings["fish_ids"]}
        calls = []

        def fake_load(**kwargs):
            calls.append(kwargs)
            print("loader progress")
            return bundles[kwargs["fish_id"]]

        namespace = {"several_loading": loading, "plot_common": plot_common,
                     "USER_SETTINGS": self.settings}
        output = io.StringIO()
        with mock.patch.object(loading.exio, "load_and_align_2p_experiment",
                               side_effect=fake_load):
            with contextlib.redirect_stdout(output):
                exec(compile(load_cell, str(NOTEBOOK), "exec"), namespace)

        result = namespace["raster_inputs"]
        self.assertEqual(list(result), [
            "settings", "timing", "fish_ids", "stim_order", "main_path",
            "analysis_path", "all_fish_data", "reference_fish_id",
            "reference_fish", "trial_aligned_traces", "stimuli_id_map",
            "stimuli_durations", "stimuli_names",
        ])
        self.assertEqual(result["fish_ids"], ["fish_b", "fish_a"])
        self.assertEqual(result["stim_order"], [8, 4])
        self.assertEqual(list(result["all_fish_data"]), ["fish_b", "fish_a"])
        self.assertEqual(result["reference_fish_id"], "fish_b")
        self.assertIs(result["reference_fish"], bundles["fish_b"])
        self.assertIs(result["trial_aligned_traces"],
                      bundles["fish_b"]["trial_aligned_traces_z_core"])
        self.assertEqual(result["main_path"], self.onedrive / "Lab" / "Data" / "2p")
        self.assertEqual(result["analysis_path"], self.onedrive / "Lab" / "Analysis")
        self.assertIsNot(result["settings"], self.settings)
        self.assertIsNot(result["timing"], self.settings["timing"])
        self.assertEqual([call["fish_id"] for call in calls], ["fish_b", "fish_a"])
        self.assertEqual(calls[0]["selected_blocks"], ["B1", "B2"])
        self.assertEqual(calls[0]["fps_2p"], 2.0)
        self.assertEqual(calls[0]["t_pre_s"], 5.0)
        self.assertEqual(calls[0]["t_post_s"], 32.0)
        self.assertNotIn("loader progress", output.getvalue())
        self.assertIn("Detected stimulus map:", output.getvalue())
        self.assertEqual(set(namespace["stimuli_colors"]), {"FLB", "FRB"})

    def test_empty_cohort_or_stimulus_order_fails_before_loading(self):
        """Keep clear configuration errors ahead of path and loader work."""
        with mock.patch.object(loading.exio, "load_and_align_2p_experiment") as loader:
            for key, message in (("fish_ids", "at least one fish"),
                                 ("stim_order", "at least one stimulus")):
                with self.subTest(key=key):
                    settings = {**self.settings, key: []}
                    with self.assertRaisesRegex(ValueError, message):
                        rsf.load_and_preflight_fish_raster_inputs(settings)
            loader.assert_not_called()

    def test_missing_data_root_fails_before_loading(self):
        """Preserve the descriptive OneDrive lookup failure."""
        with tempfile.TemporaryDirectory() as folder:
            settings = {**self.settings, "data_base": Path(folder)}
            with self.assertRaisesRegex(FileNotFoundError,
                                        "Could not find the OneDrive Lab folder"):
                rsf.load_and_preflight_fish_raster_inputs(settings)

    def test_all_load_failures_are_reported_in_cohort_order(self):
        """Do not silently return a partial cohort when some fish fail."""
        def fake_load(**kwargs):
            if kwargs["fish_id"] == "fish_b":
                raise FileNotFoundError("missing traces")
            raise ValueError("bad timing")

        with mock.patch.object(loading.exio, "load_and_align_2p_experiment",
                               side_effect=fake_load) as loader:
            with self.assertRaises(RuntimeError) as error:
                rsf.load_and_preflight_fish_raster_inputs(self.settings)
        self.assertEqual(loader.call_count, 2)
        self.assertEqual(str(error.exception),
                         "Could not load all configured fish. "
                         "fish_b: FileNotFoundError: missing traces; "
                         "fish_a: ValueError: bad timing")

    def test_unavailable_stimulus_identifies_the_fish(self):
        """Preflight every loaded fish, including those after the reference fish."""
        bundles = {"fish_b": fish_bundle(), "fish_a": fish_bundle(stimuli=(4,))}
        with mock.patch.object(loading.exio, "load_and_align_2p_experiment",
                               side_effect=lambda **kwargs: bundles[kwargs["fish_id"]]):
            with self.assertRaisesRegex(ValueError,
                                        r"Stimulus order \[8, 4\] is unavailable for fish_a"):
                rsf.load_and_preflight_fish_raster_inputs(self.settings)


class AutomaticCohortPathTests(unittest.TestCase):
    """Resolve both computer layouts without consulting the test host's folders."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.home = self.root / "Users" / "suribear"
        self.home.mkdir(parents=True)
        self.legacy = self.root / "legacy_drive" / "Alejandro" / "Data"
        self.legacy.mkdir(parents=True)
        environment = mock.patch.dict("os.environ", {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        # Model the other machine's D: base without writing to a real drive.
        constructor = mock.patch.object(
            loading, "Path",
            side_effect=lambda value: self.legacy if value == r"D:\Alejandro\Data" else Path(value),
        )
        self.path_class = constructor.start()
        self.path_class.home.return_value = self.home
        self.addCleanup(constructor.stop)

    def make_lab(self, root):
        """Create the two directories that define a complete auto-discovery root."""
        paths = (root / "Lab" / "Data" / "2p", root / "Lab" / "Analysis")
        for path in paths:
            path.mkdir(parents=True)
        return paths

    def test_environment_priority_over_home_and_legacy(self):
        commercial = self.make_lab(self.root / "commercial")
        personal = self.make_lab(self.root / "personal")
        self.make_lab(self.home / "OneDrive - Université de Lausanne")
        self.make_lab(self.legacy / "OneDrive - Universite de Lausanne")
        with mock.patch.dict("os.environ", {
            "OneDriveCommercial": str(self.root / "commercial"),
            "OneDrive": str(self.root / "personal"),
        }):
            self.assertEqual(loading._resolve_cohort_paths(None), commercial)
        with mock.patch.dict("os.environ", {"OneDrive": str(self.root / "personal")}):
            self.assertEqual(loading._resolve_cohort_paths(None), personal)

    def test_home_supports_accented_and_unaccented_names_before_legacy(self):
        self.make_lab(self.legacy / "OneDrive - Universite de Lausanne")
        for name in ("OneDrive - Université de Lausanne", "OneDrive - Universite de Lausanne"):
            with self.subTest(name=name):
                home = self.home / name
                self.path_class.home.return_value = home
                expected = self.make_lab(home / name)
                self.assertEqual(loading._resolve_cohort_paths(None), expected)

    def test_legacy_layout_when_home_and_environment_have_no_lab(self):
        expected = self.make_lab(self.legacy / "OneDrive - Université de Lausanne")
        self.assertEqual(loading._resolve_cohort_paths(None), expected)

    def test_incomplete_environment_root_is_skipped(self):
        incomplete = self.root / "incomplete"
        (incomplete / "Lab" / "Data" / "2p").mkdir(parents=True)
        expected = self.make_lab(self.home / "OneDrive - Université de Lausanne")
        with mock.patch.dict("os.environ", {"OneDriveCommercial": str(incomplete)}):
            self.assertEqual(loading._resolve_cohort_paths(None), expected)

    def test_data_and_analysis_are_never_combined_from_different_roots(self):
        data_root = self.root / "data_only"
        analysis_root = self.root / "analysis_only"
        (data_root / "Lab" / "Data" / "2p").mkdir(parents=True)
        (analysis_root / "Lab" / "Analysis").mkdir(parents=True)
        with mock.patch.dict("os.environ", {
            "OneDriveCommercial": str(data_root), "OneDrive": str(analysis_root),
        }):
            with self.assertRaisesRegex(FileNotFoundError, "containing both"):
                loading._resolve_cohort_paths(None)

    def test_missing_root_reports_unique_candidates_and_manual_override(self):
        missing = self.root / "missing"
        with mock.patch.dict("os.environ", {
            "OneDriveCommercial": str(missing), "OneDrive": str(missing),
        }):
            with self.assertRaises(FileNotFoundError) as error:
                loading._resolve_cohort_paths(None)
        self.assertEqual(str(error.exception).count(str(missing)), 1)
        self.assertIn("settings['data_base']", str(error.exception))
        self.assertIn(str(self.home), str(error.exception))
        self.assertIn(str(self.legacy), str(error.exception))

    def test_explicit_base_overrides_discovery_and_does_not_fall_back(self):
        self.make_lab(self.home / "OneDrive - Université de Lausanne")
        base = self.root / "explicit"
        expected = self.make_lab(base / "OneDrive - Universite de Lausanne")
        self.assertEqual(loading._resolve_cohort_paths(base), expected)
        with self.assertRaisesRegex(FileNotFoundError, "Could not find the OneDrive Lab folder below"):
            loading._resolve_cohort_paths(self.root / "invalid_explicit")

    def test_auto_discovery_preserves_cohort_loader_bundle(self):
        expected = self.make_lab(self.home / "OneDrive - Université de Lausanne")
        settings = {
            "experiment_name": "example", "data_base": None,
            "fish_ids": ["fish_b", "fish_a"], "selected_blocks": ["B1"],
            "timing": {"fps_2p": 2.0, "t_pre_s": 5.0, "t_post_s": 25.0},
            "stim_order": [8, 4],
        }
        with mock.patch.object(loading.exio, "load_and_align_2p_experiment",
                               side_effect=lambda **kwargs: fish_bundle()) as loader:
            result = loading.load_and_preflight_fish_raster_inputs(settings)
        self.assertEqual((result["main_path"], result["analysis_path"]), expected)
        self.assertEqual(list(result["all_fish_data"]), ["fish_b", "fish_a"])
        self.assertEqual(result["stim_order"], [8, 4])
        self.assertIsNone(settings["data_base"])
        self.assertIsNone(result["settings"]["data_base"])
        self.assertEqual(loader.call_count, 2)
        self.assertEqual(loader.call_args.kwargs["main_path"], expected[0])
        self.assertEqual(loader.call_args.kwargs["stimuli_main_path"], expected[1])


if __name__ == "__main__":
    unittest.main()

"""Check wrapper paths without generating assets or opening a display."""

import ast
from pathlib import Path
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[2]


def path_assignments(relative_path, names, namespace):
    """Execute only the actual path assignments from a wrapper's source."""
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id in names
                     for target in node.targets)]
    assert len(nodes) == len(names), relative_path
    exec(compile(ast.Module(body=nodes, type_ignores=[]), relative_path, "exec"), namespace)


class StimulusPathTests(unittest.TestCase):
    def test_generator_defaults_and_explicit_outputs(self):
        for filename, folder in (
            ("Trayectory_flicker.py", "Exp_6_mapping_positions_retina_2"),
            ("Trayectory_rocking_stimuli.py", "Exp_6_mapping_positions_retina_3"),
        ):
            relative = "scripts/stimuli/generation/" + filename
            for lab in (Path("home-layout/Lab"), Path("drive-layout/Lab")):
                with self.subTest(filename=filename, lab=lab):
                    resolver = Mock(return_value=lab)
                    namespace = {"Path": Path, "resolve_lab_root": resolver}
                    path_assignments(relative, {"DEFAULT_EXPERIMENT_CONFIG"}, namespace)
                    namespace["experiment_config"] = namespace["DEFAULT_EXPERIMENT_CONFIG"]
                    path_assignments(relative, {"configured_output", "output_path"}, namespace)
                    self.assertEqual(namespace["output_path"], lab / "Data" / "stimuli" / folder)
                    resolver.assert_called_once_with()
                    resolver.reset_mock()
                    explicit = str(ROOT / "custom-output")
                    namespace["experiment_config"] = {"output_path": explicit}
                    path_assignments(relative, {"configured_output", "output_path"}, namespace)
                    self.assertEqual(namespace["output_path"], Path(explicit))
                    resolver.assert_not_called()

    def test_playback_keeps_experiment_subfolder(self):
        lab = Path("local/Lab")
        namespace = {"resolve_lab_root": Mock(return_value=lab)}
        path_assignments("scripts/stimuli/playback/try_projection.py", {"stimuli_path"}, namespace)
        self.assertEqual(namespace["stimuli_path"], lab / "Results" / "Paper" / "2p_derived"
                         / "Exp_9_RF_memory_inhibition" / "stimuli")

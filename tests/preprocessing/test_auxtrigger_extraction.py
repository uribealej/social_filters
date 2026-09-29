"""Aux-trigger API checks with and without its TIFF reader dependency."""

from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np

from src import auxtrigger_extraction


class AuxTriggerExtractionTests(unittest.TestCase):
    """Keep the optional dependency local to the public extraction call."""

    def test_missing_reader_reports_install_action_at_call_time(self):
        with patch.dict(sys.modules, {"ScanImageTiffReader": None}):
            with self.assertRaisesRegex(ImportError, "scanimage-tiff-reader"):
                auxtrigger_extraction.extract_aux_trigger_frames(Path("unused.tif"))

    def test_reader_extracts_positive_events_by_channel_and_frame(self):
        descriptions = [
            "auxTrigger0 = [0, 0]\nauxTrigger1 = [0, 2]",
            "auxTrigger0 = [1, 0]\nauxTrigger1 = [0, 0]",
            "auxTrigger0 = [-1, 0]\nauxTrigger1 = [0, 0]",
        ]

        class FakeReader:
            def __init__(self, path):
                self.path = path

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def __len__(self):
                return len(descriptions)

            def description(self, frame_idx):
                return descriptions[frame_idx]

        reader_module = ModuleType("ScanImageTiffReader")
        reader_module.ScanImageTiffReader = FakeReader
        with patch.dict(sys.modules, {"ScanImageTiffReader": reader_module}):
            events = auxtrigger_extraction.extract_aux_trigger_frames(Path("stack.tif"))

        self.assertEqual(["auxTrigger0", "auxTrigger1"], list(events))
        self.assertEqual([1], [frame for frame, _ in events["auxTrigger0"]])
        self.assertEqual([0], [frame for frame, _ in events["auxTrigger1"]])
        np.testing.assert_array_equal(events["auxTrigger0"][0][1], [1, 0])
        np.testing.assert_array_equal(events["auxTrigger1"][0][1], [0, 2])


if __name__ == "__main__":
    unittest.main()

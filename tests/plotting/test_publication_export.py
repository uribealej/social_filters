"""Physical export size must survive user-level tight-bbox preferences."""

from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.plotting_common import save_publication_figure


class PublicationExportTests(unittest.TestCase):
    def tearDown(self):
        """Close figures after each export check."""
        plt.close("all")

    def test_vector_and_raster_exports_preserve_canvas_dimensions(self):
        """An 85 by 60 mm panel retains its size when tight bbox is global."""
        fig, ax = plt.subplots(figsize=(85 / 25.4, 60 / 25.4))
        ax.plot([0, 1], [0, 1])
        with tempfile.TemporaryDirectory() as folder, plt.rc_context({"savefig.bbox": "tight"}):
            paths = save_publication_figure(fig, Path(folder) / "panel", dpi=100)
            self.assertEqual([p.suffix for p in paths], [".pdf", ".svg", ".png"])
            self.assertTrue(all(p.stat().st_size > 0 for p in paths))
            svg = ET.parse(paths[1]).getroot()
            width, height = map(float, svg.attrib["viewBox"].split()[2:])
            self.assertAlmostEqual(width, 85 / 25.4 * 72, places=4)
            self.assertAlmostEqual(height, 60 / 25.4 * 72, places=4)
            png = plt.imread(paths[2])
            self.assertEqual(png.shape[:2], (int(60 / 25.4 * 100), int(85 / 25.4 * 100)))
            self.assertEqual(plt.rcParams["savefig.bbox"], "tight")

    def test_invalid_formats_fail_before_creating_files(self):
        """Reject unsupported exports before a partial set is written."""
        fig = plt.figure()
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                save_publication_figure(fig, Path(folder) / "panel", formats=("png", "bad"))
            self.assertEqual(list(Path(folder).iterdir()), [])


if __name__ == "__main__":
    unittest.main()

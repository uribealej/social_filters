"""Scientific display contracts for the three mirrored figure sets."""

import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import several_fish_figures as figures
from tests.analysis.test_recruitment_containment import CONDITIONS, prepared_analysis


class MirroredRecruitmentFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = prepared_analysis()

    def tearDown(self):
        plt.close("all")

    def test_recruitment_and_activity_share_side_scales_and_show_equal_fish_mean(self):
        fig, axes = figures.plot_recruitment_activity_summary(self.analysis)
        self.assertEqual(axes.shape, (2, 2))
        for row in (0, 1):
            self.assertEqual(axes[row, 0].get_ylim(), axes[row, 1].get_ylim())
        self.assertEqual([tick.get_text() for tick in axes[0, 0].get_xticklabels()], CONDITIONS)
        mean = next(line for line in axes[0, 0].lines if line.get_label() == "Fish mean")
        self.assertAlmostEqual(mean.get_ydata()[1], (2 / 3 + 1) / 2)
        self.assertEqual(len(fig.legends[0].texts), 3)
        fig.canvas.draw()

    def test_matrix_direction_color_limits_and_empty_sources(self):
        fig, axes = figures.plot_recruitment_containment_matrices(self.analysis)
        for ax in axes.ravel():
            self.assertEqual(ax.images[0].get_clim(), (0, 1))
        self.assertIn("Source", axes[0, 0].get_ylabel())
        image = axes[0, 0].images[0].get_array()
        self.assertAlmostEqual(image[0, 1], (2 / 3 + 1) / 2)
        self.assertTrue(np.ma.getmaskarray(image)[-1].all())
        fig.canvas.draw()

    def test_summary_uses_all_rocking_and_preserves_undefined_points(self):
        fig, axes = figures.plot_recruitment_containment_summary(self.analysis)
        self.assertEqual([tick.get_text() for tick in axes[1, 0].get_xticklabels()], ["F1", "F2", "F3"])
        mean = next(line for line in axes[1, 0].lines if line.get_label() == "Fish mean")
        np.testing.assert_allclose(mean.get_ydata()[:2], [1, 0])
        self.assertTrue(np.isnan(mean.get_ydata()[2]))
        self.assertEqual(axes[1, 0].get_ylim(), axes[1, 1].get_ylim())
        fig.canvas.draw()


if __name__ == "__main__":
    unittest.main()

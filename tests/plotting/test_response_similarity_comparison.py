"""Mirrored population/side comparison and fixed similarity scales."""

import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.several_fish_diagnostics import build_response_similarity_comparison
from src.several_fish_figures import plot_response_similarity_comparison
from tests.analysis.test_response_similarity_comparison import vector_example
from tests.analysis.test_recruitment_containment import CONDITIONS


class ResponseSimilarityFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = build_response_similarity_comparison(vector_example())

    def tearDown(self):
        plt.close("all")

    def test_metric_scales_population_rows_and_mirrored_stimulus_order(self):
        for metric, limits in (("jaccard", (0, 1)), ("pearson", (-1, 1)), ("cosine", (-1, 1))):
            with self.subTest(metric=metric):
                fig, axes = plot_response_similarity_comparison(self.result, metric)
                self.assertEqual(axes.shape, (2, 2))
                for row in (0, 1):
                    for column in (0, 1):
                        self.assertEqual(axes[row, column].images[0].get_clim(), limits)
                        self.assertEqual([tick.get_text() for tick in axes[row, column].get_xticklabels()], CONDITIONS)
                self.assertIn("Left", axes[0, 0].get_title())
                self.assertIn("Right", axes[0, 1].get_title())
                self.assertIn("All-rocking", axes[1, 0].get_title())
                fig.canvas.draw()

    def test_negative_similarities_and_undefined_core_pearson_are_displayed(self):
        fig, axes = plot_response_similarity_comparison(self.result, "pearson")
        image = axes[0, 0].images[0].get_array()
        self.assertAlmostEqual(image[0, 3], 0)  # Mean of -1 and +1 across fish.
        self.assertTrue(np.ma.getmaskarray(axes[1, 0].images[0].get_array()).all())
        fig.canvas.draw()


if __name__ == "__main__":
    unittest.main()

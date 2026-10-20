"""Mirrored sparseness distributions, summaries, and missing populations."""

import unittest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.several_fish_diagnostics import build_family_lifetime_sparseness
from src.several_fish_figures import plot_family_lifetime_sparseness, plot_family_sparseness_summary
from tests.analysis.test_family_lifetime_sparseness import prepared_profiles


class FamilySparsenessFigureTests(unittest.TestCase):
    def setUp(self):
        self.profiles = prepared_profiles()
        self.result = build_family_lifetime_sparseness(self.profiles, [0, .25, .5, .75, 1])

    def tearDown(self):
        plt.close("all")

    def test_each_population_has_mirrored_family_rows_and_matching_scales(self):
        for population in self.result["populations"]:
            fig, axes = plot_family_lifetime_sparseness(self.result, population)
            self.assertEqual(axes.shape, (2, 2))
            for row, family in enumerate(("Rocking", "Flicker")):
                for column, side in enumerate(("Left", "Right")):
                    ax = axes[row, column]
                    self.assertIn(side, ax.get_title())
                    self.assertIn(family, ax.get_title())
                    self.assertEqual(ax.get_xlim(), (0, 1))
                    self.assertEqual(len(ax.lines), 3)
                    self.assertEqual(ax.get_ylim(), (0, 1.025))
            fig.canvas.draw()

    def test_summary_uses_individual_fish_and_equal_fish_mean(self):
        fig, axes = plot_family_sparseness_summary(self.result)
        for row, population in enumerate(self.result["populations"]):
            for column in (0, 1):
                self.assertIn(self.result["population_labels"][population], axes[row, column].get_title())
                self.assertEqual([t.get_text() for t in axes[row, column].get_xticklabels()], ["rocking", "flicker"])
                self.assertEqual(axes[row, column].get_ylim(), (-.025, 1.025))
        fig.canvas.draw()

    def test_no_defined_values_get_annotation_and_missing_curves(self):
        for fid in self.profiles["fish_ids"]:
            self.profiles["profiles"][fid]["left"]["all_kept"]["responses"].loc[:, ["R1", "R2", "R3"]] = 0
        result = build_family_lifetime_sparseness(self.profiles, [0, 1])
        fig, axes = plot_family_lifetime_sparseness(result, "all_kept")
        self.assertIn("No defined sparseness", [text.get_text() for text in axes[0, 0].texts])
        self.assertTrue(np.isnan(axes[0, 0].lines[-1].get_ydata()).all())
        fig.canvas.draw()


if __name__ == "__main__":
    unittest.main()

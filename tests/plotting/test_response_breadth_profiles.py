"""Mirrored membership layouts and signed profile display contracts."""

import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.several_fish_diagnostics import build_response_breadth_profiles, build_response_similarity_comparison, build_pooled_response_profiles
from src.several_fish_figures import plot_response_breadth_summary, plot_response_profile_panels, plot_pooled_response_profiles
from src.plotting_specificity import plot_signed_response_profiles
from tests.analysis.test_recruitment_containment import prepared_analysis, CONDITIONS


class ResponseBreadthFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        analysis = prepared_analysis()
        cls.result = build_response_breadth_profiles(analysis, build_response_similarity_comparison(analysis))

    def tearDown(self):
        plt.close("all")

    def test_membership_modes_have_matching_scales_order_and_fish_points(self):
        for mode in ("breadth", "combinations"):
            fig, axes = plot_response_breadth_summary(self.result, mode=mode)
            self.assertEqual(axes.shape, (2, 2))
            for row, family in enumerate(("rocking", "flicker")):
                for column, side in enumerate(("Left", "Right")):
                    ax = axes[row, column]
                    self.assertIn(side, ax.get_title())
                    self.assertIn(family.capitalize(), ax.get_title())
                    self.assertEqual(ax.get_ylim(), (-0.025, 1.025))
                    self.assertEqual(len(ax.lines), 3)  # Two fish and equal-fish mean.
                    self.assertEqual(len(ax.get_xticklabels()), 4 if mode == "breadth" else 8)
            fig.canvas.draw()

    def test_both_profile_populations_share_full_signed_range_and_condition_order(self):
        for population in self.result["populations"]:
            fig, axes = plot_response_profile_panels(self.result, population, figsize=(12, 9))
            self.assertEqual(axes.shape, (2, 2))
            for row, fid in enumerate(self.result["fish_ids"]):
                for column, side in enumerate(("left", "right")):
                    ax = axes[row, column]
                    image = ax.images[0]
                    self.assertEqual(image.get_clim(), (-8, 8))
                    self.assertEqual([label.get_text() for label in ax.get_xticklabels()], CONDITIONS)
                    self.assertIn(fid, ax.get_title())
                    self.assertIn(side.capitalize(), ax.get_title())
                    expected = self.result["profiles"][fid][side][population]["responses"].to_numpy()
                    np.testing.assert_allclose(image.get_array(), expected)
            fig.canvas.draw()

    def test_empty_and_missing_profiles_render_without_inventing_values(self):
        fig, axes = plt.subplots(1, 2)
        empty = pd.DataFrame(columns=CONDITIONS, dtype=float)
        plot_signed_response_profiles(empty, axes[0], "Empty", 1)
        self.assertIn("No neurons", [text.get_text() for text in axes[0].texts])
        image = plot_signed_response_profiles(pd.DataFrame([[-1, np.nan, 2]], columns=["A", "B", "C"]),
                                              axes[1], "Missing", 2)
        self.assertTrue(np.ma.getmaskarray(image.get_array())[0, 1])
        self.assertEqual(image.get_array()[0, 0], -1)
        fig.canvas.draw()

    def test_profile_subsets_preserve_requested_order_and_global_color_scale(self):
        selected = list(reversed(self.result["fish_ids"]))[:1]
        fig, axes = plot_response_profile_panels(self.result, "all_kept", fish_ids=selected)
        self.assertEqual(axes.shape, (1, 2))
        for column, side in enumerate(("left", "right")):
            image = axes[0, column].images[0]
            self.assertEqual(image.get_clim(), (-8, 8))
            expected = self.result["profiles"][selected[0]][side]["all_kept"]["responses"]
            np.testing.assert_allclose(image.get_array(), expected)
        for invalid in ([], ["missing"], selected * 2):
            with self.assertRaises(ValueError):
                plot_response_profile_panels(self.result, "all_kept", fish_ids=invalid)

    def test_pooled_figures_keep_matching_rows_and_raw_intensities(self):
        pooled = build_pooled_response_profiles(self.result)
        for limit in (None, 2):
            panels = plot_pooled_response_profiles(pooled, figsize=(89 / 25.4, 70 / 25.4), color_limit=limit)
            self.assertEqual(list(panels), ['left', 'right'])
            for side, (fig, ax) in panels.items():
                image = ax.images[0]
                np.testing.assert_allclose(image.get_array(), pooled['responses'][side].to_numpy())
                self.assertEqual(image.get_clim(), (-8, 8) if limit is None else (-2, 2))
                self.assertEqual([t.get_text() for t in ax.get_xticklabels()], CONDITIONS)
                self.assertEqual(image.colorbar.extend, 'neither' if limit is None else 'both')
                fig.canvas.draw()


if __name__ == "__main__":
    unittest.main()

"""Independent summary panels preserve values, scales, and anonymous styling."""

import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import several_fish_diagnostics as diagnostics
from src import several_fish_figures as figures
from tests.analysis.test_recruitment_containment import prepared_analysis


def summary_cases(analysis, breadth, sparseness):
    """Return the seven notebook summary calls in their display order."""
    return [
        (figures.plot_recruitment_activity_summary, analysis, {}),
        (figures.plot_recruitment_containment_summary, analysis, {}),
        (figures.plot_response_breadth_summary, breadth, {"mode": "breadth"}),
        (figures.plot_response_breadth_summary, breadth, {"mode": "combinations"}),
        (figures.plot_family_lifetime_sparseness, sparseness, {"population": "all_kept"}),
        (figures.plot_family_lifetime_sparseness, sparseness, {"population": "all_rocking"}),
        (figures.plot_family_sparseness_summary, sparseness, {}),
    ]


class SeparateSummaryPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Prepare known-output inputs through the scientific owners once."""
        cls.analysis = prepared_analysis()
        similarity = diagnostics.build_response_similarity_comparison(cls.analysis)
        cls.breadth = diagnostics.build_response_breadth_profiles(cls.analysis, similarity)
        cls.sparseness = diagnostics.build_family_lifetime_sparseness(cls.breadth, [0, 0.25, 0.5, 0.75, 1])

    def tearDown(self):
        """Close test figures."""
        plt.close("all")

    def test_all_seven_calls_preserve_mean_values_and_side_scales(self):
        """Splitting changes presentation only; defaults retain their grid API."""
        for function, bundle, options in summary_cases(self.analysis, self.breadth, self.sparseness):
            with self.subTest(function=function.__name__, options=options):
                grid, axes = function(bundle, **options)
                self.assertEqual(axes.shape, (2, 2))
                expected = [next(line for line in ax.lines if line.get_label() in {"Fish mean", "Fish mean CDF"})
                            for ax in axes.ravel()]
                panels = function(bundle, **options, figsize=(45 / 25.4, 27.5 / 25.4),
                                  separate=True, individual_style="anonymous_points")
                self.assertEqual(len(panels), 4)
                self.assertEqual([key.rsplit("_", 1)[1] for key in panels], ["left", "right"] * 2)
                for index, (figure, ax) in enumerate(panels.values()):
                    np.testing.assert_allclose(figure.get_size_inches() * 25.4, [45, 27.5])
                    mean = next(line for line in ax.lines if line.get_label() == "Mean")
                    np.testing.assert_allclose(mean.get_xdata(), expected[index].get_xdata())
                    np.testing.assert_allclose(mean.get_ydata(), expected[index].get_ydata(), equal_nan=True)
                    np.testing.assert_allclose(ax.get_ylim(), axes.ravel()[index].get_ylim())
                    self.assertEqual([text.get_text() for text in figure.legends[0].texts], ["Fish", "Mean"])
                    self.assertTrue(ax.get_xlabel() and ax.get_ylabel())
                    fish_lines = ax.lines[:len(bundle["fish_ids"])]
                    self.assertTrue(all(line.get_linestyle() == "None" for line in fish_lines))
                    self.assertTrue(all(line.get_color() == "0.6" for line in fish_lines))
                    self.assertEqual(mean.get_color(), "black")
                    for text in [ax.title, *figure.legends[0].texts]:
                        self.assertFalse(any(fid in text.get_text() for fid in bundle["fish_ids"]))
                plt.close("all")

    def test_binary_combination_labels_preserve_membership_order(self):
        """Each binary position maps to its family variant, including none/all."""
        panels = figures.plot_response_breadth_summary(
            self.breadth, mode="combinations", separate=True, individual_style="anonymous_points")
        for family in self.breadth["families"]:
            labels = [text.get_text() for text in panels[f"combinations_{family}_left"][1].get_xticklabels()]
            self.assertEqual(labels, ["000", "100", "010", "001", "110", "101", "011", "111"])

    def test_family_responsive_panels_exclude_zero_and_keep_equal_fish_means(self):
        for mode in ("breadth", "combinations"):
            panels = figures.plot_response_breadth_summary(
                self.breadth, mode=mode, separate=True, individual_style="anonymous_points",
                denominator="family_responsive")
            table = self.breadth["tables"][f"response_{mode}_family_responsive"]
            for key, (fig, ax) in panels.items():
                _, family, side = key.split("_")
                labels = [text.get_text() for text in ax.get_xticklabels()]
                self.assertNotIn("0" if mode == "breadth" else "000", labels)
                mean = next(line for line in ax.lines if line.get_label() == "Mean")
                category = "n_active_stimuli" if mode == "breadth" else "membership"
                expected = table.loc[(table.side == side) & (table.family == family)].groupby(
                    category, sort=False).fraction_family_responsive.mean()
                np.testing.assert_allclose(mean.get_ydata(), expected.to_numpy(), equal_nan=True)
                self.assertAlmostEqual(np.nansum(mean.get_ydata()), 1)
                self.assertIn("Responsive", ax.get_ylabel())


if __name__ == "__main__":
    unittest.main()

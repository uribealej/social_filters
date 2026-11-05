"""Shared-neuron AUC difference figures retain equal-fish means and missingness."""
import unittest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.several_fish_figures import plot_shared_control_auc
from tests.analysis import test_shared_control_auc as auc_cases


class SharedAucPlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_difference_direction_both_metrics_and_shared_side_scales(self):
        analysis = auc_cases.SharedControlAucTests().prepare()
        for metric, factor in [('normalized_auc', 1), ('raw_auc', 6)]:
            panels = plot_shared_control_auc(analysis, metric=metric)
            self.assertEqual(panels['left'][1].get_ylim(), panels['right'][1].get_ylim())
            ax = panels['left'][1]
            mean = next(line for line in ax.lines if line.get_label() == 'Mean')
            np.testing.assert_allclose(mean.get_ydata()[:2], np.array([2., -2.]) * factor)
            self.assertTrue(np.isnan(mean.get_ydata()[2]))
            self.assertEqual([t.get_text() for t in ax.get_xticklabels()], ['AL', 'BL', 'EL'])
            self.assertEqual([t.get_text() for t in panels['left'][0].legends[0].get_texts()], ['Fish', 'Mean'])
            self.assertTrue(any(np.all(np.asarray(line.get_ydata()) == 0) for line in ax.lines))
            panels['left'][0].canvas.draw()


if __name__ == '__main__':
    unittest.main()

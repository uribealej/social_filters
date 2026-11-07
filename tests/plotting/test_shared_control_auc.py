"""Shared-neuron AUC difference figures retain equal-fish means and missingness."""
import unittest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.several_fish_figures import plot_shared_control_auc, plot_recruitment_strength_summary
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

    def summary_analysis(self):
        shared = auc_cases.SharedControlAucTests().prepare()
        conditions = ['Control', 'A', 'B', 'Empty']
        selections = {side: {'stimulus_ids': ids, 'groups': {'control': ids[:1]}}
                      for side, ids in [('left', [1, 2, 3, 4]), ('right', [11, 12, 13, 14])]}
        rows = [dict(fish_id=fid, side=side, condition=condition,
                     mean_normalized_auc=(j + 1) * 10 + i * 2,
                     active_fraction=(j + 1) / 10)
                for side in selections for i, fid in enumerate(shared['fish_ids'])
                for j, condition in enumerate(conditions)]
        return dict(fish_ids=shared['fish_ids'], side_selections=selections,
                    condition_labels=conditions, tables={'recruitment': pd.DataFrame(rows)}), shared

    def test_summary_permutation_keeps_values_control_and_labels_aligned(self):
        analysis, shared = self.summary_analysis()
        order = {'left': [3, 1, 4, 2], 'right': [11, 12, 13, 14]}
        original = analysis['tables']['recruitment'].copy(deep=True)
        for requested in (None, order):
            panels = plot_recruitment_strength_summary(analysis, shared, side_stimuli=requested)
            axes = panels['left'][1]
            means = [next(line for line in ax.lines if line.get_label() == 'Mean').get_ydata()
                     for ax in axes]
            indices = [0, 1, 2, 3] if requested is None else [2, 0, 3, 1]
            for actual, expected in zip(means, ([11, 21, 31, 41], [.1, .2, .3, .4],
                                                [np.nan, 2, -2, np.nan])):
                np.testing.assert_allclose(actual, np.asarray(expected)[indices], equal_nan=True)
            self.assertEqual([t.get_text() for t in axes[-1].get_xticklabels()],
                             [analysis['condition_labels'][i] for i in indices])
            self.assertEqual(next(t for t in axes[-1].texts if t.get_text() == 'n/a').get_position()[0],
                             indices.index(0))
            for row in range(3):
                self.assertEqual(axes[row].get_ylim(), panels['right'][1][row].get_ylim())
            panels['left'][0].canvas.draw()
        pd.testing.assert_frame_equal(analysis['tables']['recruitment'], original)
        self.assertEqual(analysis['side_selections']['left']['stimulus_ids'], [1, 2, 3, 4])

    def test_summary_rejects_missing_or_duplicate_conditions(self):
        analysis, shared = self.summary_analysis()
        for requested in ({'left': [1, 2, 3, 4]},
                          {'left': [1, 2, 3, 3], 'right': [11, 12, 13, 14]},
                          {'left': [1, 2, 3], 'right': [11, 12, 13, 14]}):
            with self.subTest(requested=requested), self.assertRaisesRegex(ValueError, 'side_stimuli'):
                plot_recruitment_strength_summary(analysis, shared, side_stimuli=requested)


if __name__ == '__main__':
    unittest.main()

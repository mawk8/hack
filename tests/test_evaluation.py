"""Classification accounting must agree with hand-counted examples."""
import unittest

from avito.evaluation import metrics


class EvaluationTests(unittest.TestCase):
    def test_micro_f1_counts_false_positives_on_empty_targets(self):
        score = metrics([[102, 103], [], [104]], [[102], [105], [104, 106]])
        self.assertEqual((score['tp'], score['fp'], score['fn']), (2, 2, 1))
        self.assertAlmostEqual(score['micro_f1'], 4 / 7)

    def test_no_positive_labels_has_defined_zero_f1(self):
        self.assertEqual(metrics([[]], [[]]), dict(tp=0, fp=0, fn=0, micro_f1=0.0))


if __name__ == '__main__':
    unittest.main()

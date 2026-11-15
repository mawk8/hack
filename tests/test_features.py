"""The added word branch must distinguish word order ignored by char_wb."""
import unittest

from avito.baseline import predict, train


class FeatureTests(unittest.TestCase):
    def test_word_features_learn_order_that_char_wb_cannot(self):
        examples = [('alpha beta', [102]), ('beta alpha', [103]),
                    ('pipes taps', [104]), ('cables sockets', [105]),
                    ('paint walls', [106]), ('generic enquiry', [])] * 20
        descriptions, labels = map(list, zip(*examples))
        model = train(descriptions, labels, features='char-word')
        self.assertEqual(predict(model, 'alpha beta'), [102])
        self.assertEqual(predict(model, 'beta alpha'), [103])
        self.assertEqual(predict(model, 'generic enquiry'), [])

    def test_unknown_feature_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            train(['synthetic fixture'], [[]], features='typo')


if __name__ == '__main__':
    unittest.main()

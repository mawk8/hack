"""Focused contract/leakage checks; fixtures contain no real advertisement text."""
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from avito.baseline import load_model, predict, train
from avito.data import select_training_rows, verify_freeze


class BaselineTests(unittest.TestCase):
    def test_all_evaluation_groups_excluded_and_ids_stable(self):
        rows = [dict(item_id=v, description=f"fixture {v}", found_microcategories=[])
                for v in [45, 12, 77, 90]]
        manifest = [dict(source_row=i, item_id=r['item_id'], group_id=g)
                    for i, (r, g) in enumerate(zip(rows, ['dev', 'dev', 'ctrl', 'fit']))]
        exclusions = [dict(**m, purpose='development' if m['group_id']=='dev' else 'control')
                      for m in manifest[:3]]
        fit = select_training_rows(rows, manifest, exclusions)
        self.assertEqual([(r['source_row'], r['item_id']) for r in fit], [(3, 90)])
        with self.assertRaises(ValueError):
            select_training_rows(rows, manifest, exclusions[1:])
        with self.assertRaises(ValueError):
            select_training_rows(rows, manifest, [dict(exclusions[0], item_id=999)] + exclusions[1:])

    def test_freeze_corruption_rejected_without_real_file_writes(self):
        content = b'original frozen bytes'
        checksum = hashlib.sha256(content).hexdigest()
        files = {'archive.zip': b'archive', 'frozen.bin': content, 'overlay.bin': b'overlay'}
        files['freeze_manifest.json'] = json.dumps(dict(
            archive_sha256=hashlib.sha256(b'archive').hexdigest(), sha256={'frozen.bin':checksum})).encode()
        files['training_exclusions_freeze_manifest.json'] = json.dumps(dict(
            legacy_frozen_sha256={'freeze_manifest.json':hashlib.sha256(files['freeze_manifest.json']).hexdigest()},
            sha256={'overlay.bin':hashlib.sha256(b'overlay').hexdigest()})).encode()
        def read(path):
            return files[path.name]
        with patch.object(Path, 'read_bytes', read), patch.object(Path, 'open', lambda path, *a, **k: io.BytesIO(files[path.name])):
            verify_freeze(Path('fixture'), Path('archive.zip'))
            for name in ('archive.zip', 'frozen.bin', 'overlay.bin', 'freeze_manifest.json'):
                original = files[name]
                files[name] += b'corruption'
                with self.subTest(corrupted=name), self.assertRaises(ValueError):
                    verify_freeze(Path('fixture'), Path('archive.zip'))
                files[name] = original

    def test_sources_descriptions_and_labels_validated(self):
        for source in [True, '101', None, 100, 107]:
            with self.subTest(source=source), self.assertRaises(ValueError):
                predict(None, 'fixture', source)
        for source in [102, 103, 104, 105, 106]:
            with self.subTest(source=source), self.assertRaises(NotImplementedError):
                predict(None, 'fixture', source)
        for description in [None, '', '  ', 5]:
            with self.subTest(description=description), self.assertRaises(ValueError):
                predict(None, description)
        for labels in [[[101]], [[True]], [[102, 102]], [['102']], [None]]:
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                train(['fixture'], labels)
        with self.assertRaises(ValueError):
            train(['fixture'], [])
        with self.assertRaises(ValueError):
            train(['fixture'], None)
        with self.assertRaises(ValueError):
            load_model('untrusted.joblib')

    def test_probability_threshold_and_output_validation(self):
        class FixedProbabilities:
            def __init__(self, values):
                self.values = values

            def predict_proba(self, descriptions):
                return self.values

        self.assertEqual(predict(FixedProbabilities([[.5, .49, .9, 0, 1]]), 'fixture'), [102, 104, 106])
        self.assertEqual(predict(FixedProbabilities([[0, 0, 0, 0, 0]]), 'fixture'), [])
        for probabilities in [[[0, 0]], [[float('nan'), 0, 0, 0, 0]], [[1.1, 0, 0, 0, 0]]]:
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                predict(FixedProbabilities(probabilities), 'fixture')

    def test_fixed_category_mapping_and_empty_predictions(self):
        examples = [('plumbing pipes taps', [102]), ('electrical cables sockets', [103]),
                    ('wallpaper paint walls', [104]), ('floors parquet laminate', [105]),
                    ('doors windows glazing', [106]), ('generic renovation enquiry', [])]
        descriptions, labels = zip(*(examples * 20))
        model = train(list(descriptions), list(labels))
        for description, expected in examples:
            with self.subTest(expected=expected):
                self.assertEqual(predict(model, description), expected)


if __name__ == '__main__':
    unittest.main()

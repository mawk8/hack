"""Public-only texts must be excluded at both synthetic entry points."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import uuid
import zipfile

from scripts import compare_experiments as comparison
from scripts import generate_synthetic as generator


class PublicSyntheticBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.root = generator.ROOT / 'data/test-v2' / uuid.uuid4().hex
        self.root.mkdir(parents=True)
        self.archive = self.root / 'fixture.zip'
        self.fit = [dict(item_id=7, group_id='fit', description='Исходная штукатурка',
                         found_microcategories=[{'mcId': 102}])]
        with zipfile.ZipFile(self.archive, 'w') as source:
            source.writestr('innoglobalhack_avito/public-train-annotated.jsonl',
                            json.dumps(self.fit[0], ensure_ascii=False) + '\n')
        self.example = dict(description='Выполняю штукатурку стен.', mcId=101, labels=[102],
                            evidence=[dict(mcId=102, quote='штукатурку стен')], scenario='offered')
        self.dev = [dict(description='Другое public-объявление', proposed_labels=[]),
                    dict(description='  ВЫПОЛНЯЮ ШТУКАТУРКУ СТЕН.\n', proposed_labels=[102])]

    def test_generator_rejects_public_only_copy(self):
        output = self.root / 'generation'
        with patch('sys.argv', ['generate', '--archive', str(self.archive), '--count', '1',
                                '--output-dir', str(output)]), \
             patch.object(generator, 'load_training_data', return_value=(self.fit, self.dev, {})), \
             patch.object(generator, 'request_batch', return_value={'examples': [self.example]}):
            with self.assertRaisesRegex(ValueError, 'Duplicate normalized text'):
                generator.main()
        metadata = json.loads((output / 'metadata.json').read_bytes())
        self.assertEqual((metadata['status'], metadata['accepted']), ('failed', 0))

    def test_importer_checks_full_public_before_limit(self):
        synthetic = self.root / 'synthetic.jsonl'
        row = generator.validate_example(self.example, self.fit[0], 1, set(), 'fixture')
        synthetic.write_text(json.dumps(row, ensure_ascii=False) + '\n', encoding='utf-8')
        (self.root / 'metadata.json').write_text(json.dumps(dict(status='complete', accepted=1, requested=1)),
                                                encoding='utf-8')
        with patch('sys.argv', ['compare', '--archive', str(self.archive), '--synthetic', str(synthetic),
                                '--limit', '1', '--output-dir', str(self.root / 'comparison')]), \
             patch.object(comparison, 'load_training_data', return_value=(self.fit, self.dev, {})), \
             patch.object(comparison, 'local_request', return_value={'data': [{'id': comparison.DEFAULT_MODEL}]}), \
             patch.object(comparison, 'train', side_effect=RuntimeError('unsafe training reached')):
            with self.assertRaisesRegex(ValueError, 'Duplicate normalized text'):
                comparison.main()


if __name__ == '__main__':
    unittest.main()

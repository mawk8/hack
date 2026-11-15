"""Routing must not silently train on old splits after v2 is prepared."""
import argparse
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from avito import data
from scripts import train_baseline


class DataRoutingTests(unittest.TestCase):
    def test_default_routes_to_v2(self):
        newer = data.ROOT / 'data/v2'
        with patch.object(Path, 'exists', lambda p: p == newer or p == newer / 'manifest.json'), \
             patch.object(Path, 'read_bytes', return_value=b'{"version":"data-v2"}'), \
             patch('avito.dataset_v2.load_training_data', return_value=('fit', 'dev', {})) as load:
            self.assertEqual(data.load_training_data('fixture.zip'), ('fit', 'dev', {}))
            load.assert_called_once_with('fixture.zip', newer)

    def test_incomplete_or_unknown_v2_refuses_legacy_fallback(self):
        newer = data.ROOT / 'data/v2'
        with patch.object(Path, 'exists', lambda p: p == newer):
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                data.load_training_data('fixture.zip')
        with patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'read_bytes', return_value=b'{"version":"unknown"}'):
            with self.assertRaisesRegex(ValueError, 'Unsupported'):
                data.load_training_data('fixture.zip', newer)

    def test_cli_accepts_dynamic_counts_before_fitting(self):
        # Abort deliberately at fit: this tests dataset routing/count acceptance,
        # not sklearn on toy labels or any real control contents.
        output = data.ROOT / 'data/runs/routing-fixture'
        args = argparse.Namespace(archive=Path('fixture.zip'), data_dir=Path('data/v2'), output_dir=output,
                                  features='char')
        row = {'description': 'synthetic fixture', 'found_microcategories': []}
        with patch.object(argparse.ArgumentParser, 'parse_args', return_value=args), \
             patch.object(Path, 'exists', return_value=False), patch.object(Path, 'mkdir'), \
             patch.object(train_baseline, 'load_training_data', return_value=([row], [{}] * 3, {'manifest.json': 'fixture'})) as load, \
             patch.object(train_baseline, 'train', side_effect=RuntimeError('fit sentinel')):
            with self.assertRaisesRegex(RuntimeError, 'fit sentinel'):
                train_baseline.main()
            load.assert_called_once_with(args.archive, args.data_dir)


if __name__ == '__main__':
    unittest.main()

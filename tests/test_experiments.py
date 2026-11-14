"""Local fixtures only; no measured LLM quality claims."""
import importlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.g = importlib.import_module('scripts.generate_synthetic')
        self.c = importlib.import_module('scripts.compare_experiments')
        self.seed = dict(item_id=7, group_id='fit-group', description='Исходная штукатурка')
        self.example = dict(description='Выполняю штукатурку стен.', mcId=101,
                            labels=[102], evidence=[dict(mcId=102, quote='штукатурку стен')],
                            scenario='offered')

    def test_generated_validation_duplicates_and_lineage(self):
        row = self.g.validate_example(self.example, self.seed, 1, set(), 'local')
        self.assertEqual(row['status'], 'provisional')
        self.assertEqual(self.c.validate_synthetic([row], [self.seed], set()), [row])
        for change in ({'labels': [True]}, {'labels': [102, 102]}, {'mcId': True},
                       {'evidence': [{'mcId': 102, 'quote': 'нет такой цитаты'}]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.g.validate_example(dict(self.example, **change), self.seed, 1, set(), 'local')
        with self.assertRaises(ValueError):
            self.g.validate_example(self.example, self.seed, 1,
                                    {self.g.text_hash(self.example['description'])}, 'local')
        with self.assertRaises(ValueError):
            self.c.validate_synthetic([dict(row, seed_group_id='control')], [self.seed], set())
        with self.assertRaises(ValueError):
            self.c.validate_synthetic([row, row], [self.seed], set())

    def test_comparison_rejects_cached_model_and_trains_a_on_current_fit(self):
        with patch('sys.argv', ['compare', '--archive', 'fixture.zip', '--baseline-model', 'cached.joblib']), patch('sys.stderr', new=io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                self.c.main()
        self.assertEqual(error.exception.code, 2)
        root = Path(__file__).resolve().parents[1] / 'tmp'
        root.mkdir(exist_ok=True)
        output = Path(tempfile.mkdtemp(dir=root)) / 'comparison'
        fit = [dict(self.seed, item_id=i, found_microcategories=[dict(mcId=102)]) for i in range(9637)]
        with patch('sys.argv', ['compare', '--archive', 'fixture.zip']), \
             patch.object(self.c, 'output_directory', return_value=output), \
             patch.object(self.c, 'local_request', return_value={'data': [{'id': self.c.DEFAULT_MODEL}]}), \
             patch.object(self.c, 'load_training_data', return_value=(fit, [], {})), \
             patch.object(self.c, 'train', side_effect=RuntimeError('stop after verified train input')) as train:
            with self.assertRaisesRegex(RuntimeError, 'verified train input'):
                self.c.main()
        self.assertEqual(train.call_args.args, ([r['description'] for r in fit], [[102]] * 9637))

    def test_homogeneous_empty_batch_and_imported_scenario_order_rejected(self):
        examples = [dict(self.example, description=f'Не предлагаю штукатурку {i}',
                         labels=[], evidence=[], scenario='negated') for i in range(3)]
        completion = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({'examples': examples})}}]}
        with patch.object(self.g, 'local_request', return_value=completion), self.assertRaises(ValueError):
            self.g.request_batch(self.seed, 'http://localhost:8080/v1', 'fixture', 3)
        offered = self.g.validate_example(self.example, self.seed, 1, set(), 'fixture')
        wrong_second = self.g.validate_example(dict(self.example, description='Предлагаю штукатурку стен.'),
                                              self.seed, 2, set(), 'fixture')
        with self.assertRaises(ValueError):
            self.c.validate_synthetic([offered, wrong_second], [self.seed], set())

    def test_failed_full_rows_remain_null_and_count_fn(self):
        dev = [dict(item_id=1, description='штукатурка', source_mcId=101, proposed_labels=[102])]
        with patch.object(self.c, 'analyze', side_effect=RuntimeError('offline')):
            rows = self.c.evaluate(dev, None, 'http://localhost:8080/v1', 'fixture')
        self.assertIsNone(rows[0]['predicted'])
        self.assertIn('offline', rows[0]['error']['message'])
        score = self.c.summarize(dev, rows)
        self.assertEqual((score['tp'], score['fp'], score['fn']), (0, 0, 1))
        self.assertEqual(score['failure_rate'], 1)
        self.assertEqual(score['timing']['measurements'], 1)

    def test_consistent_micro_f1_and_fixed_columns(self):
        dev = [dict(proposed_labels=[102, 103]), dict(proposed_labels=[])]
        rows = [dict(predicted=[102], elapsed_seconds=1, attempts=1, usage={}),
                dict(predicted=[104], elapsed_seconds=2, attempts=1, usage={})]
        score = self.c.summarize(dev, rows)
        self.assertEqual((score['tp'], score['fp'], score['fn']), (1, 1, 1))
        self.assertEqual(score['micro_f1'], .5)
        self.assertEqual(list(score['per_category']), ['101', '102', '103', '104', '105', '106'])
        self.assertEqual(score['timing']['p95_seconds'], 2)
        self.assertEqual(score['empty_false_positive_rows'], 1)

    def test_forced_generation_and_classifier_time_in_full_duration(self):
        dev = [dict(item_id=1, description='штукатурка', source_mcId=101, proposed_labels=[102])]
        def classifier(*args):
            time.sleep(.01)
            return [102]
        result = dict(result=dict(drafts=[dict(mcId=102)]), meta=dict(attempts=1, usage={}))
        with patch.object(self.c, 'predict', side_effect=classifier), patch.object(self.c, 'analyze', return_value=result) as llm:
            row = self.c.evaluate(dev, object(), 'http://localhost:8080/v1', 'fixture')[0]
        self.assertGreaterEqual(row['elapsed_seconds'], row['classifier_seconds'])
        self.assertGreaterEqual(row['classifier_seconds'], .009)
        self.assertEqual(llm.call_args.kwargs['forced_categories'], [102])
        with patch.object(self.c, 'predict', return_value=[]), patch.object(self.c, 'analyze') as llm:
            self.assertEqual(self.c.evaluate(dev, object(), 'http://localhost:8080/v1', 'fixture')[0]['attempts'], 0)
            llm.assert_not_called()

    def test_human_overlay_identity_and_evidence(self):
        dev = [dict(item_id=1, group_id='g', source_row=4, source_mcId=101, description='штукатурка', proposed_labels=[])]
        overlay = dict(dev[0], status='human_verified', reviewer='Tester', reviewed_at='2026-11-14T10:30:00.000Z',
                       proposed_labels=[102], evidence=[dict(mcId=102, quote='штукатурка')])
        self.assertEqual(self.c.validate_overlay(dev, [overlay])[0]['proposed_labels'], [102])
        for change in ({'description': 'другой текст'}, {'source_row': 5}, {'evidence': []},
                       {'reviewed_at': '2026-11-14'}, {'reviewed_at': 'invalid'}):
            with self.assertRaises(ValueError):
                self.c.validate_overlay(dev, [dict(overlay, **change)])

    def test_local_transport_and_generator_error_record(self):
        from tests.test_llm import fake_model
        with fake_model(content=json.dumps({'examples': [self.example]})) as (endpoint, calls):
            result = self.g.request_batch(self.seed, endpoint, 'fixture', 1)
        self.assertEqual(result['examples'], [self.example])
        self.assertEqual(len(calls), 1)
        self.assertIn(self.seed['description'], calls[0]['body']['messages'][1]['content'])
        root = Path(__file__).resolve().parents[1] / 'tmp'
        root.mkdir(exist_ok=True)
        folder = Path(tempfile.mkdtemp(dir=root))
        with patch.object(self.g, 'request_batch', side_effect=RuntimeError('offline')):
            with self.assertRaises(RuntimeError):
                self.g.generate([self.seed], set(), 3, folder, 'http://localhost:8080/v1', 'fixture')
        metadata = json.loads((folder / 'metadata.json').read_text())
        self.assertEqual(metadata['status'], 'failed')
        self.assertEqual(metadata['accepted'], 0)


if __name__ == '__main__':
    unittest.main()

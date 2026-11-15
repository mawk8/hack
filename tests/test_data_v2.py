"""Synthetic data only; fixtures are retained under ignored data/, never deleted."""
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import uuid
import zipfile

from scripts import prepare_data_v2 as builder
from avito import dataset_v2


class DataV2Tests(unittest.TestCase):
    def fixture(self, extra=14, label=102):
        import pyarrow as pa
        import pyarrow.parquet as pq
        root = Path('data/test-v2') / uuid.uuid4().hex
        root.mkdir(parents=True)
        legacy = root / 'v1'
        legacy.mkdir()
        # A public suffix joins train[0], whose old group also contains train[1].
        common = ' '.join('token%03d' % i for i in range(80))
        rows = [dict(item_id=i, description=d, found_microcategories=[{'mcId': label, 'reason': 'fixture'}])
                for i, d in enumerate([common + ' old', 'old-group-distinct', 'legacy-held-out'] +
                                      ['independent fixture %d' % j for j in range(extra)] + ['independent fixture 0'])]
        public = [dict(title='fixture', description=common + ' new', categories=['Сантехника']),
                  dict(title='fixture duplicate', description=(common + ' NEW'), categories=['Электрика'])]
        payload = io.BytesIO()
        pq.write_table(pa.Table.from_pylist(public), payload)
        archive = root / 'fixture.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr(builder.TRAIN_MEMBER, b''.join(builder.encoded(r) for r in rows))
            z.writestr(builder.PUBLIC_MEMBER, payload.getvalue())
        manifest = [dict(source_row=i, item_id=r['item_id'], group_id='old-shared' if i<2 else 'old-%d'%i,
                         excluded_from_training=i==2) for i, r in enumerate(rows)]
        exclusion = [dict(source_row=2, item_id=2, group_id='old-2', purpose='development')]
        initial = {'group_manifest.jsonl': b''.join(builder.encoded(r) for r in manifest),
                   'development_review.jsonl': builder.encoded(dict(source_row=2, group_id='old-2')),
                   'control_review.jsonl': b''}
        initial['freeze_manifest.json'] = builder.encoded(dict(version='review-v1',
            archive_sha256=builder.sha256(archive), sha256={n:builder.digest(v) for n,v in initial.items()}))
        overlay = {'training_exclusions_manifest.jsonl': b''.join(builder.encoded(r) for r in exclusion),
                   'training_filter_summary.json': builder.encoded(dict(train_rows=len(rows)))}
        overlay['training_exclusions_freeze_manifest.json'] = builder.encoded(dict(version='training-filter-v1',
            legacy_frozen_sha256={n:builder.digest(v) for n,v in initial.items()},
            sha256={n:builder.digest(v) for n,v in overlay.items()}))
        for name, content in (initial | overlay).items():
            (legacy/name).write_bytes(content)
        return root, archive, legacy, rows

    def build(self, fixture, control=4, human=2):
        root, archive, legacy, _ = fixture
        outputs = builder.build(archive, legacy, control_count=control, human_count=human)
        builder.write_outputs(root/'v2', outputs)
        return outputs

    def test_public_near_bridge_legacy_and_entire_control_groups_excluded(self):
        fixture = self.fixture()
        outputs = self.build(fixture, control=14)
        parsed = lambda name: [json.loads(v) for v in outputs[name].splitlines()]
        groups = parsed('group_manifest.jsonl')
        self.assertEqual(len({r['group_id'] for r in groups if r['dataset']=='public' or r['source_row']<2}), 1)
        exclusions = parsed('training_exclusions_manifest.jsonl')
        self.assertTrue({0,1,2} <= {r['source_row'] for r in exclusions})
        self.assertIn('public', next(r['reasons'] for r in exclusions if r['source_row']==1))
        self.assertIn('legacy', next(r['reasons'] for r in exclusions if r['source_row']==2))
        duplicated = {r['source_row'] for r in groups if r['dataset']=='train' and
                      r['group_id']==next(e['group_id'] for e in groups if e['dataset']=='train' and e['source_row']==3)}
        self.assertEqual(len(duplicated), 2)
        self.assertTrue(duplicated <= {r['source_row'] for r in exclusions if 'control' in r['reasons']})
        fit, dev, _ = dataset_v2.load_training_data(fixture[1], fixture[0]/'v2')
        self.assertFalse({r['group_id'] for r in fit} & {r['group_id'] for r in exclusions})
        self.assertEqual(len(dev), 2)
        self.assertEqual(dev[0]['proposed_labels'], [104])
        review = parsed('development_review.jsonl')
        self.assertEqual(len(review), 1)
        self.assertEqual(review[0]['original_label_variants'], [[104],[105]])
        self.assertIsNone(review[0]['proposed_labels'])

    def test_seed_reproduction_and_blind40_without_hints(self):
        fixture = self.fixture(extra=70)
        outputs = self.build(fixture, control=50, human=40)
        self.assertEqual(outputs, builder.build(fixture[1], fixture[2], control_count=50, human_count=40))
        different_labels = self.fixture(extra=70, label=106)
        relabeled = builder.build(different_labels[1], different_labels[2], control_count=50, human_count=40)
        self.assertEqual(outputs['selection_manifest.jsonl'], relabeled['selection_manifest.jsonl'])
        blind = [json.loads(v) for v in outputs['human40_blind_review.jsonl'].splitlines()]
        self.assertEqual(len(blind), 40)
        self.assertEqual(len({r['group_id'] for r in blind}), 40)
        for row in blind:
            self.assertEqual(set(row), {'item_id','group_id','source_row','description','source_mcId','purpose',
                                       'proposed_labels','evidence','questions','status','reviewer','reviewed_at'})
            self.assertIsNone(row['proposed_labels'])
            self.assertEqual(row['questions'], [])
            self.assertEqual(row['evidence'], [])
        for name in ('development_teacher_input.jsonl','control_teacher_input.jsonl'):
            self.assertTrue(all('labels' not in key and key!='categories' and key!='found_microcategories'
                                for row in map(json.loads, outputs[name].splitlines()) for key in row))

    def test_no_overwrite_partial_output_or_corruption(self):
        fixture = self.fixture()
        outputs = self.build(fixture)
        directory = fixture[0]/'v2'
        original = (directory/'train.jsonl').read_bytes()
        builder.write_outputs(directory, outputs)  # byte-identical resume is read-only
        with self.assertRaises(ValueError):
            builder.write_outputs(directory, outputs | {'train.jsonl':b'different'})
        self.assertEqual((directory/'train.jsonl').read_bytes(), original)
        partial = fixture[0]/'partial'
        partial.mkdir()
        (partial/'train.jsonl').write_bytes(original)
        with self.assertRaises(ValueError):
            builder.write_outputs(partial, outputs)
        (directory/'control_reference.jsonl').write_bytes(b'corruption')
        with self.assertRaises(ValueError):
            dataset_v2.load_training_data(fixture[1], directory)

    def test_loader_never_parses_control_files(self):
        fixture = self.fixture()
        self.build(fixture)
        read = dataset_v2.read_jsonl
        def guarded(path):
            self.assertNotIn(Path(path).name, ('control_reference.jsonl','control_teacher_input.jsonl','human40_blind_review.jsonl'))
            return read(path)
        with patch.object(dataset_v2, 'read_jsonl', guarded):
            dataset_v2.load_training_data(fixture[1], fixture[0]/'v2')

    def test_loaded_public_aliases_support_overlay_and_evaluation(self):
        from scripts.compare_experiments import evaluate, validate_overlay
        fixture = self.fixture()
        self.build(fixture)
        directory = fixture[0]/'v2'
        before = {path.name:builder.sha256(path) for path in directory.iterdir()}
        _, development, _ = dataset_v2.load_training_data(fixture[1], directory)
        self.assertEqual([r['item_id'] for r in development], ['public-0','public-1'])
        self.assertEqual([r['source_row'] for r in development], [0,1])
        self.assertTrue(all(r['source_row']==r['source_public_row'] for r in development))
        self.assertEqual(len({r['item_id'] for r in development}), len(development))
        overlay = [dict(row,status='human_verified',reviewer='synthetic fixture',reviewed_at='2026-11-15T10:00:00Z',
                        evidence=[dict(mcId=label,quote=row['description'][:8]) for label in row['proposed_labels']])
                   for row in development]
        reviewed = validate_overlay(development, overlay)
        class FixedClassifier:
            def predict_proba(self, descriptions):
                return [[1,0,0,0,0] for _ in descriptions]
        results = evaluate(reviewed, FixedClassifier(), None, None, classification_only=True)
        self.assertEqual([r['item_id'] for r in results], ['public-0','public-1'])
        self.assertEqual([r['source_row'] for r in results], [0,1])
        self.assertTrue(all(r['status']=='ok' for r in results))
        self.assertEqual(before, {path.name:builder.sha256(path) for path in directory.iterdir()})
        frozen_dev = dataset_v2.read_jsonl(directory/'development_original.jsonl')
        self.assertTrue(all('item_id' not in r and 'source_row' not in r for r in frozen_dev))

    def test_loader_rejects_incomplete_exclusions_even_with_updated_file_hash(self):
        fixture = self.fixture()
        outputs = self.build(fixture)
        exclusions = outputs['training_exclusions_manifest.jsonl'].splitlines(keepends=True)
        # Removing the old-linked train[1] must fail group coverage independently of hashes.
        outputs['training_exclusions_manifest.jsonl'] = b''.join(v for v in exclusions if json.loads(v)['source_row']!=1)
        manifest = json.loads(outputs['manifest.json'])
        manifest['sha256']['training_exclusions_manifest.jsonl'] = builder.digest(outputs['training_exclusions_manifest.jsonl'])
        outputs['manifest.json'] = builder.encoded(manifest)
        directory = fixture[0]/'incomplete-exclusions'
        builder.write_outputs(directory, outputs)
        with self.assertRaisesRegex(ValueError, 'complete public/legacy/control groups'):
            dataset_v2.load_training_data(fixture[1], directory)

    def test_unknown_public_category_and_changed_v1_rejected(self):
        with self.assertRaises(ValueError):
            builder.public_labels(['unknown'])
        fixture = self.fixture()
        (fixture[2]/'control_review.jsonl').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            builder.build(fixture[1], fixture[2], control_count=4, human_count=2)


if __name__ == '__main__':
    unittest.main()

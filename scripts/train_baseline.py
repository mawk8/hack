"""Train the fixed baseline and evaluate development only (no generation)."""
import argparse
from collections import Counter
import json
import platform
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from avito.baseline import SEED, THRESHOLD, predict, save_model, train
from avito.data import CATEGORIES, load_training_data, sha256
import joblib
import numpy
import scipy
import sklearn


def metrics(expected, predicted):
    counts = Counter()
    for truth, result in zip(expected, predicted):
        truth, result = set(truth), set(result)
        counts.update(tp=len(truth & result), fp=len(result - truth), fn=len(truth - result))
    tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
    return dict(tp=tp, fp=fp, fn=fn, micro_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, help='Default: prepared data/v2, otherwise legacy stage1')
    parser.add_argument('--features', choices=('char', 'char-word'), default='char',
                        help='E1: add word unigrams/bigrams; all classifier settings stay fixed')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data' / 'runs' / 'baseline')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to((ROOT / 'data').resolve()):
        parser.error('--output-dir must be inside ignored data/ (contains model and advertisement text)')
    if output.exists():
        parser.error('output directory already exists; choose a new directory (no overwrite)')
    fit, development, hashes = load_training_data(args.archive, args.data_dir)
    if not fit or not development:
        raise ValueError('Training and development must both be nonempty')
    version = 'v2' if 'manifest.json' in hashes else 'v1'
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    model = train([r['description'] for r in fit],
                  [[v['mcId'] for v in r['found_microcategories']] for r in fit], features=args.features)
    fit_seconds = time.perf_counter() - started
    predicted, durations = [], []
    for row in development:
        started = time.perf_counter()
        predicted.append(predict(model, row['description'], row['source_mcId']))
        durations.append(time.perf_counter() - started)
    expected = [r['proposed_labels'] for r in development]
    empty_indices = [i for i, labels in enumerate(expected) if not labels]
    empty_fp_rows = sum(bool(predicted[i]) for i in empty_indices)
    report = dict(
        mode='classification-only; no generation; not full online latency',
        label_status=('noisy-label; original public annotations; human_verified=0' if version == 'v2'
                      else 'noisy-label; provisional automatic labels; human_verified=0'),
        dataset_version=version,
        train_rows=len(fit), development_rows=len(development), source_mcId=101,
        features=args.features,
        category_order=list(CATEGORIES), threshold=THRESHOLD, seed=SEED,
        tfidf=dict(analyzer='char_wb', ngram_range=[3, 5], min_df=2, max_features=80000, sublinear_tf=True),
        word_tfidf=(dict(analyzer='word', ngram_range=[1, 2], min_df=2, max_features=80000,
                        sublinear_tf=True, combination='FeatureUnion; each branch L2; no extra normalization')
                    if args.features == 'char-word' else None),
        classifier=dict(type='OneVsRestClassifier(LogisticRegression)', C=1.0, solver='liblinear', random_state=SEED, max_iter=1000),
        **metrics(expected, predicted),
        per_category={str(c):metrics([[c] if c in values else [] for values in expected],
                                      [[c] if c in values else [] for values in predicted]) for c in CATEGORIES},
        empty_examples=dict(rows=len(empty_indices), false_positive_rows=empty_fp_rows,
                            false_positive_categories=sum(len(predicted[i]) for i in empty_indices),
                            false_positive_row_rate=empty_fp_rows/len(empty_indices) if empty_indices else None),
        errors=dict(exact_set_mismatch_rows=sum(set(a)!=set(b) for a,b in zip(expected,predicted)),
                    failed_requests=0, failure_rate=0.0),
        timing=dict(fit_seconds=fit_seconds, predict_total_seconds=sum(durations), measurements=len(durations),
                    p50_seconds=float(numpy.median(durations)), p95_seconds=sorted(durations)[int(numpy.ceil(.95*len(durations)))-1],
                    max_seconds=max(durations), p95_method='nearest-rank', passes=1,
                    limitation='one classification-only pass; no generation, retries, or sustained latency claim'),
        versions=dict(python=platform.python_version(), sklearn=sklearn.__version__, numpy=numpy.__version__,
                      scipy=scipy.__version__, joblib=joblib.__version__),
        machine=dict(platform=platform.platform(), processor=platform.processor()),
        hashes=hashes,
        code_sha256={str(path.relative_to(ROOT)):sha256(path) for path in
                     [ROOT/'avito/__init__.py', ROOT/'avito/data.py', ROOT/'avito/dataset_v2.py',
                      ROOT/'avito/baseline.py', Path(__file__)]},
        paid_api_calls=0, paid_api_cost_rub=0,
        limitations=['source101 only', 'No human-verified labels', 'No drafts or factuality evaluation', 'H3 not tested'],
    )
    save_model(model, output / 'model.joblib')
    with (output / 'development_predictions.jsonl').open('x', encoding='utf-8') as stream:
        for row, labels, duration in zip(development, predicted, durations):
            stream.write(json.dumps(dict(item_id=row['item_id'], source_row=row['source_row'], group_id=row['group_id'],
                description=row['description'], source_mcId=row['source_mcId'], expected_mcIds=row['proposed_labels'],
                predicted_mcIds=labels, classification_seconds=duration, status='ok'), ensure_ascii=False) + '\n')
    with (output / 'metrics.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()

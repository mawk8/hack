"""Compare A/B classifiers and full A/B/C pipelines on frozen development."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from avito.baseline import train, predict, save_model
from avito.data import load_training_data, validate_labels, sha256
from avito.llm import analyze, DEFAULT_ENDPOINT, DEFAULT_MODEL, validate_endpoint
from scripts.generate_synthetic import (VERSION, PROMPT_HASH, SCENARIOS, archive_text_hashes, local_request,
                                        output_directory, text_hash, validate_example)
from scripts.train_baseline import metrics


def read_rows(path):
    from avito.llm import parse_json
    return [parse_json(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def validate_synthetic(rows, fit, forbidden):
    if not rows:
        raise ValueError('Empty synthetic pilot cannot train B')
    parents = {(r['item_id'], r['group_id']): r for r in fit}
    seen, identifiers = set(forbidden), set()
    fields = ('description', 'mcId', 'labels', 'evidence', 'scenario')
    for index, row in enumerate(rows):
        if row.get('scenario') != SCENARIOS[index % 3]:
            raise ValueError('Synthetic scenario composition/order mismatch')
        parent = parents.get((row.get('seed_item_id'), row.get('seed_group_id')))
        if parent is None:
            raise ValueError('Synthetic parent is not a permitted fit row/group')
        if (row.get('status') != 'provisional' or row.get('method') != 'local_llm'
                or row.get('prompt_sha256') != PROMPT_HASH or row.get('prompt_version') != VERSION
                or type(row.get('model')) is not str or not row['model'].strip()
                or row.get('random_seed') != 42 or not isinstance(row.get('item_id'), str)
                or not row['item_id'].startswith(VERSION + '-') or row['item_id'] in identifiers):
            raise ValueError('Invalid synthetic provenance/status/identifier')
        value = validate_example({key: row[key] for key in fields}, parent, 1, seen, row['model'])
        if row.get('normalized_sha256') != value['normalized_sha256']:
            raise ValueError('Synthetic text hash mismatch')
        identifiers.add(row['item_id'])
        seen.add(value['normalized_sha256'])
    return rows


def validate_overlay(development, overlay):
    if len(overlay) != len(development):
        raise ValueError('Human development must contain the exact frozen set')
    keys = ('item_id', 'group_id', 'source_row', 'source_mcId', 'description')
    indexed = {}
    for row in overlay:
        key = row.get('item_id')
        if key in indexed:
            raise ValueError('Duplicate human development identity')
        indexed[key] = row
    result = []
    for frozen in development:
        row = indexed.get(frozen['item_id'])
        if row is None or any(type(row.get(k)) is not type(frozen[k]) or row[k] != frozen[k] for k in keys):
            raise ValueError('Human development identity/source/text mismatch')
        if (row.get('status') != 'human_verified' or type(row.get('reviewer')) is not str
                or not row['reviewer'].strip() or type(row.get('reviewed_at')) is not str):
            raise ValueError('Human development needs status, reviewer and ISO reviewed_at')
        reviewed_at = row['reviewed_at']
        if 'T' not in reviewed_at:
            raise ValueError('reviewed_at must be an ISO datetime')
        datetime.fromisoformat(reviewed_at.replace('Z', '+00:00'))
        labels = validate_labels(row['proposed_labels'])
        evidence = row.get('evidence')
        if type(evidence) is not list:
            raise ValueError('Human evidence must be a list')
        seen = set()
        for entry in evidence:
            if (type(entry) is not dict or type(entry.get('mcId')) is not int
                    or entry['mcId'] not in labels or type(entry.get('quote')) is not str
                    or not entry['quote'].strip() or entry['quote'] not in row['description']):
                raise ValueError('Invalid human evidence')
            seen.add(entry['mcId'])
        if seen != set(labels):
            raise ValueError('Missing human evidence for labels')
        result.append(dict(frozen, proposed_labels=labels, status='human_verified'))
    return result


def evaluate(development, classifier, endpoint, model, classification_only=False):
    rows = []
    for source in development:
        started = time.perf_counter()
        row = dict(item_id=source.get('item_id'), source_row=source.get('source_row'),
                   group_id=source.get('group_id'), source_mcId=source.get('source_mcId', 101),
                   description=source.get('description'), expected=source['proposed_labels'],
                   predicted=None, classifier_seconds=0, attempts=0, usage={}, status='failed')
        try:
            categories = None
            if classifier is not None:
                tick = time.perf_counter()
                try:
                    categories = predict(classifier, source['description'], source['source_mcId'])
                finally:
                    row['classifier_seconds'] = time.perf_counter() - tick
            if classification_only or categories == []:
                row['predicted'] = categories
            else:
                row['attempts'] = 1
                kwargs = {} if categories is None else dict(forced_categories=categories)
                response = analyze(source['description'], source['source_mcId'], endpoint=endpoint, model=model, **kwargs)
                row['predicted'] = [r['mcId'] for r in response['result']['drafts']]
                row['response'] = response
                row['attempts'] = response['meta']['attempts']
                row['usage'] = response['meta']['usage']
            row['status'] = 'ok'
        except Exception as error:
            row['predicted'] = None
            row['error'] = dict(type=type(error).__name__, message=str(error))
        row['elapsed_seconds'] = time.perf_counter() - started
        rows.append(row)
    return rows


def summarize(development, rows):
    if len(rows) != len(development) or not rows:
        raise ValueError('Expected nonempty complete development predictions')
    expected = [r['proposed_labels'] for r in development]
    predicted = [r['predicted'] if r['predicted'] is not None else [] for r in rows]
    durations = sorted(r['elapsed_seconds'] for r in rows)
    failed = sum(r['predicted'] is None for r in rows)
    empty = [i for i, labels in enumerate(expected) if not labels]
    return dict(**metrics(expected, predicted), rows=len(rows), failed_requests=failed,
        failure_rate=failed / len(rows),
        category_order=list(range(101, 107)), masked_categories=[101],
        per_category={str(c): metrics([[c] if c in a else [] for a in expected],
                                    [[c] if c in a else [] for a in predicted]) for c in range(101, 107)},
        empty_rows=len(empty), empty_false_positive_rows=sum(bool(predicted[i]) for i in empty),
        empty_false_positive_categories=sum(len(predicted[i]) for i in empty),
        exact_set_mismatches=sum(r['predicted'] is None or set(a) != set(b)
                                 for a, b, r in zip(expected, predicted, rows)),
        attempts=sum(r['attempts'] for r in rows),
        usage={k: sum(r['usage'].get(k, 0) for r in rows)
               for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')},
        usage_available_rows=sum(bool(r['usage']) for r in rows),
        timing=dict(total_seconds=sum(durations), measurements=len(rows),
                    classifier_seconds=sum(r.get('classifier_seconds', 0) for r in rows),
                    p50_seconds=durations[math.ceil(.5 * len(rows)) - 1],
                    p95_seconds=durations[math.ceil(.95 * len(rows)) - 1],
                    p95_method='nearest-rank; includes failed rows', passes=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/runs/comparison')
    parser.add_argument('--synthetic', type=Path)
    parser.add_argument('--human-development', type=Path)
    parser.add_argument('--endpoint', default=DEFAULT_ENDPOINT)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--limit', type=int, help='Explicitly preliminary pilot, 1..50')
    args = parser.parse_args()
    output = output_directory(args.output_dir)
    endpoint = validate_endpoint(args.endpoint)
    if args.limit is not None and not 1 <= args.limit <= 50:
        parser.error('--limit must be 1..50')
    # Fail before fitting or producing a misleading partial comparison.
    health = local_request(endpoint, '/models')
    if (type(health) is not dict or type(health.get('data')) is not list
            or not any(type(m) is dict and m.get('id') == args.model for m in health['data'])):
        raise ValueError('Local model alias unavailable in /v1/models')
    fit, development, hashes = load_training_data(args.archive)
    if args.human_development:
        development = validate_overlay(development, read_rows(args.human_development))
        hashes['human_development_sha256'] = sha256(args.human_development)
    synthetic = []
    if args.synthetic:
        # A failed partial pilot must not silently become a training set.
        metadata = json.loads(args.synthetic.with_name('metadata.json').read_text(encoding='utf-8'))
        if metadata.get('status') != 'complete' or metadata.get('accepted', 0) <= 0:
            raise ValueError('Synthetic pilot failed or empty; B training refused')
        synthetic = validate_synthetic(read_rows(args.synthetic), fit, archive_text_hashes(args.archive))
        if metadata['accepted'] != len(synthetic) or metadata.get('requested') != len(synthetic):
            raise ValueError('Synthetic pilot coverage mismatch')
        hashes['synthetic_sha256'] = sha256(args.synthetic)
        hashes['synthetic_metadata_sha256'] = sha256(args.synthetic.with_name('metadata.json'))
    full_count = len(development)
    development = development[:args.limit] if args.limit else development
    output.mkdir(parents=True, exist_ok=False)
    training_times = {}
    descriptions = [r['description'] for r in fit]
    labels = [[v['mcId'] for v in r['found_microcategories']] for r in fit]
    started = time.perf_counter()
    a = train(descriptions, labels)
    save_model(a, output / 'A-model.joblib')
    training_times['A_fit_seconds'] = time.perf_counter() - started
    models = {'A': a}
    if synthetic:
        started = time.perf_counter()
        models['B'] = train(descriptions + [r['description'] for r in synthetic], labels + [r['labels'] for r in synthetic])
        training_times['B_fit_seconds'] = time.perf_counter() - started
        save_model(models['B'], output / 'B-model.joblib')
    scores = {}
    def store(name, model, classification_only=False):
        rows = evaluate(development, model, endpoint, args.model, classification_only)
        with (output / (name + '-predictions.jsonl')).open('x', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        scores[name] = summarize(development, rows)
    for name, model in models.items():
        store(name + '_classification', model, True)
        store(name + '_full', model)
    store('C_full', None)
    import sklearn, numpy, scipy, joblib
    report = dict(version='comparison-v1', hashes=hashes, scores=scores, training_times=training_times,
        label_status='human_verified' if args.human_development else 'noisy-label diagnostics; human_verified=0',
        coverage=dict(evaluated=len(development), frozen_development=full_count, fit=len(fit), synthetic=len(synthetic)),
        preliminary=args.limit is not None, synthetic_label_status='provisional' if synthetic else 'absent',
        config=dict(endpoint=endpoint, model=args.model, random_seed=42, threshold=.5,
                    baseline_training='fresh current verified fit for A and B; no cached models',
                    classifier='scripts/train_baseline.py fixed TF-IDF + OvR LogisticRegression',
                    repeats=1, source_mcId=101),
        versions=dict(python=platform.python_version(), sklearn=sklearn.__version__, numpy=numpy.__version__, scipy=scipy.__version__, joblib=joblib.__version__),
        code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__), ROOT/'scripts/generate_synthetic.py', ROOT/'avito/data.py', ROOT/'avito/baseline.py', ROOT/'avito/llm.py']},
        paid_api_cost_rub=0, hardware_cost='unmeasured', factuality='not automatically proven; human review required',
        limitations=['single pass, no sustained latency claim', 'source101 only', 'no final control evaluation',
                     'failed rows retained as null; zero labels only for FN accounting'])
    with (output / 'metrics.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=True))


if __name__ == '__main__':
    main()

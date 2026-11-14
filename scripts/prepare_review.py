"""Prepare provisional, group-disjoint reviews without reading public-test.

Existing outputs are never overwritten. --verify checks the saved freeze.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import zipfile

from audit_data import normalize

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'review-v1'


def digest(value):
    return hashlib.sha256(value).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + '\n').encode('utf-8')


def labels(row):
    return sorted({v['mcId'] for v in row['found_microcategories']})


def build(archive):
    with zipfile.ZipFile(archive) as z:
        rows = [json.loads(s) for s in z.read('innoglobalhack_avito/public-train-annotated.jsonl').splitlines() if s.strip()]
    normalized = defaultdict(list)
    for i, row in enumerate(rows):
        normalized[normalize(row['description'])].append(i)
    texts = sorted(normalized)
    parent = list(range(len(texts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # ponytail: narrow prefix/suffix blocking misses heavily rewritten templates;
    # expand candidate retrieval only when observed leakage warrants it.
    blocks = defaultdict(list)
    for i, text in enumerate(texts):
        if len(text) >= 160:
            blocks[('prefix', text[:80])].append(i)
            blocks[('suffix', text[-80:])].append(i)
    candidates = set()
    for ids in blocks.values():
        for position, a in enumerate(ids):
            for b in ids[position+1:]:
                candidates.add((a, b))
    tokens = [set(t.split()) for t in texts]
    links = []
    for a, b in sorted(candidates):
        ta, tb = texts[a], texts[b]
        length_ratio = min(len(ta), len(tb)) / max(len(ta), len(tb))
        jaccard = len(tokens[a] & tokens[b]) / len(tokens[a] | tokens[b])
        if length_ratio >= .9 and jaccard >= .95:
            ra, rb = find(a), find(b)
            parent[max(ra, rb)] = min(ra, rb)
            links.append(dict(left_normalized_sha256=digest(ta.encode()), right_normalized_sha256=digest(tb.encode()),
                              rule='shared first/last 80 chars; length ratio >= .90; whitespace-token set Jaccard >= .95',
                              length_ratio=length_ratio, token_jaccard=jaccard))
    groups = defaultdict(list)
    for i, text in enumerate(texts):
        groups[find(i)].extend(normalized[text])
    group_rows = {}
    for indices in groups.values():
        group_id = 'g-' + digest(encoded(sorted(digest(normalize(rows[i]['description']).encode()) for i in indices)))[:24]
        group_rows[group_id] = sorted(indices)
    # One representative per leakage group; deterministic stratified round-robin.
    buckets = defaultdict(list)
    for gid, indices in group_rows.items():
        i = min(indices, key=lambda j: (digest(str(rows[j]['item_id']).encode()), j))
        row = rows[i]
        length = len(row['description'])
        band = 'short' if length < 500 else 'medium' if length < 1500 else 'long'
        buckets[(tuple(labels(row)), band)].append((gid, i))
    for bucket in buckets.values():
        bucket.sort(key=lambda pair: digest((VERSION + pair[0]).encode()))
    selected = []
    while len(selected) < 100:
        for key in sorted(buckets):
            if buckets[key] and len(selected) < 100:
                selected.append(buckets[key].pop(0))
        if not any(buckets.values()) and len(selected) < 100:
            raise ValueError('Fewer than 100 independent groups')
    reviews = {'development': [], 'control': []}
    for position, (gid, i) in enumerate(selected):
        purpose = 'development' if position % 2 == 0 else 'control'
        row = rows[i]
        reviews[purpose].append(dict(item_id=row['item_id'], source_row=i, group_id=gid, purpose=purpose,
            description=row['description'], source_mcId=101, source_mcId_provenance='task contract; absent in train row',
            original_labels=row['found_microcategories'], proposed_labels=labels(row), evidence=[],
            questions=['Automatic labels copied, not semantically checked; confirm services, negations and package constraints.'],
            status='provisional', provenance='automatic train labels copied without semantic or human verification',
            reviewer=None, reviewed_at=None))
    control_groups = {r['group_id'] for r in reviews['control']}
    manifest = [dict(item_id=rows[i]['item_id'], source_row=i, group_id=gid,
                     excluded_from_training=gid in control_groups)
                for gid, indices in sorted(group_rows.items()) for i in indices]
    excluded = [dict(source_row=i, group_id=gid, original_row=rows[i])
                for gid in sorted(control_groups) for i in group_rows[gid]]
    outputs = {f'{purpose}_review.jsonl': b''.join(encoded(r) for r in values) for purpose, values in reviews.items()}
    outputs['group_manifest.jsonl'] = b''.join(encoded(r) for r in manifest)
    outputs['control_excluded_manifest.jsonl'] = b''.join(encoded(r) for r in excluded)
    outputs['near_links_manifest.jsonl'] = b''.join(encoded(r) for r in links)
    summary = dict(version=VERSION, train_rows=len(rows), exact_descriptions=len({r['description'] for r in rows}),
        normalized_descriptions=len(texts), leakage_groups=len(groups), near_candidate_pairs=len(candidates),
        accepted_near_links=len(links), excluded_train_rows=len(excluded), additional_excluded_rows=len(excluded)-50,
        status='provisional; human review pending', selection='deterministic category-set x length-band round-robin; purposive, not representative',
        splits={p: dict(rows=len(values), categories=dict(sorted(Counter(str(k) for r in values for k in r['proposed_labels']).items())),
            empty_labels=sum(not r['proposed_labels'] for r in values),
            label_counts=dict(sorted(Counter(str(len(r['proposed_labels'])) for r in values).items())),
            length_bands=dict(Counter('short' if len(r['description'])<500 else 'medium' if len(r['description'])<1500 else 'long' for r in values)))
            for p, values in reviews.items()})
    outputs['sampling_summary.json'] = encoded(summary)
    freeze = dict(version=VERSION, archive_sha256=digest(archive.read_bytes()),
                  sha256={name:digest(data) for name, data in outputs.items()})
    outputs['freeze_manifest.json'] = encoded(freeze)
    return outputs


def verify(directory, expected):
    freeze = json.loads((directory/'freeze_manifest.json').read_text(encoding='utf-8'))
    for name, checksum in freeze['sha256'].items():
        if digest((directory/name).read_bytes()) != checksum:
            raise ValueError(f'Frozen file changed: {name}; no overwrite allowed')
    for name, content in expected.items():
        if (directory/name).read_bytes() != content:
            raise ValueError(f'Reproduction mismatch: {name}; no overwrite allowed')
    load = lambda name: [json.loads(s) for s in (directory/name).read_text(encoding='utf-8').splitlines()]
    dev, control = load('development_review.jsonl'), load('control_review.jsonl')
    assert len(dev) == len(control) == 50
    assert not ({r['group_id'] for r in dev} & {r['group_id'] for r in control})
    assert len({normalize(r['description']) for r in dev+control}) == 100
    assert all(r['status']=='provisional' and not r['evidence'] for r in dev+control)
    groups = {r['group_id'] for r in control}
    manifest = load('group_manifest.jsonl')
    excluded = load('control_excluded_manifest.jsonl')
    assert {r['source_row'] for r in excluded} == {r['source_row'] for r in manifest if r['group_id'] in groups}
    assert all(r['excluded_from_training'] == (r['group_id'] in groups) for r in manifest)
    assert all(r['original_row']['item_id'] == next(m['item_id'] for m in manifest if m['source_row']==r['source_row']) for r in excluded)
    print(json.dumps(dict(verified=True, development=50, control=50, excluded_rows=len(excluded), frozen_files=len(freeze['sha256'])), sort_keys=True))


def training_exclusions(directory, expected, verify_only=False):
    """Immutable overlay: exclude every development/control group from fitting."""
    load = lambda name: [json.loads(s) for s in expected[name].decode('utf-8').splitlines()]
    purposes = {r['group_id']: purpose for purpose in ('development', 'control')
                for r in load(f'{purpose}_review.jsonl')}
    group_manifest = load('group_manifest.jsonl')
    exclusions = [dict(source_row=r['source_row'], item_id=r['item_id'],
                       group_id=r['group_id'], purpose=purposes[r['group_id']])
                  for r in group_manifest if r['group_id'] in purposes]
    exclusions.sort(key=lambda r: r['source_row'])
    excluded_rows = {r['source_row'] for r in exclusions}
    assert len(excluded_rows) == len(exclusions)
    baseline = [r for r in group_manifest if r['source_row'] not in excluded_rows]
    assert not ({r['group_id'] for r in baseline} & set(purposes))
    assert {r['source_row'] for r in exclusions} == {
        r['source_row'] for r in group_manifest if r['group_id'] in purposes}
    summary = dict(version='training-filter-v1', train_rows=len(group_manifest),
        excluded_rows=len(exclusions), excluded_groups=len(purposes),
        excluded_rows_by_purpose=dict(sorted(Counter(r['purpose'] for r in exclusions).items())),
        baseline_train_rows=len(baseline), baseline_overlap_development_groups=0,
        baseline_overlap_control_groups=0,
        authoritative_filter='training_exclusions_manifest.jsonl',
        legacy_flag='group_manifest.excluded_from_training is deprecated: control-only')
    overlay = {'training_exclusions_manifest.jsonl': b''.join(encoded(r) for r in exclusions),
               'training_filter_summary.json': encoded(summary)}
    legacy_hashes = {name: digest(data) for name, data in expected.items()}
    overlay['training_exclusions_freeze_manifest.json'] = encoded(dict(
        version='training-filter-v1', legacy_frozen_sha256=legacy_hashes,
        sha256={name:digest(data) for name, data in overlay.items()}))
    exists = any((directory/name).exists() for name in overlay)
    if verify_only or exists:
        for name, content in overlay.items():
            if (directory/name).read_bytes() != content:
                raise ValueError(f'Training exclusion overlay changed: {name}; no overwrite allowed')
    else:
        for name, content in overlay.items():
            with (directory/name).open('xb') as output:
                output.write(content)
    # Also prove that overlay creation did not alter any original frozen byte.
    for name, checksum in legacy_hashes.items():
        if digest((directory/name).read_bytes()) != checksum:
            raise ValueError(f'Legacy freeze changed: {name}')
    print(json.dumps(dict(training_filter_verified=True, legacy_frozen_files_unchanged=len(legacy_hashes),
        excluded_rows=len(exclusions), excluded_rows_by_purpose=summary['excluded_rows_by_purpose'],
        baseline_train_rows=len(baseline), baseline_overlap_development_groups=0,
        baseline_overlap_control_groups=0), sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    directory = ROOT/'docs/stage1'
    expected = build(args.archive)
    if args.verify or any((directory/name).exists() for name in expected):
        verify(directory, expected)
        training_exclusions(directory, expected, verify_only=args.verify)
        return
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in expected.items():
        with (directory/name).open('xb') as output:
            output.write(content)
    verify(directory, expected)
    training_exclusions(directory, expected)


if __name__ == '__main__':
    main()

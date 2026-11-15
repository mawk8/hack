"""Read-only v2 loader. Closed review files are hashed as opaque bytes only."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

from scripts.audit_data import normalize

TRAIN_MEMBER = 'innoglobalhack_avito/public-train-annotated.jsonl'
PUBLIC_MEMBER = 'innoglobalhack_avito/public-test-dataset.parquet'
PUBLIC_CATEGORIES = {'Штукатурные работы': 102, 'Плиточные работы': 103,
                     'Сантехника': 104, 'Электрика': 105, 'Поклейка обоев и малярные работы': 106}
FILES = frozenset(('train.jsonl', 'development_original.jsonl', 'development_review.jsonl',
    'development_teacher_input.jsonl', 'control_reference.jsonl', 'control_teacher_input.jsonl',
    'human40_blind_review.jsonl', 'group_manifest.jsonl', 'training_exclusions_manifest.jsonl',
    'near_links_manifest.jsonl', 'selection_manifest.jsonl', 'summary.json'))


def digest(value):
    return hashlib.sha256(value).hexdigest()


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + '\n').encode('utf-8')


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_bytes().splitlines() if line.strip()]


def public_labels(categories):
    if not isinstance(categories, list) or any(not isinstance(v, str) or v not in PUBLIC_CATEGORIES for v in categories):
        raise ValueError('Unknown or malformed public category')
    if len(categories) != len(set(categories)):
        raise ValueError('Repeated public category')
    return sorted(PUBLIC_CATEGORIES[v] for v in categories)


def read_archive(archive):
    import pyarrow.parquet as pq
    from avito.data import validate_description, validate_labels
    with zipfile.ZipFile(archive) as source:
        train_payload = source.read(TRAIN_MEMBER)
        public_payload = source.read(PUBLIC_MEMBER)
    rows = [json.loads(line) for line in train_payload.splitlines() if line.strip()]
    for row in rows:
        if not isinstance(row, dict) or type(row.get('item_id')) is not int:
            raise ValueError('Malformed original train identity')
        validate_description(row.get('description'))
        annotations = row.get('found_microcategories')
        if not isinstance(annotations, list) or any(not isinstance(v, dict) or 'mcId' not in v for v in annotations):
            raise ValueError('Malformed original train labels')
        validate_labels([v['mcId'] for v in annotations])
    public = pq.read_table(io.BytesIO(public_payload)).to_pylist()
    for row in public:
        validate_description(row.get('description'))
        public_labels(row.get('categories'))
    return rows, public, {'train_member_sha256':digest(train_payload), 'public_member_sha256':digest(public_payload)}


def verify_files(archive, directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_bytes())
    if manifest.get('version') != 'data-v2' or set(manifest.get('sha256', {})) != FILES:
        raise ValueError('Incomplete or unsupported v2 manifest')
    hashes = {}
    for name, expected in manifest['sha256'].items():
        if Path(name).name != name:
            raise ValueError('Nonlocal filename in v2 manifest')
        hashes[name] = sha256(directory/name)
        if hashes[name] != expected:
            raise ValueError(f'Frozen v2 file changed: {name}')
    hashes['manifest.json'] = sha256(directory/'manifest.json')
    hashes['archive_sha256'] = sha256(archive)
    if hashes['archive_sha256'] != manifest.get('archive_sha256'):
        raise ValueError('Archive SHA256 differs from v2 provenance')
    return manifest, hashes


def load_training_data(archive, directory):
    """Return allowed original train rows and original noisy public development."""
    directory = Path(directory)
    manifest, hashes = verify_files(archive, directory)
    rows, public, members = read_archive(archive)
    if members != manifest['archive_members_sha256']:
        raise ValueError('Archive member provenance differs')
    hashes.update(members)
    summary = json.loads((directory/'summary.json').read_bytes())
    groups = read_jsonl(directory/'group_manifest.jsonl')
    by_dataset = {'train':{}, 'public':{}}
    for entry in groups:
        dataset, index = entry.get('dataset'), entry.get('source_row')
        if dataset not in by_dataset or type(index) is not int or index in by_dataset[dataset]:
            raise ValueError('Malformed/repeated v2 group row')
        source = rows if dataset == 'train' else public
        if not 0 <= index < len(source) or not isinstance(entry.get('group_id'), str):
            raise ValueError('Invalid v2 group identity')
        if entry.get('normalized_sha256') != digest(normalize(source[index]['description']).encode()):
            raise ValueError('V2 group description differs from archive')
        if dataset == 'train' and entry.get('item_id') != source[index]['item_id']:
            raise ValueError('V2 train identity differs from archive')
        by_dataset[dataset][index] = entry
    if len(by_dataset['train']) != len(rows) or len(by_dataset['public']) != len(public):
        raise ValueError('V2 group manifest is incomplete')
    public_groups = {r['group_id'] for r in by_dataset['public'].values()}
    selected = read_jsonl(directory/'selection_manifest.jsonl')
    selected_groups = set()
    for entry in selected:
        index, group = entry.get('source_row'), entry.get('group_id')
        if (type(index) is not int or index not in by_dataset['train'] or group in selected_groups
                or by_dataset['train'][index]['group_id'] != group or group in public_groups
                or entry.get('item_id') != rows[index]['item_id'] or type(entry.get('human40')) is not bool):
            raise ValueError('Invalid control selection manifest')
        selected_groups.add(group)
    exclusions = read_jsonl(directory/'training_exclusions_manifest.jsonl')
    excluded, reasons_by_group = set(), {}
    for entry in exclusions:
        index, group, reasons = entry.get('source_row'), entry.get('group_id'), entry.get('reasons')
        if (type(index) is not int or index not in by_dataset['train'] or index in excluded
                or any(entry.get(key) != by_dataset['train'][index][key] for key in ('item_id','group_id'))
                or not isinstance(reasons, list) or not reasons or reasons != sorted(set(reasons))
                or not set(reasons) <= {'public','legacy','control'}):
            raise ValueError('Invalid v2 exclusion row')
        if group in reasons_by_group and reasons_by_group[group] != reasons:
            raise ValueError('Conflicting reasons within leakage group')
        reasons_by_group[group] = reasons
        excluded.add(index)
    expected_reasons = {gid:sorted(set((['public'] if gid in public_groups else []) +
        (['legacy'] if gid in manifest['legacy_evaluation_group_ids'] else []) +
        (['control'] if gid in selected_groups else []))) for gid in {r['group_id'] for r in groups}}
    expected_excluded = {i for i, entry in by_dataset['train'].items() if expected_reasons[entry['group_id']]}
    if excluded != expected_excluded or any(reasons != expected_reasons[gid] for gid,reasons in reasons_by_group.items()):
        raise ValueError('Exclusions do not cover complete public/legacy/control groups')
    if selected_groups & set(manifest['legacy_evaluation_group_ids']):
        raise ValueError('Control selected from legacy evaluation groups')
    fit = [dict(row, source_row=i, group_id=by_dataset['train'][i]['group_id'])
           for i,row in enumerate(rows) if i not in excluded]
    if read_jsonl(directory/'train.jsonl') != fit:
        raise ValueError('V2 fit rows differ from allowed original rows')
    development = read_jsonl(directory/'development_original.jsonl')
    expected_dev = [dict(row, source_public_row=i, group_id=by_dataset['public'][i]['group_id'],
        source_mcId=101, purpose='development', proposed_labels=public_labels(row['categories']),
        status='original_public_noisy') for i,row in enumerate(public)]
    if development != expected_dev:
        raise ValueError('V2 development differs from original noisy public rows')
    measured = dict(original_train_rows=len(rows), public_rows=len(public), fit_rows=len(fit),
        excluded_train_rows=len(excluded), control_groups=len(selected), human40_rows=sum(r['human40'] for r in selected),
        public_normalized_descriptions=len({normalize(r['description']) for r in public}),
        leakage_groups=len({r['group_id'] for r in groups}), excluded_groups=len(reasons_by_group))
    if any(summary.get(key) != value for key,value in measured.items()):
        raise ValueError('V2 row/group counts differ from summary')
    if manifest.get('counts') != summary or summary.get('version') != 'data-v2':
        raise ValueError('V2 summary differs from manifest')
    # Runtime aliases preserve the existing evaluator contract without changing
    # the already frozen public rows or their source_public_row provenance.
    development = [dict(row,item_id=f"public-{row['source_public_row']}",source_row=row['source_public_row'])
                   for row in development]
    return fit, development, hashes

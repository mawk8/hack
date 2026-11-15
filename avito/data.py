"""Read train/development only; verify immutable files before selecting fit rows."""
import hashlib
import json
from pathlib import Path
import zipfile

from scripts.audit_data import normalize

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = (102, 103, 104, 105, 106)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def validate_description(description):
    if not isinstance(description, str) or not normalize(description):
        raise ValueError('description must be a nonempty string')


def validate_labels(labels):
    if not isinstance(labels, (list, tuple, set, frozenset)):
        raise ValueError('labels must be a collection of category integers')
    if any(type(label) is not int or label not in CATEGORIES for label in labels):
        raise ValueError('labels must contain only integers 102..106')
    if len(labels) != len(set(labels)):
        raise ValueError('duplicate category labels')
    return sorted(labels)


def verify_freeze(directory, archive):
    """Hash control files as opaque bytes; never parse their text or labels."""
    directory = Path(directory)
    legacy = json.loads((directory / 'freeze_manifest.json').read_bytes())
    overlay = json.loads((directory / 'training_exclusions_freeze_manifest.json').read_bytes())
    hashes = {}
    for entries in (legacy['sha256'], overlay['legacy_frozen_sha256'], overlay['sha256']):
        for name, expected in entries.items():
            if Path(name).name != name:
                raise ValueError('Frozen manifest contains a nonlocal filename')
            actual = sha256(directory / name)
            if actual != expected:
                raise ValueError(f'Frozen file changed: {name}')
            hashes[name] = actual
    hashes['training_exclusions_freeze_manifest.json'] = sha256(directory / 'training_exclusions_freeze_manifest.json')
    hashes['archive_sha256'] = sha256(archive)
    if hashes['archive_sha256'] != legacy['archive_sha256']:
        raise ValueError('Archive SHA256 differs from frozen input')
    return hashes


def select_training_rows(rows, manifest, exclusions):
    """Keep original row order and duplicates; apply the authoritative overlay."""
    by_row = {}
    for entry in manifest:
        index = entry['source_row']
        if type(index) is not int or index < 0 or index >= len(rows) or index in by_row:
            raise ValueError('Invalid or repeated source_row in group manifest')
        if entry['item_id'] != rows[index]['item_id'] or not isinstance(entry['group_id'], str):
            raise ValueError('Group manifest does not match original row identity')
        by_row[index] = entry
    if len(by_row) != len(rows):
        raise ValueError('Group manifest does not cover all original rows')
    excluded = set()
    purposes = {}
    for entry in exclusions:
        index = entry['source_row']
        group = entry['group_id']
        if (type(index) is not int or index in excluded or index not in by_row
                or any(entry[key] != by_row[index][key] for key in ('item_id', 'group_id'))
                or entry['purpose'] not in ('development', 'control')):
            raise ValueError('Invalid training exclusion row')
        if group in purposes and purposes[group] != entry['purpose']:
            raise ValueError('Evaluation group has conflicting purposes')
        purposes[group] = entry['purpose']
        excluded.add(index)
    if excluded != {index for index, entry in by_row.items() if entry['group_id'] in purposes}:
        raise ValueError('Training exclusions do not cover whole evaluation groups')
    return [dict(row, source_row=index, group_id=by_row[index]['group_id'])
            for index, row in enumerate(rows) if index not in excluded]


def load_training_data(archive, directory=None):
    if directory is None:
        newer = ROOT / 'data' / 'v2'
        directory = newer if newer.exists() else ROOT / 'docs' / 'stage1'
    directory = Path(directory)
    manifest = directory / 'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_bytes()).get('version') != 'data-v2':
            raise ValueError('Unsupported dataset manifest version')
        from .dataset_v2 import load_training_data as load_v2
        return load_v2(archive, directory)
    if directory.resolve() == (ROOT / 'data' / 'v2').resolve():
        raise ValueError('Incomplete data/v2: finish preparation before training')
    hashes = verify_freeze(directory, archive)
    with zipfile.ZipFile(archive) as source:
        payload = source.read('innoglobalhack_avito/public-train-annotated.jsonl')
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    for row in rows:
        if not isinstance(row, dict) or type(row.get('item_id')) is not int:
            raise ValueError('Invalid train row or item_id')
        validate_description(row.get('description'))
        annotations = row.get('found_microcategories')
        if not isinstance(annotations, list) or any(not isinstance(v, dict) or 'mcId' not in v for v in annotations):
            raise ValueError('Malformed found_microcategories')
        validate_labels([value['mcId'] for value in annotations])
    manifest = read_jsonl(directory / 'group_manifest.jsonl')
    exclusions = read_jsonl(directory / 'training_exclusions_manifest.jsonl')
    fit = select_training_rows(rows, manifest, exclusions)
    summary = json.loads((directory / 'training_filter_summary.json').read_text(encoding='utf-8'))
    if (len(rows), len(fit), len(exclusions)) != (summary['train_rows'], summary['baseline_train_rows'], summary['excluded_rows']):
        raise ValueError('Training row counts differ from frozen summary')
    development = read_jsonl(directory / 'development_review.jsonl')
    dev_groups = {r['group_id'] for r in exclusions if r['purpose'] == 'development'}
    for row in development:
        index = row['source_row']
        if (type(index) is not int or not 0 <= index < len(rows)
                or row['group_id'] not in dev_groups or row['item_id'] != rows[index]['item_id']
                or row['description'] != rows[index]['description']
                or row['source_mcId'] != 101 or row['purpose'] != 'development'
                or validate_labels(row['proposed_labels']) != sorted(v['mcId'] for v in rows[index]['found_microcategories'])):
            raise ValueError('Development identity or labels differ from original input')
    if len(development) != 50 or len({r['group_id'] for r in development}) != 50:
        raise ValueError('Expected 50 unique frozen development groups')
    hashes['train_member_sha256'] = hashlib.sha256(payload).hexdigest()
    return fit, development, hashes

"""Read-only audit of the supplied archive; output contains aggregates, never ads."""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.tools/python'))


def normalize(text):
    """Conservative grouping key, not a semantic equivalence guarantee."""
    return re.sub(r'\s+', ' ', text.casefold()).strip()


def group_stats(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row['description'])].append(row)
    repeated = [g for g in groups.values() if len(g) > 1]
    conflicts = [g for g in repeated if len({tuple(sorted({v['mcId'] for v in r['found_microcategories']})) for r in g}) > 1]
    return dict(unique_descriptions=len(groups), duplicate_groups=len(repeated),
                rows_in_duplicate_groups=sum(map(len, repeated)), extra_duplicate_rows=sum(len(g)-1 for g in repeated),
                conflicting_label_groups=len(conflicts), rows_in_conflicting_groups=sum(map(len, conflicts)))


def audit(archive):
    import pyarrow
    import pyarrow.parquet as pq
    prefix = 'innoglobalhack_avito/'
    with zipfile.ZipFile(archive) as z:
        rows = [json.loads(line) for line in z.read(prefix+'public-train-annotated.jsonl').splitlines() if line.strip()]
        table = pq.read_table(io.BytesIO(z.read(prefix+'public-test-dataset.parquet')))
        # Only descriptions are used in cross-dataset comparisons; test labels are not inspected.
        test_desc = table.column('description').to_pylist()
    ids = Counter(r['item_id'] for r in rows)
    label_freq = Counter(v['mcId'] for r in rows for v in {v['mcId']:v for v in r['found_microcategories']}.values())
    overlaps = {}
    for name, fn in [('exact', lambda x:x), ('normalized', normalize)]:
        train_keys = Counter(fn(r['description']) for r in rows)
        test_keys = Counter(fn(d) for d in test_desc if isinstance(d, str))
        shared = train_keys.keys() & test_keys.keys()
        overlaps[name] = dict(shared_unique_descriptions=len(shared), train_rows=sum(train_keys[k] for k in shared),
                              public_test_rows=sum(test_keys[k] for k in shared), row_pairs=sum(train_keys[k]*test_keys[k] for k in shared))
    return dict(archive_sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
        runtime=dict(python=sys.version.split()[0], pyarrow=pyarrow.__version__),
        normalization='casefold, whitespace runs→space, strip; punctuation and ё retained',
        train=dict(rows=len(rows), fields=sorted({k for r in rows for k in r}),
          field_types={k:sorted({type(r.get(k)).__name__ for r in rows}) for k in rows[0]},
          exact=group_stats(rows, lambda x:x), normalized=group_stats(rows, normalize),
          category_row_frequencies={str(k):v for k,v in sorted(label_freq.items())}, empty_labels=sum(not r['found_microcategories'] for r in rows),
          repeated_identifier_values=sum(c>1 for c in ids.values()), rows_with_repeated_identifiers=sum(c for c in ids.values() if c>1),
          duplicate_category_rows=sum(len(r['found_microcategories'])!=len({v['mcId'] for v in r['found_microcategories']}) for r in rows),
          missing_or_empty_description=sum(not r.get('description') for r in rows), explicit_source_mcId_present=any('mcId' in r for r in rows)),
        public_test=dict(rows=table.num_rows, schema=str(table.schema), fields=table.column_names,
                         explicit_source_mcId_present='mcId' in table.column_names,
                         null_descriptions=table.column('description').null_count),
        overlap=overlaps)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', required=True, type=Path)
    p.add_argument('--output', type=Path, default=ROOT/'docs/stage1/audit_summary.json')
    args = p.parse_args()
    result = audit(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()

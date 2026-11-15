"""Create immutable data/v2 from the supplied archive and verified read-only v1.

No labels influence sampling. Existing partial/different outputs are refused;
--verify reproduces bytes in memory without overwriting any file.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_data import normalize
from avito.dataset_v2 import (TRAIN_MEMBER, PUBLIC_MEMBER, digest, encoded, sha256,
    public_labels, read_archive, read_jsonl, load_training_data)

VERSION = 'data-v2'
SEED = 20261115
HUMAN_SEED = 20261116


def build(archive, legacy_directory=ROOT/'docs/stage1', control_count=200, human_count=40):
    from avito.data import verify_freeze, select_training_rows
    archive, legacy_directory = Path(archive), Path(legacy_directory)
    # This is deliberately the first data access: do not build from changed v1/input.
    frozen = verify_freeze(legacy_directory, archive)
    frozen['freeze_manifest.json'] = sha256(legacy_directory/'freeze_manifest.json')
    rows, public, members = read_archive(archive)
    old_manifest = read_jsonl(legacy_directory/'group_manifest.jsonl')
    old_exclusions = read_jsonl(legacy_directory/'training_exclusions_manifest.jsonl')
    select_training_rows(rows, old_manifest, old_exclusions)  # identity, coverage, whole groups
    old_by_row = {r['source_row']:r for r in old_manifest}
    old_evaluation = {r['source_row'] for r in old_exclusions}
    for purpose in ('development','control'):
        for row in read_jsonl(legacy_directory/f'{purpose}_review.jsonl'):
            index = row['source_row']
            if index not in old_by_row or row['group_id'] != old_by_row[index]['group_id']:
                raise ValueError('Legacy review identity differs from group manifest')
            old_evaluation.add(index)
    all_rows = rows + public
    normalized = [normalize(r['description']) for r in all_rows]
    texts = sorted(set(normalized))
    text_indices = {text:i for i,text in enumerate(texts)}
    parent = list(range(len(texts)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a,b):
        a,b = find(a),find(b)
        parent[max(a,b)] = min(a,b)
    old_groups = defaultdict(list)
    for entry in old_manifest:
        old_groups[entry['group_id']].append(text_indices[normalized[entry['source_row']]])
    for indices in old_groups.values():
        for index in indices[1:]:
            union(indices[0], index)
    blocks = defaultdict(list)
    for i,text in enumerate(texts):
        if len(text)>=160:
            blocks[('prefix',text[:80])].append(i)
            blocks[('suffix',text[-80:])].append(i)
    # ponytail: same narrow v1 candidate rule; expand only with measured leakage.
    candidates = {(a,b) for ids in blocks.values() for pos,a in enumerate(ids) for b in ids[pos+1:]}
    tokens = [set(text.split()) for text in texts]
    links = []
    for a,b in sorted(candidates):
        ratio = min(len(texts[a]),len(texts[b]))/max(len(texts[a]),len(texts[b]))
        jaccard = len(tokens[a]&tokens[b])/len(tokens[a]|tokens[b])
        if ratio>=.9 and jaccard>=.95:
            union(a,b)
            links.append(dict(left_normalized_sha256=digest(texts[a].encode()), right_normalized_sha256=digest(texts[b].encode()),
                rule='shared first/last 80 chars; length >= 160; length ratio >= .90; whitespace-token set Jaccard >= .95',
                length_ratio=ratio, token_jaccard=jaccard))
    components = defaultdict(list)
    for i,text in enumerate(texts):
        components[find(i)].append(digest(text.encode()))
    ids = {root:'g2-'+digest(encoded(sorted(values)))[:24] for root,values in components.items()}
    row_groups = [ids[find(text_indices[text])] for text in normalized]
    train_groups = defaultdict(list)
    for i,gid in enumerate(row_groups[:len(rows)]):
        train_groups[gid].append(i)
    public_groups = set(row_groups[len(rows):])
    legacy_groups = {row_groups[i] for i in old_evaluation}
    population = sorted(set(train_groups)-public_groups-legacy_groups)
    if not 0<=human_count<=control_count<=len(population):
        raise ValueError('Insufficient independent groups or invalid control/human sample size')
    rng = random.Random(SEED)
    chosen = rng.sample(population, control_count)
    representatives = [(gid,rng.choice(train_groups[gid])) for gid in chosen]
    human_groups = set(random.Random(HUMAN_SEED).sample(sorted(chosen),human_count))
    control_groups = set(chosen)
    reasons = {gid:sorted((['public'] if gid in public_groups else []) +
        (['legacy'] if gid in legacy_groups else []) + (['control'] if gid in control_groups else [])) for gid in ids.values()}
    groups = [dict(dataset='train', source_row=i, item_id=row['item_id'], group_id=row_groups[i],
        legacy_group_id=old_by_row[i]['group_id'], normalized_sha256=digest(normalized[i].encode())) for i,row in enumerate(rows)]
    groups += [dict(dataset='public',source_row=i,group_id=row_groups[len(rows)+i],
        normalized_sha256=digest(normalized[len(rows)+i].encode())) for i in range(len(public))]
    exclusions = [dict(source_row=i,item_id=row['item_id'],group_id=row_groups[i],reasons=reasons[row_groups[i]])
                  for i,row in enumerate(rows) if reasons[row_groups[i]]]
    fit = [dict(row,source_row=i,group_id=row_groups[i]) for i,row in enumerate(rows) if not reasons[row_groups[i]]]
    dev = [dict(row,source_public_row=i,group_id=row_groups[len(rows)+i],source_mcId=101,purpose='development',
        proposed_labels=public_labels(row['categories']),status='original_public_noisy') for i,row in enumerate(public)]
    public_unique = defaultdict(list)
    for i,row in enumerate(public):
        public_unique[normalize(row['description'])].append(i)
    reviews = []
    teacher = []
    for indices in sorted(public_unique.values(),key=lambda x:x[0]):
        i = indices[0]
        # Choose only the original text representative; every label variant is retained.
        identity = dict(item_id=f'public-{i}',source_public_rows=indices,group_id=row_groups[len(rows)+i],
                        description=public[i]['description'],source_mcId=101,purpose='development')
        teacher.append(identity)
        reviews.append(dict(identity,original_label_variants=[list(v) for v in sorted({tuple(public_labels(public[j]['categories'])) for j in indices})],
            proposed_labels=None,evidence=[],questions=[],status='provisional',reviewer=None,reviewed_at=None))
    control,control_teacher,blind,selection = [],[],[],[]
    for gid,i in representatives:
        row = rows[i]
        identity = dict(item_id=row['item_id'],source_row=i,group_id=gid,description=row['description'],source_mcId=101,purpose='control')
        control_teacher.append(identity)
        control.append(dict(row,**{k:v for k,v in identity.items() if k not in row},
            proposed_labels=sorted(v['mcId'] for v in row['found_microcategories']),status='original_train_noisy'))
        selection.append(dict(group_id=gid,source_row=i,item_id=row['item_id'],human40=gid in human_groups))
        if gid in human_groups:
            blind.append(dict(identity,proposed_labels=None,evidence=[],questions=[],status='provisional',reviewer=None,reviewed_at=None))
    content = {'train.jsonl':fit,'development_original.jsonl':dev,'development_review.jsonl':reviews,
        'development_teacher_input.jsonl':teacher,'control_reference.jsonl':control,'control_teacher_input.jsonl':control_teacher,
        'human40_blind_review.jsonl':blind,'group_manifest.jsonl':groups,'training_exclusions_manifest.jsonl':exclusions,
        'near_links_manifest.jsonl':links,'selection_manifest.jsonl':selection}
    outputs = {name:b''.join(encoded(row) for row in values) for name,values in content.items()}
    summary = dict(version=VERSION, original_train_rows=len(rows),public_rows=len(public),
        public_normalized_descriptions=len(reviews),leakage_groups=len(ids),fit_rows=len(fit),
        excluded_train_rows=len(exclusions),excluded_groups=len({r['group_id'] for r in exclusions}),
        excluded_rows_by_reason={reason:sum(reason in r['reasons'] for r in exclusions) for reason in ('public','legacy','control')},
        excluded_groups_by_reason={reason:sum(reason in r for r in reasons.values()) for reason in ('public','legacy','control')},
        sampling_population_groups=len(population),control_groups=len(control_groups),human40_rows=len(blind),
        near_candidate_pairs=len(candidates),accepted_near_links=len(links),
        sampling_population='unique eligible original-train leakage groups, excluding all public and legacy evaluation groups; not rows or private dataset',
        selection='uniform random groups, then uniform random original row within each selected group; labels unused',
        human_selection='independent seeded uniform sample of control groups; no labels or teacher hints',
        status='original noisy references; all new reviews provisional and unreviewed',
        normalization='scripts.audit_data.normalize; casefold and whitespace only')
    outputs['summary.json'] = encoded(summary)
    manifest = dict(version=VERSION,seed=SEED,human_seed=HUMAN_SEED,archive_sha256=frozen['archive_sha256'],
        archive_members_sha256=members,v1_sha256={k:v for k,v in frozen.items() if k!='archive_sha256'},
        legacy_evaluation_group_ids=sorted(legacy_groups),counts=summary,
        sha256={name:digest(value) for name,value in outputs.items()},
        provenance='read-only original archive and verified frozen v1; create-only outputs; no overwrite')
    outputs['manifest.json'] = encoded(manifest)
    after = verify_freeze(legacy_directory,archive)
    after['freeze_manifest.json'] = sha256(legacy_directory/'freeze_manifest.json')
    if after != frozen:
        raise ValueError('Original v1 provenance changed during preparation')
    if sha256(legacy_directory/'freeze_manifest.json') != frozen['freeze_manifest.json']:
        raise ValueError('Original v1 freeze changed during preparation')
    return outputs


def write_outputs(directory, outputs, verify_only=False):
    directory = Path(directory)
    if verify_only or directory.exists():
        if not directory.is_dir() or {p.name for p in directory.iterdir()} != set(outputs):
            raise ValueError('Partial or unexpected existing v2 output; no overwrite allowed')
        for name,value in outputs.items():
            if (directory/name).read_bytes()!=value:
                raise ValueError(f'V2 reproduction mismatch: {name}; no overwrite allowed')
        return
    directory.mkdir(parents=True,exist_ok=False)
    for name,value in outputs.items():
        with (directory/name).open('xb') as stream:
            stream.write(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',required=True,type=Path)
    parser.add_argument('--legacy-dir',type=Path,default=ROOT/'docs/stage1')
    parser.add_argument('--output',type=Path,default=ROOT/'data/v2')
    parser.add_argument('--verify',action='store_true')
    args = parser.parse_args()
    outputs = build(args.archive,args.legacy_dir)
    write_outputs(args.output,outputs,verify_only=args.verify)
    fit,development,_ = load_training_data(args.archive,args.output)
    print(json.dumps(dict(verified=True,version=VERSION,fit_rows=len(fit),development_rows=len(development),
        **{k:v for k,v in json.loads(outputs['summary.json']).items() if k in
           ('excluded_train_rows','control_groups','human40_rows','public_normalized_descriptions','sampling_population_groups')}),sort_keys=True))


if __name__=='__main__':
    main()

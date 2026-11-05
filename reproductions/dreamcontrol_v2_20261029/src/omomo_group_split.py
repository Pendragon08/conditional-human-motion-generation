"""Construct an isolated source-family split for OMOMO candidate records.

Subject identifiers are filename claims, not verified human identities. This
manifest is an auxiliary planning view, not an author split or training set.
"""
import collections
import hashlib
import json
from pathlib import Path
import re

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PAIRS=ROOT/'omomo_text_pairing_20261101/candidate_pairs.jsonl'
TIMING=ROOT/'omni_variant_caption_time_20261101'
HERE=Path(__file__).resolve().parent
POLICY={'sub3':'train','sub7':'train','sub8':'train','sub10':'train','sub12':'val','sub16':'test'}
PATTERN=re.compile(r'^robot-object/(sub(?:3|7|8|10|12|16))_largebox_(\d{3})_(original|rot_\d+|trans_\d+)\.npz$')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def classify(row):
    match=PATTERN.fullmatch(row.get('motion_member',''))
    if not match:raise ValueError('unexpected Omni filename')
    subject,num,variant=match.groups();key=f'{subject}_largebox_{num}'
    if row.get('nominal_action_key')!=key or row.get('variant')!=variant or row.get('provisional_group')!=f'OMOMO_nominal_subject:{subject}':
        raise ValueError('filename disagrees with source metadata')
    return subject,key,variant,POLICY[subject]


def main():
    pairs_sha=sha(PAIRS);time_sha=sha(TIMING/'records.jsonl')
    time_report=json.loads((TIMING/'report.json').read_text());time_audit=json.loads((TIMING/'audit.json').read_text())
    if time_report['records_sha256']!=time_sha or time_audit['records_sha256']!=time_sha or time_audit['source_report_sha256']!=sha(TIMING/'report.json'):
        raise ValueError('timing evidence changed')
    pairs=[json.loads(line) for line in PAIRS.read_text().splitlines()]
    timing={r['motion_member']:r for r in map(json.loads,(TIMING/'records.jsonl').read_text().splitlines())}
    if len(pairs)!=1826 or len(timing)!=1514 or len({r['motion_member'] for r in pairs})!=1826:
        raise ValueError('source coverage mismatch')
    action_splits={};subject_splits={};counts=collections.Counter();variant_counts=collections.Counter();action_keys=collections.defaultdict(set)
    with (HERE/'candidates.jsonl').open('x') as out:
        for row in pairs:
            subject,key,variant,split=classify(row)
            if action_splits.setdefault(key,split)!=split or subject_splits.setdefault(subject,split)!=split:
                raise ValueError('nominal source crosses split')
            time=timing.get(row['motion_member'])
            if variant=='original':
                if time is not None:raise ValueError('unexpected variant time entry for original')
            elif time is None or time['motion_member_sha256']!=row['motion_member_sha256'] or time['annotation_sha256']!=row['annotation_member_sha256'] or time['caption']!=row['caption']:
                raise ValueError('variant timeline or caption does not match source')
            result={'motion_member':row['motion_member'],'motion_member_sha256':row['motion_member_sha256'],
                    'nominal_action_key':key,'nominal_subject':subject,'source_family':'OMOMO',
                    'variant':variant,'auxiliary_split':split,'annotation_member':row['annotation_member'],
                    'annotation_sha256':row['annotation_member_sha256'],'caption':row['caption'],
                    'caption_status':'unreviewed_filename_match','variant_timeline':time['timeline'] if time else None,
                    'source_identity_verified':False,'training_ready':False}
            out.write(json.dumps(result,allow_nan=False)+'\n')
            counts[split]+=1;variant_counts[(split,'original' if variant=='original' else 'variant')]+=1;action_keys[split].add(key)
    if len(action_splits)!=312 or len(subject_splits)!=6 or sum(counts.values())!=1826:
        raise ValueError('unexpected group coverage')
    if sha(PAIRS)!=pairs_sha or sha(TIMING/'records.jsonl')!=time_sha:
        raise RuntimeError('input changed')
    report={'pair_manifest_sha256':pairs_sha,'timing_records_sha256':time_sha,'script_sha256':sha(Path(__file__)),
            'candidate_manifest_sha256':sha(HERE/'candidates.jsonl'),'policy':POLICY,'counts_by_split':dict(counts),
            'variant_counts_by_split':{s:{kind:variant_counts[(s,kind)] for kind in ('original','variant')} for s in POLICY.values()},
            'nominal_actions_by_split':{s:len(keys) for s,keys in action_keys.items()},'nominal_action_keys':len(action_splits),
            'nominal_subjects':len(subject_splits),'training_ready':False,
            'limitations':['Filename subjects are not independently verified human identities; policy is non-author auxiliary split.',
                           'No caption semantics, source rights, actual time, cross-dataset identity or physical acceptance.',
                           'This is a candidate list without processed feature bindings or train-only statistics.']}
    with (HERE/'report.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(report))


if __name__=='__main__':main()

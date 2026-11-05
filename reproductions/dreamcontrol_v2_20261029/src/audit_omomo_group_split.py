"""Recompute auxiliary split groups and archive bindings for all candidates."""
import collections
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent
ARCHIVE=ROOT/'data/OmniRetarget_Dataset/robot-object.zip'
ANNOT=ROOT/'omomo_text_pairing_20261101/omomo_text_anno.zip'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main():
    report=json.loads((HERE/'report.json').read_text())
    if sha(HERE/'candidates.jsonl')!=report['candidate_manifest_sha256']:
        raise ValueError('manifest changed')
    rows=[json.loads(line) for line in (HERE/'candidates.jsonl').read_text().splitlines()]
    if len(rows)!=1826 or len({r['motion_member'] for r in rows})!=1826:raise ValueError('coverage')
    action_split={};subject_split={};content_split={};counts=collections.Counter();actions=collections.defaultdict(set)
    actual_archive_sha=sha(ARCHIVE);actual_annot_sha=sha(ANNOT)
    with zipfile.ZipFile(ARCHIVE) as z,zipfile.ZipFile(ANNOT) as a:
        for row in rows:
            name=row['motion_member']
            if not name.startswith('robot-object/') or not name.endswith('.npz') or '/' in name[len('robot-object/'):]:
                raise ValueError('unsafe member name')
            parts=name[len('robot-object/'):-4].split('_')
            if len(parts)<4 or parts[1]!='largebox' or len(parts[2])!=3 or not parts[2].isdigit():raise ValueError('bad member')
            subject=parts[0];key='_'.join(parts[:3]);variant='_'.join(parts[3:])
            expected='val' if subject=='sub12' else 'test' if subject=='sub16' else 'train' if subject in {'sub3','sub7','sub8','sub10'} else None
            if expected is None or row['auxiliary_split']!=expected or row['nominal_action_key']!=key or row['nominal_subject']!=subject or row['variant']!=variant:
                raise ValueError('split or source metadata mismatch')
            if action_split.setdefault(key,expected)!=expected or subject_split.setdefault(subject,expected)!=expected:
                raise ValueError('source crosses splits')
            raw=z.read(name);text=a.read(row['annotation_member'])
            raw_hash=hashlib.sha256(raw).hexdigest()
            if raw_hash!=row['motion_member_sha256'] or hashlib.sha256(text).hexdigest()!=row['annotation_sha256']:
                raise ValueError('member hash mismatch')
            if json.loads(text)[key]!=row['caption'] or row['caption_status']!='unreviewed_filename_match' or row['training_ready'] is not False:
                raise ValueError('caption status mismatch')
            if content_split.setdefault(raw_hash,expected)!=expected:raise ValueError('equal source bytes cross splits')
            counts[expected]+=1;actions[expected].add(key)
    if dict(counts)!=report['counts_by_split'] or {s:len(keys) for s,keys in actions.items()}!=report['nominal_actions_by_split']:
        raise ValueError('count disagreement')
    if sha(ARCHIVE)!=actual_archive_sha or sha(ANNOT)!=actual_annot_sha or sha(HERE/'candidates.jsonl')!=report['candidate_manifest_sha256']:
        raise RuntimeError('input changed')
    result={'report_sha256':sha(HERE/'report.json'),'manifest_sha256':report['candidate_manifest_sha256'],
            'archive_sha256':actual_archive_sha,'annotation_sha256':actual_annot_sha,'script_sha256':sha(Path(__file__)),
            'records':len(rows),'nominal_actions':len(action_split),'nominal_subjects':len(subject_split),
            'split_counts':dict(counts),'equal_raw_bytes_cross_splits':0,'training_ready':False,
            'limitations':['Filename group disjointness is not proof of subject or cross-dataset independence.',
                           'No actual feature bindings, training statistics, semantics, real time, rights or physics acceptance.']}
    with (HERE/'audit.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()

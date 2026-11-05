"""Recompute GRAB pilot and batch selection coverage from immutable receipts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--registry',required=True,type=Path)
    p.add_argument('--selection',required=True,type=Path)
    p.add_argument('--prior-selection',required=True,type=Path)
    p.add_argument('--records',required=True,type=Path)
    p.add_argument('--report',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    paths=(a.registry,a.selection,a.prior_selection,a.records,a.report)
    hashes={str(x):sha(x) for x in paths}
    selection=json.loads(a.selection.read_text(encoding='utf8'))
    prior=json.loads(a.prior_selection.read_text(encoding='utf8'))
    report=json.loads(a.report.read_text(encoding='utf8'))
    rows=[json.loads(x) for x in a.records.read_text(encoding='utf8').splitlines()]
    registry={r['member']:r for r in map(json.loads,a.registry.read_text(encoding='utf8').splitlines())}
    if (selection['source_registry_sha256']!=hashes[str(a.registry)] or report['registry_sha256']!=hashes[str(a.registry)]
        or report['selection_sha256']!=hashes[str(a.selection)] or report['records_sha256']!=hashes[str(a.records)]
        or report['requested']!=len(selection['members']) or [r['member'] for r in rows]!=selection['members']
        or len(set(selection['members']))!=len(selection['members']) or set(selection['members'])&set(prior['members'])):
        raise ValueError('selection or records mismatch')
    status=Counter();subjects=Counter();frames=0
    for i,row in enumerate(rows):
        source=registry[row['member']]
        if row['selection_index']!=i or row['source_member_sha256']!=source['member_sha256'] or row['training_ready'] is not False:
            raise ValueError('source provenance mismatch')
        status[row['status']]+=1;subjects[source['subject']]+=1
        if row['status']=='reconstructed_unreviewed':
            if (row['frames']!=source['frames'] or row['declared_fps']!=source['fps_declared']
                or row['archive_sha256']!=source['archive_sha256'] or row['template_sha256']!=source['template_sha256']
                or row['object_mesh_sha256']!=source['object_mesh_sha256']):raise ValueError('source metadata mismatch')
            frames+=row['frames']
    if any(sha(x)!=hashes[str(x)] for x in paths):raise RuntimeError('audit input changed')
    result={'selected':len(rows),'status_counts':dict(status),'source_subject_counts':dict(subjects),
            'reconstructed_source_frames':frames,'inputs_sha256':hashes,'script_sha256':sha(Path(__file__)),
            'training_ready':False,'limitations':['Recomputes registry bindings from receipts; does not re-read archive or output bytes.',
                'No verification of actual capture clock, semantics, contact, robot conversion or physics.']}
    with a.output.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('selected','status_counts','reconstructed_source_frames','training_ready')}))


if __name__=='__main__':main()

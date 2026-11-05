"""Build a hash-bound source-member queue, excluding the audited 20-item pilot."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
INVENTORY=ROOT/'data/bmlmovi_smplx_inventory_20261102/records.jsonl'
SUMMARY=ROOT/'data/bmlmovi_smplx_inventory_20261102/summary.json'
PILOT=ROOT/'data/bmlmovi_diverse_selection_20261102.json'
PILOT_AUDIT=ROOT/'reports/bmlmovi_full_human_audit_20261102.json'
OUT=ROOT/'data/bmlmovi_full_stream_queue_20261102.json'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    paths=(INVENTORY,SUMMARY,PILOT,PILOT_AUDIT);hashes={str(x):sha(x) for x in paths}
    summary=json.loads(SUMMARY.read_text());pilot=json.loads(PILOT.read_text(encoding='utf8'))
    audit=json.loads(PILOT_AUDIT.read_text());selected={x['id']:x for x in pilot['members']}
    if (summary['event']!='complete' or summary['dataset']!='BMLmovi' or len(selected)!=20
        or {x['id'] for x in audit['cases'] if x['status']=='technical_sample_checked'}!=set(selected)):
        raise ValueError('incomplete inventory or pilot audit')
    excluded={x['member'] for x in selected.values()};members=[];seen=set()
    for line in INVENTORY.read_text(encoding='utf8').splitlines():
        row=json.loads(line)
        if row['member'] in seen or row.get('duplicate_key') or 'error' in row or row['training_accepted'] is not False:
            raise ValueError('duplicate/unresolved inventory')
        seen.add(row['member'])
        if row['member'] in excluded:continue
        if row['member_size']>64*1024**2 or row['poses_shape'][0]<8 or row['declared_fps']!=120.:
            raise ValueError('unsupported input or source rate')
        members.append({'member':row['member'],'source_sha256':row['member_sha256'],
            'member_size':row['member_size'],'source_frames':row['poses_shape'][0],
            'declared_fps':row['declared_fps'],'nominal_rows':row['nominal_rows']})
    if not excluded.issubset(seen):
        raise ValueError('pilot exclusion mismatch')
    if len(seen)!=summary['counts']['matched_members'] or len(members)+len(excluded)!=len(seen):
        raise ValueError('inventory coverage mismatch')
    if any(sha(x)!=hashes[str(x)] for x in paths):raise RuntimeError('queue inputs changed')
    result={'members':members,'excluded_pilot_members':sorted(excluded),'inputs_sha256':hashes,
            'expected_source_frames':sum(r['source_frames'] for r in members),
            'raw_22point_float32_bytes':sum(r['source_frames'] for r in members)*22*3*4,
            'script_sha256':sha(Path(__file__)),'training_ready':False,
            'limitations':['Source names and declared FPS are not verified identity or actual clock.',
              'Only a source reconstruction queue; no captions, robot states, physics or training acceptance.']}
    with OUT.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'members':len(members),'frames':result['expected_source_frames'],
                      'raw_points_bytes':result['raw_22point_float32_bytes'],'training_ready':False}))


if __name__=='__main__':main()

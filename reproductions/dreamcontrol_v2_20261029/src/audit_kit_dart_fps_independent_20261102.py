"""Independently recompute the candidate KIT↔DART FPS and bounds join."""
import collections,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/'data';REPORT=ROOT/'reports/kit_dart_fps_comparison_v2_20261102.json'
OUT=ROOT/'reports/kit_dart_fps_comparison_audit_20261102.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def unique_pairs(pairs):
    d={}
    for k,v in pairs:
        if k in d:raise ValueError('duplicate JSON dictionary key')
        d[k]=v
    return d
def normalize(parts):
    if len(parts)!=3 or parts[0]!='KIT':raise ValueError('unexpected path')
    name=parts[2]
    for suffix in ('_stageii.npz','_poses.npz','_poses.npy'):
        if name.lower().endswith(suffix):name=name[:-len(suffix)];break
    else:raise ValueError('unknown file suffix')
    # Inputs are ASCII; use membership rather than the producer's regex.
    return (parts[1],''.join(c for c in name.lower() if c in 'abcdefghijklmnopqrstuvwxyz0123456789'))
def main():
    old=DATA/'kit_smplx_stream_inventory_20261102/records.jsonl';large=DATA/'kit_smplx_large_members_20261102/records.jsonl'
    allpath=DATA/'dart_fps_metadata_20261102/fps_dict_all.json';subpath=DATA/'dart_fps_metadata_20261102/fps_dict.json'
    r=json.loads(REPORT.read_text()); oldhash=sha(old);largeh=sha(large)
    if oldhash!=r['source_inventory_sha256'] or sha(allpath)!=r['dart_all_sha256'] or sha(subpath)!=r['dart_subset_sha256']:raise ValueError('input hash')
    dart=json.loads(allpath.read_text(),object_pairs_hook=unique_pairs)
    sub=json.loads(subpath.read_text(),object_pairs_hook=unique_pairs)
    table={};subtable=collections.defaultdict(list)
    for raw,fps in dart.items():
        parts=raw.replace('\\','/').split('/')
        if len(parts)!=3 or parts[0]!='KIT':continue
        key=normalize(parts)
        if key in table:raise ValueError('ambiguous DART candidate')
        table[key]=(raw,float(fps))
    for raw,fps in sub.items():
        parts=raw.replace('\\','/').split('/')
        if len(parts)>=4 and parts[0:2]==['KIT','KIT']:parts=parts[1:]
        if len(parts)==3 and parts[0]=='KIT':subtable[normalize(parts)].append(float(fps))
    replacements={x['member']:x for x in map(json.loads,large.read_text().splitlines())}
    if len(replacements)!=33:raise ValueError('oversize coverage')
    rows=[json.loads(x) for x in old.read_text().splitlines()];seen=set();matched=0;submatched=0;fpscount=collections.Counter();headers=collections.Counter();bounds=collections.Counter();ids=set()
    for row in rows:
        member=row['member'];key=normalize(member.split('/'));seen.add(key)
        if key not in table:raise ValueError('missing DART candidate')
        fps=table[key][1];fpscount[str(fps)]+=1;matched+=1
        if key in subtable:
            submatched+=1
            if any(x!=fps for x in subtable[key]):raise ValueError('DART dictionary disagreement')
        actual=replacements.get(member,row);n=actual['poses_shape'][0];header=float(actual['declared_fps']);headers[str(header)]+=1
        for x in row['nominal_rows']:
            mid=x['motion_id']
            if mid in ids:raise ValueError('duplicate ID')
            ids.add(mid);start=int(x['start']);end=int(x['end']);length=len(range(0,n,int(fps/20)))
            if 0<=start<end<=length:bounds['within_dart_rate']+=1
            else:bounds['outside_dart_rate']+=1
    if (matched!=r['matched_members'] or len(ids)!=r['matched_nominal_ids'] or dict(fpscount)!=r['matched_fps_counts'] or
        submatched!=r['matched_subset_members'] or len(set(table)-seen)!=len(r['dart_only_keys'])):raise ValueError('original result mismatch')
    if sha(old)!=oldhash or sha(large)!=largeh:raise RuntimeError('inventory changed')
    result={'first_audit_sha256':sha(REPORT),'source_inventory_sha256':oldhash,'oversize_inventory_sha256':largeh,
      'dart_all_sha256':sha(allpath),'script_sha256':sha(Path(__file__)),'matched_members':matched,'nominal_ids':len(ids),
      'dart_fps_counts':dict(fpscount),'source_declared_fps_counts':dict(headers),'dart_subset_matches':submatched,
      'nominal_bounds_under_dart_fps':dict(bounds),'training_ready':False,
      'limitations':['Matched names are not independently established recording identities.',
        'DART dictionary is an independent correction, not original C3D ground truth; real capture rate remains uncertain.',
        'Interval coverage does not validate text semantics, time alignment or training.']}
    with OUT.open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result))
if __name__=='__main__':main()

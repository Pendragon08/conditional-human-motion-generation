"""Compare S2 contact-array width with verified staged PLY vertex counts."""
import collections
import json
from pathlib import Path, PurePosixPath

BASE=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
STAGE=BASE/'grab_mesh_stage_20261030'
OUT=BASE/'grab_s2_contact_mesh_sizes_20261030'

def vertices(path):
    with path.open('rb') as f:
        if f.readline().strip()!=b'ply': raise ValueError('not a PLY file')
        count=None
        for _ in range(500):
            line=f.readline()
            if not line or len(line)>8192: raise ValueError('invalid PLY header')
            words=line.strip().split()
            if words[:2]==[b'element',b'vertex']:
                if count is not None: raise ValueError('duplicate vertex count')
                count=int(words[2])
            if words==[b'end_header']:
                if count is None or count<1: raise ValueError('missing vertex count')
                return count
        raise ValueError('oversized PLY header')

def main():
    docs=[BASE/'grab_body_audit_20261030/all_structure_report.json',BASE/'grab_oversize_audit_20261030/report.json']
    sequences=[]
    for doc in docs:sequences.extend(json.loads(doc.read_text())['sequences'])
    if len(sequences)!=93 or len({r['member'] for r in sequences})!=93:
        raise ValueError('incomplete or duplicate S2 sequence records')
    rows=[]; cache={}
    for r in sequences:
        p=PurePosixPath(r['object_mesh'])
        if p.is_absolute() or '..' in p.parts or p.parts[:2]!=('tools','object_meshes'):
            raise ValueError('unexpected mesh reference')
        if str(p) not in cache:cache[str(p)]=vertices(STAGE/str(p))
        n=cache[str(p)]; contact=r['object_contact_shape'][1]
        rows.append({'sequence':r['member'],'mesh':str(p),'ply_vertices':n,
                     'contact_array_width':contact,'counts_match':n==contact})
    OUT.mkdir(mode=0o700)
    with (OUT/'rows.jsonl').open('x',encoding='utf8') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    result={'sequences':len(rows),'distinct_meshes':len(cache),
            'matched':sum(r['counts_match'] for r in rows),
            'mismatched':[r for r in rows if not r['counts_match']],
            'limitations':['Equality of counts does not prove matching vertex order or geometric registration.',
                           'Only S2; other subjects contact fields not audited here.',
                           'No contact-force, physical dynamics, retargeting or training validation.']}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()

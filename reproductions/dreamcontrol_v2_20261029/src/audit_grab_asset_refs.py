"""Read-only asset-reference audit for verified GRAB archives; trusted inputs only."""
import collections
import io
import json
from pathlib import Path, PurePosixPath
import pickle
import zipfile
import numpy as np
from grab_restricted_reader import read_object_npy

BASE=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
ARCHIVES=BASE/'data/GRAB'
STAGE=BASE/'grab_mesh_stage_20261030'
OUT=BASE/'grab_asset_refs_audit_20261030'


def safe_path(text):
    p=PurePosixPath(text)
    if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0]!='tools':
        raise ValueError('unsafe asset path')
    return p


def field(z, key):
    info=z.getinfo(key+'.npy')
    if info.file_size > 128*1024**2:
        raise ValueError('object field exceeds bound')
    return read_object_npy(z.read(info))


def main():
    if not (STAGE/'manifest.json').is_file(): raise ValueError('missing staged asset manifest')
    OUT.mkdir(mode=0o700)
    rows=[]; failures=[]; status=collections.Counter(); objects=collections.Counter(); subjects=collections.Counter()
    for subject in range(1,11):
        archive=ARCHIVES/f'grab__s{subject}.zip'
        with zipfile.ZipFile(archive) as z:
            members=sorted(i for i in z.namelist() if i.endswith('.npz'))
            for idx, member in enumerate(members):
                try:
                    if PurePosixPath(member).parts[0] != f's{subject}': raise ValueError('subject path mismatch')
                    info=z.getinfo(member)
                    if info.file_size>128*1024**2: raise ValueError('outer sequence exceeds bound')
                    with zipfile.ZipFile(io.BytesIO(z.read(info))) as nested:
                        gender=str(np.load(io.BytesIO(nested.read('gender.npy')),allow_pickle=False).item())
                        sid=str(np.load(io.BytesIO(nested.read('sbj_id.npy')),allow_pickle=False).item())
                        body=field(nested,'body'); obj=field(nested,'object')
                        template=safe_path(body['vtemp']); mesh=safe_path(obj['object_mesh'])
                        if sid!=f's{subject}' or gender not in ('male','female'):
                            raise ValueError('subject or gender mismatch')
                        if template != PurePosixPath(f'tools/subject_meshes/{gender}/{sid}.ply'):
                            raise ValueError('subject template reference mismatch')
                        template_exists=(STAGE/str(template)).is_file()
                        mesh_exists=(STAGE/str(mesh)).is_file()
                        objects[str(mesh)]+=1; subjects[sid]+=1
                        row={'member':member,'subject':sid,'gender':gender,'template':str(template),'object':str(mesh),
                             'template_exists':template_exists,'object_exists':mesh_exists}
                        rows.append(row);status['inspected']+=1
                        if not template_exists: status['missing_template']+=1
                        if not mesh_exists:status['missing_object']+=1
                except (ValueError,KeyError,TypeError,IndexError,OSError,EOFError,pickle.UnpicklingError,zipfile.BadZipFile) as exc:
                    failures.append({'member':member,'type':type(exc).__name__,'error':str(exc)})
            print(f'subject {subject} inspected={len(members)} cumulative_ok={len(rows)} errors={len(failures)}',flush=True)
    with (OUT/'references.jsonl').open('x',encoding='utf8') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    result={'counts':dict(status),'subjects':dict(subjects),'distinct_object_references':len(objects),
            'objects':dict(objects),'failures':failures,
            'limitations':['Reader reduces but does not eliminate pickle risks; only verified trusted GRAB inputs.',
                           'File existence is not shape/topology or vertex-index correspondence.',
                           'This is not robot retargeting, physical validation or license approval.']}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print('FINISHED '+json.dumps({'counts':dict(status),'failures':len(failures)}),flush=True)

if __name__=='__main__':main()

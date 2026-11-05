"""Re-read GRAB source, output and sampled model joints; not a physical audit."""
import argparse
import contextlib
import hashlib
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import smplx
import torch
import trimesh
from grab_restricted_reader import read_grab_field

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
MODEL = Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead')
REGISTRY = ROOT/'grab_source_registry_20261030/source_registry.jsonl'
STAGE = ROOT/'grab_mesh_stage_20261030'
REGISTRY_SHA = '27d74b72fec830afa01cc05f788d921ef3d42fc1115aad08565d26604cdec8e4'
FIELDS = ('global_orient','body_pose','left_hand_pose','right_hand_pose','jaw_pose','leye_pose','reye_pose','expression','transl')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def asset(ref, digest):
    p=PurePosixPath(ref)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('invalid asset path')
    path=(STAGE/ref).resolve(strict=True)
    if not path.is_relative_to(STAGE.resolve()) or sha(path)!=digest:
        raise ValueError('asset hash mismatch')
    return path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--selection',type=Path,required=True)
    ap.add_argument('--batch',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    torch.set_num_threads(1)
    if shutil.disk_usage(ROOT).free<8*1024**3+64*1024**2:
        raise RuntimeError('disk safety threshold reached')
    selection=json.loads(a.selection.read_text())['members']
    report=json.loads((a.batch/'report.json').read_text())
    if (sha(REGISTRY)!=REGISTRY_SHA or report['registry_sha256']!=REGISTRY_SHA or
        sha(a.selection)!=report['selection_sha256'] or sha(a.batch/'records.jsonl')!=report['records_sha256']):
        raise ValueError('batch bindings differ')
    registry={r['member']:r for r in map(json.loads,REGISTRY.read_text().splitlines())}
    records=[json.loads(x) for x in (a.batch/'records.jsonl').read_text().splitlines()]
    if [r['member'] for r in records]!=selection or len(selection)!=report['requested']:
        raise ValueError('selection coverage mismatch')
    cases=[]
    archive_cache={}
    for r in records:
        if r['status']!='reconstructed_unreviewed':
            cases.append({'member':r['member'],'status':'source_batch_failed','error':r.get('error')})
            continue
        src=registry[r['member']]
        if r['source_member_sha256']!=src['member_sha256']:
            raise ValueError('member binding mismatch')
        archive=ROOT/'data/GRAB'/src['archive']
        if src['archive'] not in archive_cache:
            archive_cache[src['archive']]=sha(archive)
        if archive_cache[src['archive']]!=src['archive_sha256']:
            raise ValueError('archive hash mismatch')
        with zipfile.ZipFile(archive) as z:
            raw=z.read(r['member'])
            if hashlib.sha256(raw).hexdigest()!=src['member_sha256']:
                raise ValueError('member hash mismatch')
            body=read_grab_field(z,r['member'],'body')
            obj=read_grab_field(z,r['member'],'object')
            with np.load(io.BytesIO(raw),allow_pickle=False) as meta:
                fps=float(meta['framerate'].item());n=int(meta['n_frames'].item())
                gender=str(meta['gender'].item());components=int(meta['n_comps'].item())
        target=(a.batch/r['artifact']).resolve(strict=True)
        if not target.is_relative_to(a.batch.resolve()) or sha(target)!=r['artifact_sha256']:
            raise ValueError('artifact path or hash mismatch')
        with np.load(target,allow_pickle=False) as f:
            human=f['human_joints'].copy();frames=f['source_frames'].copy()
            trans=f['object_transl'].copy();orient=f['object_orient_rotvec'].copy();stored_fps=float(f['declared_fps'])
        if (human.shape!=(n,22,3) or not np.isfinite(human).all() or not np.array_equal(frames,np.arange(n))
            or fps!=stored_fps or fps!=src['fps_declared'] or n!=src['frames'] or gender!=src['gender']
            or not np.array_equal(trans,np.asarray(obj['params']['transl']))
            or not np.array_equal(orient,np.asarray(obj['params']['global_orient']))):
            raise ValueError('frame, clock declaration or object data mismatch')
        model_path=MODEL/'smplx'/('SMPLX_'+gender.upper()+'.npz')
        if sha(model_path)!=r['model_sha256']:
            raise ValueError('model hash mismatch')
        mesh=trimesh.load(str(asset(src['template'],src['template_sha256'])),process=False)
        asset(src['object_mesh'],src['object_mesh_sha256'])
        idx=np.array([0,n//2,n-1],dtype=int)
        params={k:torch.as_tensor(np.asarray(body['params'][k])[idx],dtype=torch.float32) for k in FIELDS}
        with contextlib.redirect_stdout(sys.stderr):
            model=smplx.create(str(MODEL),model_type='smplx',gender=gender,num_betas=10,
                num_pca_comps=components,batch_size=3,use_pca=True,ext='npz').cpu()
            if not np.array_equal(np.asarray(mesh.faces),np.asarray(model.faces)):
                raise ValueError('template topology mismatch')
            model.v_template.copy_(torch.as_tensor(np.asarray(mesh.vertices),dtype=model.v_template.dtype))
            with torch.no_grad():
                check=model(**params,return_verts=False).joints[:,:22].cpu().numpy()
        error=float(np.max(np.abs(check-human[idx])))
        if not np.isfinite(error) or error>1e-5:
            raise ValueError('source joint recomputation mismatch')
        cases.append({'member':r['member'],'status':'technical_sample_checked','source_sha256':src['member_sha256'],
                      'artifact_sha256':sha(target),'source_frames':n,'sampled_indices':idx.tolist(),
                      'max_sampled_joint_difference_m':error,'full_object_array_equal':True})
    out={'cases':cases,'batch_report_sha256':sha(a.batch/'report.json'),'registry_sha256':REGISTRY_SHA,
         'script_sha256':sha(Path(__file__)),'training_ready':False,
         'limitations':['Same source/body model and subject template, not independent physical truth.',
                        'Only three human frames recomputed per sequence; declared clock unverified.',
                        'Object arrays compared in full; contact fields, text and robot conversion unaccepted.']}
    with a.output.open('x',encoding='utf8') as f:
        json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps({'cases':len(cases),'checked':sum(c['status']=='technical_sample_checked' for c in cases),
                      'failed':sum(c['status']!='technical_sample_checked' for c in cases),'training_ready':False}))


if __name__=='__main__':
    main()

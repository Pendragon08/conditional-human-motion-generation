"""Reconstruct predetermined S5 source frames on CPU; no training acceptance."""
import contextlib,hashlib,io,json,sys,zipfile
from pathlib import Path,PurePosixPath
import numpy as np
import torch,smplx,trimesh
from grab_restricted_reader import read_grab_field

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent
MODEL=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead')
SOURCE='s5/cup_pour_1.npz'
REGISTRY=ROOT/'grab_source_registry_20261030/source_registry.jsonl'
REGISTRY_SHA='27d74b72fec830afa01cc05f788d921ef3d42fc1115aad08565d26604cdec8e4'

def digest(f):
    h=hashlib.sha256()
    for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def sha(p):
    with Path(p).open('rb') as f:return digest(f)

def main():
    torch.set_num_threads(1)
    if sha(REGISTRY)!=REGISTRY_SHA:raise ValueError('source registry changed')
    matched=[r for r in map(json.loads,REGISTRY.read_text().splitlines()) if r['member']==SOURCE]
    if len(matched)!=1:raise ValueError('source registry coverage')
    row=matched[0];archive=ROOT/'data/GRAB'/row['archive']
    if sha(archive)!=row['archive_sha256']:raise ValueError('archive changed')
    with zipfile.ZipFile(archive) as z:
        with z.open(SOURCE) as f:
            if digest(f)!=row['member_sha256']:raise ValueError('member changed')
        body=read_grab_field(z,SOURCE,'body');obj=read_grab_field(z,SOURCE,'object');contact=read_grab_field(z,SOURCE,'contact')
        with np.load(io.BytesIO(z.read(SOURCE)),allow_pickle=False) as d:
            fps=float(d['framerate'].item());n=int(d['n_frames'].item());gender=str(d['gender'].item());pca=int(d['n_comps'].item());subject=str(d['sbj_id'].item())
    if subject!='s5' or gender!=row['gender'] or fps!=row['fps_declared'] or n!=row['frames'] or pca!=24 or n<120:
        raise ValueError('metadata mismatch')
    stage=ROOT/'grab_mesh_stage_20261030';refs=(body['vtemp'],obj['object_mesh']);assets=[]
    for ref,expected in zip(refs,(row['template_sha256'],row['object_mesh_sha256'])):
        p=PurePosixPath(ref)
        if p.is_absolute() or '..' in p.parts:raise ValueError('unsafe asset')
        resolved=(stage/ref).resolve(strict=True)
        if not resolved.is_relative_to(stage.resolve()) or sha(resolved)!=expected:raise ValueError('asset hash changed')
        assets.append(resolved)
    if refs!=(row['template'],row['object_mesh']):raise ValueError('asset provenance mismatch')
    mesh=trimesh.load(str(assets[0]),process=False);vertices=np.asarray(mesh.vertices)
    if vertices.shape!=(10475,3) or not np.isfinite(vertices).all():raise ValueError('invalid template')
    start=n//2-60;frames=np.arange(start,start+120,dtype=np.int64)
    fields=('global_orient','body_pose','left_hand_pose','right_hand_pose','jaw_pose','leye_pose','reye_pose','expression','transl')
    tensors={k:torch.as_tensor(np.asarray(body['params'][k])[frames],dtype=torch.float32) for k in fields}
    with contextlib.redirect_stdout(sys.stderr):
        model=smplx.create(str(MODEL),model_type='smplx',gender=gender,num_betas=10,num_pca_comps=pca,batch_size=120,use_pca=True,ext='npz')
        if not np.array_equal(np.asarray(mesh.faces),np.asarray(model.faces)):raise ValueError('topology mismatch')
        model.v_template.copy_(torch.as_tensor(vertices,dtype=model.v_template.dtype))
        with torch.no_grad():human=model(**tensors,return_verts=False).joints[:,:22].cpu().numpy()
    if human.shape!=(120,22,3) or not np.isfinite(human).all():raise ValueError('invalid reconstructed joints')
    labels=np.asarray(contact['object'])[frames]
    if labels.shape!=(120,row['object_vertices']):raise ValueError('contact shape mismatch')
    right=np.any((labels==22)|((labels>=41)&(labels<=55)),axis=1);left=np.any((labels==21)|((labels>=26)&(labels<=40)),axis=1)
    trans=np.asarray(obj['params']['transl'])[frames];orient=np.asarray(obj['params']['global_orient'])[frames]
    if trans.shape!=(120,3) or orient.shape!=trans.shape or not (np.isfinite(trans).all() and np.isfinite(orient).all()):raise ValueError('invalid object timeline')
    artifact=HERE/'human_source.npz'
    with artifact.open('xb') as f:np.savez_compressed(f,human_joints=human,input_frame_indices=frames,input_fps=np.array(fps),object_transl=trans,object_orient_rotvec=orient,right_hand_labels=right,left_hand_labels=left)
    out={'archive_sha256':row['archive_sha256'],'source_member_sha256':row['member_sha256'],'registry_sha256':REGISTRY_SHA,'source_member':SOURCE,'source_frames':n,
         'gender':gender,'fps':fps,'selected_first_frame':int(frames[0]),'selected_last_frame':int(frames[-1]),
         'right_hand_labeled_frames':int(right.sum()),'left_hand_labeled_frames':int(left.sum()),'object_mesh':refs[1],
         'template_sha256':row['template_sha256'],'object_mesh_sha256':row['object_mesh_sha256'],
         'model_sha256':sha(MODEL/'smplx'/('SMPLX_'+gender.upper()+'.npz')),'script_sha256':sha(__file__),'artifact_sha256':sha(artifact),
         'warning':'One S5 middle-second cup-pour span. FPS is declared, not independently measured; locked-head SMPL-X equivalence unknown. Labels do not establish contact force; no robot or physics acceptance.'}
    with (HERE/'report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps(out),flush=True)

if __name__=='__main__':main()

"""Eight-frame CPU witness: subject template applied to an existing SMPL-X model."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import torch
import smplx
import trimesh
from grab_restricted_reader import read_grab_field

BASE=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
ARCHIVE=BASE/'data/GRAB/grab__s2.zip'
STAGE=BASE/'grab_mesh_stage_20261030'
MODEL_ROOT=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    torch.set_num_threads(1)
    expected='d77eb7d13cfb3e3d0b81f08ccf9d8d79320fa6541352e6e65173c56d4b01b9b6'
    if sha(ARCHIVE)!=expected:raise ValueError('archive hash differs from verified Subject 2')
    with zipfile.ZipFile(ARCHIVE) as z:
        member='s2/airplane_fly_1.npz';body=read_grab_field(z,member,'body')
        with np.load(io.BytesIO(z.read(member)),allow_pickle=False) as x:
            gender=str(x['gender'].item());ncomps=int(x['n_comps'].item());nframes=int(x['n_frames'].item())
        if gender!='male' or ncomps!=24 or body['vtemp']!='tools/subject_meshes/male/s2.ply':
            raise ValueError('unexpected metadata')
        frames=np.array([0,6,12,18,24,30,36,42])
        names=('global_orient','body_pose','left_hand_pose','right_hand_pose','jaw_pose','leye_pose','reye_pose','expression','transl')
        tensors={k:torch.as_tensor(np.asarray(body['params'][k])[frames],dtype=torch.float32) for k in names}
    template=STAGE/body['vtemp']; loaded=trimesh.load(str(template),process=False)
    vertices=np.asarray(loaded.vertices);faces=np.asarray(loaded.faces)
    if vertices.shape!=(10475,3) or not np.isfinite(vertices).all():raise ValueError('unexpected template vertices')
    with contextlib.redirect_stdout(sys.stderr):
        model=smplx.create(str(MODEL_ROOT),model_type='smplx',gender=gender,num_betas=10,num_pca_comps=ncomps,
                           batch_size=len(frames),use_pca=True,ext='npz')
        if not np.array_equal(faces,np.asarray(model.faces)):raise ValueError('template face indexing differs from model')
        with torch.no_grad():
            generic=model(**tensors,return_verts=False).joints.detach().cpu().numpy()
            model.v_template.copy_(torch.as_tensor(vertices,dtype=model.v_template.dtype))
            personalized=model(**tensors,return_verts=False).joints.detach().cpu().numpy()
    if personalized.shape!=generic.shape or not np.isfinite(personalized).all():raise ValueError('invalid joint output')
    delta=np.linalg.norm(personalized[:,:22]-generic[:,:22],axis=-1)
    print(json.dumps({'member':member,'subject_template_sha256':sha(template),'body_model_sha256':sha(MODEL_ROOT/'smplx/SMPLX_MALE.npz'),
        'archive_sha256':expected,'frames':frames.tolist(),'source_frames':nframes,'template_vertices':len(vertices),
        'template_faces':len(faces),'template_face_index_matches_model':True,'joints_shape':list(personalized.shape),
        'generic_vs_personalized_22_joint_displacement_mean_m':float(delta.mean()),
        'generic_vs_personalized_22_joint_displacement_max_m':float(delta.max()),
        'limitations':['Existing locked-head model may differ from official GRAB SMPL-X configuration.',
                       'Plumbing test for eight frames, not independent ground truth or validated full sequence.',
                       'No object contact, retargeting, dynamics, or v2 training.']},allow_nan=False))

if __name__=='__main__':main()

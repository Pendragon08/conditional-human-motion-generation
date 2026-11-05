"""CPU-only SMPL-X plumbing test with a generic pre-existing template.

The GRAB subject-specific v_template is missing. A pre-existing male
locked-head model with default shape is used solely to check eight frames of
data/model plumbing. The output is not GRAB-accurate data or robot retargeting.
"""
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
from grab_restricted_reader import read_grab_field


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def main():
    torch.set_num_threads(1)
    archive=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/data/GRAB/grab__s2.zip')
    model_root=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead')
    model_file=model_root/'smplx/SMPLX_MALE.npz'
    expected='d77eb7d13cfb3e3d0b81f08ccf9d8d79320fa6541352e6e65173c56d4b01b9b6'
    if digest(archive)!=expected: raise ValueError('archive hash mismatch')
    with zipfile.ZipFile(archive) as z:
        member='s2/airplane_fly_1.npz'
        body=read_grab_field(z,member,'body')
        with np.load(io.BytesIO(z.read(member)),allow_pickle=False) as x:
            gender=str(x['gender'].item()); ncomps=int(x['n_comps'].item()); frame_count=int(x['n_frames'].item())
        if gender!='male' or ncomps!=24: raise ValueError('unexpected gender or hand PCA count')
        params=body['params']; frames=np.array([0,6,12,18,24,30,36,42])
        names=('global_orient','body_pose','left_hand_pose','right_hand_pose','jaw_pose','leye_pose','reye_pose','expression','transl')
        tensors={name:torch.as_tensor(np.asarray(params[name])[frames],dtype=torch.float32) for name in names}
    with contextlib.redirect_stdout(sys.stderr):
        model=smplx.create(str(model_root),model_type='smplx',gender=gender,num_betas=10,num_pca_comps=ncomps,
                           batch_size=len(frames),use_pca=True,ext='npz')
        with torch.no_grad(): output=model(**tensors,return_verts=False)
    joints=output.joints.detach().cpu().numpy()
    if joints.ndim!=3 or joints.shape[0]!=len(frames) or joints.shape[1]<22 or joints.shape[2]!=3 or not np.isfinite(joints).all():
        raise ValueError('invalid SMPL-X joint output')
    print(json.dumps({'member':member,'archive_sha256':expected,'body_model_sha256':digest(model_file),
                      'body_model':'existing locked-head male SMPL-X, generic template and default shape',
                      'subject_template_reference':body['vtemp'],'subject_template_used':False,
                      'input_frames':frame_count,'sample_frame_indices':frames.tolist(),
                      'hand_pca_components':ncomps,'joint_output_shape':list(joints.shape),
                      'warning':'8-frame CPU smoke test, generic shape and locked-head model are approximations; no GRAB subject template or object meshes, no FK accuracy, retargeting, dynamics or training.'},allow_nan=False))


if __name__=='__main__': main()

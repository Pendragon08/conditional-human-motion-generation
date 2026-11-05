"""Recheck selected frames of the ACCAD 120-Hz reconstruction."""
import contextlib
import hashlib
import json
import sys
import zipfile
from pathlib import Path
import numpy as np
import smplx
import torch

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
MODEL=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead')
HUMANML=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D')
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    info=json.loads((HERE/'human_report.json').read_text());source=Path(info['source_candidate_path'])
    assert sha(source)==info['source_sha256'] and sha(HERE/'human_120hz.npz')==info['artifact_sha256']
    assert sha(HERE/'reconstruct_v2.py')==info['script_sha256']
    assert sha(MODEL/'smplx'/('SMPLX_'+info['gender'].upper()+'.npz'))==info['model_sha256']
    assert info['id'] in (HUMANML/'train.txt').read_text().splitlines()
    assert all(info['id'] not in (HUMANML/(s+'.txt')).read_text().splitlines() for s in ('val','test'))
    with zipfile.ZipFile(HUMANML/'texts.zip') as z:raw=z.read('texts/'+info['id']+'.txt')
    assert hashlib.sha256(raw).hexdigest()==info['caption_member_sha256']
    assert [line.split('#')[0] for line in raw.decode('utf8').splitlines()]==info['captions']
    with np.load(source,allow_pickle=False) as f:pose=f['poses'].copy();trans=f['trans'].copy();betas=f['betas'].copy()
    with np.load(HERE/'human_120hz.npz',allow_pickle=False) as f:
        frames=f['source_frames'].copy();times=f['source_times'].copy();saved=f['human'].copy()
    np.testing.assert_array_equal(frames,np.arange(355))
    np.testing.assert_allclose(times,frames/120.,atol=0,rtol=0)
    idx=np.array([0,177,354]);n=len(idx)
    with contextlib.redirect_stdout(sys.stderr):
        model=smplx.create(str(MODEL),model_type='smplx',gender=info['gender'],num_betas=16,
                           use_pca=False,batch_size=n,ext='npz').cpu()
    convert=lambda x:torch.tensor(np.asarray(x),dtype=torch.float32)
    zeros=lambda width:torch.zeros((n,width),dtype=torch.float32)
    with torch.no_grad():
        actual=model(betas=convert(np.broadcast_to(betas,(n,16)).copy()),global_orient=convert(pose[idx,:3]),
            body_pose=convert(pose[idx,3:66]),transl=convert(trans[idx]),left_hand_pose=zeros(45),
            right_hand_pose=zeros(45),jaw_pose=zeros(3),leye_pose=zeros(3),reye_pose=zeros(3),
            expression=zeros(model.num_expression_coeffs),return_verts=False).joints[:,:22].numpy()
    error=float(np.max(np.abs(actual-saved[idx])))
    assert error<1e-6
    result={'human_report_sha256':sha(HERE/'human_report.json'),'artifact_sha256':sha(HERE/'human_120hz.npz'),
            'source_sha256':sha(source),'source_frames_checked':idx.tolist(),'max_component_difference_m':error,
            'training_ready':False,'limitations':['Uses the same SMPL-X library; no official SMPL+H identity, true FPS, semantic or physical validation.']}
    with (HERE/'human_audit.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()

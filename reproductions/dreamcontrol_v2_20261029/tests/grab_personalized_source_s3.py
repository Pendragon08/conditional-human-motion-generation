"""Reconstruct one fixed GRAB S3 120-frame source span; not retargeting."""
import contextlib,hashlib,io,json,sys,zipfile
from pathlib import Path,PurePosixPath
import numpy as np
import torch,smplx,trimesh
from grab_restricted_reader import read_grab_field

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
MODEL=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    torch.set_num_threads(1)
    archive=ROOT/'data/GRAB/grab__s3.zip'; expected='c09458512f01e2c7415fb8a4d343e158041e0f8fa1bb8da08822a2bb30dd1a66'
    if sha(archive)!=expected:raise ValueError('source archive hash mismatch')
    member='s3/apple_eat_1.npz'
    with zipfile.ZipFile(archive) as z:
        body=read_grab_field(z,member,'body')
        with np.load(io.BytesIO(z.read(member)),allow_pickle=False) as x:
            fps=float(x['framerate'].item()); n=int(x['n_frames'].item()); gender=str(x['gender'].item());
            pca=int(x['n_comps'].item()); sid=str(x['sbj_id'].item())
    if sid!='s3' or gender not in ('male','female') or fps!=120 or pca!=24 or n<120:
        raise ValueError('unexpected source metadata')
    ref=PurePosixPath(body['vtemp'])
    if ref.is_absolute() or '..' in ref.parts or ref.parts[:2]!=('tools','subject_meshes') or ref.name!='s3.ply':
        raise ValueError('unexpected template path')
    template=ROOT/'grab_mesh_stage_20261030'/str(ref); mesh=trimesh.load(str(template),process=False)
    vertices=np.asarray(mesh.vertices)
    if vertices.shape!=(10475,3) or not np.isfinite(vertices).all():raise ValueError('invalid template')
    start=max(0,n//2-60); frames=np.arange(start,start+120,dtype=np.int64)
    keys=('global_orient','body_pose','left_hand_pose','right_hand_pose','jaw_pose','leye_pose','reye_pose','expression','transl')
    tensors={k:torch.as_tensor(np.asarray(body['params'][k])[frames],dtype=torch.float32) for k in keys}
    with contextlib.redirect_stdout(sys.stderr):
        model=smplx.create(str(MODEL),model_type='smplx',gender=gender,num_betas=10,num_pca_comps=pca,batch_size=120,use_pca=True,ext='npz')
        if not np.array_equal(np.asarray(mesh.faces),np.asarray(model.faces)):raise ValueError('face-index mismatch')
        model.v_template.copy_(torch.as_tensor(vertices,dtype=model.v_template.dtype))
        with torch.no_grad(): human=model(**tensors,return_verts=False).joints[:,:22].detach().cpu().numpy()
    if human.shape!=(120,22,3) or not np.isfinite(human).all():raise ValueError('invalid human joints')
    out=Path(__file__).with_name('human_source.npz')
    with out.open('xb') as f:np.savez_compressed(f,human_joints=human,input_frame_indices=frames,input_fps=np.array(fps))
    doc={'archive_sha256':expected,'source_member':member,'source_frames':n,'gender':gender,'fps':fps,
         'selected_first_frame':int(frames[0]),'selected_last_frame':int(frames[-1]),
         'template_sha256':sha(template),'model_sha256':sha(MODEL/'smplx'/('SMPLX_'+gender.upper()+'.npz')),
         'script_sha256':sha(__file__),'artifact_sha256':sha(out),
         'warning':'One predetermined middle-second S3 apple sequence. Existing locked-head SMPL-X version may differ from GRAB original. No robot retargeting, object contact, dynamics or data acceptance.'}
    with Path(__file__).with_name('report.json').open('x',encoding='utf8') as f:json.dump(doc,f,indent=2,allow_nan=False)
    print(json.dumps(doc),flush=True)

if __name__=='__main__':main()

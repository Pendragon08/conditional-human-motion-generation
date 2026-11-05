"""Independently verify selected S4 event geometry from the original inputs."""
import contextlib
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path
import numpy as np
import smplx
import torch
import trimesh
from scipy.spatial.distance import cdist
from grab_restricted_reader import read_grab_field
from grab_multisubject_contact_witness import ROOT, MODEL, REGISTRY, safe_asset, sha

def row_rotation(v):
    angle=np.linalg.norm(v)
    if angle==0:return np.eye(3)
    x,y,z=v/angle;K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*(K@K)
def main():
    torch.set_num_threads(1); report=Path(sys.argv[1]);npz=Path(sys.argv[2]);r=json.loads(report.read_text());assert sha(npz)==r['artifact_sha256'] and sha(REGISTRY)==r['registry_sha256']
    rows=[x for x in map(json.loads,REGISTRY.read_text().splitlines()) if x['member']==r['member']];assert len(rows)==1
    row=rows[0];archive=ROOT/'data/GRAB'/row['archive'];assert sha(archive)==r['archive_sha256']==row['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        raw=z.read(row['member']);assert hashlib.sha256(raw).hexdigest()==r['member_sha256']
        body=read_grab_field(z,row['member'],'body');obj=read_grab_field(z,row['member'],'object');label=read_grab_field(z,row['member'],'contact')['object']
        with np.load(io.BytesIO(raw),allow_pickle=False) as meta:gender=str(meta['gender'].item());pca=int(meta['n_comps'].item())
    assert sha(MODEL/'smplx'/('SMPLX_'+gender.upper()+'.npz'))==r['model_sha256']
    bodymesh=trimesh.load(str(safe_asset(row['template'],r['template_sha256'])),process=False); mesh=trimesh.load(str(safe_asset(row['object_mesh'],r['object_mesh_sha256'])),process=False)
    with np.load(npz,allow_pickle=False) as d:frames=d['frames'];human=d['human'];rot=d['object_rotvec'];trans=d['object_translation']
    assert list(frames)==list(range(180,195)) and np.array_equal(rot,np.asarray(obj['params']['global_orient'])[frames]) and np.array_equal(trans,np.asarray(obj['params']['transl'])[frames])
    frame=185; i=list(frames).index(frame);keys=('global_orient','body_pose','left_hand_pose','right_hand_pose','jaw_pose','leye_pose','reye_pose','expression','transl')
    inputs={k:torch.as_tensor(np.asarray(body['params'][k])[[frame]],dtype=torch.float32) for k in keys}
    with contextlib.redirect_stdout(sys.stderr):
        model=smplx.create(str(MODEL),model_type='smplx',gender=gender,num_betas=10,num_pca_comps=pca,batch_size=1,use_pca=True,ext='npz')
        assert np.array_equal(bodymesh.faces,model.faces);model.v_template.copy_(torch.as_tensor(np.asarray(bodymesh.vertices),dtype=model.v_template.dtype))
        with torch.no_grad():h=model(**inputs,return_verts=True).vertices.cpu().numpy()[0]
    forward_error=float(np.max(np.abs(h-human[i])));assert forward_error<1e-5
    checked=0;maxerr=0.
    for i,frame in enumerate(frames):
        a=label[frame];right=np.isin(a,[22,*range(41,56)]);left=np.isin(a,[21,*range(26,41)]);record=r['frames'][i]
        assert int(frame)==record['frame'] and right.sum()==record['right_count'] and left.sum()==record['left_count']
        if frame in (183,185,186,187,189) and right.any():
            world=np.asarray(mesh.vertices)[right]@row_rotation(rot[i])+trans[i]; chunks=[]
            for start in range(0,len(world),32):chunks.extend(cdist(world[start:start+32],human[i]).min(axis=1))
            values=np.asarray(chunks);err=max(abs(float(values.min())-record['right_min_m']),abs(float(np.median(values))-record['right_median_m']),abs(float(values.max())-record['right_max_m']))
            maxerr=max(maxerr,err);assert err<1e-8;checked+=1
    out={'status':'passed','report_sha256':sha(report),'artifact_sha256':sha(npz),'frames_checked':len(frames),'frames_brute_force_checked':checked,'max_nearest_distance_error_m':maxerr,'frame_185_smplx_forward_max_abs_m':forward_error,'audit_sha256':sha(Path(__file__)),'limitations':'Same SMPL-X library and locked-head model, one independently reconstructed frame. Other source frames use saved human vertices; brute force checks labeled object points only. No real contact, physics or FPS verification.'}
    with Path(sys.argv[3]).open('x',encoding='utf8') as f:json.dump(out,f,indent=2)
if __name__=='__main__':main()

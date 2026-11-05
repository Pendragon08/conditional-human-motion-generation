"""Inspect human-model geometry around one short GRAB label run on CPU."""
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
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from grab_restricted_reader import read_grab_field
from grab_multisubject_contact_witness import ROOT, MODEL, REGISTRY, REGISTRY_HASH, safe_asset, sha

MEMBER='s4/waterbottle_drink_1.npz'
FRAMES=np.arange(180,195)
def main():
    torch.set_num_threads(1)
    assert sha(REGISTRY)==REGISTRY_HASH
    rows=[r for r in map(json.loads,REGISTRY.read_text().splitlines()) if r['member']==MEMBER];assert len(rows)==1
    row=rows[0];archive=ROOT/'data/GRAB'/row['archive'];assert sha(archive)==row['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        raw=z.read(MEMBER);assert hashlib.sha256(raw).hexdigest()==row['member_sha256']
        body=read_grab_field(z,MEMBER,'body');obj=read_grab_field(z,MEMBER,'object');contact=read_grab_field(z,MEMBER,'contact')
        with np.load(io.BytesIO(raw),allow_pickle=False) as d:
            fps=float(d['framerate'].item()); gender=str(d['gender'].item()); pca=int(d['n_comps'].item());frames=int(d['n_frames'].item())
    assert fps==row['fps_declared']==120 and frames==row['frames'] and gender==row['gender']
    assert body['vtemp']==row['template'] and obj['object_mesh']==row['object_mesh']
    template=trimesh.load(str(safe_asset(row['template'],row['template_sha256'])),process=False)
    mesh=trimesh.load(str(safe_asset(row['object_mesh'],row['object_mesh_sha256'])),process=False)
    vertices=np.asarray(mesh.vertices);labels=np.asarray(contact['object']); assert labels.shape==(frames,len(vertices))
    fields=('global_orient','body_pose','left_hand_pose','right_hand_pose','jaw_pose','leye_pose','reye_pose','expression','transl')
    params={k:torch.as_tensor(np.asarray(body['params'][k])[FRAMES],dtype=torch.float32) for k in fields}
    with contextlib.redirect_stdout(sys.stderr):
        model=smplx.create(str(MODEL),model_type='smplx',gender=gender,num_betas=10,num_pca_comps=pca,batch_size=len(FRAMES),use_pca=True,ext='npz')
        assert np.array_equal(template.faces,model.faces)
        model.v_template.copy_(torch.as_tensor(np.asarray(template.vertices),dtype=model.v_template.dtype))
        with torch.no_grad():human=model(**params,return_verts=True).vertices.cpu().numpy()
    rot=np.asarray(obj['params']['global_orient'])[FRAMES];trans=np.asarray(obj['params']['transl'])[FRAMES]
    records=[]
    for i,frame in enumerate(FRAMES):
        world=Rotation.from_rotvec(rot[i]).inv().apply(vertices)+trans[i]
        right=np.isin(labels[frame],[22,*range(41,56)]); left=np.isin(labels[frame],[21,*range(26,41)])
        dist=cKDTree(human[i]).query(world,workers=1)[0];d=dist[right]
        records.append({'frame':int(frame),'right_count':int(right.sum()),'left_count':int(left.sum()),'right_min_m':float(d.min()) if len(d) else None,'right_max_m':float(d.max()) if len(d) else None,'right_median_m':float(np.median(d)) if len(d) else None,'all_object_min_to_body_vertex_m':float(dist.min())})
    artifact=Path(sys.argv[1])
    with artifact.open('xb') as f:np.savez_compressed(f,frames=FRAMES,human=human,object_rotvec=rot,object_translation=trans)
    result={'member':MEMBER,'member_sha256':row['member_sha256'],'archive_sha256':row['archive_sha256'],'registry_sha256':REGISTRY_HASH,'template_sha256':row['template_sha256'],'object_mesh_sha256':row['object_mesh_sha256'],'model_sha256':sha(MODEL/'smplx'/('SMPLX_'+gender.upper()+'.npz')),'fps_declared':fps,'frames':records,'artifact_sha256':sha(artifact),'script_sha256':sha(Path(__file__)),'reader_sha256':sha(Path(sys.modules['grab_restricted_reader'].__file__)),'limitations':'15 source frames; fitted locked-head model; object vertices to nearest body vertices, not point-surface distance. Labels, real sampling rate, physical contact, topology and force are unverified. No robot, dynamics or training validation.'}
    with Path(sys.argv[2]).open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
if __name__=='__main__':main()

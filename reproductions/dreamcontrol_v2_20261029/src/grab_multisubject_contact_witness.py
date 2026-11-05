"""CPU-only human/object labeled-vertex geometry, not physical acceptance."""
import contextlib, hashlib, io, json, sys, zipfile
from pathlib import Path, PurePosixPath
import numpy as np
import smplx
import torch
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from grab_restricted_reader import read_grab_field

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
OUT=Path(__file__).resolve().parent
STAGE=ROOT/'grab_mesh_stage_20261030'
MODEL=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead')
REGISTRY=ROOT/'grab_source_registry_20261030/source_registry.jsonl'
REGISTRY_HASH='27d74b72fec830afa01cc05f788d921ef3d42fc1115aad08565d26604cdec8e4'
MEMBERS=('s1/teapot_pour_1.npz','s3/apple_eat_1.npz','s4/waterbottle_drink_1.npz','s5/cup_pour_1.npz')

def digest_stream(stream):
    h=hashlib.sha256()
    for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def sha(path):
    with Path(path).open('rb') as f:return digest_stream(f)

def safe_asset(name,expected):
    ref=PurePosixPath(name)
    if ref.is_absolute() or '..' in ref.parts:raise ValueError('unsafe asset')
    path=(STAGE/name).resolve(strict=True)
    if not path.is_relative_to(STAGE.resolve()) or sha(path)!=expected:raise ValueError('asset hash mismatch')
    return path

def summary(x):
    if len(x)==0:return None
    return {'count':int(len(x)),'median_m':float(np.median(x)),
            'p95_m':float(np.percentile(x,95)),'max_m':float(np.max(x))}

def main():
    torch.set_num_threads(1)
    if sha(REGISTRY)!=REGISTRY_HASH:raise ValueError('registry hash mismatch')
    records={r['member']:r for r in map(json.loads,REGISTRY.read_text().splitlines()) if r['member'] in MEMBERS}
    if set(records)!=set(MEMBERS):raise ValueError('selected source coverage')
    results=[]
    for member in MEMBERS:
        r=records[member];archive=ROOT/'data/GRAB'/r['archive']
        if sha(archive)!=r['archive_sha256']:raise ValueError('archive changed')
        with zipfile.ZipFile(archive) as z:
            with z.open(member) as f:
                if digest_stream(f)!=r['member_sha256']:raise ValueError('source member changed')
            body=read_grab_field(z,member,'body');obj=read_grab_field(z,member,'object');contact=read_grab_field(z,member,'contact')
            with np.load(io.BytesIO(z.read(member)),allow_pickle=False) as meta:
                gender=str(meta['gender'].item());pca=int(meta['n_comps'].item());nframes=int(meta['n_frames'].item())
        if gender!=r['gender'] or nframes!=r['frames'] or body['vtemp']!=r['template'] or obj['object_mesh']!=r['object_mesh']:
            raise ValueError('source record mismatch')
        template=safe_asset(r['template'],r['template_sha256']);mesh=safe_asset(r['object_mesh'],r['object_mesh_sha256'])
        body_mesh=trimesh.load(str(template),process=False);object_mesh=trimesh.load(str(mesh),process=False)
        vertices=np.asarray(object_mesh.vertices);body_vertices=np.asarray(body_mesh.vertices)
        labels=np.asarray(contact['object'])
        if labels.shape!=(nframes,len(vertices)) or labels.dtype.kind not in 'biu' or len(vertices)!=r['object_vertices']:
            raise ValueError('object contact shape mismatch')
        candidates=np.flatnonzero((labels!=0).sum(axis=1)>=50)
        if len(candidates)<5:raise ValueError('insufficient object-labeled frames: '+member)
        selected=candidates[np.linspace(0,len(candidates)-1,5,dtype=int)]
        keys=('global_orient','body_pose','left_hand_pose','right_hand_pose','jaw_pose','leye_pose','reye_pose','expression','transl')
        params={k:torch.as_tensor(np.asarray(body['params'][k])[selected],dtype=torch.float32) for k in keys}
        model_path=MODEL/'smplx'/('SMPLX_'+gender.upper()+'.npz')
        with contextlib.redirect_stdout(sys.stderr):
            model=smplx.create(str(MODEL),model_type='smplx',gender=gender,num_betas=10,num_pca_comps=pca,
                               batch_size=len(selected),use_pca=True,ext='npz')
            if not np.array_equal(np.asarray(body_mesh.faces),np.asarray(model.faces)):raise ValueError('template faces mismatch')
            model.v_template.copy_(torch.as_tensor(body_vertices,dtype=model.v_template.dtype))
            with torch.no_grad():human=model(**params,return_verts=True).vertices.cpu().numpy()
        rot=np.asarray(obj['params']['global_orient'])[selected];trans=np.asarray(obj['params']['transl'])[selected]
        frame_rows=[]
        for i,frame in enumerate(selected):
            # Official ObjectModel uses row-vector vertex @ R + translation.
            world=Rotation.from_rotvec(rot[i]).inv().apply(vertices)+trans[i]
            distances=cKDTree(human[i]).query(world,workers=1)[0];positive=labels[frame]!=0
            frame_rows.append({'frame':int(frame),'seconds_if_declared_fps_correct':float(frame/r['fps_declared']),
                               'labeled_object_vertices_to_nearest_body_vertex':summary(distances[positive]),
                               'unlabeled_object_vertices_to_nearest_body_vertex':summary(distances[~positive])})
        results.append({'member':member,'member_sha256':r['member_sha256'],'gender':gender,'declared_fps':r['fps_declared'],
                        'template_sha256':r['template_sha256'],'object_mesh_sha256':r['object_mesh_sha256'],
                        'model_sha256':sha(model_path),'source_frames_with_at_least_50_labeled_vertices':int(len(candidates)),
                        'frames':frame_rows})
        print(json.dumps({'completed':member,'sampled_frames':len(selected)}),flush=True)
    report={'registry_sha256':REGISTRY_HASH,'script_sha256':sha(__file__),'reader_sha256':sha(OUT/'grab_restricted_reader.py'),
            'samples':results,'limitations':'Only four preselected subject/object motions and five label-bearing frames per sequence. Locked-head model and contact vertex topology equivalence are not proven. Distances are object vertices to nearest body vertices, not surface distance, contact force, true ground, real FPS, dynamics, retargeting or training acceptance. Reader is limited to trusted archives.'}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({'complete':True,'sequences':len(results)}),flush=True)

if __name__=='__main__':main()

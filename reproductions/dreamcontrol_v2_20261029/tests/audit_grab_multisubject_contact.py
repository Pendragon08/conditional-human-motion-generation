"""Recheck human-object witness with row-vector matrices and brute-force probes."""
import contextlib, hashlib, io, json, sys, zipfile
from pathlib import Path
import numpy as np
import smplx, torch, trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from grab_restricted_reader import read_grab_field

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
OUT=Path(__file__).resolve().parent
STAGE=ROOT/'grab_mesh_stage_20261030'
MODEL=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead')
EXPECTED=('s1/teapot_pour_1.npz','s3/apple_eat_1.npz','s4/waterbottle_drink_1.npz','s5/cup_pour_1.npz')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    torch.set_num_threads(1)
    report_path=OUT/'report.json';report=json.loads(report_path.read_text())
    if sha(OUT/'grab_multisubject_contact_witness.py')!=report['script_sha256'] or sha(OUT/'grab_restricted_reader.py')!=report['reader_sha256']:
        raise ValueError('source code changed')
    reg=ROOT/'grab_source_registry_20261030/source_registry.jsonl'
    if sha(reg)!=report['registry_sha256'] or report['registry_sha256']!='27d74b72fec830afa01cc05f788d921ef3d42fc1115aad08565d26604cdec8e4':raise ValueError('registry changed')
    records={r['member']:r for r in map(json.loads,reg.read_text().splitlines()) if r['member'] in EXPECTED}
    if tuple(x['member'] for x in report['samples'])!=EXPECTED:raise ValueError('sample set mismatch')
    max_stat=max_probe=max_transform=0.;count_frames=0;summary=[]
    for entry in report['samples']:
        name=entry['member'];r=records[name];archive=ROOT/'data/GRAB'/r['archive']
        if sha(archive)!=r['archive_sha256'] or entry['member_sha256']!=r['member_sha256']:raise ValueError('source mismatch')
        with zipfile.ZipFile(archive) as z:
            body=read_grab_field(z,name,'body');obj=read_grab_field(z,name,'object');labels=np.asarray(read_grab_field(z,name,'contact')['object'])
            with np.load(io.BytesIO(z.read(name)),allow_pickle=False) as meta:pca=int(meta['n_comps'].item())
        template=STAGE/r['template'];mesh=STAGE/r['object_mesh'];model_path=MODEL/'smplx'/('SMPLX_'+entry['gender'].upper()+'.npz')
        if sha(template)!=r['template_sha256'] or sha(mesh)!=r['object_mesh_sha256'] or sha(model_path)!=entry['model_sha256']:
            raise ValueError('asset hash mismatch')
        bm=trimesh.load(str(template),process=False);om=trimesh.load(str(mesh),process=False)
        verts=np.asarray(om.vertices);n=r['frames']
        if labels.shape!=(n,len(verts)):raise ValueError('shape mismatch')
        candidates=np.flatnonzero((labels!=0).sum(axis=1)>=50)
        picked=candidates[np.linspace(0,len(candidates)-1,5,dtype=int)]
        if [int(x) for x in picked]!=[x['frame'] for x in entry['frames']]:raise ValueError('frame selection mismatch')
        keys=('global_orient','body_pose','left_hand_pose','right_hand_pose','jaw_pose','leye_pose','reye_pose','expression','transl')
        parms={k:torch.as_tensor(np.asarray(body['params'][k])[picked],dtype=torch.float32) for k in keys}
        with contextlib.redirect_stdout(sys.stderr):
            model=smplx.create(str(MODEL),model_type='smplx',gender=entry['gender'],num_betas=10,num_pca_comps=pca,batch_size=5,use_pca=True,ext='npz')
            if not np.array_equal(np.asarray(bm.faces),np.asarray(model.faces)):raise ValueError('faces mismatch')
            model.v_template.copy_(torch.as_tensor(np.asarray(bm.vertices),dtype=model.v_template.dtype))
            with torch.no_grad():human=model(**parms,return_verts=True).vertices.cpu().numpy()
        medians=[]
        for i,frame in enumerate(picked):
            # Independent expression of official row-vector rotation.
            rot=Rotation.from_rotvec(np.asarray(obj['params']['global_orient'])[frame]).as_matrix()
            trans=np.asarray(obj['params']['transl'])[frame]
            world=verts@rot+trans
            roundtrip=(world-trans)@rot.T
            max_transform=max(max_transform,float(np.max(np.abs(roundtrip-verts))))
            dist=cKDTree(human[i]).query(world,workers=1)[0];mask=labels[frame]!=0
            for label,selected in (('labeled_object_vertices_to_nearest_body_vertex',mask),('unlabeled_object_vertices_to_nearest_body_vertex',~mask)):
                arr=dist[selected];record=entry['frames'][i][label]
                if len(arr)!=record['count']:raise ValueError('count mismatch')
                for key,observed in (('median_m',np.median(arr)),('p95_m',np.percentile(arr,95)),('max_m',np.max(arr))):
                    max_stat=max(max_stat,abs(float(observed)-record[key]))
            medians.append(float(np.median(dist[mask])))
            points=np.flatnonzero(mask);probes=points[np.linspace(0,len(points)-1,min(16,len(points)),dtype=int)]
            for p in probes:
                direct=np.sqrt(np.min(np.sum((human[i]-world[p])**2,axis=1)))
                max_probe=max(max_probe,abs(float(direct)-float(dist[p])))
            count_frames+=1
        summary.append({'member':name,'median_range_m':[min(medians),max(medians)]})
        print(json.dumps({'verified':name}),flush=True)
    if max_stat>1e-8 or max_probe>1e-8 or max_transform>1e-8:raise ValueError('independent geometry mismatch')
    out={'script_sha256':sha(__file__),'source_report_sha256':sha(report_path),'verified_frames':count_frames,'sequence_summaries':summary,
         'max_stat_difference_m':max_stat,'max_brute_probe_difference_m':max_probe,'max_transform_roundtrip_error_m':max_transform,
         'limitations':'Same locked-head SMPL-X and assets as witness; numerical and provenance check, not independent ground-truth body mesh, real FPS, topology, physics, contact force or robot acceptance.'}
    with (OUT/'independent_audit.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps(out),flush=True)

if __name__=='__main__':main()

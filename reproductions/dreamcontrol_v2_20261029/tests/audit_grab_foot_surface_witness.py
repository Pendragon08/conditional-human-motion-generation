"""Recompute 12 selected human foot-proxy frames and provenance independently."""
import contextlib,hashlib,io,json,sys,zipfile
from pathlib import Path
import numpy as np
import smplx,torch,trimesh
from grab_restricted_reader import read_grab_field

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');OUT=Path(__file__).resolve().parent
STAGE=ROOT/'grab_mesh_stage_20261030'
MODEL=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead')
REGISTRY=ROOT/'grab_source_registry_20261030/source_registry.jsonl'
INPUTS={'s1/teapot_pour_1.npz':'grab_s1_pour_human_source_20261030/human_source.npz',
        's2/airplane_fly_1.npz':'grab_human_robot_pilot_20261030/pilot_arrays.npz',
        's3/apple_eat_1.npz':'grab_s3_human_source_20261030/human_source.npz',
        's4/waterbottle_drink_1.npz':'grab_s4_drink_human_source_20261030/human_source.npz'}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    torch.set_num_threads(1);report_path=OUT/'report.json';report=json.loads(report_path.read_text())
    if sha(REGISTRY)!=report['registry_sha256'] or sha(OUT/'grab_restricted_reader.py')!=report['reader_sha256'] or sha(OUT/'audit_grab_human_foot_surface_proxy.py')!=report['script_sha256']:
        raise ValueError('provenance mismatch')
    rows={r['member']:r for r in map(json.loads,REGISTRY.read_text().splitlines()) if r['member'] in INPUTS}
    if {r['member'] for r in report['subjects']}!=set(INPUTS):raise ValueError('subject mismatch')
    errors=[]
    for rec in report['subjects']:
        member=rec['member'];row=rows[member];archive=ROOT/'data/GRAB'/row['archive'];source=ROOT/INPUTS[member]
        if sha(archive)!=row['archive_sha256'] or sha(source)!=rec['source_npz_sha256'] or rec['member_sha256']!=row['member_sha256']:
            raise ValueError('source changed')
        with np.load(source,allow_pickle=False) as d:frame_ids=d['input_frame_indices'];joints_reference=d['human_joints'];fps=float(d['input_fps'])
        frames=[r['frame'] for r in rec['frames']]
        if frames!=frame_ids[::6].tolist() or len(frames)!=20:raise ValueError('frame selection mismatch')
        for foot in ('left','right'):
            values=[x[foot+'_skinning_region_min_z_m'] for x in rec['frames']]
            if [min(values),max(values)]!=rec[foot+'_min_z_range_m']:raise ValueError('summary mismatch')
        with zipfile.ZipFile(archive) as z:
            body=read_grab_field(z,member,'body')
            with np.load(io.BytesIO(z.read(member)),allow_pickle=False) as d:gender=str(d['gender'].item());pca=int(d['n_comps'].item())
        template=STAGE/row['template'];modelpath=MODEL/'smplx'/('SMPLX_'+gender.upper()+'.npz')
        if sha(template)!=rec['template_sha256'] or sha(modelpath)!=rec['model_sha256']:raise ValueError('asset mismatch')
        mesh=trimesh.load(str(template),process=False);sample=np.array([frames[i] for i in (0,10,19)])
        keys=('global_orient','body_pose','left_hand_pose','right_hand_pose','jaw_pose','leye_pose','reye_pose','expression','transl')
        args={k:torch.as_tensor(np.asarray(body['params'][k])[sample],dtype=torch.float32) for k in keys}
        with contextlib.redirect_stdout(sys.stderr):
            model=smplx.create(str(MODEL),model_type='smplx',gender=gender,num_betas=10,num_pca_comps=pca,batch_size=3,use_pca=True,ext='npz')
            if not np.array_equal(np.asarray(mesh.faces),np.asarray(model.faces)):raise ValueError('topology mismatch')
            model.v_template.copy_(torch.as_tensor(np.asarray(mesh.vertices),dtype=model.v_template.dtype))
            weights=model.lbs_weights.cpu().numpy()
            indices=[np.flatnonzero(weights[:,7]+weights[:,10]>=.5),np.flatnonzero(weights[:,8]+weights[:,11]>=.5)]
            if [len(x) for x in indices]!=rec['skinning_region_vertex_counts']:raise ValueError('mask mismatch')
            with torch.no_grad():out=model(**args,return_verts=True);verts=out.vertices.cpu().numpy();joints=out.joints[:,:22].cpu().numpy()
        err_j=float(np.max(np.abs(joints-joints_reference[[0,60,114]])))
        max_z=max(abs(float(verts[k,ids,2].min())-rec['frames'][i][side+'_skinning_region_min_z_m'])
                  for k,i in enumerate((0,10,19)) for side,ids in zip(('left','right'),indices))
        if err_j>2e-5 or max_z>2e-5:raise ValueError('reconstruction mismatch')
        errors.append({'member':member,'frames':sample.tolist(),'max_z_difference_m':max_z,'max_joint_difference_m':err_j})
    result={'script_sha256':sha(__file__),'source_report_sha256':sha(report_path),'independent_recomputed_frames':12,'checks':errors,
            'limitations':'Same approximate body model and weight-based anatomical proxy; this checks arithmetic/inputs, not foot segmentation correctness, ground height, contact force, source FPS or physical validation.'}
    with (OUT/'independent_audit.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()

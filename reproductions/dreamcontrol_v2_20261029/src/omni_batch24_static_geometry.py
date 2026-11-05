"""Batch static geometry diagnostics, not physics or training acceptance.

The fixed MuJoCo robot, default hands and assumed plane are not the source
scene; in particular the source boxes are not represented in this model.
"""
import hashlib,json
from pathlib import Path
import mujoco,numpy as np
from measure_eight_static_robot_quality import SCENE,MODEL_XML,asset_hashes,mid
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
SOURCE=ROOT/'omni_gpu_batch24_20261101'; REPAIRS=ROOT/'omni_batch24_projection_20261101'
HERE=Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    r=json.loads((SOURCE/'report.json').read_text());audit=json.loads((SOURCE/'audit_v2.json').read_text())
    repair=json.loads((REPAIRS/'report.json').read_text())
    assert audit['report_sha256']==sha(SOURCE/'report.json') and repair['source_audit_sha256']==sha(SOURCE/'audit_v2.json')
    assert r['records_sha256']==sha(SOURCE/'records.jsonl')
    records=[json.loads(x) for x in (SOURCE/'records.jsonl').read_text().splitlines()]
    corrections={x['sample_id']:x for x in repair['cases'] if x['action']=='bounded_projection_preview'}
    if len(records)!=24 or len(corrections)!=3:raise ValueError('unexpected batch coverage')
    model=mujoco.MjModel.from_xml_path(str(SCENE));data=mujoco.MjData(model)
    root=mid(model,mujoco.mjtObj.mjOBJ_JOINT,'floating_base_joint');base=int(model.jnt_qposadr[root])
    floor=mid(model,mujoco.mjtObj.mjOBJ_GEOM,'floor')
    feet=[mid(model,mujoco.mjtObj.mjOBJ_BODY,x+'_ankle_roll_link') for x in ('left','right')]
    spheres=[[g for g in range(model.ngeom) if int(model.geom_bodyid[g])==body and
             int(model.geom_type[g])==int(mujoco.mjtGeom.mjGEOM_SPHERE) and
             (model.geom_contype[g]!=0 or model.geom_conaffinity[g]!=0)] for body in feet]
    if any(len(s)!=4 for s in spheres):raise ValueError('foot geometry changed')
    cases=[]
    with (HERE/'frames.jsonl').open('x',buffering=1) as stream:
        for rec in records:
            sid=rec['sample_id']
            if sid in audit['failed'] and sid not in corrections:continue
            if sid in corrections:
                path=REPAIRS/(sid+'_state.npz');expected=corrections[sid]['state_sha256'];variant='projected'
            else:
                path=SOURCE/rec['state20'];expected=rec['state20_sha256'];variant='original'
            if sha(path)!=expected:raise ValueError('state hash mismatch')
            with np.load(path,allow_pickle=False) as f:
                roots=f['root_wxyz_xyz']; joints=f['joint_pos'];times=f['sample_times'];names=list(f['joint_names'])
            addresses=[int(model.jnt_qposadr[mid(model,mujoco.mjtObj.mjOBJ_JOINT,n)]) for n in names]
            floor_frames=self_frames=0; mingaps=[];deepest=0.
            for i in range(len(times)):
                mujoco.mj_resetData(model,data);data.qpos[base:base+3]=roots[i,4:];data.qpos[base+3:base+7]=roots[i,:4]
                data.qpos[addresses]=joints[i];mujoco.mj_forward(model,data)
                normal=data.geom_xmat[floor].reshape(3,3)[:,2];origin=data.geom_xpos[floor]
                gaps=[min(float((data.geom_xpos[g]-origin)@normal-model.geom_size[g,0]) for g in group) for group in spheres]
                ground=[];self_contacts=[]
                for j in range(data.ncon):
                    c=data.contact[j]
                    if c.dist>=0:continue
                    geom1,geom2=int(c.geom1),int(c.geom2)
                    bodies=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) for g in (geom1,geom2)]
                    item={'bodies':bodies,'dist_m':float(c.dist)}
                    (ground if floor in (geom1,geom2) else self_contacts).append(item)
                    if floor not in (geom1,geom2):deepest=min(deepest,float(c.dist))
                floor_frames+=bool(ground);self_frames+=bool(self_contacts);mingaps.append(gaps)
                stream.write(json.dumps({'sample_id':sid,'frame':i,'time_s':float(times[i]),'foot_gap_m':gaps,
                    'floor_contacts':ground,'self_contacts':self_contacts},allow_nan=False)+'\n')
            cases.append({'sample_id':sid,'state_sha256':expected,'state_variant':variant,'frames':len(times),
                'minimum_foot_gap_m':np.min(mingaps,axis=0).tolist(),'maximum_foot_gap_m':np.max(mingaps,axis=0).tolist(),
                'floor_penetration_frames':floor_frames,'self_penetration_frames':self_frames,
                'worst_self_penetration_m':-deepest,'training_ready':False})
    if len(cases)!=23:raise ValueError('unexpected non-quarantined technical coverage')
    result={'source_report_sha256':sha(SOURCE/'report.json'),'source_audit_sha256':sha(SOURCE/'audit_v2.json'),
        'repairs_report_sha256':sha(REPAIRS/'report.json'),'scene_sha256':sha(SCENE),'model_sha256':sha(MODEL_XML),
        'model_meshes':asset_hashes(),'script_sha256':sha(Path(__file__)),'frames_sha256':sha(HERE/'frames.jsonl'),
        'mujoco_version':mujoco.__version__,'cases':cases,'training_ready':False,
        'limitations':['Static MuJoCo only; default hands and assumed ground; no object geometry or forces.',
            'No PhysX equivalence, source ground calibration, real clock, text semantic or license acceptance.',
            'Not an author filtering criterion; no automatic exclusions.']}
    with (HERE/'report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'cases':len(cases),'frames':sum(x['frames'] for x in cases),
        'samples_with_ground_penetration':sum(x['floor_penetration_frames']>0 for x in cases),
        'samples_with_self_penetration':sum(x['self_penetration_frames']>0 for x in cases),'training_ready':False}))
if __name__=='__main__':main()

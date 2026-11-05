"""Recount all batch static contacts; replay first/middle/last in same MuJoCo."""
import hashlib,json
from pathlib import Path
import mujoco,numpy as np
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'omni_gpu_batch24_20261101';REPAIRS=ROOT/'omni_batch24_projection_20261101'
SCENE=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Sim2Real/resources/robots/g1_description/scene_full.xml')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def jid(m,t,n):
    v=mujoco.mj_name2id(m,t,n)
    if v<0:raise ValueError('missing '+n)
    return v
def main():
    report=json.loads((HERE/'report.json').read_text());rows=[json.loads(s) for s in (HERE/'frames.jsonl').read_text().splitlines()]
    assert sha(HERE/'frames.jsonl')==report['frames_sha256'] and sha(SCENE)==report['scene_sha256']
    inputs={x['sample_id']:x for x in map(json.loads,(SOURCE/'records.jsonl').read_text().splitlines())}
    index={(x['sample_id'],x['frame']):x for x in rows}; assert len(index)==len(rows)
    model=mujoco.MjModel.from_xml_path(str(SCENE));data=mujoco.MjData(model)
    base=int(model.jnt_qposadr[jid(model,mujoco.mjtObj.mjOBJ_JOINT,'floating_base_joint')]);floor=jid(model,mujoco.mjtObj.mjOBJ_GEOM,'floor')
    replayed=0;nonhand_frames=0;floor_frames=0;self_frames=0
    for x in rows:
        floor_frames+=bool(x['floor_contacts']);self_frames+=bool(x['self_contacts'])
        nonhand_frames+=any(not any(('hand' in (n or '') or 'wrist' in (n or '')) for n in c['bodies']) for c in x['self_contacts'])
    for case in report['cases']:
        sid=case['sample_id'];source=inputs[sid]
        path=(REPAIRS/(sid+'_state.npz')) if case['state_variant']=='projected' else (SOURCE/source['state20'])
        assert sha(path)==case['state_sha256']
        subset=[x for x in rows if x['sample_id']==sid]
        assert len(subset)==case['frames']
        assert sum(bool(x['floor_contacts']) for x in subset)==case['floor_penetration_frames']
        assert sum(bool(x['self_contacts']) for x in subset)==case['self_penetration_frames']
        with np.load(path,allow_pickle=False) as f:roots=f['root_wxyz_xyz'];joints=f['joint_pos'];names=list(f['joint_names'])
        addresses=[int(model.jnt_qposadr[jid(model,mujoco.mjtObj.mjOBJ_JOINT,n)]) for n in names]
        for i in sorted({0,len(roots)//2,len(roots)-1}):
            mujoco.mj_resetData(model,data);data.qpos[base:base+3]=roots[i,4:];data.qpos[base+3:base+7]=roots[i,:4]
            data.qpos[addresses]=joints[i];mujoco.mj_forward(model,data)
            ground=selfc=0
            for k in range(data.ncon):
                c=data.contact[k]
                if c.dist<0:
                    if floor in (int(c.geom1),int(c.geom2)):ground+=1
                    else:selfc+=1
            frame=index[(sid,i)]
            assert ground==len(frame['floor_contacts']) and selfc==len(frame['self_contacts'])
            replayed+=1
    result={'report_sha256':sha(HERE/'report.json'),'frames_sha256':sha(HERE/'frames.jsonl'),
        'audit_script_sha256':sha(Path(__file__)),'cases':len(report['cases']),'frames':len(rows),
        'replayed_same_engine_frames':replayed,'floor_contact_frames':floor_frames,'self_contact_frames':self_frames,
        'nonhand_nonwrist_contact_frames':nonhand_frames,'training_ready':False,
        'limitations':['Replay shares same MuJoCo model and engine, not an independent physical validation.',
            'Default hands and uncalibrated plane; no object geometry, forces, actual physical acceptance.']}
    with (HERE/'audit.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()

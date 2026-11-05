"""Compare original and projected poses in the same assumed MuJoCo scene.

This evaluates consequences of the experimental correction; the floor is not a
measured ground plane and contacts are static geometry, not forces or training.
"""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
INPUT = ROOT/'four_semantic_subset_20261101'
DATA = ROOT/'six_family_disjoint_preview_v2_20261101'
PROJECT = ROOT/'vertical_support_projection_20261101'
BOUND = ROOT/'vertical_support_bound_20261101'
SCENE = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Sim2Real/resources/robots/g1_description/scene_full.xml')
HERE = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def mid(model, kind, name):
    i = mujoco.mj_name2id(model, kind, name)
    if i < 0:
        raise ValueError(name)
    return i

def main():
    proj = json.loads((PROJECT/'report.json').read_text())
    audit = json.loads((PROJECT/'audit.json').read_text())
    bound = json.loads((BOUND/'report.json').read_text())
    assert sha(PROJECT/'report.json') == audit['projection_report_sha256']
    assert sha(INPUT/'manifest.json') == proj['input_manifest_sha256']
    assert sha(SCENE) == bound['scene_sha256']
    states = {r['sample_id']: r for r in json.loads((INPUT/'state_manifest.json').read_text())['records']}
    model = mujoco.MjModel.from_xml_path(str(SCENE)); data = mujoco.MjData(model)
    root = mid(model,mujoco.mjtObj.mjOBJ_JOINT,'floating_base_joint')
    adr = int(model.jnt_qposadr[root]); floor = mid(model,mujoco.mjtObj.mjOBJ_GEOM,'floor')
    spheres = []
    for side in ('left','right'):
        body = mid(model,mujoco.mjtObj.mjOBJ_BODY,side+'_ankle_roll_link')
        group = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body and
                 model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE and (model.geom_contype[g] or model.geom_conaffinity[g])]
        assert len(group) == 4
        spheres.append(group)
    cases = []
    with (HERE/'frames.jsonl').open('x') as stream:
        for case in proj['cases']:
            if case['status'] != 'projected':
                continue
            sid = case['sample_id']; state_path = DATA/states[sid]['state_path']; path = PROJECT/(sid+'.npz')
            assert sha(state_path) == case['source_state_sha256'] and sha(path) == case['output_sha256']
            with np.load(state_path, allow_pickle=False) as old, np.load(path, allow_pickle=False) as new:
                a=old['root_wxyz_xyz'].astype(float); b=new['root_wxyz_xyz'].astype(float)
                q=new['joint_pos'].copy(); times=new['sample_times'].copy(); shift=new['height_shift_m'].copy()
                np.testing.assert_array_equal(q,old['joint_pos']); np.testing.assert_array_equal(times,old['sample_times'])
                addresses=[int(model.jnt_qposadr[mid(model,mujoco.mjtObj.mjOBJ_JOINT,str(n))]) for n in new['joint_names']]
            stats = {'before_floor':0,'after_floor':0,'before_self':0,'after_self':0,'self_changed':0,
                     'before_below':[0,0],'after_below':[0,0]}
            worst_gap_error=0.; min_gap=[np.full(2,np.inf),np.full(2,np.inf)]
            for i in range(len(times)):
                poses=[]
                for roots in (a,b):
                    mujoco.mj_resetData(model,data)
                    data.qpos[adr:adr+3]=roots[i,4:]; data.qpos[adr+3:adr+7]=roots[i,:4]
                    data.qpos[addresses]=q[i]; mujoco.mj_forward(model,data)
                    origin=data.geom_xpos[floor]; normal=data.geom_xmat[floor].reshape(3,3)[:,2]
                    gaps=[min(float((data.geom_xpos[g]-origin)@normal-model.geom_size[g,0]) for g in group) for group in spheres]
                    floor_d=[]; self_d=[]
                    for k in range(data.ncon):
                        c=data.contact[k]
                        if c.dist < -1e-9:
                            (floor_d if floor in (int(c.geom1),int(c.geom2)) else self_d).append(float(c.dist))
                    poses.append({'gaps_m':gaps,'floor_contacts':len(floor_d),'self_contacts':len(self_d),
                                  'worst_floor_m':min(floor_d,default=0.),'worst_self_m':min(self_d,default=0.)})
                before,after=poses
                predicted=np.asarray(before['gaps_m'])+shift[i]
                err=float(np.max(np.abs(np.asarray(after['gaps_m'])-predicted)));worst_gap_error=max(err,worst_gap_error)
                assert err < 1e-8
                for j,(tag,p) in enumerate(zip(('before','after'),poses)):
                    min_gap[j]=np.minimum(min_gap[j],p['gaps_m'])
                    stats[tag+'_floor']+=bool(p['floor_contacts']);stats[tag+'_self']+=bool(p['self_contacts'])
                    stats[tag+'_below']=[x+int(g < 0) for x,g in zip(stats[tag+'_below'],p['gaps_m'])]
                stats['self_changed']+=before['self_contacts']!=after['self_contacts']
                stream.write(json.dumps({'sample_id':sid,'frame':i,'shift_m':float(shift[i]),'before':before,'after':after},allow_nan=False)+'\n')
            cases.append({'sample_id':sid,'frames':len(times),'old_state_sha256':sha(state_path),'new_state_sha256':sha(path),
                          'foot_min_gap_before_m':min_gap[0].tolist(),'foot_min_gap_after_m':min_gap[1].tolist(),
                          'max_gap_shift_error_m':worst_gap_error,**stats})
    out={'projection_sha256':sha(PROJECT/'report.json'),'projection_audit_sha256':sha(PROJECT/'audit.json'),
         'scene_sha256':sha(SCENE),'script_sha256':sha(Path(__file__)),'frames_sha256':sha(HERE/'frames.jsonl'),
         'cases':cases,'training_ready':False,
         'limitations':['Static geometry, assumed z=0 floor and same old model; no measured contacts or forces.',
                        'Default hand angles and model collision filters; not Isaac equivalence or training acceptance.',
                        'Discrepancies caused by numerical contact tolerances are recorded, not equated to physical effects.']}
    with (HERE/'report.json').open('x') as f:
        json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'cases':len(cases),'frames':sum(c['frames'] for c in cases),'training_ready':False}))

if __name__ == '__main__':
    main()

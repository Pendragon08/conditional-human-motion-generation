"""Evaluate a ground-height-independent necessary bound on four preview clips."""
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from vertical_support_bound import vertical_support_bound

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
INPUT=ROOT/'four_semantic_subset_20261101'
DATA=ROOT/'six_family_disjoint_preview_v2_20261101'
OLD=ROOT/'eight_static_robot_quality_20261031'
SCENE=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Sim2Real/resources/robots/g1_description/scene_full.xml')
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def mid(m,kind,name):
    i=mujoco.mj_name2id(m,kind,name)
    if i<0:raise ValueError('missing model name '+name)
    return i


def main():
    audit=json.loads((INPUT/'audit.json').read_text());manifest=INPUT/'manifest.json'
    assert sha(manifest)==audit['manifest_sha256']
    records=json.loads(manifest.read_text())['records']
    states={x['sample_id']:x for x in json.loads((INPUT/'state_manifest.json').read_text())['records']}
    old=json.loads((OLD/'report.json').read_text());assert sha(SCENE)==old['scene_sha256']
    model=mujoco.MjModel.from_xml_path(str(SCENE));data=mujoco.MjData(model)
    joint=mid(model,mujoco.mjtObj.mjOBJ_JOINT,'floating_base_joint');root=int(model.jnt_bodyid[joint]);adr=int(model.jnt_qposadr[joint])
    if model.body_parentid[root]!=0 or model.jnt_type[joint]!=mujoco.mjtJoint.mjJNT_FREE:raise ValueError('expected free-root robot')
    bodies=[i for i in range(model.nbody) if model.body_rootid[i]==root]
    mass=float(sum(model.body_mass[i] for i in bodies))
    if not np.isclose(mass,model.body_subtreemass[root],rtol=0,atol=1e-9):raise ValueError('robot mass tree mismatch')
    output={};cases=[]
    for rec in records:
        sid=rec['sample_id'];state=states[sid];p=DATA/state['state_path']
        assert sha(p)==state['state_sha256'] and state['feature_sha256']==rec['feature_sha256']
        with np.load(p,allow_pickle=False) as f:
            roots=f['root_wxyz_xyz'].copy();q=f['joint_pos'].copy();t=f['sample_times'].copy();names=[str(x) for x in f['joint_names']]
        addresses=[int(model.jnt_qposadr[mid(model,mujoco.mjtObj.mjOBJ_JOINT,n)]) for n in names]
        if roots.shape!=(len(t),7) or q.shape!=(len(t),27) or len(set(addresses))!=27:raise ValueError('invalid state')
        com=[]
        for root_state,angles in zip(roots,q):
            mujoco.mj_resetData(model,data);data.qpos[adr:adr+3]=root_state[4:];data.qpos[adr+3:adr+7]=root_state[:4]
            data.qpos[addresses]=angles;mujoco.mj_forward(model,data);com.append(data.subtree_com[root].copy())
        com=np.asarray(com);bound=vertical_support_bound(t,com,model.opt.gravity);specific=bound['required_vertical_support_n_per_kg']
        worst=int(np.argmin(specific))+1
        output[sid+'_times']=t;output[sid+'_com']=com;output[sid+'_vertical_support']=specific
        cases.append({'sample_id':sid,'state_sha256':sha(p),'frames':len(t),'interior_intervals':len(specific),
          'min_required_vertical_support_n_per_kg':float(specific.min()),'worst_center_sample':worst,
          'worst_center_nominal_time_s':float(t[worst]),'negative_intervals':int((specific<0).sum()),
          'below_minus_1e-6_intervals':int((specific < -1e-6).sum())})
    path=HERE/'trajectories.npz'
    with path.open('xb') as f:np.savez_compressed(f,**output)
    report={'manifest_sha256':sha(manifest),'state_manifest_sha256':sha(INPUT/'state_manifest.json'),
        'scene_sha256':sha(SCENE),'robot_mass_kg':mass,'robot_bodies':len(bodies),'gravity_m_s2':model.opt.gravity.tolist(),
        'mujoco_version':mujoco.__version__,'module_sha256':sha(HERE/'vertical_support_bound.py'),
        'script_sha256':sha(Path(__file__)),'trajectories_sha256':sha(path),'cases':cases,'training_ready':False,
        'assumptions':['No downward external force other than gravity; support contacts can push but not pull.',
                       'Robot model mass distribution and 20 Hz sample times are treated as given for this diagnostic.',
                       'Second differences describe sampled approximate COM motion; no ground height or contact labels assumed.'],
        'limitations':['A passing necessary bound does not establish contact, friction, joint torque, stability, collision or feasibility.',
                       'Unknown true FPS, filtering, interpolation and model differences can change inferred acceleration.',
                       'No physical or overall training acceptance.']}
    with (HERE/'report.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({'cases':len(cases),'negative_intervals':sum(x['negative_intervals'] for x in cases),'training_ready':False}))


if __name__=='__main__':main()

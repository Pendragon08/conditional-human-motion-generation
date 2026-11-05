"""Recompute COM from inertial body positions, then both gravity hypotheses."""
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
INPUT=ROOT/'four_semantic_subset_20261101'
DATA=ROOT/'six_family_disjoint_preview_v2_20261101'
SCENE=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Sim2Real/resources/robots/g1_description/scene_full.xml')
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def nameid(m,kind,name):
    i=mujoco.mj_name2id(m,kind,name)
    if i<0:raise ValueError(name)
    return i


def main():
    report=json.loads((HERE/'report.json').read_text());artifact=HERE/'trajectories.npz'
    assert sha(artifact)==report['trajectories_sha256'] and sha(SCENE)==report['scene_sha256']
    assert sha(INPUT/'manifest.json')==report['manifest_sha256'] and sha(INPUT/'state_manifest.json')==report['state_manifest_sha256']
    states={x['sample_id']:x for x in json.loads((INPUT/'state_manifest.json').read_text())['records']}
    model=mujoco.MjModel.from_xml_path(str(SCENE));d=mujoco.MjData(model)
    root_joint=nameid(model,mujoco.mjtObj.mjOBJ_JOINT,'floating_base_joint')
    root=int(model.jnt_bodyid[root_joint]);adr=int(model.jnt_qposadr[root_joint])
    bodies=[i for i in range(model.nbody) if model.body_rootid[i]==root]
    mass=sum(float(model.body_mass[i]) for i in bodies);assert abs(mass-report['robot_mass_kg'])<1e-9
    results=[];worst_com=0.;worst_force=0.
    with np.load(artifact,allow_pickle=False) as saved:
        for case in report['cases']:
            sid=case['sample_id'];state=states[sid];path=DATA/state['state_path'];assert sha(path)==case['state_sha256']
            with np.load(path,allow_pickle=False) as f:
                roots=f['root_wxyz_xyz'].copy();q=f['joint_pos'].copy();t=f['sample_times'].copy();names=[str(x) for x in f['joint_names']]
            addresses=[int(model.jnt_qposadr[nameid(model,mujoco.mjtObj.mjOBJ_JOINT,n)]) for n in names]
            com=[]
            for root_state,angles in zip(roots,q):
                mujoco.mj_resetData(model,d);d.qpos[adr:adr+3]=root_state[4:];d.qpos[adr+3:adr+7]=root_state[:4]
                for address,angle in zip(addresses,angles):d.qpos[address]=angle
                mujoco.mj_forward(model,d)
                # Independent mass-weighted sum over body inertial origins, not subtree_com.
                com.append([sum(float(model.body_mass[i])*float(d.xipos[i,k]) for i in bodies)/mass for k in range(3)])
            com=np.asarray(com);reference=saved[sid+'_com'];error=float(np.max(np.abs(com-reference)));worst_com=max(worst_com,error)
            np.testing.assert_allclose(com,reference,rtol=0,atol=1e-12)
            np.testing.assert_allclose(t,saved[sid+'_times'],rtol=0,atol=1e-12)
            # Direct scalar differences and recorded gravity, no shared bound function.
            actual=np.array([(float(com[i+1,2])-2*float(com[i,2])+float(com[i-1,2]))/float(t[1]-t[0])**2-float(model.opt.gravity[2])
                             for i in range(1,len(t)-1)])
            err=float(np.max(np.abs(actual-saved[sid+'_vertical_support'])));worst_force=max(worst_force,err)
            np.testing.assert_allclose(actual,saved[sid+'_vertical_support'],rtol=0,atol=1e-9)
            assert int((actual<0).sum())==case['negative_intervals']
            assert abs(float(actual.min())-case['min_required_vertical_support_n_per_kg'])<1e-9
            earth=actual+(-9.81-float(model.opt.gravity[2]))*-1
            results.append({'sample_id':sid,'frames_recomputed':len(t),
                'scene_g_negative_intervals':int((actual < -1e-6).sum()),'scene_g_min_n_per_kg':float(actual.min()),
                'hypothetical_9_81_negative_intervals':int((earth < -1e-6).sum()),'hypothetical_9_81_min_n_per_kg':float(earth.min())})
    assert abs(float(model.opt.gravity[2])-(-7.5))<1e-12
    assert 'Ran 3 tests' in (HERE/'tests.log').read_text() and 'OK' in (HERE/'tests.log').read_text()
    out={'report_sha256':sha(HERE/'report.json'),'trajectories_sha256':sha(artifact),'all_frames_recomputed':sum(x['frames_recomputed'] for x in results),
         'max_com_difference_m':worst_com,'max_support_difference_n_per_kg':worst_force,'cases':results,
         'training_ready':False,'script_sha256':sha(Path(__file__)),
         'limitations':'Same MuJoCo asset and sample states; 9.81 is an alternative assumption, not an author setting; no forces or full dynamics measured.'}
    with (HERE/'audit.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out))


if __name__=='__main__':main()

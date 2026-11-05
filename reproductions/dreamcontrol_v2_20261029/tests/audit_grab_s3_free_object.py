"""Independent numerical and provenance checks for free-apple pilot."""
import json,sys,zipfile
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
sys.path.insert(0,str(ROOT/'grab_s3_scene_free_object_20261030'))
from grab_s3_scene_free_object import sha,read_grab_field,load_table,object_hulls,prepare_scene

def main():
    d=ROOT/'grab_s3_scene_free_object_20261030';report=json.loads((d/'report.json').read_text())
    if sha(d/'grab_s3_scene_free_object.py')!=report['script_sha256'] or sha(d/'reference_20hz.npz')!=report['reference_sha256']:raise ValueError('artifact changed')
    if report['reference_sha256']!=json.loads((ROOT/'grab_s3_scene_pd_diagnostic_20261030/report.json').read_text())['reference_sha256']:raise ValueError('not same robot reference')
    source=ROOT/'grab_s3_human_source_20261030';meta=json.loads((source/'report.json').read_text());archive=ROOT/'data/GRAB/grab__s3.zip'
    if sha(archive)!=report['archive_sha256'] or sha(source/'human_source.npz')!=report['source_sha256']:raise ValueError('source changed')
    with zipfile.ZipFile(archive) as z:obj=read_grab_field(z,meta['source_member'],'object')
    table=load_table(archive,meta['source_member']);hulls,paths,bound=object_hulls(obj,table)
    if any(sha(p)!=report['mesh_sha256'][n] for n,p in paths.items()) or abs(bound-report['apple_inner_hull_hausdorff_upper_bound_m'])>1e-12:raise ValueError('mesh changed')
    scene=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Sim2Real/resources/robots/g1_description/scene_full.xml')
    if sha(scene)!=report['scene_sha256']:raise ValueError('scene changed')
    model=prepare_scene(scene,hulls);body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'grab_apple')
    joint=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'grab_apple_free')
    if min(body,joint)<0 or model.jnt_type[joint]!=mujoco.mjtJoint.mjJNT_FREE:raise ValueError('apple is not free')
    mass=float(model.body_mass[body])
    if abs(mass-.15)>1e-9:raise ValueError('apple mass mismatch')
    pos=np.asarray(obj['params']['transl'],float);rot=Rotation.from_rotvec(np.asarray(obj['params']['global_orient'],float)).inv()
    vel=(pos[265]-pos[264])*120.;wlocal=(rot[264].inv()*rot[265]).as_rotvec()*120.
    modes={x['mode']:x for x in report['results']}
    if set(modes)!={'free_zero_velocity','free_source_velocity'}:raise ValueError('scenario mismatch')
    init=np.asarray(modes['free_source_velocity']['initial_apple_velocity_qvel']);zero=np.asarray(modes['free_zero_velocity']['initial_apple_velocity_qvel'])
    linear_diff=float(np.max(np.abs(init[:3]-vel)));angular_diff=float(np.max(np.abs(init[3:]-wlocal)))
    if np.max(np.abs(zero))>1e-12 or linear_diff>1e-8 or angular_diff>1e-7:raise ValueError('initial velocity mismatch')
    for x in modes.values():
        if x['steps']!=450 or abs(x['dt']-.002)>1e-12:raise ValueError('bad timestep')
        if any(not np.isfinite(x[k]) for k in ('apple_translation_error_mean_m','apple_translation_error_final_m','mean_root_error_m')):raise ValueError('nonfinite results')
        if any(v<0 or v>1 for v in x['contact_step_fraction'].values()):raise ValueError('bad contact fractions')
    # Isolated no-contact mass under the same gravity; confirms that a free body
    # is integrated, independently of the recorded apple trajectory.
    test=mujoco.MjModel.from_xml_string('<mujoco><option timestep=".002" gravity="0 0 -7.5"/><worldbody><body pos="0 0 2"><freejoint/><geom type="sphere" size=".05" mass=".15"/></body></worldbody></mujoco>')
    data=mujoco.MjData(test);start=float(data.qpos[2])
    for _ in range(50):mujoco.mj_step(test,data)
    drop=start-float(data.qpos[2])
    if not (.03<drop<.05):raise ValueError('free body gravity test failed')
    out={'report_sha256':sha(d/'report.json'),'script_sha256':sha(Path(__file__)),'apple_model_mass_kg':mass,
         'source_velocity_linear_max_diff_m_s':linear_diff,'source_velocity_local_angular_max_diff_rad_s':angular_diff,
         'isolated_free_body_drop_0_1s_m':drop,'warnings':'Provenance, initialization, and a synthetic gravity sanity check only. Does not validate assumed mass/friction, real grasp, scene, force, control or Isaac Lab.'}
    with Path(__file__).with_name('audit_report.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps(out),flush=True)
if __name__=='__main__':main()

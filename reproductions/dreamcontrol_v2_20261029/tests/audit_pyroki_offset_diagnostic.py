"""Independently verify instrumentation, numeric output and convergence logs."""
import hashlib,json,re,sys
from pathlib import Path
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'foot_contact_fit_20261030'))
from g1_fk import URDFFK

LINKS=('pelvis_contour_link','left_hip_pitch_link','right_hip_pitch_link','left_knee_link','right_knee_link',
       'left_ankle_roll_link','right_ankle_roll_link','left_shoulder_roll_link','right_shoulder_roll_link',
       'left_elbow_link','right_elbow_link','left_wrist_yaw_link','right_wrist_yaw_link')
HUMAN=(0,1,2,4,5,7,8,16,17,18,19,20,21)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    report_path=OUT/'report.json';report=json.loads(report_path.read_text());manifest=json.loads((OUT/'manifest.json').read_text())
    source=ROOT/'grab_s4_drink_human_source_20261030/human_source.npz';base_dir=ROOT/'pyroki_grab_s4_drink_windowed_20261030';baseline=base_dir/'window_00.npz'
    original=ROOT/'pyroki_grab_temporal_20261030/retarget_helpers/_utils.py';instrumented=OUT/'retarget_helpers/_utils.py'
    old='    return transform, solution[var_joints]\n';new='    return transform, solution[var_joints], offset, solution[var_Ts_world_root]\n'
    before=original.read_text();after=instrumented.read_text()
    if before.count(old)!=1 or after!=before.replace(old,new):raise ValueError('helper differs beyond return')
    if sha(original)!=manifest['base_helper_sha256'] or sha(instrumented)!=manifest['instrumented_helper_sha256'] or sha(instrumented)!=report['instrumented_helper_sha256']:
        raise ValueError('helper hash mismatch')
    for path,key in ((source,'source_sha256'),(baseline,'baseline_sha256'),(OUT/'diagnostic.npz','artifact_sha256'),(OUT/'pyroki_offset_diagnostic.py','driver_sha256')):
        if sha(path)!=report[key]:raise ValueError('provenance mismatch '+key)
    urdf=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
    variant=json.loads((ROOT/'pyroki_grab_temporal_20261030/report.json').read_text())
    if sha(urdf)!=variant['urdf_sha256']:raise ValueError('URDF changed')
    with np.load(OUT/'diagnostic.npz',allow_pickle=False) as d:
        frames=d['frames'];offset=d['offset'];pre=d['raw_root'];post=d['final_root'];joints=d['joints']
    with np.load(baseline,allow_pickle=False) as d:
        ref_frames=d['frames'];ref_root=d['root_wxyz_xyz'];ref_joints=d['joint_pos'];names=d['joint_names'].tolist()
    with np.load(source,allow_pickle=False) as d:
        indices=d['input_frame_indices'];human=d['human_joints']
    if not np.array_equal(frames,ref_frames) or not np.array_equal(frames,indices[2:9]):raise ValueError('timeline differs')
    if not np.allclose(post[:,4:],pre[:,4:]+offset,atol=1e-7,rtol=0):raise ValueError('composition mismatch')
    root_diff=float(np.max(np.linalg.norm(post[:,4:]-ref_root[:,4:],axis=-1)))
    quat_diff=float(np.max(np.minimum(np.linalg.norm(post[:,:4]-ref_root[:,:4],axis=-1),np.linalg.norm(post[:,:4]+ref_root[:,:4],axis=-1))))
    joint_diff=float(np.max(np.abs(joints-ref_joints)))
    if max(root_diff,quat_diff,joint_diff)>1e-6:raise ValueError('instrumented output differs from baseline')
    fk=URDFFK(urdf);target=human[2:9][:,HUMAN,:]
    err=[]
    for root in (pre,post):
        points=fk.forward(root[:,4:],root[:,[1,2,3,0]],joints,names,links=LINKS)
        err.append(float(np.linalg.norm(points-target,axis=-1).mean()))
    if abs(err[0]-report['unscaled_world_keypoint_distance_before_offset_mean_m'])>1e-8 or abs(err[1]-report['unscaled_world_keypoint_distance_after_offset_mean_m'])>1e-8:
        raise ValueError('FK distance mismatch')
    # Pair each termination marker with the next WINDOW line; no claim of solver optimality.
    log=(base_dir/'solver.log').read_text();parts=re.split(r'WINDOW (\d+) of 29 elapsed [^\n]*',log)
    if len(parts)!=59:raise ValueError('missing window log markers')
    statuses=[]
    for i in range(29):
        if int(parts[2*i+1])!=i:raise ValueError('window log ordering')
        m=re.findall(r'Terminated @ iteration #(\d+):[^\n]*criteria=\[([01 ]+)\]',parts[2*i])
        if len(m)!=1:raise ValueError('termination markers')
        flags=[int(x) for x in m[0][1].split()]
        if len(flags)!=3:raise ValueError('criterion format')
        statuses.append({'window':i,'iteration':int(m[0][0]),'flags':flags})
    debug_log=(OUT/'solver.log').read_text()
    m=re.findall(r'Terminated @ iteration #(\d+):[^\n]*criteria=\[([01 ]+)\]',debug_log)
    if len(m)!=1 or int(m[0][0])!=statuses[0]['iteration'] or [int(x) for x in m[0][1].split()]!=statuses[0]['flags']:
        raise ValueError('instrumented stopping differs')
    out={'script_sha256':sha(__file__),'source_report_sha256':sha(report_path),'source_log_sha256':sha(base_dir/'solver.log'),
         'source_sha256':sha(source),'urdf_sha256':sha(urdf),'artifact_sha256':sha(OUT/'diagnostic.npz'),
         'helper_only_return_changed':True,'baseline_max_position_diff_m':root_diff,'baseline_max_quaternion_sign_invariant_diff':quat_diff,
         'baseline_max_joint_diff_rad':joint_diff,'recomputed_before_after_mean_m':err,
         'offset_first_xyz_m':offset[0].tolist(),'offset_max_variation_m':float(np.max(np.abs(offset-offset[0]))),
         'window_stopping_flags':statuses,'windows_with_all_flags_false':sum(not any(s['flags']) for s in statuses),
         'limitations':'Numerical consistency and source log audit only. One 7-frame S4 window; root-offset decomposition is legacy approximate code, assumed ground and proxy contacts. JAXLS flags and point distances do not prove optimization quality or physical validity.'}
    with (OUT/'independent_audit.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps(out),flush=True)

if __name__=='__main__':main()

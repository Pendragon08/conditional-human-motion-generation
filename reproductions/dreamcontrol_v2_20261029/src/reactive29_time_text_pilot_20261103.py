"""Convert two published G1 trajectories under explicitly labelled FPS hypotheses.

Outputs are diagnostic, not accepted training samples; no retargeting or filtering.
"""
import hashlib, json, pickle, shutil, sys
from pathlib import Path
import numpy as np
R = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
P = Path('/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/data/amass_g1_36dim/full_train.pkl')
U = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf')
T = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/TrajGen')
sys.path[:0] = [str(R), str(R/'ssm_new6_20hz_resume_20261101'), str(R/'antialias_v3_20261030/src'), str(T)]
from g1_fk import URDFFK
from feature263_signfix import encode_signfixed
from resample_antialias_omniretarget import resample_antialias
NAMES = tuple(n+'_joint' for n in ('left_hip_pitch','left_hip_roll','left_hip_yaw','left_knee','left_ankle_pitch','left_ankle_roll','right_hip_pitch','right_hip_roll','right_hip_yaw','right_knee','right_ankle_pitch','right_ankle_roll','waist_yaw','waist_roll','waist_pitch','left_shoulder_pitch','left_shoulder_roll','left_shoulder_yaw','left_elbow','left_wrist_roll','left_wrist_pitch','left_wrist_yaw','right_shoulder_pitch','right_shoulder_roll','right_shoulder_yaw','right_elbow','right_wrist_roll','right_wrist_pitch','right_wrist_yaw'))
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()
def main():
    if shutil.disk_usage(R).free < 8*1024**3+356*1024**2: raise RuntimeError('disk reserve')
    if sha(P) != '53a3fff8c2928d3e36b06b1b8bce1148675854ff3c5ef5bad0e56f295abe302d': raise ValueError('corpus hash mismatch')
    if sha(U) != '572b54991d8dccd6d3e091570788fb307bd0877b6a062426583f0a17fb56cc95': raise ValueError('URDF hash mismatch')
    with P.open('rb') as f: data = pickle.load(f)['data_dict']
    fk = URDFFK(U)
    if set(fk.actuated) != set(NAMES): raise ValueError('joint name mismatch')
    out = R/'reactive29_time_text_pilot_20261103'
    out.mkdir(exist_ok=False)
    records = []
    for sid in ('004271','012314'):
        entry = data[sid]; motion = np.asarray(entry['motion'])
        if motion.ndim != 2 or motion.shape[1] != 36 or not np.isfinite(motion).all(): raise ValueError('invalid input')
        qpos = np.column_stack((motion[:,[6,3,4,5]], motion[:,:3], motion[:,7:]))
        for fps in (60., 20.):
            q = resample_antialias(qpos, fps, 20., max_seconds=10.)
            points = fk.forward(q[:,4:7], q[:,[1,2,3,0]], q[:,7:36], NAMES)
            features = encode_signfixed(points, foot_velocity_squared_threshold=.002)
            if features.shape != (len(q)-1, 263) or not np.isfinite(features).all(): raise ValueError('invalid features')
            dest = out/f'{sid}_assume{int(fps)}hz.npz'
            with dest.open('xb') as f:
                np.savez_compressed(f, qpos29_wxyz_xyz=q, points_z_up=points, features=features,
                                    time_seconds=np.arange(len(q))/20., joint_names=np.asarray(NAMES))
            with np.load(dest, allow_pickle=False) as check:
                if not np.array_equal(check['qpos29_wxyz_xyz'], q) or not np.array_equal(check['features'], features): raise ValueError('readback')
            records.append({'id':sid,'source_motion_sha256':hashlib.sha256(motion.tobytes()).hexdigest(),
                            'source_frames':len(motion),'source_assumed_fps':fps,'source_assumption':'publisher_documented' if fps==60 else 'sensitivity_comparison_only',
                            'captions':entry['text'],'sampled_states':len(q),'feature_frames':len(features),
                            'output':dest.name,'output_sha256':sha(dest),'accepted':False})
    report = {'source_sha256':sha(P),'urdf_sha256':sha(U),'target_fps':20,'records':records,'training_ready':False,
              'limitations':['60Hz is publisher-documented, not per-record independently verified. 20Hz is only a sensitivity hypothesis.',
                             'Caption binding comes from the source record; semantic match, timing and license applicability are not accepted.',
                             '29 joint states retained; 263D encodes 22 keypoints and cannot reconstruct all joints losslessly.',
                             'Quaternion filtering and feature encoding are approximations, not DreamControl-v2 author implementation.']}
    with (out/'report.json').open('x',encoding='utf8') as f: json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps({'out':str(out),'records':[{k:r[k] for k in ('id','source_assumed_fps','source_frames','sampled_states','feature_frames')} for r in records]}))
if __name__ == '__main__': main()

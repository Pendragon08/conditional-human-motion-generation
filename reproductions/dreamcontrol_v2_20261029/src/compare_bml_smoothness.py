"""Compare fixed smoothness alternatives on the same BML human frames."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from g1_fk import URDFFK

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
URDF=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
LINKS=['left_hip_pitch_link','right_hip_pitch_link','left_knee_link','right_knee_link',
       'left_ankle_pitch_link','right_ankle_pitch_link','left_shoulder_pitch_link',
       'right_shoulder_pitch_link','left_elbow_link','right_elbow_link','left_wrist_yaw_link','right_wrist_yaw_link']
IDX=[1,2,4,5,7,8,16,17,18,19,20,21]


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    label=sys.argv[1]
    if label not in ('S04_Expert','S08_Novice'):raise ValueError('invalid label')
    sources=[ROOT/'bml_global_candidates_20261031'/label,ROOT/'bml_joint_smoothness_20261031'/label]
    fk=URDFFK(URDF); result=[]
    reference=None
    for source in sources:
        report=json.loads((source/'report.json').read_text())
        audit=json.loads((source/'audit_report_v2.json').read_text())
        assert sha(source/'trajectory.npz')==audit['trajectory_sha256']==report['artifact_sha256']
        assert sha(source/'human.npz')==audit['input_sha256']==report['source_sha256']
        assert sha(URDF)==report['urdf_sha256']
        with np.load(source/'trajectory.npz',allow_pickle=False) as d:
            root=d['root_wxyz_xyz'].copy();q=d['joint_pos'].copy(); names=[str(x) for x in d['joint_names']]
            frames=d['source_frames'].copy()
        with np.load(source/'human.npz',allow_pickle=False) as d:
            human=d['human'][1:].copy();np.testing.assert_array_equal(frames,d['source_frames'][1:])
        if reference is not None: np.testing.assert_array_equal(reference,frames)
        reference=frames
        points=fk.forward(root[:,4:],root[:,[1,2,3,0]],q,names,links=LINKS)
        error=np.linalg.norm(points-human[:,IDX],axis=-1)
        mean=float(error.mean());maximum=float(error.max())
        assert abs(mean-report['unscaled_world_point_mean_m'])<1e-6
        assert abs(maximum-report['unscaled_world_point_max_m'])<1e-6
        result.append({'trajectory_sha256':sha(source/'trajectory.npz'),'audit_sha256':sha(source/'audit_report_v2.json'),
                       'max_adjacent_joint_rad':audit['max_adjacent_joint_rad'],
                       'max_adjacent_root_m':audit['max_adjacent_root_m'],
                       'max_joint_limit_violation_rad':audit['max_joint_limit_violation_rad'],
                       'unscaled_world_point_mean_m':mean,'unscaled_world_point_max_m':maximum})
    out={'label':label,'old':result[0],'smoothness_4x':result[1],'training_ready':False,
         'limitations':['Same FK and world alignment as the solver, not ground truth error.',
                        'No physical/contact, text, source, license or robot tracking quality acceptance.',
                        '60-point excerpts and a single exploratory coefficient, not author method.']}
    with (Path.cwd()/(label+'.comparison.json')).open('x',encoding='utf8') as f:
        json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(out),flush=True)


if __name__=='__main__':main()

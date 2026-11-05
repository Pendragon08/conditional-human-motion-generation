"""Audit two isolated 29-to-27 robot fits; this does not grant training approval."""
import hashlib, io, json, zipfile
from pathlib import Path
import numpy as np
from g1_fk import URDFFK
from g1_joint_limit_gate import read_joint_limits, check_joint_limits
from motion_bridge import SOURCE_NAMES
from omniretarget_bridge import convert_layout

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'data/OmniRetarget_Dataset/robot-object.zip'
ROBOT = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1')
LINKS = ('left_hip_pitch_link','right_hip_pitch_link','left_knee_link','right_knee_link',
         'left_ankle_roll_link','right_ankle_roll_link','left_shoulder_roll_link',
         'right_shoulder_roll_link','left_elbow_link','right_elbow_link',
         'left_wrist_yaw_link','right_wrist_yaw_link')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def main():
    report = json.loads((HERE/'report.json').read_text())
    assert sha(ARCHIVE) == report['source_archive_sha256']
    assert sha(ROBOT/'g1_29dof.urdf') == report['urdf29_sha256']
    assert sha(ROBOT/'g1_27dof.urdf') == report['urdf27_sha256']
    assert sha(HERE/'fit_omni_27_gpu_candidates.py') == report['script_sha256']
    assert sha(HERE/'waist_lock_root_ik.py') == report['fitter_sha256']
    assert sha(ROOT/'omomo_text_pairing_20261101/candidate_pairs.jsonl') == report['pair_manifest_sha256']
    results=[]
    fk29, fk27 = URDFFK(ROBOT/'g1_29dof.urdf'), URDFFK(ROBOT/'g1_27dof.urdf')
    with zipfile.ZipFile(ARCHIVE) as z:
        for case in report['cases']:
            raw=z.read(case['source_member'])
            assert hashlib.sha256(raw).hexdigest() == case['source_sha256']
            with np.load(io.BytesIO(raw), allow_pickle=False) as data:
                motion,obj=convert_layout(data['qpos'])
                assert float(data['fps']) == case['declared_fps'] == 30.
            path=HERE/case['output']; assert sha(path)==case['output_sha256']
            with np.load(path,allow_pickle=False) as data:
                root=data['root_wxyz_xyz']; joints=data['joint_pos']; names=list(data['joint_names'])
                assert root.shape==(len(motion),7) and joints.shape==(len(motion),27)
                assert np.array_equal(data['object_wxyz_xyz'],obj)
                assert np.array_equal(data['sample_times'],np.arange(len(motion))/30.)
                assert np.isfinite(root).all() and np.isfinite(joints).all()
                assert np.max(np.abs(np.linalg.norm(root[:,:4],axis=1)-1)) < 1e-10
            limits=check_joint_limits(joints,names,read_joint_limits(ROBOT/'g1_27dof.urdf'),tolerance_rad=1e-10)
            assert limits['joint_limits_passed']
            source=fk29.forward(motion[:,:3],motion[:,3:7],motion[:,7:],list(SOURCE_NAMES),links=LINKS)
            fitted=fk27.forward(root[:,4:],root[:,[1,2,3,0]],joints,names,links=LINKS)
            error=np.linalg.norm(source-fitted,axis=-1)
            assert abs(float(error.mean())-case['final_mean_keypoint_error_m']) < 1e-8
            assert abs(float(error.max())-case['final_max_keypoint_error_m']) < 1e-8
            results.append({'sample_id':case['sample_id'],'frames':len(motion),
                'independent_fk_mean_error_m':float(error.mean()),'independent_fk_max_error_m':float(error.max()),
                'joint_limit':limits,'object_and_time_equal':True})
    assert len(results)==2
    out={'audit_script_sha256':sha(Path(__file__)),'source_report_sha256':sha(HERE/'report.json'),
         'cases':results,'limitations':['Independent URDF FK, but common source conversion helper.',
         'Text semantics, actual timing, source permissions, collisions and dynamics were not accepted.'],
         'training_ready':False}
    with (HERE/'audit.json').open('x') as f: json.dump(out,f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()

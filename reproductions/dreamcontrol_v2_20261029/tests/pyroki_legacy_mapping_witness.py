"""Check legacy hard-coded link-index mapping against current PyRoki model."""
import hashlib
import json
from pathlib import Path
import sys
import jax
import numpy as np
import pyroki
import yourdfpy

PROJECT=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl')
sys.path.insert(0,str(PROJECT/'TrajGen/sample'))
from retarget_helpers._utils import get_humanoid_retarget_indices_g1_27dof, SMPL_JOINT_NAMES


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    if any(x.platform!='cpu' for x in jax.devices()): raise ValueError('CPU required')
    path=PROJECT/'Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf'
    robot=pyroki.Robot.from_urdf(yourdfpy.URDF.load(str(path),load_meshes=False))
    human,indices=map(np.asarray,get_humanoid_retarget_indices_g1_27dof())
    expected=['pelvis_contour_link','left_hip_pitch_link','right_hip_pitch_link',
              'left_knee_link','right_knee_link','left_ankle_roll_link','right_ankle_roll_link',
              'left_shoulder_roll_link','right_shoulder_roll_link','left_elbow_link',
              'right_elbow_link','left_wrist_yaw_link','right_wrist_yaw_link']
    actual=[robot.links.names[i] for i in indices]
    rows=[{'human':SMPL_JOINT_NAMES[h], 'index':int(i), 'expected_robot':e, 'actual_robot':a,
           'matches':e==a} for h,i,e,a in zip(human,indices,expected,actual)]
    result={'urdf_sha256':sha(path),'helper_sha256':sha(PROJECT/'TrajGen/sample/retarget_helpers/_utils.py'),
            'script_sha256':sha(Path(__file__)),'matched':sum(x['matches'] for x in rows),
            'total':len(rows),'rows':rows,
            'warning':'Index mapping only; source model, PyRoki targets, ground contact, objective and dynamics remain unvalidated.'}
    with Path(__file__).with_name('legacy_mapping.json').open('x',encoding='utf8') as f:
        json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()

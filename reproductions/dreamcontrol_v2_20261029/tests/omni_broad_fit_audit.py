"""Compare two experimental root-IK settings on 12 source-distinct clips.

Selection aims at different subjects / source IDs but is not a random sample or
proof of dataset independence. This emits diagnostics, never training files.
"""
import io
import json
from pathlib import Path
import re
import zipfile
import numpy as np
import torch

from g1_fk import URDFFK
from motion_bridge import SOURCE_NAMES
from omniretarget_bridge import convert_layout
from resample_omniretarget import resample
from waist_lock_ik import FIT_LINKS
from waist_lock_root_ik import fit_root_and_waist


def select_members(archive_name, names):
    candidates = sorted(n for n in names if n.endswith('.npz'))
    chosen = {}
    for member in candidates:
        basename = Path(member).name
        if archive_name == 'robot-object.zip':
            match = re.match(r'(sub\d+)_.*_original\.npz$', basename)
        elif archive_name == 'robot-terrain.zip':
            match = re.match(r'(climb_\d+)_z_scale_1\.0\.npz$', basename)
        else:
            match = re.match(r'(Take .*?)_Skeleton.*_chair_2_stage\.npz$', basename)
        if match:
            chosen.setdefault(match.group(1), member)
    if len(chosen) < 4:
        raise ValueError(f'not enough distinct-looking source groups for {archive_name}')
    return list(chosen.values())[:4]


def main():
    torch.set_num_threads(1)
    repo = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation')
    assets = repo/'third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1'
    u29, u27 = assets/'g1_29dof.urdf', assets/'g1_27dof.urdf'
    base = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/data/OmniRetarget_Dataset')
    fk29, fk27 = URDFFK(u29), URDFFK(u27)
    rows = []
    for archive_name in ['robot-object.zip', 'robot-terrain.zip', 'robot-object-terrain.zip']:
        with zipfile.ZipFile(base/archive_name) as archive:
            members = select_members(archive_name, archive.namelist())
            for member in members:
                with np.load(io.BytesIO(archive.read(member)), allow_pickle=False) as data:
                    sample = resample(data['qpos'], float(data['fps']))
                motion, _ = convert_layout(sample)
                truth = fk29.forward(motion[:, :3], motion[:, 3:7], motion[:, 7:], SOURCE_NAMES, links=FIT_LINKS)
                source_vel = np.diff(truth[:, 4:6], axis=0)*20
                contact = np.linalg.norm(source_vel, axis=-1) < .03
                metrics = {}
                for label, foot, contact_weight in [('unweighted', 0., 0.), ('foot_2_contact_10', 2., 10.)]:
                    fit = fit_root_and_waist(motion, u29, u27, steps=120,
                                             foot_position_weight=foot, contact_velocity_weight=contact_weight)
                    pred = fk27.forward(fit['root_pos'], fit['root_quat_xyzw'], fit['joint_pos'], fit['joint_names'], links=FIT_LINKS)
                    errors = np.linalg.norm(pred-truth, axis=-1)
                    vel = np.diff(pred[:, 4:6], axis=0)*20
                    vel_error = np.linalg.norm(vel-source_vel, axis=-1)
                    metrics[label] = {'mean_keypoint_error_m': float(errors.mean()), 'max_keypoint_error_m': float(errors.max()),
                                      'mean_foot_error_m': float(errors[:, 4:6].mean()), 'max_foot_error_m': float(errors[:, 4:6].max()),
                                      'mean_wrist_error_m': float(errors[:, 10:12].mean()),
                                      'contact_proxy_velocity_error_mps': float(vel_error[contact].mean()) if contact.any() else None,
                                      'max_root_translation_m': fit['max_root_translation_m'],
                                      'max_root_rotation_rad': fit['max_root_rotation_rad']}
                rows.append({'archive': archive_name, 'member': member, 'frames': len(motion),
                             'contact_proxy_count': int(contact.sum()), 'methods': metrics})
    print(json.dumps({'fps': 20, 'steps': 120, 'samples': rows,
                      'caveat': '12 nonrandom clips; proxy contact threshold 0.03m/s; no collision, physical or paper-fidelity validation.'}))


if __name__ == '__main__':
    main()

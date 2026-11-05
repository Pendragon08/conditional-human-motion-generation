"""Compare exploratory foot-loss weights on three distinct OmniRetarget clips."""
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import torch

from g1_fk import URDFFK
from motion_bridge import SOURCE_NAMES
from omniretarget_bridge import convert_layout
from resample_omniretarget import resample
from waist_lock_ik import FIT_LINKS
from waist_lock_root_ik import fit_root_and_waist


def main():
    torch.set_num_threads(1)
    repo = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation')
    assets = repo/'third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1'
    u29, u27 = assets/'g1_29dof.urdf', assets/'g1_27dof.urdf'
    base = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/data/OmniRetarget_Dataset')
    fk29, fk27 = URDFFK(u29), URDFFK(u27)
    samples = [
        ('robot-object.zip', 'robot-object/sub10_largebox_000_original.npz'),
        ('robot-terrain.zip', 'robot-terrain/climb_00_z_scale_0.8.npz'),
        ('robot-object-terrain.zip', 'robot-object-terrain/Take 2025-09-02 08.48.33 PM_Skeleton 001_joint_positions_f0-2618_chair_2_stage.npz')]
    configs = [('unweighted', 0., 0.), ('foot_0.5_contact_3', .5, 3.),
               ('foot_0.5_contact_30', .5, 30.), ('foot_2_contact_10', 2., 10.)]
    rows = []
    for archive_name, member in samples:
        with zipfile.ZipFile(base/archive_name) as archive:
            with np.load(io.BytesIO(archive.read(member)), allow_pickle=False) as data:
                sample = resample(data['qpos'], float(data['fps']))
        motion, _ = convert_layout(sample)
        truth = fk29.forward(motion[:, :3], motion[:, 3:7], motion[:, 7:], SOURCE_NAMES, links=FIT_LINKS)
        contact = np.linalg.norm(np.diff(truth[:, 4:6], axis=0), axis=-1)*20 < .03
        metrics = {}
        for name, foot_weight, contact_weight in configs:
            fit = fit_root_and_waist(motion, u29, u27, steps=120, foot_position_weight=foot_weight,
                                     contact_velocity_weight=contact_weight)
            pred = fk27.forward(fit['root_pos'], fit['root_quat_xyzw'], fit['joint_pos'], fit['joint_names'], links=FIT_LINKS)
            errors = np.linalg.norm(pred-truth, axis=-1)
            speeds = np.linalg.norm(np.diff(pred[:, 4:6], axis=0), axis=-1)*20
            metrics[name] = {'mean_keypoint_error_m': float(errors.mean()),
                             'max_keypoint_error_m': float(errors.max()),
                             'mean_foot_error_m': float(errors[:, 4:6].mean()),
                             'max_foot_error_m': float(errors[:, 4:6].max()),
                             'contact_proxy_mean_speed_mps': float(speeds[contact].mean()) if contact.any() else None,
                             'max_root_translation_m': fit['max_root_translation_m'],
                             'max_root_rotation_rad': fit['max_root_rotation_rad']}
            if abs(metrics[name]['mean_keypoint_error_m'] - fit['final_mean_keypoint_error_m']) > 1e-5:
                raise AssertionError('independent FK disagreement')
        rows.append({'archive': archive_name, 'member': member, 'frames': len(motion),
                     'contact_proxy_count': int(contact.sum()), 'configs': metrics})
    print(json.dumps({'fps': 20, 'steps': 120, 'samples': rows,
                      'warning': 'Three sampled motions only. Contact proxy is source ankle speed <0.03m/s; not a contact or dynamics validation.'}))


if __name__ == '__main__':
    main()

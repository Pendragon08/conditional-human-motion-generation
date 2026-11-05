"""Compare fitting on six real OmniRetarget clips; exploratory CPU witness."""
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
from waist_lock_ik import FIT_LINKS, fit_locked_waist
from waist_lock_root_ik import fit_root_and_waist


def main():
    torch.set_num_threads(1)
    repo = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation')
    assets = repo/'third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1'
    u29, u27 = assets/'g1_29dof.urdf', assets/'g1_27dof.urdf'
    data_root = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/data/OmniRetarget_Dataset')
    fk29, fk27 = URDFFK(u29), URDFFK(u27)
    rows = []
    for name in ['robot-object.zip', 'robot-terrain.zip', 'robot-object-terrain.zip']:
        with zipfile.ZipFile(data_root/name) as archive:
            members = sorted(x for x in archive.namelist() if x.endswith('.npz'))[:2]
            for member in members:
                with np.load(io.BytesIO(archive.read(member)), allow_pickle=False) as f:
                    sample = resample(f['qpos'], float(f['fps']))
                motion, _ = convert_layout(sample)
                truth = fk29.forward(motion[:, :3], motion[:, 3:7], motion[:, 7:], SOURCE_NAMES, links=FIT_LINKS)
                fits = {'joint_only': fit_locked_waist(motion, u29, u27, steps=120, learning_rate=.015,
                                                       nominal_weight=.002, velocity_weight=.01),
                        'bounded_root_joint': fit_root_and_waist(motion, u29, u27, steps=120),
                        'foot_aware_root_joint': fit_root_and_waist(motion, u29, u27, steps=120,
                                                                    foot_position_weight=.5,
                                                                    contact_velocity_weight=3.)}
                results = {}
                for label, fit in fits.items():
                    pred = fk27.forward(fit.get('root_pos', motion[:, :3]), fit.get('root_quat_xyzw', motion[:, 3:7]),
                                        fit['joint_pos'], fit['joint_names'], links=FIT_LINKS)
                    err = np.linalg.norm(pred-truth, axis=-1)
                    source_speed = np.linalg.norm(np.diff(truth[:, 4:6], axis=0), axis=-1)*20
                    fitted_speed = np.linalg.norm(np.diff(pred[:, 4:6], axis=0), axis=-1)*20
                    contact = source_speed < .03  # Exploratory only.
                    results[label] = {'mean_error_m': float(err.mean()), 'max_error_m': float(err.max()),
                                      'mean_foot_error_m': float(err[:, 4:6].mean()),
                                      'max_foot_error_m': float(err[:, 4:6].max()),
                                      'contact_proxy_count': int(contact.sum()),
                                      'contact_proxy_mean_fit_speed_mps': float(fitted_speed[contact].mean()) if contact.any() else None,
                                      'max_root_translation_m': fit.get('max_root_translation_m', 0.),
                                      'max_root_rotation_rad': fit.get('max_root_rotation_rad', 0.)}
                    if abs(results[label]['mean_error_m'] - fit['final_mean_keypoint_error_m']) > 1e-5:
                        raise AssertionError('independent FK disagrees with optimizer')
                rows.append({'archive': name, 'member': member, 'frames': len(motion), 'methods': results})
    print(json.dumps({'fps': 20, 'steps_per_method': 120, 'learning_rate': .015,
                      'foot_position_weight': .5, 'contact_velocity_weight': 3., 'samples': rows,
                      'warning': 'Interpolation without antialiasing; 6 samples only; no physics validation.'}))


if __name__ == '__main__':
    main()

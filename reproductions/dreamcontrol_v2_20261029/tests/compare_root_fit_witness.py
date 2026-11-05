"""Compare experimental 29→27 fitting methods on a few existing clips.

The pickle is trusted local project data and may execute code on load. These
samples are exploratory and cannot establish physics or dataset-wide quality.
"""
import json
import pickle
from pathlib import Path
import numpy as np
import torch

from g1_fk import URDFFK
from motion_bridge import SOURCE_NAMES
from waist_lock_ik import FIT_LINKS, fit_locked_waist
from waist_lock_root_ik import fit_root_and_waist


def main():
    torch.set_num_threads(1)
    root = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation')
    assets = root / 'third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1'
    u29, u27 = assets/'g1_29dof.urdf', assets/'g1_27dof.urdf'
    source = '/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/data/amass_g1_36dim/full_train.pkl'
    with open(source, 'rb') as f:
        data = pickle.load(f)['data_dict']
    fk29, fk27 = URDFFK(u29), URDFFK(u27)
    rows = []
    for key in ['000000', '000004', '000005', '000007', '000009']:
        motion = data[key]['motion'][::3]
        truth = fk29.forward(motion[:, :3], motion[:, 3:7], motion[:, 7:], SOURCE_NAMES, links=FIT_LINKS)
        baseline = fit_locked_waist(motion, u29, u27, steps=120, learning_rate=.015,
                                    nominal_weight=.002, velocity_weight=.01)
        bounded = fit_root_and_waist(motion, u29, u27, steps=120)
        entries = {}
        for label, result in [('joint_only', baseline), ('bounded_root_joint', bounded)]:
            root_pos = result.get('root_pos', motion[:, :3])
            root_quat = result.get('root_quat_xyzw', motion[:, 3:7])
            pred = fk27.forward(root_pos, root_quat, result['joint_pos'], result['joint_names'], links=FIT_LINKS)
            err = np.linalg.norm(pred-truth, axis=-1)
            foot_speed_source = np.linalg.norm(np.diff(truth[:, 4:6], axis=0), axis=-1)*20
            foot_speed_fit = np.linalg.norm(np.diff(pred[:, 4:6], axis=0), axis=-1)*20
            contact = foot_speed_source < .03  # Explicit exploratory proxy, not paper threshold.
            entries[label] = {'mean_error_m': float(err.mean()), 'max_error_m': float(err.max()),
                              'mean_foot_error_m': float(err[:, 4:6].mean()),
                              'max_foot_error_m': float(err[:, 4:6].max()),
                              'mean_wrist_error_m': float(err[:, 10:12].mean()),
                              'contact_proxy_count': int(contact.sum()),
                              'contact_proxy_mean_fit_speed_mps': float(foot_speed_fit[contact].mean()) if contact.any() else None,
                              'max_root_translation_m': result.get('max_root_translation_m', 0.),
                              'max_root_rotation_rad': result.get('max_root_rotation_rad', 0.)}
            if abs(entries[label]['mean_error_m'] - result['final_mean_keypoint_error_m']) > 1e-5:
                raise AssertionError('independent world-space FK disagrees with optimizer')
        rows.append({'id': key, 'frames': len(motion), 'methods': entries})
    print(json.dumps({'fps': 20, 'source': source, 'steps_per_method': 120, 'learning_rate': .015, 'samples': rows,
                      'warning': 'Exploratory sample comparison, not paper-faithful, not physical validation.'}))


if __name__ == '__main__':
    main()

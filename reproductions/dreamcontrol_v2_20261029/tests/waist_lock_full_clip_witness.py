"""Exploratory full-clip CPU fit; not the authors' undisclosed optimizer."""
import json
import pickle
from pathlib import Path
import numpy as np
import torch
from waist_lock_ik import fit_locked_waist
from g1_fk import URDFFK
from motion_bridge import prepare_fitted_clip
from feature263 import encode
from data_loaders.humanml.scripts.motion_process import recover_from_ric

torch.set_num_threads(1)
root = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation')
assets = root / 'third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1'
source = '/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/data/amass_g1_36dim/full_train.pkl'
with open(source, 'rb') as f:
    corpus = pickle.load(f)
motion = corpus['data_dict']['000000']['motion'][::3]
result = fit_locked_waist(motion, assets / 'g1_29dof.urdf', assets / 'g1_27dof.urdf',
                          steps=40, learning_rate=0.02, nominal_weight=0.002, velocity_weight=0.01)
clip = prepare_fitted_clip(motion, result['joint_pos'], result['joint_names'], URDFFK(assets / 'g1_27dof.urdf'))
features = encode(clip['points_z_up_world'], foot_velocity_squared_threshold=0.002)
decoded = recover_from_ric(torch.from_numpy(features), 22).detach().cpu().numpy()
from feature263 import canonicalize_z_up
errors = np.linalg.norm(decoded - canonicalize_z_up(clip['points_z_up_world'])[:-1], axis=-1)
print(json.dumps({'id': '000000', 'frames': len(motion), 'steps': 40,
                  'source_max_waist_angle_rad': float(np.abs(motion[:, [20, 21]]).max()),
                  'initial_mean_keypoint_error_m': result['initial_mean_keypoint_error_m'],
                  'final_mean_keypoint_error_m': result['final_mean_keypoint_error_m'],
                  'final_max_keypoint_error_m': result['final_max_keypoint_error_m'],
                  'joint_shape': list(result['joint_pos'].shape),
                  'feature_shape': list(features.shape),
                  'reconstruction_max_error_m': float(errors.max()),
                  'warning': 'experimental 29-to-27 fitting only; no physical or paper-fidelity claim'}))

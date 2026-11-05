"""CPU smoke on six frames of an existing trusted G1 corpus clip."""
import json
import pickle
from pathlib import Path
import numpy as np
import torch
from waist_lock_ik import fit_locked_waist
from g1_fk import URDFFK
from motion_bridge import prepare_fitted_clip
from feature263 import encode

torch.set_num_threads(1)
root = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation')
assets = root / 'third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1'
source = '/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/data/amass_g1_36dim/full_train.pkl'
with open(source, 'rb') as f:
    corpus = pickle.load(f)
item = corpus['data_dict']['000000']
motion = item['motion'][:18:3]
result = fit_locked_waist(motion, assets / 'g1_29dof.urdf', assets / 'g1_27dof.urdf',
                          steps=20, learning_rate=0.02, nominal_weight=0.002, velocity_weight=0.01)
clip = prepare_fitted_clip(motion, result['joint_pos'], result['joint_names'], URDFFK(assets / 'g1_27dof.urdf'))
features = encode(clip['points_z_up_world'], foot_velocity_squared_threshold=0.002)
print(json.dumps({'id': '000000', 'frames': len(motion), 'steps': 20,
                  'source_max_waist_angle_rad': float(np.abs(motion[:, [20, 21]]).max()),
                  'initial_mean_keypoint_error_m': result['initial_mean_keypoint_error_m'],
                  'final_mean_keypoint_error_m': result['final_mean_keypoint_error_m'],
                  'final_max_keypoint_error_m': result['final_max_keypoint_error_m'],
                  'joint_shape': list(result['joint_pos'].shape),
                  'feature_shape': list(features.shape), 'finite_features': bool(np.isfinite(features).all()),
                  'warning': 'experimental fit only; not v2 retargeting or full-clip validation'}))

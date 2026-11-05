# Append g1_fk.py, motion_bridge.py and feature263.py to run this witness.
import json
import pickle
import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric

root = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation')
model = URDFFK(root / 'third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
source = '/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/data/amass_g1_36dim/full_train.pkl'
with open(source, 'rb') as f:
    corpus = pickle.load(f)
results = []
for key, item in corpus['data_dict'].items():
    if np.max(np.abs(item['motion'][:, [20, 21]])) <= 0.01:
        clip = prepare_near_locked_clip(item['motion'], model, source_fps=60, target_fps=20, max_locked_angle_rad=0.01)
        features = encode(clip['points_z_up_world'], foot_velocity_squared_threshold=0.002)
        decoded = recover_from_ric(torch.from_numpy(features), 22).detach().cpu().numpy()
        expected = canonicalize_z_up(clip['points_z_up_world'])[:-1]
        errors = np.linalg.norm(decoded - expected, axis=-1)
        results.append({'id': key, 'frames': len(features), 'feature_shape': list(features.shape),
                        'finite': bool(np.isfinite(features).all()),
                        'reconstruction_mean_error_m': float(errors.mean()),
                        'reconstruction_max_error_m': float(errors.max())})
print(json.dumps({'count': len(results), 'samples': results, 'threshold': 0.002,
                  'approximate_waist_lock': True,
                  'status': 'experimental encode smoke; not verified author pipeline'}))

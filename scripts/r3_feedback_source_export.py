"""Read-only source identity export in the existing official retarget runtime."""
import json
import pickle
import hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/'artifacts/feedback_world_model_20261006'
refs = np.load(ROOT/'artifacts/sim2sim_mujoco/interfaces/Kick_references.npz')
human_path = ART/'sources/Kick_official_results.npy'
assert hashlib.sha256(human_path.read_bytes()).hexdigest() == '98d40fae4e8436ee5018ff83fbb589a176f81dd025faf92ee5a8eb2296436396'
human = np.load(human_path, allow_pickle=True).item()['motion'].transpose(0, 3, 1, 2)
source = ROOT/'third_party/DreamControl/TrajGen/sample/Kick_sim2'
mapping = {}
for path in source.glob('*.pkl'):
    data = pickle.load(path.open('rb'))
    joints = np.asarray(data['joints'])[:200]
    errors = np.max(np.abs(refs['dof_pos']-joints[None, :, :]), axis=(1, 2))
    index = int(np.argmin(errors))
    assert errors[index] < 1e-6 and index not in mapping
    mapping[index] = {'reference': index, 'source_sample': int(path.stem.split('_')[0]),
                      'pkl': path.name, 'joint_identity_max_abs_error': float(errors[index])}
assert len(mapping) == 66
# Refinement inserted 10 pause +10 interpolation frames; retarget discarded
# original human frame0. These facts come from the fixed original scripts.
paired = np.stack([human[mapping[i]['source_sample'], 1:181] for i in range(66)])
np.savez_compressed(ART/'sources/paired_human_xyz.npz', human_xyz=paired)
(ART/'sources/mapping.json').write_text(json.dumps({'status': 'PASS', 'entries': list(mapping.values()),
    'official_human_sha256': hashlib.sha256(human_path.read_bytes()).hexdigest(),
    'robot_frame_range': [20, 200], 'human_frame_range': [1, 181],
    'retarget_dropped_frames': 1, 'refine_pause_frames': 10, 'refine_interpolation_frames': 10,
    'reference_period_s': .05, 'human_period_s': .05}, indent=2)+'\n')
print('FEEDBACK_SOURCE_PAIRING_PASS', len(mapping))

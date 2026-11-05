"""Exploratory time-corrected SSM dynamics, using the older MuJoCo PD scene."""
import json
import os
from pathlib import Path
import numpy as np
import torch

from ssm_robot_time_fit_audit import (ROOT, PAIR, SOURCE_NAMES, digest, convert_layout,
                                      resample, resample_antialias, fit_root_and_waist)
from mujoco_dynamic_contact_witness import simulate


def main():
    torch.set_num_threads(1)
    ids = [json.loads(x) for x in (ROOT / 'ssm_id_alignment_20261030/ids.jsonl').read_text().splitlines()]
    names = json.loads((PAIR / 'code/configs/g1_codec_skeleton.json').read_text())['joint_names']
    if len(names) != 29 or set(names) != set(SOURCE_NAMES):
        raise ValueError('unexpected source joint layout')
    order = [names.index(n) for n in SOURCE_NAMES]
    project = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl')
    assets = project / 'Training/HumanoidVerse/humanoidverse/data/robots/g1'
    u29, u27 = assets / 'g1_29dof.urdf', assets / 'g1_27dof.urdf'
    scene = project / 'Sim2Real/resources/robots/g1_description/scene_full.xml'
    gains = project / 'Sim2Real/deploy/deploy_mujoco/configs/g1_full_body.yaml'
    results = []
    for ident in ('002511', '002941', '007410'):
        row = next(x for x in ids if x['motion_id'] == ident and not x['mirrored'])
        rec = json.loads(Path(row['source_receipt']).read_text())
        source = PAIR / 'ssm_batch_v1' / rec['retargeted_source']
        if digest(source) != rec['retargeted_source_sha256']:
            raise ValueError('source hash mismatch')
        with np.load(source, allow_pickle=False) as data:
            native, frames = data['native'].copy(), data['reference_raw_frames'].copy()
        if not np.array_equal(frames, np.arange(0, row['reference_raw_frame_count'], row['recorded_stride'])):
            raise ValueError('frame index mismatch')
        native = native[row['start']:row['stop']]
        q = np.concatenate((native[:, 3:7], native[:, :3], native[:, 13:42][:, order]), axis=1)
        fps = row['reference_fps'] / row['recorded_stride']
        if abs(fps - 30.) > .01:
            raise ValueError('not a near-30 Hz source')
        filtered = resample(resample_antialias(q, 30., 20.), fps * 2 / 3, 20.)[:200]
        motion, _ = convert_layout(filtered)
        fit = fit_root_and_waist(motion, u29, u27, steps=120, learning_rate=.015,
              foot_position_weight=2., contact_velocity_weight=10., ground_penetration_weight=10.,
              ground_plane_z=0., ground_contact_weight=10., ground_contact_band_m=.005)
        metrics = simulate(scene, gains, u27, fit, seconds=1.)
        results.append({'motion_id': ident, 'source_path': row['source_path'],
                        'retargeted_source_sha256': digest(source), 'metrics': metrics})
        print('completed dynamic diagnostic', ident, flush=True)
    report = {'samples': results, 'scene_sha256': digest(scene), 'gains_sha256': digest(gains),
              'urdf27_sha256': digest(u27), 'script_sha256': digest(__file__),
              'warning': 'Three motions from same subject; FIR applied to old robot retargeting. Existing MuJoCo scene uses gravity -7.5 m/s2, 0.002s step, and extra hand joints; simple joint PD with zero initial velocities, no balance controller or task object. Source floor unverified. Not Isaac Lab, source physical ground truth, a general controllability verdict, or training acceptance.'}
    with Path(__file__).with_name('dynamic_report.json').open('x', encoding='utf-8') as f:
        json.dump(report, f, allow_nan=False, indent=2)
        f.flush()
        os.fsync(f.fileno())


if __name__ == '__main__':
    main()

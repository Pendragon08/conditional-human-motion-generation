"""Exploratory SSM robot retiming and 29->27 fit; not data acceptance."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import mujoco

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PAIR = Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021')
sys.path.insert(0, str(ROOT / 'foot_contact_fit_20261030'))
sys.path.insert(1, str(ROOT / 'antialias_mujoco_20261030'))
from foot_geometry_audit import foot_spheres, sphere_centers, compare
from g1_fk import URDFFK
from motion_bridge import SOURCE_NAMES
from mujoco_kinematic_contacts import scan
from omniretarget_bridge import convert_layout
from resample_omniretarget import resample
from resample_antialias_omniretarget import resample_antialias
from waist_lock_root_ik import fit_root_and_waist


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    torch.set_num_threads(1)
    ids_file = ROOT / 'ssm_id_alignment_20261030/ids.jsonl'
    chosen = ('002511', '002941', '007410')  # hopping, walking, punching
    rows = [json.loads(line) for line in ids_file.read_text().splitlines()]
    selected = [next(x for x in rows if x['motion_id'] == n and not x['mirrored']) for n in chosen]
    skeleton_file = PAIR / 'code/configs/g1_codec_skeleton.json'
    skeleton = json.loads(skeleton_file.read_text())
    names = skeleton['joint_names']
    if len(names) != 29 or len(set(names)) != 29 or set(names) != set(SOURCE_NAMES):
        raise ValueError('codec joint names do not exactly match source G1')
    order = [names.index(n) for n in SOURCE_NAMES]
    project = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl')
    assets = project / 'Training/HumanoidVerse/humanoidverse/data/robots/g1'
    u29, u27 = assets / 'g1_29dof.urdf', assets / 'g1_27dof.urdf'
    scene = project / 'Sim2Real/resources/robots/g1_description/scene_full.xml'
    offsets, radii = foot_spheres(u29)
    offsets27, radii27 = foot_spheres(u27)
    if not (np.allclose(offsets, offsets27, atol=1e-12, rtol=0) and np.allclose(radii, radii27, atol=1e-12, rtol=0)):
        raise ValueError('foot geometry mismatch')
    fk29, fk27 = URDFFK(u29), URDFFK(u27)
    model = mujoco.MjModel.from_xml_path(str(scene))
    results = []
    for row in selected:
        print('checking', row['motion_id'], row['source_path'], flush=True)
        receipt_path = Path(row['source_receipt'])
        receipt = json.loads(receipt_path.read_text())
        if receipt['timeline']['status'] != 'FRAME_IDENTITY_VERIFIED_BY_MEASURED_MARKERS':
            raise ValueError('reference markers not verified')
        for key, expected in [('reference_file', 'reference_sha256'), ('smplx_file', 'smplx_sha256')]:
            if digest(receipt[key]) != receipt[expected]:
                raise ValueError('original input hash changed: ' + key)
        source = PAIR / 'ssm_batch_v1' / receipt['retargeted_source']
        if digest(source) != receipt['retargeted_source_sha256']:
            raise ValueError('retargeted input hash changed')
        with np.load(source, allow_pickle=False) as d:
            native = d['native'].copy()
            frames = d['reference_raw_frames'].copy()
        stride = row['recorded_stride']
        expected_frames = np.arange(0, row['reference_raw_frame_count'], stride)
        if not np.array_equal(frames, expected_frames) or len(native) != len(frames):
            raise ValueError('retargeted frames do not match source sampling')
        if native.ndim != 2 or native.shape[1] < 42:
            raise ValueError('unexpected native feature layout')
        # The old velocity features were calculated at an assumed 20 Hz. Do not use them.
        native = native[row['start']:row['stop']]
        q = np.concatenate([native[:, 3:7], native[:, :3], native[:, 13:42][:, order]], axis=1)
        if not np.isfinite(q).all() or not np.allclose(np.linalg.norm(q[:, :4], axis=1), 1, atol=1e-4, rtol=0):
            raise ValueError('invalid positions or root quaternions')
        fps = float(row['reference_fps']) / stride
        if abs(fps - 30) > .01:
            raise ValueError('expected the near-30 Hz recording cohort')
        linear = resample(q, fps, 20.)
        nominal_filtered = resample_antialias(q, 30., 20.)
        actual_filtered_rate = fps * (2. / 3.)
        filtered = resample(nominal_filtered, actual_filtered_rate, 20.)
        count = min(len(linear), len(filtered), 200)
        if count < 2:
            raise ValueError('too short')
        item = {'motion_id': row['motion_id'], 'source_path': row['source_path'],
                'reference_fps': row['reference_fps'], 'input_effective_fps': fps,
                'filtered_intermediate_fps': actual_filtered_rate, 'output_fps': 20.,
                'input_frames': len(q), 'linear_frames': len(linear), 'filtered_frames': len(filtered),
                'compared_frames': count, 'receipt_sha256': digest(receipt_path),
                'retargeted_source_sha256': digest(source), 'methods': []}
        for label, value in [('linear', linear[:count]), ('fir', filtered[:count])]:
            motion, _ = convert_layout(value)
            source_feet = sphere_centers(fk29, motion[:, :3], motion[:, 3:7], motion[:, 7:], SOURCE_NAMES, offsets)
            fit = fit_root_and_waist(motion, u29, u27, steps=120, learning_rate=.015,
                    foot_position_weight=2., contact_velocity_weight=10.,
                    ground_penetration_weight=10., ground_plane_z=0.,
                    ground_contact_weight=10., ground_contact_band_m=.005)
            target_feet = sphere_centers(fk27, fit['root_pos'], fit['root_quat_xyzw'],
                                         fit['joint_pos'], fit['joint_names'], offsets27)
            item['methods'].append({'name': label,
                'max_keypoint_error_m': fit['final_max_keypoint_error_m'],
                'mean_keypoint_error_m': fit['final_mean_keypoint_error_m'],
                'foot_geometry': compare(source_feet, target_feet, radii27, fps=20, plane_z=0.,
                                         height_tolerance=.02, max_vertical_speed=.1),
                'static_contacts': scan(model, fit)})
            print('finished', row['motion_id'], label, flush=True)
        results.append(item)
    report = {'samples': results, 'ids_sha256': digest(ids_file), 'skeleton_sha256': digest(skeleton_file),
              'urdf29_sha256': digest(u29), 'urdf27_sha256': digest(u27), 'scene_sha256': digest(scene),
              'script_sha256': digest(__file__), 'mujoco_version': mujoco.__version__,
              'warning': 'Exploratory old GMR robot qpos, not paper v2 data. All three motions have the same subject. FIR applied after retargeting is approximate and is not the same as filtering human motion before retargeting. Old velocities are discarded. Static geometry assumes flat z=0 with no task object; source floor is unverified; collision filters differ from Isaac Lab. No dynamics or training acceptance.'}
    output = Path(__file__).with_name('report.json')
    with open(output, 'x', encoding='utf-8') as f:
        json.dump(report, f, allow_nan=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    print('complete', output, flush=True)


if __name__ == '__main__':
    main()

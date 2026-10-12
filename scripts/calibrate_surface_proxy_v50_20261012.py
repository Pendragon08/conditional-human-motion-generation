"""Analytical rigid-body contact controls, NOT simulator efficacy or contact GT.

The frozen V21 metric is exercised on independently prescribed no-slip pivot
kinematics. Inputs use the four actual collision boxes; no human trial is
reclassified, and no threshold or source file from earlier runs is modified.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from physics_guided_metric_v21_20261012 import surface_metrics, rotate_xyzw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/surface_proxy_calibration_v50_20261012'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output = OUT / 'first_attempt'
    assert not output.exists(), 'No retry or overwrite; retain first attempt'
    output.mkdir(parents=True)
    meta_path = ROOT / 'results/native_reference_velocity_v46_20261012/public/raw/val_native_replay_s11/metadata.json'
    threshold_path = ROOT / 'results/native_reference_velocity_v46_20261012/frozen_thresholds.json'
    meta = json.loads(meta_path.read_text())
    thresholds = json.loads(threshold_path.read_text())['thresholds']
    cases = []
    arrays = {}
    for name, original in meta['foot_geometry'].items():
        points = np.asarray(original['points'], dtype=float)
        assert len(points) == 8 and not any(original['radii'])
        k = meta['names'].index(name)
        com = np.asarray(meta['native_body_properties'][0][k]['com'], dtype=float)
        geometry = {name: dict(original, com_local=com.tolist())}
        for label, angles, angular_speed, translation, airborne in (
            ('static', np.zeros(60), 0., 0., False),
            ('flat_sliding', np.zeros(60), 0., .2, False),
            ('positive_rocking', np.linspace(.03, .8, 60), 3., 0., False),
            ('negative_rocking', np.linspace(-.8, -.03, 60), 3., 0., False),
            ('rocking_plus_sliding', np.linspace(.03, .8, 60), 3., .2, False),
            ('airborne', np.linspace(.03, .8, 60), 3., 0., True),
        ):
            quat = np.zeros((60, 1, 4)); quat[:, 0, 1] = np.sin(angles / 2); quat[:, 0, 3] = np.cos(angles / 2)
            # Choose the geometric rocking edge from analytic box bounds,
            # independently of the metric's argmin surface-point selection.
            lo, hi = points.min(axis=0), points.max(axis=0)
            pivot_local = np.tile(lo, (60, 1))
            pivot_local[angles > 0, 0] = hi[0]
            pivot = rotate_xyzw(quat[:, 0], pivot_local)
            position = np.zeros((60, 1, 3)); position[:, 0, 2] = -pivot[:, 2] + (1. if airborne else 0.)
            omega = np.zeros((60, 1, 3)); omega[:, 0, 1] = angular_speed
            # Prescribe the instantaneous contact material point's velocity
            # independently: v_origin = -omega x r_pivot + sliding velocity.
            origin_velocity = -np.cross(omega[:, 0], pivot)
            origin_velocity[:, 0] += translation
            linear_velocity = (origin_velocity + np.cross(omega[:, 0], rotate_xyzw(quat[:, 0], com)))[:, None]
            force = np.zeros((60, 1, 3)); force[:, 0, 2] = 0. if airborne else 49.05
            result = surface_metrics(position, quat, linear_velocity, omega, force, [name], geometry, thresholds)
            expected_sliding = translation > thresholds['sliding_speed_m_s']
            if airborne:
                passed = result['support_observations'] == 0 and result['surface_slip_proxy_fraction'] is None
            else:
                passed = result['support_observations'] == 60 and result['surface_slip_proxy_fraction'] == float(expected_sliding)
            cases.append(dict(foot=name, control=label, expected_contact_xy_speed_m_s=translation,
                              prescribed_angular_speed_rad_s=angular_speed, expected_supported=not airborne,
                              check_pass=bool(passed), metrics=result))
            for field, value in dict(position=position, quaternion=quat, com_velocity=linear_velocity,
                                     angular_velocity=omega, prescribed_force=force).items():
                arrays[name + '_' + label + '_' + field] = value
    np.savez_compressed(output / 'analytical_controls.npz', **arrays)
    result = dict(status='PASS_ANALYTICAL_CONTROLS' if all(x['check_pass'] for x in cases) else 'FAIL_RETAINED',
                  checks=len(cases), cases=cases, source_sha256=sha(__file__),
                  input_hashes={str(p.relative_to(ROOT)): sha(p) for p in (
                      meta_path, threshold_path, ROOT / 'scripts/physics_guided_metric_v21_20261012.py')},
                  arrays_sha256=sha(output / 'analytical_controls.npz'),
                  numpy_version=np.__version__, generation_calls=0, simulation_calls=0, gpu_calls=0,
                  thresholds_changed=False, previous_scores_changed=False, innovation_confirmed=False,
                  scope='Prescribed instantaneous rigid-body kinematics, synthetic unit controls. Force is prescribed, not measured. Angles are parameter samples, not an integrated time trajectory. Positive/negative box rocking about a lowest edge is distinguished from translation. This does not establish native contact locations, contact-force distribution, simulator rolling accuracy, or whether human trajectories actually slide. Angular-speed association alone is insufficient to invalidate the old proxy.')
    (output / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print('SURFACE_PROXY_ANALYTICAL_CONTROLS', result['status'], len(cases), flush=True)


if __name__ == '__main__':
    main()

"""Read-only contact geometry diagnosis; no simulation or threshold revision."""
import hashlib
import html
import json
from pathlib import Path
import re

import numpy as np
from physics_guided_core import rotate_xyzw, surface_metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/contact_slip_diagnostic_v13_20261012'
INPUT = ROOT / 'results/contact_cpu_validation_v12_20261012'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantiles(values):
    return np.quantile(values, [.05, .5, .95]).tolist() if len(values) else None


def box_check(thresholds):
    path = ROOT / 'results/contact_force_fixture_v3_20261012/repeat_a/fixture_arrays.npz'
    with np.load(path, allow_pickle=False) as a:
        state = a['state']; force = a['net_contact_force']
    points = [[x, y, z] for x in (-.1, .1) for y in (-.1, .1) for z in (-.05, .05)]
    geometry = dict(box=dict(points=points, radii=[0.] * 8))
    rows = []
    for name, i, interval in [('known_stationary', 0, slice(60, 120)),
                              ('known_frictionless_slide', 1, slice(60, 120)),
                              ('known_airborne', 2, slice(0, 20))]:
        s = state[interval, i, None]
        row = surface_metrics(s[..., :3], s[..., 3:7], s[..., 7:10],
                              s[..., 10:13], force[interval, i, None], ['box'], geometry, thresholds)
        rows.append(dict(name=name, **row))
    checks = dict(stationary_classified_not_sliding=rows[0]['surface_slip_proxy_fraction'] == 0,
                  known_slide_classified_sliding=rows[1]['surface_slip_proxy_fraction'] == 1,
                  airborne_has_no_support=rows[2]['support_observations'] == 0)
    assert all(checks.values()), 'Frozen proxy fails known box examples'
    return dict(checks=checks, rows=rows, input_sha256=sha(path),
                scope='Translating non-articulated box; does not certify rolling feet or humanoid quality')


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'diagnostic.json').exists(), 'Preserve previous analysis attempt'
    thresholds = json.loads((INPUT / 'frozen_thresholds.json').read_text())['thresholds']
    summary = json.loads((INPUT / 'validation_summary.json').read_text())
    assert summary['attempts'] == 432 and summary['full_quality_floor']
    all_rows = []
    inputs = {str(INPUT.relative_to(ROOT) / name): sha(INPUT / name)
              for name in ('frozen_thresholds.json', 'validation_summary.json')}
    recompute_error = 0.
    for entry in summary['entries']:
        label = 'val_%s_s%d' % (entry['mode'], entry['seed'])
        path = INPUT / 'raw' / label
        meta = json.loads((path / 'metadata.json').read_text())
        rows = json.loads((path / 'metrics.json').read_text())
        assert sha(path / 'metadata.json') == entry['metadata_sha256']
        assert sha(path / 'metrics.json') == entry['metrics_sha256']
        for name in ('metadata.json', 'metrics.json', 'trajectories.npz'):
            inputs[str((path / name).relative_to(ROOT))] = sha(path / name)
        with np.load(path / 'trajectories.npz', allow_pickle=False) as a:
            for case, row in enumerate(rows):
                valid = a['valid'][:, case]
                start = min(meta['context_steps'], int(valid.sum()) - 1)
                pos = a['actual'][valid, case][start:]
                quat = a['quaternion'][valid, case][start:]
                vel = a['linvel'][valid, case][start:]
                omega = a['angvel'][valid, case][start:]
                force = a['contact_force'][valid, case][start:]
                original = surface_metrics(pos, quat, vel, omega, force, meta['names'], meta['foot_geometry'], thresholds)
                recompute_error = max(recompute_error, abs(original['surface_slip_proxy_fraction'] - row['surface_slip_proxy_fraction']))
                speeds = []; optimistic = []; origins = []; differences = []; supports = []; angular = []
                per_body = []
                for name, geometry in meta['foot_geometry'].items():
                    i = meta['names'].index(name)
                    points = np.asarray(geometry['points'], dtype=float)
                    radii = np.asarray(geometry['radii'], dtype=float)
                    rotated = rotate_xyzw(quat[:, i, None], points[None])
                    world = pos[:, i, None] + rotated
                    world[..., 2] -= radii[None]
                    lowest = world[..., 2].min(axis=1)
                    index = world[..., 2].argmin(axis=1)
                    lever = world - pos[:, i, None]
                    point_vel = vel[:, i, None] + np.cross(omega[:, i, None], lever)
                    point_speed = np.linalg.norm(point_vel[..., :2], axis=-1)
                    selected = point_speed[np.arange(len(pos)), index]
                    support = (force[:, i, 2] > thresholds['contact_force_n']) & (lowest < thresholds['surface_near_ground_m'])
                    # Diagnostic lower bound only: selecting the slowest point
                    # would be optimistic. Never substitute it for old scores.
                    near_lowest = world[..., 2] <= lowest[:, None] + .002
                    slowest = np.where(near_lowest, point_speed, np.inf).min(axis=1)
                    origin_speed = np.linalg.norm(vel[:, i, :2], axis=-1)
                    body_diff = np.linalg.norm(np.gradient(pos[:, i].astype(float), meta['dt'], axis=0) - vel[:, i], axis=-1)
                    speeds.extend(selected[support]); optimistic.extend(slowest[support]); origins.extend(origin_speed[support])
                    differences.extend(body_diff[support]); angular.extend(np.linalg.norm(omega[:, i], axis=-1)[support]); supports.extend(support.tolist())
                    per_body.append(dict(name=name, support_observations=int(support.sum()),
                        selected_speed_quantiles_m_s=quantiles(selected[support]),
                        linear_speed_quantiles_m_s=quantiles(origin_speed[support])))
                speeds = np.asarray(speeds); optimistic = np.asarray(optimistic)
                assert len(speeds) > 0
                all_rows.append(dict(label=label, case_id=row['case_id'], source_id=row['source_id'],
                    support_observations=len(speeds), saved_proxy_fraction=row['surface_slip_proxy_fraction'],
                    selected_point_speed_quantiles_m_s=quantiles(speeds),
                    optimistic_near_lowest_fraction=float((optimistic > thresholds['sliding_speed_m_s']).mean()),
                    optimistic_near_lowest_speed_quantiles_m_s=quantiles(optimistic),
                    body_linear_speed_quantiles_m_s=quantiles(np.asarray(origins)),
                    angular_speed_quantiles_rad_s=quantiles(np.asarray(angular)),
                    coarse_origin_difference_vs_exported_velocity_quantiles_m_s=quantiles(np.asarray(differences)),
                    per_body=per_body))
    assert len(all_rows) == 432 and recompute_error < 1e-12
    sdk = Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/isaacgym/docs')
    sdk_evidence = {}
    for relative, needles in [('api/python/struct_py.html', ['isaacgym.gymapi.RigidContact.local_pos0', 'isaacgym.gymapi.RigidContact.lambda', 'isaacgym.gymapi.RigidBodyProperties.com']),
                              ('api/python/gym_py.html', ['isaacgym.gymapi.Gym.get_env_rigid_contacts'])]:
        file = sdk / relative
        text = file.read_text()
        snippets = {}
        for needle in needles:
            match = re.search(r'<dt[^>]*id="' + re.escape(needle) + r'".*?</dd>', text, re.S)
            snippets[needle] = html.unescape(re.sub('<[^>]+>', ' ', match.group(0)))[:1500] if match else 'Not found in bundled documentation'
        sdk_evidence[relative] = dict(sha256=sha(file), snippets=snippets)
    result = dict(status='DIAGNOSTIC_ONLY', no_simulation_or_model_calls=True, inputs=432,
        known_box_proxy_calibration=box_check(thresholds), original_metric_recompute_max_abs_error=recompute_error,
        original_proxy_pass=sum(r['saved_proxy_fraction'] <= thresholds['sliding_fraction'] for r in all_rows),
        optimistic_near_lowest_pass=sum(r['optimistic_near_lowest_fraction'] <= thresholds['sliding_fraction'] for r in all_rows),
        interpretation='Optimistic alternative is a diagnostic, not contact truth or a replacement score. Coarse finite differences do not identify COM-versus-body velocity origin by themselves.',
        next_required='Known articulated support/rolling examples and native contact-point availability; do not relax frozen thresholds or claim efficacy.',
        thresholds_changed=False, innovation_confirmed=False, overleaf_results_sync=False,
        sdk_evidence=sdk_evidence, rows=all_rows, input_hashes=inputs,
        source_sha256=sha(Path(__file__)))
    (OUT / 'diagnostic.json').write_text(json.dumps(result, indent=2))
    print('SLIP_PROXY_OFFLINE_DIAGNOSTIC_PASS', json.dumps({key: result[key] for key in ('inputs','original_proxy_pass','optimistic_near_lowest_pass','original_metric_recompute_max_abs_error')}), flush=True)


if __name__ == '__main__':
    main()

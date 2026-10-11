"""Known geometric fixtures for the contact proxy, not simulator force truth."""
import hashlib
import json
from pathlib import Path
import numpy as np
from physics_guided_core import surface_metrics

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'results/contact_sampling_v5_20261011/raw/repeat_a/metadata.json'
THRESHOLDS = ROOT / 'results/contact_sampling_v5_20261011/frozen_thresholds.json'
OUT = ROOT / 'results/contact_geometry_calibration_20261011'


def main():
    meta = json.loads(INPUT.read_text())
    thresholds = json.loads(THRESHOLDS.read_text())['thresholds']
    names = meta['names']
    geometry = meta['foot_geometry']
    n = len(names)
    pos = np.zeros((60, n, 3))
    quat = np.zeros((60, n, 4)); quat[..., 3] = 1
    vel = np.zeros_like(pos); omega = np.zeros_like(pos)
    force = np.zeros_like(pos)
    for name, g in geometry.items():
        index = names.index(name)
        floor = min(float(p[2])-float(radius) for p, radius in zip(g['points'], g['radii']))
        pos[:, index, 2] = -floor
        force[:, index, 2] = 50.0

    def evaluate(p, v, w, f):
        return surface_metrics(p, quat, v, w, f, names, geometry, thresholds)

    result = {'stationary_grounded': evaluate(pos, vel, omega, force)}
    assert result['stationary_grounded']['surface_slip_proxy_fraction'] == 0
    assert result['stationary_grounded']['penetration_fraction'] == 0
    assert result['stationary_grounded']['airborne_fraction'] == 0
    sliding = vel.copy(); sliding[..., 0] = 0.2
    result['known_horizontal_slide'] = evaluate(pos, sliding, omega, force)
    assert result['known_horizontal_slide']['surface_slip_proxy_fraction'] == 1
    lifted = pos.copy(); lifted[..., 2] += 0.3
    result['airborne_no_force'] = evaluate(lifted, vel, omega, force*0)
    assert result['airborne_no_force']['surface_slip_proxy_fraction'] is None
    assert result['airborne_no_force']['airborne_fraction'] == 1
    assert result['airborne_no_force']['penetration_fraction'] == 0
    sunk = pos.copy(); sunk[..., 2] -= 0.03
    result['known_surface_penetration'] = evaluate(sunk, vel, omega, force)
    assert result['known_surface_penetration']['penetration_fraction'] == 1
    # For the current material-point proxy, pure rolling at the lowest point
    # has zero tangent velocity despite nonzero body-center linear speed.
    rolling_v = vel.copy(); rolling_w = omega.copy()
    for name, g in geometry.items():
        i = names.index(name)
        points = np.asarray(g['points']); radii = np.asarray(g['radii'])
        world = pos[:, i, None, :] + points[None]
        world[..., 2] -= radii
        closest = np.argmin(world[..., 2], axis=1)
        lever = world[np.arange(len(pos)), closest] - pos[:, i]
        rolling_w[:, i, 1] = 1.0
        rolling_v[:, i] = -np.cross(rolling_w[:, i], lever)
    result['rolling_zero_material_point_speed'] = evaluate(pos, rolling_v, rolling_w, force)
    assert result['rolling_zero_material_point_speed']['surface_slip_proxy_fraction'] == 0
    result.update(status='PASS_GEOMETRY_ONLY', thresholds_changed=False,
        force_source='Manually specified analytical fixtures; NOT recorded simulation forces',
        limitations=['Does not calibrate PhysX force reporting or actual contact point truth',
                     'Fixtures are not a feasible humanoid motion dataset',
                     'Airborne is a state label, not a floating-error judgment',
                     'End-to-end measurement and physical thresholds remain unvalidated'],
        inputs_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (INPUT, THRESHOLDS, Path(__file__))})
    OUT.mkdir(exist_ok=False)
    (OUT/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print('CONTACT_GEOMETRY_FIXTURES_PASS_5', flush=True)


if __name__ == '__main__':
    main()

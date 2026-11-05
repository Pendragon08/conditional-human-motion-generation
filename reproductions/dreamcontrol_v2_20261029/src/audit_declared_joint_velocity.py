"""Compare average joint rates to declared URDF limits; never accept a dataset.

Intervals are nominal and the URDF limits are declarations, not measured motor
performance or DreamControl-v2 filtering criteria. A passing average says nothing
about possible instantaneous peaks between samples.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def read_velocity_limits(path):
    limits = {}
    for joint in ET.parse(Path(path)).getroot().findall('joint'):
        if joint.get('type') != 'revolute':
            continue
        name = joint.get('name')
        if not name or name in limits:
            raise ValueError('missing or duplicate URDF joint name')
        element = joint.find('limit')
        value = None if element is None else element.get('velocity')
        if value is None:
            limits[name] = None
            continue
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError('URDF velocity must be finite and positive')
        limits[name] = value
    return limits


def audit_velocity(q, times, names, limits):
    q = np.asarray(q, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if (q.ndim != 2 or q.shape[0] < 2 or times.shape != (q.shape[0],) or
        len(names) != q.shape[1] or len(set(names)) != len(names) or
        not np.isfinite(q).all() or not np.isfinite(times).all() or
        np.any(np.diff(times) <= 0)):
        raise ValueError('invalid joint states, names or times')
    if any(name not in limits for name in names):
        raise ValueError('state joint not found in URDF')
    missing = [name for name in names if limits[name] is None]
    if missing:
        return {'status': 'missing_declared_limits', 'missing_joints': missing,
                'training_ready': False}
    cap = np.array([limits[name] for name in names], dtype=np.float64)
    if not np.isfinite(cap).all() or np.any(cap <= 0):
        raise ValueError('invalid velocity limits')
    speed = np.abs(np.diff(q, axis=0)) / np.diff(times)[:, None]
    ratios = speed / cap
    frame, joint = np.unravel_index(np.argmax(ratios), ratios.shape)
    return {'status': 'nominal_average_rates_checked', 'intervals': len(times)-1,
            'joint_interval_pairs_above_limit': int((ratios > 1).sum()),
            'intervals_with_any_joint_above_limit': int((ratios > 1).any(axis=1).sum()),
            'maximum_ratio': float(ratios[frame,joint]), 'worst_interval': int(frame),
            'worst_joint': names[joint], 'worst_average_rad_s': float(speed[frame,joint]),
            'worst_declared_limit_rad_s': float(cap[joint]), 'training_ready': False,
            'limitations': ['Nominal sample times are not verified capture times.',
                            'URDF velocity limits are declarations, not verified runtime actuator limits.',
                            'Averaging cannot bound instantaneous speed or prove physical feasibility.',
                            'Not an author v2 filtering rule or training acceptance.']}

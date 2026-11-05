"""Recompute and reject joint-limit violations on sampled robot trajectories.

Tolerance is required explicitly and is a project numerical allowance, not a
paper feasibility threshold. Passing this check alone never grants training
acceptance; callers must also validate provenance, time, captions and quality.
"""
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def read_joint_limits(path):
    limits = {}
    for node in ET.parse(Path(path)).getroot().findall('joint'):
        name, kind = node.get('name'), node.get('type')
        if kind == 'continuous':
            bounds = (-math.inf, math.inf)
        elif kind == 'revolute':
            limit = node.find('limit')
            if limit is None:
                raise ValueError('revolute joint lacks limits')
            bounds = (float(limit.get('lower')), float(limit.get('upper')))
            if not all(math.isfinite(v) for v in bounds) or bounds[0] > bounds[1]:
                raise ValueError('invalid URDF limits')
        else:
            continue
        if not name or name in limits:
            raise ValueError('missing or duplicate joint name')
        limits[name] = bounds
    if len(limits) not in (27, 29):
        raise ValueError('expected 27 or 29 actuated G1 joints')
    return limits


def check_joint_limits(joint_pos, joint_names, limits, *, tolerance_rad):
    if isinstance(tolerance_rad, bool) or not math.isfinite(tolerance_rad) or tolerance_rad < 0:
        raise ValueError('explicit nonnegative finite tolerance required')
    q = np.asarray(joint_pos, dtype=np.float64)
    names = list(joint_names)
    joint_count = len(limits)
    if joint_count not in (27, 29):
        raise ValueError('expected 27 or 29 actuated G1 joints')
    if q.ndim != 2 or q.shape[1] != joint_count or not len(q) or not np.isfinite(q).all():
        raise ValueError('joint positions must be nonempty finite (T,{}) radians'.format(joint_count))
    if len(names) != joint_count or len(set(names)) != joint_count or set(names) != set(limits):
        raise ValueError('joint order must name every URDF joint exactly once')
    low = np.array([limits[name][0] for name in names], dtype=np.float64)
    high = np.array([limits[name][1] for name in names], dtype=np.float64)
    if np.isnan(low).any() or np.isnan(high).any() or (low > high).any():
        raise ValueError('invalid limits')
    excess = np.maximum(np.maximum(low - q, q - high), 0.)
    row, col = np.unravel_index(excess.argmax(), excess.shape)
    max_excess = float(excess[row, col])
    return {'joint_limits_passed': bool(max_excess <= tolerance_rad),
            'tolerance_rad': float(tolerance_rad), 'max_excess_rad': max_excess,
            'worst_row': int(row), 'worst_joint': names[col],
            'violations_above_tolerance': int(np.count_nonzero(excess > tolerance_rad)),
            'training_ready': False}


def require_joint_limits(joint_pos, joint_names, limits, *, tolerance_rad):
    result = check_joint_limits(joint_pos, joint_names, limits, tolerance_rad=tolerance_rad)
    if not result['joint_limits_passed']:
        raise ValueError('G1 joint limits exceeded: {} rad at row {} joint {}'.format(
            result['max_excess_rad'], result['worst_row'], result['worst_joint']))
    return result

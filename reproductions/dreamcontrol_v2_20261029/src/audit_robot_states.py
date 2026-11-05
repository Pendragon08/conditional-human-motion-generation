"""Validate sampled robot state evidence against 263D features and G1 limits.

This is one necessary technical check. A pass is not physical, licensing,
source, caption, or overall training acceptance.
"""
import hashlib
from pathlib import Path

import numpy as np
from g1_joint_limit_gate import check_joint_limits, read_joint_limits
from audit_declared_joint_velocity import audit_velocity, read_velocity_limits


def _sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _resolve(root, name, expected):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError('artifact path must be relative')
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file() or path.suffix != '.npz':
        raise ValueError('artifact path must resolve to an NPZ inside its root')
    if not isinstance(expected, str) or len(expected) != 64 or _sha(path) != expected.lower():
        raise ValueError('artifact hash mismatch')
    return path


def audit_robot_states(records, *, feature_root, state_root, urdf_path, tolerance_rad,
                       fk_decode_tolerance_m, reencoding_tolerance, fk=None, decode=None, canonicalize=None, reencode=None):
    if not isinstance(records, list) or not records:
        raise ValueError('nonempty manifest records required')
    if not np.isfinite(fk_decode_tolerance_m) or fk_decode_tolerance_m <= 0:
        raise ValueError('explicit positive FK/decode tolerance required')
    if not np.isfinite(reencoding_tolerance) or reencoding_tolerance <= 0:
        raise ValueError('explicit positive re-encoding tolerance required')
    feature_root = Path(feature_root).resolve(strict=True)
    state_root = Path(state_root).resolve(strict=True)
    urdf_path = Path(urdf_path).resolve(strict=True)
    limits = read_joint_limits(urdf_path)
    velocity_limits = read_velocity_limits(urdf_path)
    if fk is None:
        from g1_fk import URDFFK
        fk = URDFFK(urdf_path).forward
    if decode is None:
        import torch
        from data_loaders.humanml.scripts.motion_process import recover_from_ric
        def decode(value):
            with torch.no_grad():
                return recover_from_ric(torch.from_numpy(value), 22).numpy()
    if canonicalize is None:
        from feature263 import canonicalize_z_up
        canonicalize = canonicalize_z_up
    # A named encoding is explicit per record.  The sign-continuous encoder is
    # a project approximation and remains subject to the overall source and
    # physical quality gate.  An injected callable retains the old test API;
    # it cannot be used to claim a named alternative encoding.
    if reencode is None:
        injected_callable = False
        from feature263 import encode as legacy_encode
        encoders = {'legacy': legacy_encode}
        if any(r.get('encoding_variant', 'legacy') == 'sign_continuous_v1' for r in records):
            from feature263_signfix import encode_signfixed
            encoders['sign_continuous_v1'] = encode_signfixed
    elif callable(reencode):
        injected_callable = True
        if any(r.get('encoding_variant', 'legacy') != 'legacy' for r in records):
            raise ValueError('named alternative encoding requires an explicit encoder map')
        encoders = {'legacy': reencode}
    elif isinstance(reencode, dict):
        injected_callable = False
        encoders = reencode
    else:
        raise ValueError('invalid re-encoding implementation')
    rows, ids = [], set()
    for rec in records:
        sample = rec.get('sample_id')
        if not isinstance(sample, str) or not sample or sample in ids:
            raise ValueError('missing or duplicate sample ID')
        ids.add(sample)
        variant = rec.get('encoding_variant', 'legacy')
        if not isinstance(variant, str) or variant not in {'legacy', 'sign_continuous_v1'} or not callable(encoders.get(variant)):
            raise ValueError('unsupported or unavailable encoding variant')
        feature_path = _resolve(feature_root, rec.get('feature_path'), rec.get('feature_sha256'))
        state_path = _resolve(state_root, rec.get('state_path'), rec.get('state_sha256'))
        with np.load(feature_path, allow_pickle=False) as f:
            raw = np.asarray(f['features'])
            if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
                raise ValueError('features must be real numeric')
            features = np.asarray(raw, dtype=np.float32)
        with np.load(state_path, allow_pickle=False) as s:
            if 'root_wxyz_xyz' in s and 'joint_pos' in s:
                if 'qpos29_source_layout_20hz' in s:
                    raise ValueError('ambiguous robot state layout')
                root = np.asarray(s['root_wxyz_xyz'], dtype=np.float64)
                q = np.asarray(s['joint_pos'], dtype=np.float64)
            elif 'qpos29_source_layout_20hz' in s:
                packed = np.asarray(s['qpos29_source_layout_20hz'], dtype=np.float64)
                if packed.ndim != 2 or packed.shape[1] not in (36, 43) or len(limits) != 29:
                    raise ValueError('invalid direct 29-DoF state layout')
                root, q = packed[:, :7], packed[:, 7:36]
            else:
                raise ValueError('missing robot state layout')
            times = np.asarray(s['sample_times'], dtype=np.float64)
            names = [str(n) for n in s['joint_names']]
        length = len(features)
        if (features.ndim != 2 or features.shape[1] != 263 or length < 2 or
            not np.isfinite(features).all() or root.shape != (length + 1, 7) or
            q.shape != (length + 1, len(limits)) or times.shape != (length + 1,) or
            not np.isfinite(root).all() or not np.isfinite(q).all() or not np.isfinite(times).all() or
            not np.allclose(np.diff(times), .05, rtol=0, atol=1e-8) or
            not np.allclose(np.linalg.norm(root[:, :4], axis=1), 1., rtol=0, atol=1e-5)):
            raise ValueError('invalid feature, state, quaternion or nominal 20 Hz timeline')
        limits_check = check_joint_limits(q, names, limits, tolerance_rad=tolerance_rad)
        points = np.asarray(fk(root[:, 4:], root[:, [1, 2, 3, 0]], q, names))
        if points.shape != (length + 1, 22, 3) or not np.isfinite(points).all():
            raise ValueError('FK returned invalid points')
        target, decoded = np.asarray(canonicalize(points))[:-1], np.asarray(decode(features))
        if (target.shape != (length, 22, 3) or decoded.shape != target.shape or
            not np.isfinite(target).all() or not np.isfinite(decoded).all()):
            raise ValueError('decoded points invalid')
        err = float(np.linalg.norm(target - decoded, axis=-1).max())
        if not np.isfinite(err) or err > fk_decode_tolerance_m:
            raise ValueError('state/feature FK consistency failed: {}'.format(sample))
        foot_threshold = rec.get('foot_velocity_squared_threshold')
        if (isinstance(foot_threshold, bool) or not isinstance(foot_threshold, (int, float)) or
            not np.isfinite(foot_threshold) or foot_threshold <= 0):
            raise ValueError('explicit positive foot velocity proxy threshold required')
        rebuilt = np.asarray(encoders[variant](points, foot_velocity_squared_threshold=foot_threshold))
        if rebuilt.shape != features.shape or not np.isfinite(rebuilt).all():
            raise ValueError('invalid re-encoded features')
        reencode_error = float(np.abs(rebuilt.astype(np.float64) - features.astype(np.float64)).max())
        if reencode_error > reencoding_tolerance or not np.array_equal(rebuilt[:, -4:], features[:, -4:]):
            raise ValueError('full feature re-encoding failed: {}'.format(sample))
        if _sha(feature_path) != rec['feature_sha256'].lower() or _sha(state_path) != rec['state_sha256'].lower():
            raise RuntimeError('artifact changed while auditing')
        rows.append({'sample_id': sample, 'feature_sha256': rec['feature_sha256'],
                     'state_sha256': rec['state_sha256'], 'fk_decode_max_error_m': err,
                     'encoding_variant': 'custom_callable_unidentified' if injected_callable else variant,
                     'feature_reencoding_max_abs_error': reencode_error,
                     'foot_velocity_squared_threshold': float(foot_threshold),
                     'joint_check': limits_check,
                     'declared_velocity_diagnostic': audit_velocity(q, times, names, velocity_limits)})
    failed = [row['sample_id'] for row in rows if not row['joint_check']['joint_limits_passed']]
    return {'state_technical_checks_passed': not failed, 'limit_failed_samples': failed,
            'cases': rows, 'urdf_sha256': _sha(urdf_path), 'training_ready': False,
            'limitations': ['Caller-supplied tolerances are not paper feasibility criteria.',
                            'Full feature re-encoding shares the experimental encoder and is not independent verification.',
                            'The configured foot velocity proxy is not physical contact.',
                            'Nominal 20 Hz and FK consistency do not verify actual timing or original-source identity.',
                            'No caption, source license, collision, contact or physical acceptance is established.']}

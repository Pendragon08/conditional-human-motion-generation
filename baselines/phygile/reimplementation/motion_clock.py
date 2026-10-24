"""Explicit training/control clock bridge for the independent 262D codec.

Interpolation preserves physical sample times. It is an implementation choice,
not an author release or a learned motion-quality result.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from codec262 import SCHEMA, derivative, matrix_quat, matrix_rotvec, quat_matrix

CONTRACT = 'phygile-feature-clock-v1'


def positive_rate(value):
    if isinstance(value, (bool, str)) or not np.isscalar(value):
        raise ValueError('An explicit numeric sample rate is required')
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError('A finite positive sample rate is required')
    return value


def validate_feature_contract(contract):
    if not isinstance(contract, dict) or contract.get('version') != CONTRACT:
        raise ValueError('Missing/versioned feature clock; legacy checkpoints require an audited migration')
    rate = positive_rate(contract.get('feature_fps'))
    if contract.get('coordinate_frame') != 'initial_root_xy_yaw':
        raise ValueError('This generator bridge requires the audited canonical root XY/yaw frame')
    return rate


def manifest_feature_contract(manifest, manifest_path):
    """Bind every real sequence's codec clock/frame metadata into the checkpoint.

    Do not guess FPS from a file length, text, control frequency or file name.
    Older codec sidecars lack a canonicalize flag; their actual canonical initial
    XY/yaw is checked explicitly. All sidecar hashes are retained for audit.
    """
    if manifest['purpose'] == 'unit_test':
        return {'version': CONTRACT, 'feature_fps': positive_rate(manifest['feature_fps']),
                'coordinate_frame': 'synthetic_unit_test', 'metadata': []}
    base = Path(manifest_path).resolve().parent
    expected_schema = manifest.get('codec_schema')
    expected_skeleton = manifest.get('codec_skeleton_sha256')
    if expected_schema != SCHEMA or not expected_skeleton:
        raise ValueError('Real training needs an explicit codec schema and skeleton')
    rows = []
    rate = None
    for split, entries in manifest['splits'].items():
        for entry in entries:
            if not entry.get('codec_metadata'):
                raise ValueError('Missing per-sequence codec clock/frame metadata')
            path = (base / entry['codec_metadata']).resolve(strict=True)
            raw = path.read_bytes()
            metadata = json.loads(raw)
            if metadata.get('schema') != expected_schema or metadata.get('skeleton_sha256') != expected_skeleton:
                raise ValueError('Sequence codec identity mismatch')
            fps = positive_rate(metadata.get('fps'))
            if rate is not None and not np.isclose(rate, fps, atol=1e-10, rtol=0):
                raise ValueError('Mixed feature clocks cannot share this training run')
            rate = fps
            gate = entry.get('sample_clock_gate')
            if gate is not None and (not gate.get('eligible') or not np.isclose(
                    positive_rate(gate.get('feature_fps')), fps, atol=1e-10, rtol=0)):
                raise ValueError('Sidecar and sample-clock gate disagree')
            initial = np.asarray(metadata['initial_root_pos'], dtype=float)
            rot = quat_matrix(metadata['initial_root_quat'])
            yaw = np.arctan2(rot[1, 0], rot[0, 0])
            if initial.shape != (3,) or not np.isfinite(initial).all() or max(np.abs(initial[:2])) > 1e-6 or abs(yaw) > 1e-6:
                raise ValueError('Training sequence is not in its initial root XY/yaw frame')
            transform = np.asarray(metadata['canonical_rotation'], dtype=float)
            offset = np.asarray(metadata['canonical_offset'], dtype=float)
            if (transform.shape != (3, 3) or not np.isfinite(transform).all() or
                    not np.allclose(transform @ transform.T, np.eye(3), atol=1e-7) or
                    not np.isclose(np.linalg.det(transform), 1, atol=1e-7) or
                    not np.allclose(transform[2], [0, 0, 1], atol=1e-7) or
                    offset.shape != (3,) or not np.isfinite(offset).all() or abs(offset[2]) > 1e-7):
                raise ValueError('Invalid canonical XY/yaw side information')
            rows.append({'split': split, 'features_sha256': entry['sha256'],
                         'codec_metadata_sha256': hashlib.sha256(raw).hexdigest()})
    if rate is None:
        raise ValueError('No real codec clock metadata')
    contract = {'version': CONTRACT, 'feature_fps': rate,
                'coordinate_frame': 'initial_root_xy_yaw',
                'root_velocity_frame': 'canonical_world',
                'metadata': rows, 'author_exact_mapping_verified': False}
    validate_feature_contract(contract)
    return contract


def resample_native(native, source_times, query_times):
    """Linear translation/joints and shortest-arc quaternion SLERP, no extrapolation.

    Native 71D root velocities are world-frame; new velocities are finite
    differences on the *query* clock, rather than interpolated old velocities.
    """
    native = np.asarray(native, dtype=np.float64)
    times = np.asarray(source_times, dtype=np.float64)
    query = np.asarray(query_times, dtype=np.float64)
    if (native.ndim != 2 or native.shape[1] != 71 or len(native) < 2 or
            times.shape != (len(native),) or query.ndim != 1 or len(query) < 2 or
            not all(np.isfinite(x).all() for x in (native, times, query)) or
            not (np.diff(times) > 0).all() or not (np.diff(query) > 0).all()):
        raise ValueError('Finite native poses and increasing time arrays required')
    if query[0] < times[0]-1e-10 or query[-1] > times[-1]+1e-10:
        raise ValueError('Clock bridge refuses extrapolation or time stretching')
    query = np.clip(query, times[0], times[-1])
    dt = np.diff(query)
    if not np.allclose(dt, dt[0], atol=1e-10, rtol=1e-9):
        raise ValueError('The output native sequence requires a uniform clock')
    fps = 1 / dt[0]
    pos = np.stack([np.interp(query, times, native[:, i]) for i in range(3)], -1)
    joints = np.stack([np.interp(query, times, native[:, i]) for i in range(13, 42)], -1)
    quat = native[:, 3:7].copy()
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    if (norm < 1e-10).any():
        raise ValueError('Zero root quaternion')
    quat /= norm
    index = np.clip(np.searchsorted(times, query, side='right')-1, 0, len(times)-2)
    fraction = ((query-times[index])/(times[index+1]-times[index]))[:, None]
    left, right = quat[index], quat[index+1]
    dot = (left*right).sum(-1, keepdims=True)
    right = np.where(dot < 0, -right, right)
    angle = np.arccos(np.clip(np.abs(dot), 0, 1))
    sine = np.sin(angle)
    qa = np.sin((1-fraction)*angle)/np.maximum(sine, 1e-15)
    qb = np.sin(fraction*angle)/np.maximum(sine, 1e-15)
    interpolated = np.where(sine > 1e-7, qa*left+qb*right, (1-fraction)*left+fraction*right)
    interpolated /= np.linalg.norm(interpolated, axis=1, keepdims=True)
    # Preserve exact source poses at exact sample times (including quaternion sign).
    for k, t in enumerate(query):
        exact = np.flatnonzero(np.isclose(times, t, atol=1e-10, rtol=0))
        if len(exact):
            interpolated[k] = quat[exact[0]]
    rot = quat_matrix(interpolated)
    omega = matrix_rotvec(rot[1:] @ rot[:-1].transpose(0, 2, 1))*fps
    return np.concatenate([pos, interpolated, derivative(pos, fps),
                           np.concatenate([omega, omega[-1:]], 0),
                           joints, derivative(joints, fps)], -1)


def prepare_context(codec, prefix, feature_fps, *, canonicalize, seconds=1.):
    """Feature grid ends exactly at the latest actual control sample, time zero.

    At cold start there may be only one observation. Missing history is an
    explicitly recorded held-pose initialization, never invented execution.
    """
    rate = positive_rate(feature_fps)
    control = positive_rate(prefix.fps)
    duration = min((len(prefix)-1)/control, seconds)
    intervals = int(np.floor(duration*rate+1e-9))
    padded = max(0, 1-intervals)
    if intervals < 1:
        native = np.repeat(prefix.states[-1:], 2, axis=0)
        # A held-pose history has zero differences, not claimed actual history.
        native[:, 7:13] = 0
        native[:, 42:71] = 0
        grid = np.array([-1/rate, 0.])
    else:
        grid = np.arange(-intervals, 1, dtype=float)/rate
        actual_times = np.arange(-(len(prefix)-1), 1, dtype=float)/control
        native = resample_native(prefix.states, actual_times, grid)
    native[-1] = prefix.states[-1]
    features, metadata = codec.encode(native, rate, canonicalize=canonicalize)
    transform = np.asarray(metadata['canonical_rotation'])
    # Actual observed boundary velocities must be expressed in the training frame.
    features[-1, :3] = transform @ prefix.states[-1, 10:13]
    features[-1, 3:6] = transform @ prefix.states[-1, 7:10]
    return features, metadata, {'control_fps': control, 'feature_fps': rate,
        'context_sample_times_relative_to_boundary_s': grid.tolist(),
        'held_initialization_intervals': padded, 'actual_history_invented': False,
        'canonicalized_for_training_frame': bool(canonicalize)}


def generation_grid(new_frames, control_fps, feature_fps, final):
    if isinstance(new_frames, bool) or not isinstance(new_frames, (int, np.integer)) or new_frames < 1:
        raise ValueError('A positive integer control frame count is required')
    control, rate = positive_rate(control_fps), positive_rate(feature_fps)
    duration = new_frames/control
    steps = int(np.ceil(duration*rate-1e-9))
    # A hard terminal pose needs a feature-grid endpoint; otherwise reject rather
    # than shift a 1.02s target to 1.00s or stretch the generated sequence.
    if final and not np.isclose(duration*rate, round(duration*rate), atol=1e-8, rtol=0):
        raise ValueError('Final terminal time is off the feature grid; choose an aligned control frame count')
    return np.arange(steps+1)/rate, np.arange(new_frames+1)/control


def terminal_features(codec, suffix, metadata):
    """Encode one terminal state in the *same* frame as the conditioned prefix.

    Root XY and orientation are integrated codec side information, so their
    terminal equality is additionally enforced as an explicit output projection.
    Multi-frame terminal windows need a separate time-grid contract and fail.
    """
    if len(suffix) != 1:
        raise ValueError('This clock bridge currently requires a one-frame terminal suffix')
    native = np.repeat(suffix.states, 2, axis=0)
    transform = np.asarray(metadata['canonical_rotation'])
    offset = np.asarray(metadata['canonical_offset'])
    native[:, :3] = (native[:, :3]-offset) @ transform.T
    native[:, 3:7] = matrix_quat(transform @ quat_matrix(native[:, 3:7]))
    features, _ = codec.encode(native, metadata['fps'], canonicalize=False)
    features[:, :3] = suffix.states[:, 10:13] @ transform.T
    features[:, 3:6] = suffix.states[:, 7:10] @ transform.T
    return features[-1]

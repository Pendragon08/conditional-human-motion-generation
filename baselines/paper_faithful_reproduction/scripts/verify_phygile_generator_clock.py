"""Read-only real-pair clock/codec witness; no neural training or physics steps.

Only hashes and numerical summaries are written. Restricted motion arrays stay
at the authorized data location. Mathematical resampling is not tracking quality.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import traceback
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'STARTING', 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'evaluation_type': 'numerical_clock_and_codec_only', 'optimizer_updates': 0,
              'physics_control_calls': 0, 'trained_generator': False, 'source_data_modified': False}
    observed = {}

    def bind(path):
        p = Path(path).resolve(strict=True)
        observed[str(p)] = sha(p)
        return p

    try:
        test = bind(a.phygile_code/'reimplementation/test_motion_clock.py')
        command = [sys.executable, str(test), '-v']
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        (a.output/'unit_stdout.txt').write_text(result.stdout, encoding='utf-8')
        (a.output/'unit_stderr.txt').write_text(result.stderr, encoding='utf-8')
        report.update(unit_tests_command=command, unit_tests_exit_code=result.returncode)
        if result.returncode:
            raise RuntimeError('Clock/proposer counterexample tests failed; full stderr retained')
        sys.path.insert(0, str(a.phygile_code/'reimplementation'))
        from codec262 import G1Codec, quat_matrix
        from motion_clock import manifest_feature_contract, prepare_context, resample_native
        from physics_prefix import Motion
        from train import load_manifest
        manifest_path = bind(a.manifest)
        manifest, training = load_manifest(manifest_path, 'research_training')
        contract = manifest_feature_contract(manifest, manifest_path)
        if manifest.get('sample_clock_gate', {}).get('status') != 'NUMERIC_CLOCK_CONSISTENCY_ONLY':
            raise ValueError('Expected the existing audited clock-gated manifest')
        codec = G1Codec(bind(a.phygile_code/'configs/g1_codec_skeleton.json'))
        for name in ['codec262.py', 'motion_clock.py', 'train.py', 'diffusion_proposer.py']:
            bind(a.phygile_code/'reimplementation'/name)
        bind(__file__)
        rows = []
        for split, entries in manifest['splits'].items():
            for entry in entries:
                feature = bind(manifest_path.parent/entry['features'])
                sidecar = bind(manifest_path.parent/entry['codec_metadata'])
                metadata = json.loads(sidecar.read_text(encoding='utf-8'))
                native, _, _ = codec.decode(np.load(feature, allow_pickle=False), metadata, project=False, restore_world=True)
                source_rate = float(metadata['fps'])
                times = np.arange(len(native))/source_rate
                query = np.arange(int(np.floor(times[-1]*50+1e-9))+1)/50
                control = resample_native(native, times, query)
                # For 20/50 Hz the shared grid is 10 Hz. Check against the actual
                # source poses at those times, including nonzero world XY/yaw.
                indices = [k for k, t in enumerate(query) if np.isclose(t*source_rate, round(t*source_rate), atol=1e-9)]
                source_indices = np.rint(query[indices]*source_rate).astype(int)
                position = float(np.max(np.abs(control[indices, :3]-native[source_indices, :3])))
                joint = float(np.max(np.abs(control[indices, 13:42]-native[source_indices, 13:42])))
                rotation = float(np.max(np.abs(quat_matrix(control[indices, 3:7])-quat_matrix(native[source_indices, 3:7]))))
                if max(position, joint, rotation) > 1e-9:
                    raise ValueError('Shared physical sample times changed their source pose')
                prefix_native = control[:min(len(control), 52)]
                pos, _ = codec.fk(prefix_native[:, :3], prefix_native[:, 3:7], prefix_native[:, 13:42])
                prefix = Motion(prefix_native, np.stack([pos[n] for n in codec.body_names], 1),
                                50., 'root13_q29_dq29_wxyz_world', codec.body_names)
                f, meta, context_clock = prepare_context(codec, prefix, source_rate, canonicalize=True)
                rebuilt, _, _ = codec.decode(f, meta, project=False, restore_world=True)
                grid = np.asarray(context_clock['context_sample_times_relative_to_boundary_s'])
                actual_times = np.arange(-(len(prefix)-1), 1)/50
                if context_clock['held_initialization_intervals']:
                    truth = np.repeat(prefix.states[-1:], 2, 0)
                else:
                    truth = resample_native(prefix.states, actual_times, grid)
                error = {'position_m': float(np.max(np.abs(rebuilt[:, :3]-truth[:, :3]))),
                         'joint_rad': float(np.max(np.abs(rebuilt[:, 13:42]-truth[:, 13:42]))),
                         'rotation_matrix': float(np.max(np.abs(quat_matrix(rebuilt[:, 3:7])-quat_matrix(truth[:, 3:7]))))}
                if max(error.values()) > 1e-5:
                    raise ValueError('Canonical feature context did not return to its original world poses')
                rows.append({'split': split, 'feature_sha256': sha(feature), 'source_frames': len(native),
                    'feature_fps': source_rate, 'control_fps': 50., 'control_frames': len(control),
                    'source_last_sample_s': float(times[-1]), 'control_last_sample_s': float(query[-1]),
                    'omitted_tail_s': float(max(0, times[-1]-query[-1])),
                    'shared_time_samples': len(indices), 'shared_time_pose_max_abs':
                        {'position_m': position, 'joint_rad': joint, 'rotation_matrix': rotation},
                    'canonical_context_world_roundtrip_max_abs': error,
                    'context_boundary_is_actual_time': context_clock['context_sample_times_relative_to_boundary_s'][-1] == 0,
                    'time_stretched': False, 'extrapolation': False})
        if any(sha(p) != digest for p, digest in observed.items()):
            raise ValueError('An input changed during the read-only witness')
        report.update(status='PASSED_NUMERICAL_CLOCK_WITNESS', sequences=len(rows), records=rows,
            training_sequences=len(training), training_frames=sum(len(x) for x, _ in training),
            feature_contract=contract, inputs_unchanged=True,
            input_hashes=[{'name': Path(p).name, 'sha256': h} for p, h in observed.items()],
            maximum_omitted_tail_s=max(r['omitted_tail_s'] for r in rows),
            maximum_context_world_roundtrip=max(max(r['canonical_context_world_roundtrip_max_abs'].values()) for r in rows),
            interpretation='Real paired features used solely for numerical clock/frame checks. '
                'No trained generator sampling, physics evaluation, course labels or motion-quality result.')
    except BaseException as exc:
        report.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (a.output/'verification.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ['records', 'input_hashes', 'feature_contract']}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['manifest', 'phygile-code', 'output']:
        p.add_argument('--'+name, required=True, type=Path)
    run(p.parse_args())

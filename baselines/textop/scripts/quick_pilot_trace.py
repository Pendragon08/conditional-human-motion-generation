"""Private incremental execution poses, captured BEFORE IsaacLab autoresets.

The caller supplies physics state from termination_manager.compute. Fetching
state after env.step returns would instead record reset poses on terminal steps.
Binary journals retain partial trials if an external supervisor ends a process.
"""
import hashlib
import json
import pathlib
import numpy as np
from strict_tracking_coverage import verify_trial


FIELDS = [('reference_frame', (1,)), ('root_pos', (3,)), ('root_quat_wxyz', (4,)),
          ('joint_pos', (29,)), ('reference_joint_pos', (29,)), ('actions', (29,)),
          ('body_pos', (14, 3)), ('reference_body_pos', (14, 3))]
WIDTH = sum(int(np.prod(shape)) for _, shape in FIELDS)


class ExecutionTrace:
    def __init__(self, folder, motion_id, reference_frames, joint_names,
                 runtime_kind='isaaclab_physics', budget=None):
        self.folder = pathlib.Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        assert len(joint_names) == 29 and len(set(joint_names)) == 29
        assert len(motion_id) == 64 and all(c in '0123456789abcdef' for c in motion_id)
        assert reference_frames >= 2 and runtime_kind in ['isaaclab_physics', 'synthetic_cpu_fixture']
        self.budget = budget if budget is not None else {'used': 0, 'maximum': 256 * 1024**2}
        self.meta = dict(motion_id=motion_id, reference_frames=reference_frames,
                        joint_names=list(joint_names), control_hz=50, runtime_kind=runtime_kind,
                        phase='INITIAL_THEN_PRE_AUTORESET_TERMINATION_COMPUTE',
                        dtype='little_endian_float32', row_width=WIDTH,
                        fields=[{'name': k, 'shape': list(s)} for k, s in FIELDS],
                        status='INCOMPLETE', samples=0, actual_execution_video_created=False)
        self.path = self.folder/'poses.private.bin'
        self.stream = self.path.open('xb')
        self._meta()

    def _meta(self):
        path = self.folder/'trace.private.json'
        temp = path.with_name(path.name+'.partial')
        temp.write_text(json.dumps(self.meta, indent=2))
        temp.replace(path)

    def append(self, state):
        assert not self.stream.closed and self.meta['status'] == 'INCOMPLETE'
        parts = []
        for key, shape in FIELDS:
            value = np.asarray(state[key], dtype='<f4')
            assert value.shape == shape and np.isfinite(value).all(), key
            parts.append(value.reshape(-1))
        row = np.concatenate(parts).astype('<f4', copy=False)
        assert 0.99 < np.linalg.norm(state['root_quat_wxyz']) < 1.01
        assert self.meta['samples'] < self.meta['reference_frames']
        # The pinned command advances AFTER termination and possible autoreset:
        # initial frame0, then consumed targets0..steps-1 (zero-order hold).
        expected_frame = max(0, self.meta['samples']-1)
        assert float(state['reference_frame'][0]) == expected_frame, 'Unexpected target advance/reset inside physical trial'
        assert self.budget['used']+row.nbytes <= self.budget['maximum'], 'Private trace storage budget reached'
        self.stream.write(row.tobytes()); self.stream.flush()
        self.budget['used'] += row.nbytes
        self.meta['samples'] += 1
        # Header is written incrementally too: a timeout is never a success.
        self._meta()

    def finish(self, trial):
        verify_trial(trial, self.meta['reference_frames'], 0)
        assert self.meta['samples'] == trial['control_steps']+1
        self.stream.close()
        raw = np.fromfile(self.path, dtype='<f4').reshape(self.meta['samples'], WIDTH)
        arrays = {}; offset = 0
        for key, shape in FIELDS:
            size = int(np.prod(shape))
            arrays[key] = raw[:, offset:offset+size].reshape((len(raw),)+shape).copy()
            offset += size
        arrays['time_seconds'] = np.arange(len(raw), dtype=np.float64)/50.
        arrays['joint_names'] = np.asarray(self.meta['joint_names'])
        arrays['runtime_kind'] = np.asarray(self.meta['runtime_kind'])
        arrays['capture_phase'] = np.asarray(self.meta['phase'])
        # Worst-case reserve before serialization; the binary journal is retained.
        upper = raw.nbytes+256*1024
        assert self.budget['used']+upper <= self.budget['maximum'], 'Compressed trace reserve unavailable'
        target = self.folder/'executed_poses.private.npz'
        assert not target.exists()
        with target.open('xb') as stream: np.savez_compressed(stream, **arrays)
        self.budget['used'] += target.stat().st_size
        assert self.budget['used'] <= self.budget['maximum']
        body_error = np.linalg.norm(arrays['body_pos'][1:]-arrays['reference_body_pos'][1:], axis=-1)
        joint_error = arrays['joint_pos'][1:]-arrays['reference_joint_pos'][1:]
        metrics = dict(aligned_14_body_position_error_mean_m=float(body_error.mean()),
                       aligned_14_body_position_error_max_m=float(body_error.max()),
                       joint_rmse_rad=float(np.sqrt(np.mean(joint_error**2))),
                       definition='Own error against current control targets before autoreset; zero-order-hold timing, not final endpoint or paper aggregate scores')
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        self.meta.update(status='FINISHED_PHYSICAL_TRIAL' if self.meta['runtime_kind']=='isaaclab_physics' else 'FINISHED_SYNTHETIC_FIXTURE',
                         trial=trial, metrics=metrics, binary_sha256=sha(self.path),
                         npz_sha256=sha(target), npz_bytes=target.stat().st_size,
                         consumed_reference_frame_min=0,consumed_reference_frame_max=trial['control_steps']-1,
                         reference_final_frame=self.meta['reference_frames']-1,
                         final_reference_endpoint_comparison_performed=False)
        self._meta()
        return dict(folder=str(self.folder), samples=self.meta['samples'],
                    npz_sha256=self.meta['npz_sha256'], metrics=metrics,
                    runtime_kind=self.meta['runtime_kind'], physical_failure=trial['failed'])

    def close_incomplete(self):
        if not self.stream.closed: self.stream.close()
        if self.meta['status'] == 'INCOMPLETE':
            self.meta['binary_sha256'] = hashlib.sha256(self.path.read_bytes()).hexdigest()
            self._meta()

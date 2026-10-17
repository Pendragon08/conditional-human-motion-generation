"""Independent transactional physics-prefix orchestration, not an author release.

Motion.states has an explicit caller-owned schema. It is NOT implicitly 262D.
A production proposer still needs a trained generator and a verified codec.
Simulator contract: snapshot/restore ALL state (including controller, RNG and
command state); rollout returns post-step, pre-reset continuation samples.
"""
from dataclasses import dataclass
import hashlib
import numpy as np


@dataclass(frozen=True)
class Motion:
    states: np.ndarray
    body_positions: np.ndarray
    fps: float
    state_schema: str
    body_names: tuple

    def __post_init__(self):
        states = np.array(self.states, dtype=np.float64, copy=True)
        bodies = np.array(self.body_positions, dtype=np.float64, copy=True)
        if states.ndim != 2 or not len(states) or states.shape[1] == 0:
            raise ValueError('Expected nonempty [frames,state_dimensions]')
        if bodies.shape != (len(states), len(self.body_names), 3) or not self.body_names:
            raise ValueError('Body positions must have explicit ordered names')
        if len(set(self.body_names)) != len(self.body_names):
            raise ValueError('Repeated body names')
        if not np.isfinite(states).all() or not np.isfinite(bodies).all():
            raise ValueError('Nonfinite motion')
        if not np.isfinite(self.fps) or self.fps <= 0 or not self.state_schema:
            raise ValueError('Explicit positive fps and state schema required')
        states.setflags(write=False)
        bodies.setflags(write=False)
        object.__setattr__(self, 'states', states)
        object.__setattr__(self, 'body_positions', bodies)
        object.__setattr__(self, 'body_names', tuple(self.body_names))

    def __len__(self):
        return len(self.states)

    def slice(self, start, end=None):
        return Motion(self.states[start:end], self.body_positions[start:end],
                      self.fps, self.state_schema, self.body_names)

    def compatible(self, other):
        if (self.fps, self.state_schema, self.body_names, self.states.shape[1]) != (
                other.fps, other.state_schema, other.body_names, other.states.shape[1]):
            raise ValueError('Coordinate/state schema, body order or fps mismatch')

    def append(self, other):
        self.compatible(other)
        return Motion(np.concatenate([self.states, other.states]),
                      np.concatenate([self.body_positions, other.body_positions]),
                      self.fps, self.state_schema, self.body_names)

    def digest(self):
        h = hashlib.sha256()
        h.update(repr((self.fps, self.state_schema, self.body_names, self.states.shape)).encode())
        h.update(self.states.tobytes())
        h.update(self.body_positions.tobytes())
        return h.hexdigest()


@dataclass(frozen=True)
class Execution:
    continuation: Motion
    terminated: np.ndarray
    start_state: np.ndarray
    candidate_sha256: str
    timing: str = 'post_step_pre_reset_current_reference'


def assess_execution(candidate, prefix, execution, tolerance_m, terminal_frames=0,
                     terminal_tolerance_m=None):
    """Use raw metre errors; incomplete/terminated/nonfinite runs cannot pass."""
    prefix.compatible(candidate)
    if not np.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError('Positive finite tolerance required')
    expected = len(candidate) - len(prefix)
    if expected <= 0 or not np.array_equal(candidate.states[:len(prefix)], prefix.states) or not np.array_equal(
            candidate.body_positions[:len(prefix)], prefix.body_positions):
        raise ValueError('Candidate must preserve the committed prefix exactly')
    if terminal_frames < 0 or terminal_frames > expected:
        raise ValueError('Invalid terminal window')
    if terminal_frames and (terminal_tolerance_m is None or not np.isfinite(terminal_tolerance_m)
                            or terminal_tolerance_m <= 0):
        raise ValueError('Final suffix requires an explicit terminal tolerance')
    prefix.compatible(execution.continuation)
    done = np.asarray(execution.terminated)
    if done.dtype != np.bool_ or done.shape != (len(execution.continuation),):
        raise ValueError('Termination flags must be boolean and match every recorded frame')
    if execution.timing != 'post_step_pre_reset_current_reference':
        raise ValueError('Post-reset or ambiguous timing is not accepted')
    if execution.candidate_sha256 != candidate.digest():
        raise ValueError('Execution is not bound to this candidate')
    if not np.array_equal(execution.start_state, prefix.states[-1]):
        raise ValueError('Simulator did not start at the accepted prefix boundary')
    count = len(execution.continuation)
    if count > expected:
        raise ValueError('Extra execution frames')
    delta = candidate.body_positions[len(prefix):len(prefix)+count] - execution.continuation.body_positions
    per_frame = np.linalg.norm(delta, axis=-1).mean(axis=-1)
    reasons = []
    if count != expected:
        reasons.append('incomplete_continuation')
    if done.any():
        reasons.append('failure_termination')
    mean = float(per_frame.mean())
    if mean > tolerance_m:
        reasons.append('mpjpe_above_tolerance')
    terminal_max = None
    if terminal_frames and count == expected:
        terminal_max = float(per_frame[-terminal_frames:].max())
        if terminal_max > terminal_tolerance_m:
            reasons.append('terminal_suffix_above_tolerance')
    return {'accepted': not reasons, 'reasons': reasons, 'expected_frames': expected,
            'recorded_frames': count, 'mpjpe_m': mean, 'terminal_max_frame_mpjpe_m': terminal_max,
            'candidate_sha256': candidate.digest(), 'execution_sha256': execution.continuation.digest()}


class PrefixRejected(RuntimeError):
    def __init__(self, prefix, attempts):
        super().__init__('Retry budget exhausted; no rejected segment was committed')
        self.prefix = prefix
        self.attempts = attempts


def roll_physics_prefix(prefix, suffix, proposer, simulator, *, total_new_frames,
                        chunk_frames, max_attempts, tolerance_m, terminal_tolerance_m):
    """Bounded propose/simulate/reject/commit loop with rollback on every failure.

    proposer(prefix, suffix, new_frames, final, attempt) returns a whole window.
    simulator.rollout(window, prefix_frames) returns ONLY the new actual states.
    Rejected attempts are restored; accepted ACTUAL states become the next prefix.
    Generator parameters are never trained here. GMT PPO fine-tuning is separate.
    """
    prefix.compatible(suffix)
    for value in (total_new_frames, chunk_frames, max_attempts):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError('Positive integer budgets required')
    if len(suffix) > min(chunk_frames, total_new_frames):
        raise ValueError('Suffix must fit the final chunk')
    if total_new_frames % chunk_frames and total_new_frames % chunk_frames < len(suffix):
        raise ValueError('Final partial chunk must contain the suffix')
    attempts = []
    remaining = total_new_frames
    while remaining:
        count = min(chunk_frames, remaining)
        final = count == remaining
        checkpoint = simulator.snapshot()
        accepted = False
        for attempt in range(max_attempts):
            simulator.restore(checkpoint)
            try:
                candidate = proposer(prefix, suffix, count, final, attempt)
                prefix.compatible(candidate)
                if len(candidate) != len(prefix) + count:
                    raise ValueError('Proposer returned wrong window length')
                if not np.array_equal(candidate.states[:len(prefix)], prefix.states) or not np.array_equal(
                        candidate.body_positions[:len(prefix)], prefix.body_positions):
                    raise ValueError('Proposer altered the committed prefix')
                if final and (not np.array_equal(candidate.states[-len(suffix):], suffix.states)
                              or not np.array_equal(candidate.body_positions[-len(suffix):], suffix.body_positions)):
                    raise ValueError('Final proposal did not preserve the terminal suffix')
                execution = simulator.rollout(candidate, len(prefix))
                record = assess_execution(candidate, prefix, execution, tolerance_m,
                                          len(suffix) if final else 0,
                                          terminal_tolerance_m if final else None)
            except Exception:
                simulator.restore(checkpoint)
                raise
            record.update(prefix_frames=len(prefix), attempt=attempt, final=final)
            attempts.append(record)
            if record['accepted']:
                prefix = prefix.append(execution.continuation)
                remaining -= count
                accepted = True
                break
        if not accepted:
            simulator.restore(checkpoint)
            raise PrefixRejected(prefix, attempts)
    return prefix, attempts

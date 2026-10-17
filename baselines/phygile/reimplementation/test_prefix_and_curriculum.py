"""Regression checks for actual patched methods and transactional prefix logic.

All inputs in this file are synthetic software fixtures, never motion benchmarks.
"""
import argparse
import ast
from pathlib import Path
import hashlib
import json
import math
import os
import sys
from types import SimpleNamespace as NS, MethodType
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from prepare_guarded_upstream import patched_sources, MODULE
from physics_prefix import Motion, Execution, PrefixRejected, assess_execution, roll_physics_prefix


def raises(kind, func):
    try:
        func()
    except kind:
        return
    raise AssertionError(f'Expected {kind.__name__}')


def load_method(source, name):
    tree = ast.parse(source)
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(matches) == 1, name
    node = matches[0]
    node.returns = None
    for arg in node.args.args:
        arg.annotation = None
    namespace = {'torch': torch, 'math': math, 'os': os}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual patched method>', 'exec'), namespace)
    return namespace[name]


def run():
    checks = []
    def passed(name):
        checks.append(name)
    sources = patched_sources()
    for source in sources.values():
        compile(source, '<patched source>', 'exec')
    runner = sources[MODULE+'utils/my_on_policy_runner.py']
    commands = sources[MODULE+'tasks/tracking/mdp/commands.py']
    convergence = load_method(runner, '_check_metric_convergence')
    fixture = NS(_metric_conv_evals=3, _metric_conv_threshold=.03)
    for history, expected in [([(1, 1), (.99, .99), (.98, .98), (.97, .97)], True),
                              ([(1, 1), (2, 2), (3, 3), (4, 4)], False),
                              ([(1, 1), (2, 2), (.9, .9), (.99, .99)], False),
                              ([(1, 1), (1, 1), (float('nan'), 1), (1, 1)], False),
                              ([(1, 1), (.5, .5), (.25, .25), (.125, .125)], False)]:
        fixture._eval_metric_history = history
        assert convergence(fixture)[0] == expected
    passed('Convergence rejects worsening, cancellation, nonfinite and still-improving windows')

    accumulate = load_method(commands, '_accumulate_clip_errors')
    f = NS(cfg=NS(file_error_mpjae_weight=0), metrics={'error_body_pos': torch.tensor([.2, .3])},
           joint_pos=torch.ones(2, 29), robot_joint_pos=torch.zeros(2, 29),
           clip_error_accum=torch.zeros(2), clip_mpjpe_accum=torch.zeros(2),
           clip_mpjae_accum=torch.zeros(2), clip_step_count=torch.zeros(2))
    accumulate(f)
    torch.testing.assert_close(f.clip_error_accum, torch.tensor([.2, .3]))
    for invalid in [float('nan'), float('inf'), -float('inf')]:
        before = f.clip_error_accum.clone()
        f.metrics['error_body_pos'][0] = invalid
        raises(FloatingPointError, lambda: accumulate(f))
        assert torch.equal(before, f.clip_error_accum)
    f.clip_error_accum[0] = float('inf')
    finalize = load_method(commands, '_finalize_clip_errors')
    raises(FloatingPointError, lambda: finalize(f, torch.tensor([0])))
    passed('Nonfinite instantaneous and accumulated errors stop before statistics mutation')

    for name, source in [('run_curriculum_evaluation', commands),
                         ('_run_curriculum_evaluation_subprocess', runner)]:
        fn = load_method(source, name)
        if name.startswith('_'):
            raises(RuntimeError, lambda: fn(NS(), NS(), 1, ['clip']))
        else:
            raises(RuntimeError, lambda: fn(NS(), None, set()))
    passed('Both unsafe evaluator entrypoints fail explicitly before stepping')

    ema = load_method(runner, '_run_curriculum_evaluation_from_ema')
    motion = NS(cfg=NS(curriculum_min_attempts=5), motion_file_paths=['a', 'b'], device='cpu',
                file_mpjpe_ema=torch.tensor([0., .03]), file_mpjae_ema=torch.tensor([0., .04]),
                file_success_count=torch.tensor([5., 5.]), file_failed_count=torch.zeros(2))
    assert ema(NS(log_dir=None), motion, ['a', 'b'], set())[0] < .1
    motion.file_success_count[1] = 0
    assert math.isinf(ema(NS(log_dir=None), motion, ['a', 'b'], set())[0])
    motion.file_success_count[1] = 5
    motion.file_mpjpe_ema[1] = float('nan')
    assert math.isinf(ema(NS(log_dir=None), motion, ['a', 'b'], set())[0])
    motion.file_mpjpe_ema[1] = .03
    assert math.isinf(ema(NS(log_dir=None), motion, ['a', 'missing'], set())[0])
    passed('EMA requires every requested file and enough attempts; valid zero error remains valid')

    manager = load_method(sources[MODULE+'utils/curriculum_manager.py'], 'report_evaluation_result')
    manager_f = NS(max_level=1, cfg=NS(mpjpe_threshold=.1, mpjae_threshold=.1,
                     auto_upgrade_after_iters=1, auto_upgrade_per_level=None, max_difficulty=10),
                   _level_unlocked_iter={1: 0})
    for invalid in [float('nan'), float('inf'), -.1]:
        assert not manager(manager_f, invalid, .01, 10000)
        assert manager_f.max_level == 1
    passed('Curriculum manager cannot promote invalid metrics even after timeout')

    # Invoke the real patched outer method with controlled evaluator returns.
    # This checks its wiring only; the public subprocess remains blocked above.
    promotion_calls = []
    cfg = NS(curriculum_use_subprocess_eval=True, curriculum_auto_upgrade_after_iters=1)
    mt = NS(cfg=cfg, set_training_iteration=lambda i: None,
            should_run_curriculum_evaluation=lambda: True, get_curriculum_level=lambda: 1,
            get_curriculum_files_for_evaluation=lambda frozen, level: ['a'],
            get_curriculum_level_unlocked_iter=lambda level: 0,
            report_curriculum_evaluation=lambda *args: promotion_calls.append(args) or False)
    outer = NS(log_dir='synthetic_fixture', writer=None, _get_motion_term=lambda: mt,
               _get_frozen_files=lambda mt: set(), _write_curriculum_eval_log=lambda *a: None,
               _eval_metric_history=[], _metric_conv_enabled=False, _metric_conv_min_iters=3000,
               _metric_conv_evals=3, _metric_conv_threshold=.03)
    outer._check_metric_convergence = MethodType(convergence, outer)
    check_eval = load_method(runner, '_run_curriculum_evaluation_if_needed')
    outer._run_curriculum_evaluation_subprocess = lambda **kw: (float('inf'), float('inf'), [])
    check_eval(outer, 10000)
    assert not promotion_calls
    outer._run_curriculum_evaluation_subprocess = lambda **kw: (.01, .01, [])
    check_eval(outer, 10001)
    assert len(promotion_calls) == 1
    passed('Outer evaluation branch has initialized fields, unknown success and invalid-timeout rejection')

    def motion(values):
        values = np.asarray(values, dtype=float).reshape(-1, 1)
        return Motion(values, np.repeat(values[:, :, None], 3, axis=2), 50., 'synthetic_scalar_v1', ('body',))
    prefix, suffix = motion([0]), motion([0])
    seen = []
    class Sim:
        def __init__(self):
            self.state = prefix.states[-1].copy()
        def snapshot(self):
            return self.state.copy()
        def restore(self, saved):
            self.state = saved.copy()
        def rollout(self, candidate, start):
            initial = self.state.copy()
            segment = candidate.slice(start)
            offset = 1. if segment.states[0, 0] == 10 else .01
            actual = Motion(segment.states+offset, segment.body_positions+offset,
                            segment.fps, segment.state_schema, segment.body_names)
            self.state = actual.states[-1].copy()
            return Execution(actual, np.zeros(len(actual), bool), initial, candidate.digest())
    def proposer(p, s, n, final, attempt):
        seen.append(p)
        addition = motion([10 if attempt == 0 else 1]*n)
        if final:
            addition = addition.slice(0, n-len(s)).append(s)
        return p.append(addition)
    sim = Sim()
    result, attempts = roll_physics_prefix(prefix, suffix, proposer, sim, total_new_frames=6,
        chunk_frames=3, max_attempts=2, tolerance_m=.1, terminal_tolerance_m=.1)
    assert len(result) == 7 and [x['accepted'] for x in attempts] == [False, True, False, True]
    assert seen[2].digest() == result.slice(0, 4).digest()
    assert seen[2].states[-1, 0] == 1.01  # actual, not proposed 1.0
    passed('Two-window loop rolls back rejected attempts and conditions on accepted actual states')

    sim = Sim()
    try:
        roll_physics_prefix(prefix, suffix, proposer, sim, total_new_frames=3, chunk_frames=3,
                            max_attempts=1, tolerance_m=.1, terminal_tolerance_m=.1)
    except PrefixRejected as exc:
        assert exc.prefix.digest() == prefix.digest() and len(exc.attempts) == 1
        np.testing.assert_array_equal(sim.state, prefix.states[-1])
    else:
        raise AssertionError('Rejected candidate committed')
    passed('Retry exhaustion returns the unchanged prefix and restores simulator state')

    candidate = prefix.append(motion([1, 1, 0]))
    actual = candidate.slice(1)
    good = Execution(actual, np.zeros(3, bool), prefix.states[-1], candidate.digest())
    assert assess_execution(candidate, prefix, good, .1)['accepted']
    partial = Execution(actual.slice(0, 2), np.zeros(2, bool), prefix.states[-1], candidate.digest())
    assert not assess_execution(candidate, prefix, partial, .1)['accepted']
    failed = Execution(actual, np.array([False, False, True]), prefix.states[-1], candidate.digest())
    assert not assess_execution(candidate, prefix, failed, .1)['accepted']
    raises(ValueError, lambda: assess_execution(candidate, prefix, Execution(actual, np.zeros(3, bool),
           prefix.states[-1], 'wrong'), .1))
    raises(ValueError, lambda: assess_execution(candidate, prefix, Execution(actual, np.zeros(3, bool),
           np.array([9]), candidate.digest()), .1))
    raises(ValueError, lambda: assess_execution(candidate, prefix, Execution(actual, np.zeros(3, bool),
           prefix.states[-1], candidate.digest(), 'post_reset'), .1))
    passed('Acceptance rejects truncation, failure, wrong candidate, wrong boundary and post-reset timing')

    terminal_bad = Motion(actual.states, actual.body_positions + np.array([0, 0, .15])[:, None, None],
                          50., actual.state_schema, actual.body_names)
    verdict = assess_execution(candidate, prefix, Execution(terminal_bad, np.zeros(3, bool),
            prefix.states[-1], candidate.digest()), 1., terminal_frames=1, terminal_tolerance_m=.1)
    assert verdict['reasons'] == ['terminal_suffix_above_tolerance']
    passed('Terminal error cannot hide inside the whole-window average')
    sim = Sim()
    def broken(*args):
        sim.state[:] = 99
        raise RuntimeError('generator/adapter failure')
    raises(RuntimeError, lambda: roll_physics_prefix(prefix, suffix, broken, sim, total_new_frames=3,
        chunk_frames=3, max_attempts=1, tolerance_m=.1, terminal_tolerance_m=.1))
    np.testing.assert_array_equal(sim.state, prefix.states[-1])
    passed('Unexpected proposer/adapter exceptions restore the checkpoint')
    raises(ValueError, lambda: motion([float('nan')]))
    raises(ValueError, lambda: prefix.append(Motion(np.zeros((1, 1)), np.zeros((1, 1, 3)),
           60, 'synthetic_scalar_v1', ('body',))))
    passed('Nonfinite states and incompatible frame rates fail before use')
    return checks, sources


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    checks, sources = run()
    report = {'status': 'PASSED_REGRESSION_CHECKS', 'evaluation_type': 'synthetic_software_fixtures',
              'checks': checks, 'torch_version': torch.__version__, 'trained_model': False,
              'physics_simulation': False, 'full_curriculum_training': False,
              'patched_source_sha256': {name: hashlib.sha256(s.encode()).hexdigest() for name, s in sources.items()},
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [Path(__file__), ROOT/'reimplementation/physics_prefix.py', ROOT/'scripts/prepare_guarded_upstream.py']}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))

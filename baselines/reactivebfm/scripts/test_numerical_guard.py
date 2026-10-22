"""Numerical and collective-skip regression tests, never motion performance data."""
import argparse
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest

import torch
import torch.distributed as distributed
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'upstream'))
from numerical_guard import norms64, clip64, RecoveryGuard


class GuardTests(unittest.TestCase):
    def test_finite_large_gradients_do_not_overflow_norm_or_clip_to_zero(self):
        p = torch.nn.Parameter(torch.ones(2048))
        p.grad = torch.full_like(p, 1e20)
        self.assertFalse(bool(torch.isfinite(torch.linalg.vector_norm(p.grad))))
        gradient, _ = norms64([p])
        self.assertTrue(torch.isfinite(torch.tensor(gradient, dtype=torch.float64)))
        clip64([p], gradient, 1.0)
        self.assertAlmostEqual(norms64([p])[0], 1.0, places=5)
        self.assertGreater(int(torch.count_nonzero(p.grad)), 0)

    def test_regular_norm_and_clipping_match_reference(self):
        torch.manual_seed(42)
        p = torch.nn.Parameter(torch.randn(500))
        q = torch.nn.Parameter(p.detach().clone())
        p.grad = torch.randn_like(p); q.grad = p.grad.clone()
        clip64([p], norms64([p])[0], 1.0)
        torch.nn.utils.clip_grad_norm_([q], 1.0)
        torch.testing.assert_close(p.grad, q.grad)

    def test_watchdog_counts_rejected_updates_and_writes_before_raising(self):
        with tempfile.TemporaryDirectory() as directory:
            p = torch.nn.Parameter(torch.ones(3)); opt = torch.optim.AdamW([p])
            model = torch.nn.ParameterList([p])
            guard = RecoveryGuard(SimpleNamespace(master_params=[p], model=model, max_grad_norm=1.0), opt, directory, 2)
            for i in range(2):
                guard.begin_attempt(); p.grad = torch.full_like(p, float('inf'))
                self.assertFalse(guard.optimize()); self.assertFalse(guard.last_step_applied)
                if i == 0: guard.finish_attempt(i)
                else:
                    with self.assertRaises(FloatingPointError): guard.finish_attempt(i)
            row = json.loads((Path(directory)/'health_rank0.json').read_text())
            self.assertEqual(row['optimizer_updates'], 0)
            self.assertEqual(row['status'], 'NUMERICAL_WATCHDOG_STOP')
            self.assertEqual(len(opt.state), 0)

    def test_post_adam_corruption_is_not_counted_as_validated_update(self):
        with tempfile.TemporaryDirectory() as directory:
            p = torch.nn.Parameter(torch.ones(3)); opt = torch.optim.AdamW([p])
            p.grad = torch.ones_like(p); opt.step()
            opt.state[p]['exp_avg_sq'].fill_(-1.0)
            guard = RecoveryGuard(SimpleNamespace(master_params=[p],model=torch.nn.ParameterList([p]),max_grad_norm=1.0),opt,directory,1)
            p.grad = torch.ones_like(p)
            with self.assertRaises(FloatingPointError): guard.optimize()
            row = json.loads((Path(directory)/'health_rank0.json').read_text())
            self.assertEqual(row['optimizer_step_calls'],2)
            self.assertEqual(row['validated_optimizer_updates'],1)
            self.assertIn('AFTER_STEP_NO_ROLLBACK',row['reason'])

    def test_bad_ema_aborts_before_checkpoint_path(self):
        with tempfile.TemporaryDirectory() as directory:
            p = torch.nn.Parameter(torch.ones(3)); opt = torch.optim.AdamW([p])
            guard = RecoveryGuard(SimpleNamespace(master_params=[p],model=torch.nn.ParameterList([p]),max_grad_norm=1.0),opt,directory,1)
            with self.assertRaises(FloatingPointError):
                guard.validate_ema([torch.tensor(float('nan'))],'BEFORE_UPDATE')
            row = json.loads((Path(directory)/'health_rank0.json').read_text())
            self.assertEqual(row['reason'],'NONFINITE_EMA_BEFORE_UPDATE')


def distributed_test(post_step=False, output='.'):
    distributed.init_process_group('gloo')
    rank = distributed.get_rank()
    p = torch.nn.Parameter(torch.ones(3)); opt = torch.optim.AdamW([p])
    guard = RecoveryGuard(SimpleNamespace(master_params=[p], model=torch.nn.ParameterList([p]), max_grad_norm=1.0), opt, output, 3)
    if post_step:
        original_step = opt.step
        def poison_step(*args, **kwargs):
            original_step(*args, **kwargs)
            if rank == 1: p.data.fill_(float('inf'))
        opt.step = poison_step
        p.grad = torch.ones_like(p)
        try: guard.optimize()
        except FloatingPointError: pass
        else: raise AssertionError('Post-step nonfinite state must abort both ranks')
        if guard.step_calls != 1 or guard.updates != 0: raise AssertionError('Called and validated counters confused')
    else:
        p.grad = torch.full_like(p, float('inf') if rank == 1 else 1.0)
        applied = guard.optimize()
        if applied or opt.state or not torch.equal(p.detach(), torch.ones(3)):
            raise AssertionError('Every rank must reject a step when one rank is nonfinite')
    distributed.barrier(); distributed.destroy_process_group()
    print(json.dumps({'rank':rank,'status':'ALL_RANKS_ABORTED_POST_STEP_NO_VALIDATED_UPDATE' if post_step else 'ALL_RANKS_REJECTED_NONFINITE_WITHOUT_MUTATION'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--post-step',action='store_true')
    parser.add_argument('--output',default='.')
    arguments, remaining = parser.parse_known_args()
    if arguments.distributed: distributed_test(arguments.post_step,arguments.output)
    else: unittest.main(argv=[__file__]+remaining)

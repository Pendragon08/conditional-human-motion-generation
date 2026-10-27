"""Supplementary, opt-in recovery guards; vendor source and objective stay intact.

Use FP64 reductions for norms (not FP64 model training), make skip decisions
collective, and stop sustained failures instead of consuming an entire schedule.
"""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as distributed


def collective_true(value, device):
    if not distributed.is_available() or not distributed.is_initialized():
        return bool(value)
    flag = torch.tensor(int(bool(value)), device=device, dtype=torch.int32)
    distributed.all_reduce(flag, op=distributed.ReduceOp.MIN)
    return bool(flag.item())


@torch.no_grad()
def norms64(parameters):
    """Avoid FP32 sum-of-squares overflow for individually finite gradients."""
    grad_squares = 0.0
    param_squares = 0.0
    for parameter in parameters:
        if not parameter.requires_grad:
            continue
        param_squares += torch.linalg.vector_norm(parameter.detach().double()).item() ** 2
        if parameter.grad is not None:
            grad_squares += torch.linalg.vector_norm(parameter.grad.detach().double()).item() ** 2
    return math.sqrt(grad_squares), math.sqrt(param_squares)


@torch.no_grad()
def clip64(parameters, grad_norm, maximum):
    if maximum <= 0:
        return
    coefficient = min(1.0, float(maximum) / (float(grad_norm) + 1e-6))
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(coefficient)


def optimizer_steps(optimizer):
    steps = {int(value['step']) for value in optimizer.state.values() if 'step' in value}
    if len(steps) > 1:
        raise RuntimeError('Optimizer state has inconsistent parameter step counters')
    return next(iter(steps), 0)


def tensors_finite(values):
    return all(bool(torch.isfinite(value).all()) for value in values if torch.is_tensor(value))


class RecoveryGuard:
    def __init__(self, trainer, optimizer, output, maximum_skips=3):
        self.trainer = trainer
        self.optimizer = optimizer
        self.output = Path(output)
        self.maximum_skips = maximum_skips
        self.rank = int(os.getenv('RANK', '0'))
        self.initial_updates = optimizer_steps(optimizer)
        self.updates = self.initial_updates
        self.step_calls = self.initial_updates
        self.attempts = 0
        self.skips = 0
        self.consecutive_skips = 0
        self.started = time.monotonic()
        self.last_step_applied = False
        self.step_called = False
        self.ema_validated = False
        self.begin_attempt()

    def begin_attempt(self):
        self.last_step_applied = False
        self.step_called = False
        self.ema_validated = False
        self.reason = 'NO_OPTIMIZER_STEP_OR_NONFINITE_PRIMITIVE_LOSS'
        self.gradient_norm = None
        self.clipped_gradient_norm = None
        self.parameter_norm = None
        self.offending_parameters = []

    def fatal(self, reason):
        self.reason = reason
        self.last_step_applied = False
        step = self.loop.total_step() if hasattr(self, 'loop') else self.attempts
        self.finish_attempt(step, force_stop=True)

    def validate_ema(self, parameters, phase):
        parameters = list(parameters)
        device = self.trainer.master_params[0].device
        if not collective_true(tensors_finite(parameters), device):
            self.fatal('NONFINITE_EMA_'+phase)
        self.ema_validated = phase == 'AFTER_UPDATE'

    def optimize(self):
        from reactivebfm.model.motion_planner.objectives.diffusion import logger
        parameters = list(self.trainer.master_params)
        device = next(p.device for p in parameters if p.requires_grad)
        gradient, parameter = norms64(parameters)
        self.gradient_norm = gradient if math.isfinite(gradient) else None
        self.parameter_norm = parameter if math.isfinite(parameter) else None
        if not collective_true(math.isfinite(gradient) and math.isfinite(parameter), device):
            self.reason = 'NONFINITE_NORM_BEFORE_CLIPPING'
            for name, p in self.trainer.model.named_parameters():
                for kind, tensor in [('parameter', p.detach()), ('gradient', p.grad)]:
                    if tensor is None: continue
                    finite = torch.isfinite(tensor)
                    if not bool(finite.all()):
                        self.offending_parameters.append({'name':name, 'kind':kind,
                            'nan_count':int(torch.isnan(tensor).sum()),
                            'inf_count':int(torch.isinf(tensor).sum())})
                if len(self.offending_parameters) >= 8: break
            for p in parameters:
                p.grad = None
            return False
        clip64(parameters, gradient, self.trainer.max_grad_norm)
        clipped, parameter = norms64(parameters)
        self.clipped_gradient_norm = clipped if math.isfinite(clipped) else None
        if not collective_true(math.isfinite(clipped) and math.isfinite(parameter), device):
            self.reason = 'NONFINITE_NORM_AFTER_CLIPPING'
            for p in parameters:
                p.grad = None
            return False
        logger.logkv_mean('grad_norm_pre_clip', gradient)
        logger.logkv_mean('grad_norm', clipped)
        logger.logkv_mean('param_norm', parameter)
        self.optimizer.step()
        self.step_called = True
        self.step_calls += 1
        finite_after = tensors_finite(parameters) and all(
            tensors_finite(state.values()) for state in self.optimizer.state.values())
        if not collective_true(finite_after, device):
            self.fatal('NONFINITE_PARAMETER_OR_OPTIMIZER_STATE_AFTER_STEP_NO_ROLLBACK')
        self.updates += 1
        self.last_step_applied = True
        self.reason = 'OPTIMIZER_STEP_APPLIED'
        return True

    def finish_attempt(self, loop_step, force_stop=False):
        self.attempts += 1
        if self.last_step_applied:
            self.consecutive_skips = 0
        else:
            self.skips += 1
            self.consecutive_skips += 1
        stopped = force_stop or self.consecutive_skips >= self.maximum_skips
        row = {
            'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'rank': self.rank, 'status': 'NUMERICAL_WATCHDOG_STOP' if stopped else 'TRAINING',
            'loop_step': loop_step, 'initial_optimizer_updates': self.initial_updates,
            'optimizer_updates': self.updates, 'resumed_updates': self.updates-self.initial_updates,
            'optimizer_step_calls': self.step_calls, 'validated_optimizer_updates': self.updates,
            'attempts_since_resume': self.attempts, 'skipped_attempts': self.skips,
            'consecutive_skips': self.consecutive_skips, 'step_applied': self.step_called,
            'step_validated': self.last_step_applied, 'ema_validated': self.ema_validated,
            'reason': self.reason, 'grad_norm_pre_clip_fp64': self.gradient_norm,
            'grad_norm_post_clip_fp64': self.clipped_gradient_norm,
            'param_norm_fp64': self.parameter_norm,
            'offending_parameters': self.offending_parameters,
            'elapsed_seconds': time.monotonic()-self.started,
            'guard_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'full_training_completed': False,
        }
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / f'health_rank{self.rank}.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        destination = self.output / f'health_rank{self.rank}.json'
        temporary = destination.with_suffix('.partial')
        temporary.write_text(json.dumps(row, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        os.replace(temporary, destination)
        if stopped:
            raise FloatingPointError('Recovery watchdog: consecutive rejected optimizer steps; evidence preserved')


def install_guards(output, maximum_skips=3, save_after_resumed_updates=20, expected_initial_updates=None):
    from reactivebfm.model.motion_planner.objectives.diffusion.fp16_util import MixedPrecisionTrainer
    from reactivebfm.train.scheduled_forcing_loop import ScheduledForcingLoop

    initialize = ScheduledForcingLoop.__init__
    zero = MixedPrecisionTrainer.zero_grad
    optimize = MixedPrecisionTrainer._optimize_normal
    average = ScheduledForcingLoop.update_average_model
    anneal = ScheduledForcingLoop._anneal_lr

    def guarded_initialize(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        self.mp_trainer.recovery_guard = RecoveryGuard(self.mp_trainer, self.opt, output, maximum_skips)
        self.mp_trainer.recovery_guard.loop = self
        if expected_initial_updates is not None and self.mp_trainer.recovery_guard.initial_updates != expected_initial_updates:
            raise ValueError('GPU-loaded optimizer count differs from the recovery manifest')
        self._recovery_early_saved = False

    def guarded_zero(self):
        guard = getattr(self, 'recovery_guard', None)
        if guard is not None:
            guard.begin_attempt()
        zero(self)

    def guarded_optimize(self, optimizer):
        guard = getattr(self, 'recovery_guard', None)
        return guard.optimize() if guard is not None else optimize(self, optimizer)

    def guarded_average(self):
        guard = getattr(self.mp_trainer, 'recovery_guard', None)
        if guard is None or guard.last_step_applied:
            if guard is not None and self.model_avg is not None:
                guard.validate_ema(self.model_avg.parameters(), 'BEFORE_UPDATE')
            average(self)
            if guard is not None and self.model_avg is not None:
                guard.validate_ema(self.model_avg.parameters(), 'AFTER_UPDATE')

    def guarded_anneal(self):
        guard = self.mp_trainer.recovery_guard
        guard.finish_attempt(self.total_step())
        anneal(self)
        if (guard.last_step_applied and not self._recovery_early_saved
                and guard.updates-guard.initial_updates == save_after_resumed_updates):
            if distributed.is_initialized(): distributed.barrier()
            if guard.rank == 0: self.save()
            if distributed.is_initialized(): distributed.barrier()
            self._recovery_early_saved = True

    ScheduledForcingLoop.__init__ = guarded_initialize
    MixedPrecisionTrainer.zero_grad = guarded_zero
    MixedPrecisionTrainer._optimize_normal = guarded_optimize
    ScheduledForcingLoop.update_average_model = guarded_average
    ScheduledForcingLoop._anneal_lr = guarded_anneal

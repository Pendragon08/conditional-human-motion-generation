"""Exact warm restoration and original TMR updates; no imports create models."""
import copy
import hashlib
import io
import json
import os
import pathlib
import shutil
from tmr100_capacity_resume_protocol import ROOT, HOME, BASE_SPEC, BASE_SHA, EMERGENCY, EMERGENCY_SHA, origin, check_metadata


class Session:
    def __init__(self, spec, request, report):
        import numpy as np
        import torch
        from tmr69d_core import make_model, Sampler, rng_state, restore_rng, step
        from tmr69d_training_inputs import Inputs
        from generator_training_state import equal, module_sha
        self.torch, self.np = torch, np
        self.equal, self.module_sha = equal, module_sha
        self.rng, self.restore_rng, self.original_step = rng_state, restore_rng, step
        self.s, self.sp, self.report = spec, request, report
        self.base, self.source = origin(spec)
        torch.set_num_threads(4)
        assert torch.cuda.device_count() == 1
        torch.cuda.set_per_process_memory_fraction(.65, 0)
        assert torch.cuda.mem_get_info()[0] >= 20 * 1024**3
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.inputs = Inputs(self.base['inputs'], 'cuda:0', lambda **kw: report(input_preparation=kw))
        assert self.inputs.signature_sha256 == self.base['accepted_input_signature_sha256']
        self.model, self.optimizer, self.ast_audit = make_model(pathlib.Path(self.base['upstream_model_dir']), self.base['upstream_model_sha256'], 'cuda:0')
        self.sampler = Sampler(len(self.inputs.gradient), self.base['seed'] + 1)
        state = torch.load(EMERGENCY, map_location='cpu')
        check_metadata(state, self.base)
        assert state['optimizer_updates'] == 83703 and state['completed_epochs'] == 82 and state['pending_validation'] is False
        self.validate(state)
        self.source_content_sha256 = self.content_hash(state)
        self.restore(state)
        assert self.equal(self.snapshot(), state)
        self.initial_next_batch_sha256 = self.next_batch_sha()
        self.initial_model_sha256 = self.module_sha(self.model)
        self.last_good = self.snapshot()
        del state
        report(actual_input_signature_sha256=self.inputs.signature_sha256,
               actual_input_signature=self.inputs.signature,
               source_restored_content_sha256=self.source_content_sha256,
               source_restored_model_sha256=self.initial_model_sha256,
               initial_next_batch_sha256=self.initial_next_batch_sha256,
               exact_source_model_adamw_rng_sampler_nextbatch_restored=True)

    def tree(self, value, cpu=False):
        if self.torch.is_tensor(value):
            return value.detach().cpu().clone() if cpu else value.detach().clone()
        if isinstance(value, dict):
            return {k: self.tree(v, cpu) for k, v in value.items()}
        if isinstance(value, tuple):
            return tuple(self.tree(v, cpu) for v in value)
        if isinstance(value, list):
            return [self.tree(v, cpu) for v in value]
        return copy.deepcopy(value)

    def content_hash(self, value):
        h = hashlib.sha256()
        def visit(x):
            h.update(type(x).__name__.encode())
            if self.torch.is_tensor(x):
                y = x.detach().cpu().contiguous()
                h.update(str(y.dtype).encode()); h.update(str(tuple(y.shape)).encode()); h.update(y.numpy().tobytes())
            elif isinstance(x, self.np.ndarray):
                h.update(str(x.dtype).encode()); h.update(str(x.shape).encode()); h.update(x.tobytes())
            elif isinstance(x, dict):
                for key in sorted(x, key=repr):
                    visit(key); visit(x[key])
            elif isinstance(x, (list, tuple)):
                for v in x:
                    visit(v)
            else:
                h.update(repr(x).encode())
        visit(value)
        return h.hexdigest()

    def snapshot(self):
        return dict(model=self.tree(self.model.state_dict()), optimizer=self.tree(self.optimizer.state_dict()),
            sampler=self.sampler.state(), all_rng=self.rng(), optimizer_updates=self.updates,
            completed_epochs=self.completed, pending_validation=self.pending,
            input_signature_sha256=self.inputs.signature_sha256, request_path=str(BASE_SPEC),
            request_sha256=BASE_SHA, runtime_binding_sha256=self.base['runtime_binding_sha256'])

    def validate(self, state):
        check_metadata(state, self.base)
        t = self.torch
        assert all(t.isfinite(v).all() for v in state['model'].values())
        assert len(state['optimizer']['state']) == len(list(self.model.parameters()))
        for row in state['optimizer']['state'].values():
            assert set(row) == {'step', 'exp_avg', 'exp_avg_sq'} and int(row['step']) == state['optimizer_updates']
            assert all(t.isfinite(v).all() for v in row.values() if t.is_tensor(v))
        for group in state['optimizer']['param_groups']:
            assert group['lr'] == .0001 and group['weight_decay'] == .01

    def restore(self, state):
        self.model.load_state_dict(state['model'], strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        self.sampler.restore(state['sampler'])
        self.restore_rng(state['all_rng'])
        self.updates, self.completed, self.pending = state['optimizer_updates'], state['completed_epochs'], state['pending_validation']
        self.model.train(True)
        assert self.equal(self.tree(self.model.state_dict(), True), self.tree(state['model'], True))
        assert self.equal(self.tree(self.optimizer.state_dict(), True), self.tree(state['optimizer'], True))
        assert self.equal(self.rng(), state['all_rng']) and self.equal(self.sampler.state(), state['sampler'])
        for param, row in self.optimizer.state.items():
            assert row['exp_avg'].shape == row['exp_avg_sq'].shape == param.shape

    def next_batch_sha(self):
        prior, rng = self.sampler.state(), self.rng()
        _, digest = self.inputs.batch(self.sampler.take(32))
        self.sampler.restore(prior)
        assert self.equal(self.rng(), rng)
        return digest

    def update(self):
        assert not self.pending and self.completed < 100
        batch, digest = self.inputs.batch(self.sampler.take(32))
        values = self.original_step(self.model, self.optimizer, batch, training=True)
        self.updates += 1
        self.pending = self.sampler.cursor == self.sampler.count
        state = self.snapshot()
        self.validate(state)
        self.last_good = state
        return values, digest

    def finish_validation(self):
        assert self.pending and self.sampler.cursor == 32462 and self.sampler.epoch == self.completed
        before_rng, before_model = self.rng(), self.module_sha(self.model)
        total, count = {}, 0
        try:
            for first in range(0, len(self.inputs.internal), 32):
                ids = list(range(first, min(first + 32, len(self.inputs.internal))))
                batch, _ = self.inputs.batch(ids, split='internal_validation')
                values = self.original_step(self.model, self.optimizer, batch, training=False)
                for key, value in values.items():
                    total[key] = total.get(key, 0.) + value * len(ids)
                count += len(ids)
        finally:
            self.restore_rng(before_rng)
            self.model.train(True)
        assert count == 1606 and self.module_sha(self.model) == before_model
        assert self.equal(self.rng(), before_rng)
        assert all(int(row['step']) == self.updates for row in self.optimizer.state.values())
        self.completed += 1
        self.pending = False
        self.last_good = self.snapshot()
        return {k: v / count for k, v in total.items()}

    def unchanged(self):
        from ldm100k_verified_input import sha
        self.inputs.verify_unchanged()
        assert sha(EMERGENCY) == EMERGENCY_SHA
        origin(self.s)

    def witness(self):
        original = self.tree(self.last_good, True)
        records = [self.update(), self.update()]
        saved = self.tree(self.snapshot(), True)
        buffer = io.BytesIO()
        self.torch.save(saved, buffer)
        raw = buffer.getvalue()
        assert len(raw) <= 201326592
        reference = self.update()
        expected = self.tree(self.snapshot(), True)
        loaded = self.torch.load(io.BytesIO(raw), map_location='cpu')
        assert self.equal(loaded, saved)
        self.restore(loaded)
        replay = self.update()
        assert reference == replay and self.equal(self.tree(self.snapshot(), True), expected)
        replay_hash = self.content_hash(expected)
        self.restore(original)
        assert self.equal(self.tree(self.snapshot(), True), original)
        assert self.next_batch_sha() == self.initial_next_batch_sha256
        self.unchanged()
        return dict(actual_diagnostic_optimizer_calls=4, selected_logical_diagnostic_steps=3,
            actual_new_formal_optimizer_updates=0, final_restored_optimizer_updates=83703,
            exact_third_step_loss_fullstate_replay=True, source_inputs_and_checkpoint_unchanged=True,
            replayed_content_sha256=replay_hash, diagnostic_losses=[x[0] for x in records + [reference, replay]],
            diagnostic_batch_sha256=[x[1] for x in records + [reference, replay]],
            diagnostic_weights_discarded=True, memory_fullstate_bytes=len(raw), persistent_checkpoint_writes=0)

    def home_guard(self, emergency=False):
        root = pathlib.Path(self.s['checkpoint_dir'])
        planned = ['tmr_epoch_100_resumed.pth', 'tmr_resume_emergency.pth']
        missing = 1 if emergency else sum(not (root / name).exists() for name in planned)
        required = self.s['reserve_bytes'] + self.s['log_allowance_bytes'] + missing * self.s['checkpoint_upper_bytes']
        free = shutil.disk_usage(root).free
        assert free >= required, (free, required)
        return dict(free_bytes=free, required_bytes=required, remaining_slots=missing)

    def save_checkpoint(self, emergency=False):
        from ldm100k_verified_input import sha
        root = pathlib.Path(self.s['checkpoint_dir'])
        name = 'tmr_resume_emergency.pth' if emergency else 'tmr_epoch_100_resumed.pth'
        path = root / name
        partial = path.with_name(name + '.partial')
        assert not path.exists() and not partial.exists()
        capacity = self.home_guard(emergency)
        state = self.tree(self.last_good, True)
        self.validate(state)
        state['capacity_resume'] = dict(request_path=str(self.sp), request_sha256=sha(self.sp),
            runtime_binding_sha256=self.s['runtime_binding_sha256'], source_checkpoint_sha256=EMERGENCY_SHA,
            original_safe_updates=83703, original_failed_attempts=83704)
        if not emergency:
            assert state['optimizer_updates'] == 101500 and state['completed_epochs'] == 100 and state['pending_validation'] is False
            state['next_batch_sha256'] = self.next_batch_sha()
        with partial.open('xb') as f:
            os.chmod(partial, 0o600)
            self.torch.save(state, f); f.flush(); os.fsync(f.fileno())
        assert partial.stat().st_size <= self.s['checkpoint_upper_bytes']
        loaded = self.torch.load(partial, map_location='cpu')
        assert self.equal(loaded, state)
        self.validate(loaded)
        assert shutil.disk_usage(root).free >= self.s['reserve_bytes'] + self.s['log_allowance_bytes']
        partial.rename(path)
        fd = os.open(str(root), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        if not emergency:
            self.restore(loaded)
            assert self.next_batch_sha() == loaded['next_batch_sha256']
            assert self.equal(self.tree(self.snapshot(), True), {k: v for k, v in loaded.items() if k not in ['next_batch_sha256', 'capacity_resume']})
        return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path),
            epoch=loaded['completed_epochs'], optimizer_updates=loaded['optimizer_updates'],
            pending_validation=loaded['pending_validation'], emergency=emergency,
            exact_fullstate_deserialization=True, production_model_adamw_rng_sampler_nextbatch_reloaded=not emergency,
            capacity_before=capacity)

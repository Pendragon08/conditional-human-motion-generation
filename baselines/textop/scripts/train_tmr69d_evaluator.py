"""Fresh TRAIN-only TMR evaluator; generator/teacher weights are never loaded.

Manual lifecycle around unchanged original model/loss/AdamW methods. Immutable
full states at epochs100/500 and at most one emergency slot; all versions stay.
"""
import argparse
import copy
import json
import os
import pathlib
import shutil
import signal
import time
import traceback
from ldm100k_verified_input import ROOT, read, sha, owned

p = argparse.ArgumentParser()
p.add_argument('--request', required=True)
a = p.parse_args()
sp = owned(a.request)
s = read(sp)
assert os.getuid() == 1013 and sha(__file__) == s['executor_sha256'] and s['formal_training_authorized']
assert set(os.sched_getaffinity(0)) == {0, 1, 2, 3} and os.environ.get('CUDA_VISIBLE_DEVICES') == '1'
assert s['epochs'] == 500 and s['batch_size'] == 32 and s['expected_optimizer_updates'] == 507500
assert s['save_epochs'] == [100, 500] and s['checkpoint_upper_bytes'] == 201326592
for name, value in s['bound_sources'].items():
    assert sha(owned(ROOT / name)) == value
parent = owned(ROOT / 'evaluator_tmr_20261029')
out = parent / s['run']
out.mkdir(mode=0o700, exist_ok=False)
ckroot = owned(s['checkpoint_dir'], pathlib.Path('/home/ouyanghao'))
assert not list(ckroot.iterdir())
planned = ['tmr_epoch_100.pth', 'tmr_epoch_500.pth', 'tmr_emergency.pth']
started = time.monotonic()
receipt = dict(status='PREPARING_FRESH_TMR69D_FORMAL_EVALUATOR', run=s['run'], source_sha256=sha(__file__),
    request_sha256=sha(sp), output_writers_closed=False, formal_optimizer_updates=0, attempted_optimizer_updates=0,
    skipped_updates=0, completed_epochs=0, target_epochs=500, expected_optimizer_updates=507500,
    generator_or_teacher_training_updates=0, original_validation_sources_used_for_gradient=0,
    semantic_metrics_computed=False, checkpoints=[])
last_good = None
metrics = None


def save_receipt():
    receipt['elapsed_seconds'] = time.monotonic() - started
    receipt['home_free_bytes'] = shutil.disk_usage(ckroot).free
    receipt['data_free_bytes'] = shutil.disk_usage(out).free
    tmp = out / 'receipt.partial'
    with tmp.open('w') as f:
        json.dump(receipt, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(out / 'receipt.json')


def home_guard(emergency=False):
    missing = 1 if emergency else sum(not (ckroot / n).exists() for n in planned)
    required = s['reserve_bytes'] + s['home_log_allowance_bytes'] + missing * s['checkpoint_upper_bytes']
    free = shutil.disk_usage(ckroot).free
    assert free >= required, (free, required)
    return dict(free_bytes=free, required_bytes=required, missing_slots=missing)


def interrupted(signum, frame):
    raise RuntimeError('Owned TMR evaluator interrupted with signal {}'.format(signum))


for sig in [signal.SIGTERM, signal.SIGINT]:
    signal.signal(sig, interrupted)
save_receipt()
try:
    import random
    import numpy as np
    import torch
    from tmr69d_training_inputs import Inputs
    from tmr69d_core import make_model, Sampler, rng_state, restore_rng, step
    from generator_training_state import equal, module_sha
    torch.set_num_threads(4)
    assert torch.cuda.device_count() == 1
    torch.cuda.set_per_process_memory_fraction(.65, 0)
    assert torch.cuda.mem_get_info()[0] >= 20 * 1024**3
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    def progress(**value):
        receipt['input_preparation'] = value
        save_receipt()

    inputs = Inputs(s['inputs'], 'cuda:0', progress)
    assert inputs.signature_sha256 == s['accepted_input_signature_sha256']
    receipt['actual_input_signature'] = inputs.signature
    receipt['actual_input_signature_sha256'] = inputs.signature_sha256
    home_guard()
    random.seed(s['seed'])
    np.random.seed(s['seed'])
    torch.manual_seed(s['seed'])
    torch.cuda.manual_seed_all(s['seed'])
    model, optimizer, audit = make_model(owned(s['upstream_model_dir']), s['upstream_model_sha256'], 'cuda:0')
    receipt['initial_model_sha256'] = module_sha(model)
    assert receipt['initial_model_sha256'] != s['diagnostic_initial_model_sha256']
    receipt['original_class_AST_audit'] = audit
    sampler = Sampler(len(inputs.gradient), s['seed'] + 1)
    updates = completed = 0
    metrics = (out / 'metrics.jsonl').open('x')

    def tree(value, to_cpu):
        if torch.is_tensor(value):
            return value.detach().cpu().clone() if to_cpu else value.detach().clone()
        if isinstance(value, dict):
            return {k: tree(v, to_cpu) for k, v in value.items()}
        if isinstance(value, tuple):
            return tuple(tree(v, to_cpu) for v in value)
        if isinstance(value, list):
            return [tree(v, to_cpu) for v in value]
        return copy.deepcopy(value)

    def snapshot(pending_validation=False):
        return dict(model=tree(model.state_dict(), False), optimizer=tree(optimizer.state_dict(), False),
            sampler=sampler.state(), all_rng=rng_state(), optimizer_updates=updates, completed_epochs=completed,
            pending_validation=pending_validation, input_signature_sha256=inputs.signature_sha256,
            request_path=str(sp), request_sha256=sha(sp), runtime_binding_sha256=s['runtime_binding_sha256'])

    def validate_state(state):
        assert state['input_signature_sha256'] == inputs.signature_sha256 and state['request_sha256'] == sha(sp)
        assert state['sampler']['steps'] == state['optimizer_updates']
        if state['optimizer_updates']:
            assert all(int(x['step']) == state['optimizer_updates'] for x in state['optimizer']['state'].values())
        assert all(torch.isfinite(v).all() for v in state['model'].values())
        assert all(torch.isfinite(v).all() for row in state['optimizer']['state'].values() for v in row.values() if torch.is_tensor(v))

    def next_batch_sha():
        prior = sampler.state()
        ids = sampler.take(s['batch_size'])
        _, result = inputs.batch(ids)
        sampler.restore(prior)
        return result

    def checkpoint(name, state, emergency=False):
        assert name in planned
        path = ckroot / name
        partial = path.with_name(path.name + '.partial')
        assert not path.exists() and not partial.exists()
        capacity = home_guard(emergency)
        cpu_state = tree(state, True)
        validate_state(cpu_state)
        if not emergency:
            cpu_state['next_batch_sha256'] = next_batch_sha()
        with partial.open('xb') as f:
            os.chmod(partial, 0o600)
            torch.save(cpu_state, f)
            f.flush()
            os.fsync(f.fileno())
        assert partial.stat().st_size <= s['checkpoint_upper_bytes']
        loaded = torch.load(partial, map_location='cpu')
        assert equal(loaded, cpu_state)
        validate_state(loaded)
        assert shutil.disk_usage(ckroot).free >= s['reserve_bytes'] + s['home_log_allowance_bytes']
        partial.rename(path)
        fd = os.open(str(ckroot), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        if not emergency:
            model.load_state_dict(loaded['model'], strict=True)
            optimizer.load_state_dict(loaded['optimizer'])
            sampler.restore(loaded['sampler'])
            restore_rng(loaded['all_rng'])
            assert equal(tree(model.state_dict(), True), loaded['model']) and equal(tree(optimizer.state_dict(), True), loaded['optimizer'])
            assert equal(rng_state(), loaded['all_rng']) and next_batch_sha() == loaded['next_batch_sha256']
            assert equal(sampler.state(), loaded['sampler']) and equal(rng_state(), loaded['all_rng'])
        value = dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path), epoch=loaded['completed_epochs'],
            optimizer_updates=loaded['optimizer_updates'], pending_validation=loaded['pending_validation'], emergency=emergency,
            exact_fullstate_deserialization=True, production_model_adamw_rng_sampler_nextbatch_reloaded=not emergency,
            capacity_before=capacity)
        receipt['checkpoints'].append(value)
        save_receipt()
        return value

    def log(row):
        metrics.write(json.dumps(row, allow_nan=False) + '\n')
        metrics.flush()

    last_good = snapshot()
    initial_rng = rng_state()
    baseline_batch, baseline_batch_sha = inputs.batch(list(range(32)), split='internal_validation')
    receipt['initial_internal_validation_batch32_loss'] = step(model, optimizer, baseline_batch, training=False)
    restore_rng(initial_rng)
    model.train(True)
    assert not optimizer.state and module_sha(model) == receipt['initial_model_sha256'] and equal(rng_state(), initial_rng)
    receipt['initial_internal_validation_batch_sha256'] = baseline_batch_sha
    receipt.update(status='RUNNING_FRESH_TMR69D_EVALUATOR_TRAINING', fresh_model_and_adamw_initialized=True,
        diagnostic_weights_used_for_initialization=False, initial_epoch=0, initial_optimizer_updates=0,
        optimizer=dict(type='torch.optim.AdamW',lr=.0001,weight_decay=.01))
    save_receipt()
    while completed < s['epochs']:
        ids = sampler.take(s['batch_size'])
        batch, batch_sha = inputs.batch(ids)
        receipt['attempted_optimizer_updates'] += 1
        values = step(model, optimizer, batch, training=True)
        assert all(torch.isfinite(v).all() for row in optimizer.state.values() for v in row.values() if torch.is_tensor(v))
        updates += 1
        assert sampler.steps == updates and all(int(x['step']) == updates for x in optimizer.state.values())
        receipt['formal_optimizer_updates'] = updates
        epoch_end = sampler.cursor == sampler.count
        last_good = snapshot(pending_validation=epoch_end)
        if updates % 100 == 0 or epoch_end:
            home_guard()
            log(dict(kind='train', updates=updates, completed_epochs=completed, batch_size=len(ids), batch_sha256=batch_sha, losses=values))
            receipt.update(last_train_loss=values, sampler_epoch=sampler.epoch, sampler_cursor=sampler.cursor)
            save_receipt()
        if epoch_end:
            before_rng = rng_state()
            train_model_sha = module_sha(model)
            val_sum, val_count = {}, 0
            try:
                for first in range(0, len(inputs.internal), s['batch_size']):
                    vi = list(range(first, min(first + s['batch_size'], len(inputs.internal))))
                    vb, _ = inputs.batch(vi, split='internal_validation')
                    vv = step(model, optimizer, vb, training=False)
                    for key, value in vv.items():
                        val_sum[key] = val_sum.get(key, 0.) + value * len(vi)
                    val_count += len(vi)
            finally:
                restore_rng(before_rng)
                model.train(True)
            assert val_count == 1606 and module_sha(model) == train_model_sha
            assert equal(rng_state(), before_rng) and all(int(x['step']) == updates for x in optimizer.state.values())
            completed += 1
            receipt['completed_epochs'] = completed
            receipt['last_internal_validation_loss'] = {key: value/val_count for key,value in val_sum.items()}
            receipt['internal_validation_optimizer_updates'] = 0
            log(dict(kind='internal_validation', completed_epochs=completed, updates=updates,
                losses=receipt['last_internal_validation_loss'], actual_optimizer_updates=0, training_rng_restored=True))
            last_good = snapshot()
            save_receipt()
            if completed in s['save_epochs']:
                inputs.verify_unchanged()
                checkpoint('tmr_epoch_' + str(completed) + '.pth', last_good)
    assert updates == sampler.steps == s['expected_optimizer_updates']
    assert receipt['attempted_optimizer_updates'] == updates and receipt['skipped_updates'] == 0
    assert receipt['checkpoints'][-1]['epoch'] == 500 and not receipt['checkpoints'][-1]['pending_validation']
    inputs.verify_unchanged()
    metrics.flush()
    os.fsync(metrics.fileno())
    metrics.close()
    metrics = None
    receipt.update(status='TMR69D_FORMAL_EVALUATOR_500_EPOCHS_FINISHED', output_writers_closed=True,
        selected_checkpoint=receipt['checkpoints'][-1], original_class_method_AST_unchanged=True,
        final_model_sha256=module_sha(model), actual_original_validation_source_reads=0,
        exact_final_model_adamw_all_rng_sampler_nextbatch_disk_reload=True,
        model_sources_and_training_inputs_unchanged=True, inference_export_and_heldout_semantic_metrics_accepted=False)
    save_receipt()
except BaseException as error:
    receipt.update(status='FAILED_TMR69D_FORMAL_EVALUATOR_ALL_STATES_RETAINED', error=repr(error), traceback=traceback.format_exc())
    if 'optimizer' in locals():
        try:
            receipt['observed_adam_step_counts_at_failure'] = sorted({int(x['step']) for x in optimizer.state.values()})
        except BaseException as step_error:
            receipt['adam_step_observation_error'] = repr(step_error)
    if last_good is not None:
        try:
            receipt['emergency_checkpoint'] = checkpoint('tmr_emergency.pth', last_good, emergency=True)
        except BaseException as emergency_error:
            receipt['emergency_error'] = repr(emergency_error)
    if metrics is not None:
        metrics.flush()
        metrics.close()
    save_receipt()
    raise

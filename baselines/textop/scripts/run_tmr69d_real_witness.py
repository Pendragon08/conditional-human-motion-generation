"""Real TRAIN69D/frozen-text GPU gradients and exact full-state replay witness."""
import argparse
import copy
import hashlib
import io
import json
import os
import pathlib
import time
import traceback
from ldm100k_verified_input import ROOT, read, sha, owned

p = argparse.ArgumentParser()
p.add_argument('--request', required=True)
a = p.parse_args()
sp = owned(a.request)
s = read(sp)
assert os.getuid() == 1013 and sha(__file__) == s['executor_sha256']
assert set(os.sched_getaffinity(0)) == {0, 1, 2, 3} and os.environ.get('CUDA_VISIBLE_DEVICES') == '1'
assert s['role'] in ['author', 'document'] and not s['formal_training_authorized']
for name, value in s['bound_sources'].items():
    assert sha(owned(ROOT / name)) == value
parent = owned(ROOT / 'evaluator_tmr_20261029')
out = parent / s['run']
out.mkdir(mode=0o700, exist_ok=False)
started = time.monotonic()
rec = dict(status='STARTING_TMR69D_REAL_DATA_GPU_WITNESS', run=s['run'], role=s['role'], source_sha256=sha(__file__),
    request_sha256=sha(sp), runtime_binding_sha256=s['runtime_binding_sha256'], output_writers_closed=False,
    actual_diagnostic_optimizer_calls=0, actual_formal_evaluator_updates=0, generator_or_teacher_updates=0,
    persistent_fullstate_writes=0, semantic_metrics_computed=False)


def save():
    rec['elapsed_seconds'] = time.monotonic() - started
    tmp = out / 'receipt.partial'
    with tmp.open('w') as f:
        json.dump(rec, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(out / 'receipt.json')


def progress(**kw):
    rec['input_preparation'] = kw
    save()


save()
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
    inputs = Inputs(s['inputs'], 'cuda:0', progress)
    rec['actual_input_signature'] = inputs.signature
    rec['actual_input_signature_sha256'] = inputs.signature_sha256
    # New model initialization follows frozen-cache preparation, without using
    # weights from either the earlier synthetic contract or the other role.
    random.seed(s['seed'])
    np.random.seed(s['seed'])
    torch.manual_seed(s['seed'])
    torch.cuda.manual_seed_all(s['seed'])
    model, optimizer, class_audit = make_model(owned(s['upstream_model_dir']), s['upstream_model_sha256'], 'cuda:0')
    initial_sha = module_sha(model)
    sampler = Sampler(len(inputs.gradient), s['seed'] + 1)

    def cpu(value):
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {k: cpu(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cpu(v) for v in value]
        if isinstance(value, tuple):
            return tuple(cpu(v) for v in value)
        return copy.deepcopy(value)

    def snapshot(logical):
        return dict(model=cpu(model.state_dict()), optimizer=cpu(optimizer.state_dict()), sampler=sampler.state(),
            all_rng=rng_state(), logical_optimizer_steps=logical, input_signature_sha256=inputs.signature_sha256,
            runtime_binding_sha256=s['runtime_binding_sha256'])

    def content_hash(value):
        h = hashlib.sha256()
        def visit(x):
            h.update(type(x).__name__.encode())
            if torch.is_tensor(x):
                y = x.detach().cpu().contiguous()
                h.update(str(y.dtype).encode())
                h.update(str(tuple(y.shape)).encode())
                h.update(y.numpy().tobytes())
            elif isinstance(x, np.ndarray):
                h.update(str(x.dtype).encode())
                h.update(str(x.shape).encode())
                h.update(x.tobytes())
            elif isinstance(x, dict):
                for key in sorted(x, key=repr):
                    visit(key)
                    visit(x[key])
            elif isinstance(x, (list, tuple)):
                for element in x:
                    visit(element)
            else:
                h.update(repr(x).encode())
        visit(value)
        return h.hexdigest()

    def update():
        ids = sampler.take(32)
        batch, batch_sha = inputs.batch(ids)
        values = step(model, optimizer, batch, training=True)
        assert all(torch.isfinite(v).all() for row in optimizer.state.values() for v in row.values() if torch.is_tensor(v))
        rec['actual_diagnostic_optimizer_calls'] += 1
        save()
        return values, batch_sha

    records = [update(), update()]
    saved = snapshot(2)
    buffer = io.BytesIO()
    torch.save(saved, buffer)
    raw = buffer.getvalue()
    assert len(raw) <= 201326592
    reference_values, reference_batch_sha = update()
    reference = snapshot(3)
    loaded = torch.load(io.BytesIO(raw), map_location='cpu')
    assert equal(loaded, saved)
    model.load_state_dict(loaded['model'], strict=True)
    optimizer.load_state_dict(loaded['optimizer'])
    sampler.restore(loaded['sampler'])
    restore_rng(loaded['all_rng'])
    assert equal(cpu(model.state_dict()), loaded['model']) and equal(cpu(optimizer.state_dict()), loaded['optimizer'])
    assert equal(rng_state(), loaded['all_rng'])
    replay_values, replay_batch_sha = update()
    assert reference_values == replay_values and reference_batch_sha == replay_batch_sha
    assert equal(snapshot(3), reference)
    assert all(int(x['step']) == 3 for x in optimizer.state.values())
    before_validation = snapshot(3)
    validation_batch, validation_sha = inputs.batch(list(range(32)), split='internal_validation')
    validation_values = step(model, optimizer, validation_batch, training=False)
    restore_rng(before_validation['all_rng'])
    model.train(True)
    assert equal(snapshot(3), before_validation)
    inputs.verify_unchanged()
    assert module_sha(model) != initial_sha
    rec.update(status='TMR69D_REAL_DATA_GPU_WITNESS_VERIFIED', output_writers_closed=True,
        original_class_AST_audit=class_audit, original_class_method_AST_unchanged=True,
        selected_logical_optimizer_steps=3, actual_random_tmr_model_constructions=1, frozen_public_text_model_constructions=2,
        actual_real_train_source_families=5361, actual_original_validation_source_reads=0,
        memory_fullstate_bytes=len(raw), memory_fullstate_sha256=hashlib.sha256(raw).hexdigest(),
        exact_model_adamw_all_rng_sampler_nextbatch_loss_replay=True,
        actual_initial_model_sha256=initial_sha, actual_final_model_sha256=module_sha(model),
        final_model_adamw_rng_sampler_content_sha256=content_hash(snapshot(3)),
        actual_losses=[x[0] for x in records] + [reference_values, replay_values],
        actual_batch_sha256=[x[1] for x in records] + [reference_batch_sha, replay_batch_sha],
        internal_validation_loss=validation_values, internal_validation_batch_sha256=validation_sha,
        actual_validation_optimizer_updates=0, validation_restored_training_rng=True,
        source_arrays_norm_and_public_model_fingerprints_unchanged=True,
        optimizer=dict(type='torch.optim.AdamW',lr=.0001,weight_decay=.01),
        diagnostic_weights_discarded_not_formal_initialization=True,
        gpu_name=torch.cuda.get_device_name(0), cuda_visible_devices='1', headless=True)
    save()
except BaseException as error:
    rec.update(status='FAILED_TMR69D_REAL_DATA_GPU_WITNESS_ALL_OUTPUTS_RETAINED', error=repr(error), traceback=traceback.format_exc())
    save()
    raise

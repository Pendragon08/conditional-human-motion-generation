"""Actual CPU synthetic original-loss/AdamW/memory-restore contract probe.

Four diagnostic optimizer calls select logical step3 after exact replay. No real
robot/text data, public weight payload, saved checkpoint or formal seed is used.
"""
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
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and set(os.sched_getaffinity(0)) == {0, 1, 2, 3}
assert sha(ROOT / 'scripts/tmr69d_core.py') == s['core_sha256']
parent = owned(ROOT / 'evaluator_tmr_20261029')
out = parent / s['run']
out.mkdir(mode=0o700, exist_ok=False)
started = time.monotonic()
rec = dict(status='STARTING_TMR69D_SYNTHETIC_CPU_CONTRACT', run=s['run'], source_sha256=sha(__file__),
    request_sha256=sha(sp), output_writers_closed=False, real_robot_or_babel_text_reads=0,
    pretrained_weight_payload_reads=0, actual_formal_evaluator_updates=0, actual_diagnostic_optimizer_calls=0,
    formal_training_authorized=False, semantic_metrics_computed=False)


def save():
    rec['elapsed_seconds'] = time.monotonic() - started
    tmp = out / 'receipt.partial'
    with tmp.open('w') as f:
        json.dump(rec, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(out / 'receipt.json')


save()
try:
    import random
    import numpy as np
    import torch
    from tmr69d_core import make_model, Sampler, rng_state, restore_rng, step
    from generator_training_state import equal, module_sha
    random.seed(20261029)
    np.random.seed(20261029)
    torch.manual_seed(20261029)
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized() and not torch.cuda.is_available()
    model, optimizer, ast_audit = make_model(owned(s['upstream_model_dir']), s['upstream_model_sha256'], 'cpu')
    initial_model_sha = module_sha(model)
    sampler = Sampler(17, 20261029)

    def batch():
        ids = sampler.take(3)
        lengths = [2 + x % 4 for x in ids]
        text_lengths = [2 + x % 3 for x in ids]
        motion = torch.randn(3, max(lengths), 69)
        text = torch.randn(3, max(text_lengths), 768)
        mm = torch.arange(max(lengths))[None] < torch.tensor(lengths)[:, None]
        tm = torch.arange(max(text_lengths))[None] < torch.tensor(text_lengths)[:, None]
        motion[~mm] = 0
        text[~tm] = 0
        result = dict(motion_x_dict=dict(x=motion, mask=mm), text_x_dict=dict(x=text, mask=tm),
            sent_emb=torch.nn.functional.normalize(torch.randn(3, 768), dim=-1))
        h = hashlib.sha256()
        for value in [motion, text, mm, tm, result['sent_emb']]:
            h.update(value.numpy().tobytes())
        return result, ids, h.hexdigest()

    losses = []
    for _ in range(2):
        b, _, _ = batch()
        losses.append(step(model, optimizer, b, training=True))
        rec['actual_diagnostic_optimizer_calls'] += 1
    state = dict(model=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()),
                 sampler=sampler.state(), all_rng=rng_state(), logical_optimizer_steps=2)
    memory = io.BytesIO()
    torch.save(state, memory)
    checkpoint_bytes = memory.getvalue()
    assert len(checkpoint_bytes) <= 201326592
    b, reference_ids, reference_batch_sha = batch()
    reference_loss = step(model, optimizer, b, training=True)
    rec['actual_diagnostic_optimizer_calls'] += 1
    reference = dict(model=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()),
                     sampler=sampler.state(), all_rng=rng_state(), logical_optimizer_steps=3)
    restored = torch.load(io.BytesIO(checkpoint_bytes), map_location='cpu')
    assert equal(restored, state)
    model.load_state_dict(restored['model'], strict=True)
    optimizer.load_state_dict(restored['optimizer'])
    sampler.restore(restored['sampler'])
    restore_rng(restored['all_rng'])
    assert equal(model.state_dict(), restored['model']) and equal(optimizer.state_dict(), restored['optimizer'])
    assert equal(rng_state(), restored['all_rng'])
    b, replay_ids, replay_batch_sha = batch()
    assert reference_ids == replay_ids and reference_batch_sha == replay_batch_sha
    replay_loss = step(model, optimizer, b, training=True)
    rec['actual_diagnostic_optimizer_calls'] += 1
    replay = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), sampler=sampler.state(),
                  all_rng=rng_state(), logical_optimizer_steps=3)
    assert equal(reference, replay) and reference_loss == replay_loss
    assert module_sha(model) != initial_model_sha and not torch.cuda.is_initialized()
    assert all(int(x['step']) == 3 for x in optimizer.state_dict()['state'].values())
    rec.update(status='TMR69D_SYNTHETIC_CPU_CORE_CONTRACT_VERIFIED', output_writers_closed=True,
        original_class_AST_audit=ast_audit, actual_random_cpu_model_constructions=1,
        selected_logical_optimizer_steps=3, memory_checkpoint_bytes=len(checkpoint_bytes),
        memory_checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(), persistent_checkpoint_writes=0,
        exact_model_adamw_all_rng_sampler_nextbatch_loss_replay=True, original_method_bodies_unchanged=True,
        optimizer=dict(type='torch.optim.AdamW', lr=optimizer.defaults['lr'], weight_decay=optimizer.defaults['weight_decay']),
        actual_losses=losses + [reference_loss, replay_loss], synthetic_nextbatch_sha256=reference_batch_sha,
        final_model_sha256=module_sha(model), cuda_initialized=False,
        diagnostic_weights_discarded_not_formal_initialization=True, real_data_or_semantic_acceptance_claimed=False)
    save()
except BaseException as error:
    rec.update(status='FAILED_TMR69D_SYNTHETIC_CPU_CONTRACT_ALL_OUTPUTS_RETAINED', error=repr(error), traceback=traceback.format_exc())
    save()
    raise

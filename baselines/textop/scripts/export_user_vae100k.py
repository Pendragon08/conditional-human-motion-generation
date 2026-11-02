"""Future CPU-only inference export from the real closed filtered user-shortened VAE100k.

All original full states remain private and unchanged. No optimizer or CUDA RNG
is restored by this exporter, and no data or model is published by this script.
"""
import argparse
import importlib.metadata
import json
import os
import pathlib
import sys
import time
import traceback
from user_vae100k_completion import ROOT, sha, read, owned, accepted_vae100k as completed_vae

p = argparse.ArgumentParser()
p.add_argument('--run', required=True)
a = p.parse_args()
assert a.run.isascii() and a.run.isidentifier()
assert os.getuid() == 1013 and set(os.sched_getaffinity(0)) == {0, 1, 2, 3}
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
sp = ROOT / 'configs/user-vae100k-export-spec.json'
s = read(sp)
assert s['schema'] == 'textop.user_vae100k_cpu_export.v1' and s['export_authorized']
assert sha(__file__) == s['executor_sha256']
for n, h in s['bound_sources'].items():
    assert sha(ROOT / n) == h, n
base = read(ROOT / 'configs/paper69-env-spec.json')
assert sys.version.split()[0] == base['python_version']
assert {k: importlib.metadata.version(k) for k in base['required_versions']} == base['required_versions']
v = completed_vae(s['vae_training'])
out = ROOT / 'filtered15000_20261027/vae_export_runs' / a.run
out.mkdir(mode=0o700, parents=True, exist_ok=False)
started = time.monotonic()
r = dict(status='STARTING_FILTERED15000_VAE_CPU_EXPORT', run=a.run,
         source_sha256=sha(__file__), spec_sha256=sha(sp), output_writers_closed=False,
         vae_training_receipt_sha256=v['receipt_sha256'], new_gpu_contexts=0,
         new_sdk_runs=0, optimizer_updates=0, source_full_state_preserved=True)


def save():
    r['elapsed_seconds'] = time.monotonic() - started
    t = out / 'receipt.partial'
    t.write_text(json.dumps(r, indent=2, allow_nan=False))
    t.replace(out / 'receipt.json')


save()
try:
    import torch
    import numpy as np
    from generator_training_state import equal, module_sha
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    ck = v['checkpoint']
    assert sha(ck['path']) == ck['sha256']
    state = torch.load(ck['path'], map_location='cpu')
    assert state['schema'] == 'textop.generator_training_state.v1' and state['step'] == 100000
    assert state['binding']['stage'] == 'vae' and state['binding']['actual_strict_input']
    assert not state['binding']['prefilter_data'] and state['binding']['feature_dim'] == 69
    assert state['binding']['corpus_receipt_sha256'] == v['spec']['corpus_receipt_sha256']
    assert state['binding']['numeric_sha256'] == v['runtime_binding_sha256']
    assert state['primitive_clock']['completed_primitive_attempts'] == 100000
    assert state['primitive_clock']['actual_optimizer_updates'] == v['receipt']['optimizer_updates']
    assert state['primitive_clock']['skipped_updates'] == v['receipt']['skipped_updates']
    # The unchanged VAE runtime drains native10 validation primitives through
    # complete four-primitive batches. Its closed counter can therefore be-2.
    # Preserve that signed field exactly; only a positive pending count blocks
    # export. Do not invent a zero counter or modify the trained checkpoint.
    pending_eval = state['manager_fields']['pending_evaluation_steps']
    assert not state['ema_models'] and pending_eval <= 0
    tensor_count = [0]

    def finite(x):
        if torch.is_tensor(x):
            assert x.device.type == 'cpu' and torch.isfinite(x).all()
            tensor_count[0] += 1
        elif isinstance(x, dict):
            for value in x.values():
                finite(value)
        elif isinstance(x, (list, tuple)):
            for value in x:
                finite(value)
        elif isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number):
            assert np.isfinite(x).all()

    finite(state)
    steps = [int(x['step']) for x in state['optimizer']['state'].values()]
    assert steps and min(steps) == max(steps) == v['receipt']['optimizer_updates']
    sys.path.insert(0, str(ROOT / 'upstream/TextOpRobotMDAR'))
    from robotmdar.model.mld_vae import AutoMldVae
    model = AutoMldVae(nfeats=69, latent_dim=[1, 128], h_dim=512,
                       ff_size=1024, num_layers=9, num_heads=4).cpu().eval()
    model.load_state_dict(state['vae'], strict=True)
    assert equal(model.state_dict(), state['vae'])
    assert module_sha(model) == v['receipt']['trained_model_sha256']
    allowed = pathlib.Path(s['export_allowed_root'])
    parent = owned(s['export_dir'], allowed)
    path = parent / 'vae100000_inference.private.pth'
    assert not path.exists() and not path.with_suffix('.partial').exists()
    import shutil
    assert shutil.disk_usage(parent).free >= s['reserve_bytes'] + s['log_allowance_bytes'] + s['export_upper_bound_bytes']
    payload = dict(schema='textop.filtered15000_vae_inference.v1', vae=model.state_dict(),
                   completed_primitive_attempts=100000, actual_optimizer_updates=v['receipt']['optimizer_updates'],
                   source_full_state_sha256=ck['sha256'], source_training_receipt_sha256=v['receipt_sha256'],
                   corpus_receipt_sha256=v['spec']['corpus_receipt_sha256'],
                   data_acceptance_sha256=v['spec']['data_acceptance_sha256'],
                   strict_audit_receipt_sha256=v['spec']['strict_audit_receipt_sha256'],
                   trained_model_sha256=module_sha(model))
    partial = path.with_suffix('.partial')
    with partial.open('xb') as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    partial.replace(path)
    assert path.stat().st_size <= s['export_upper_bound_bytes']
    loaded = torch.load(path, map_location='cpu')
    assert equal(payload, loaded) and 'optimizer' not in loaded and 'rng' not in loaded
    fresh = AutoMldVae(nfeats=69, latent_dim=[1, 128], h_dim=512,
                       ff_size=1024, num_layers=9, num_heads=4).cpu().eval()
    fresh.load_state_dict(loaded['vae'], strict=True)
    assert module_sha(fresh) == module_sha(model)
    assert sha(ck['path']) == ck['sha256'] and not torch.cuda.is_initialized()
    r.update(status='USER_VAE100000_CPU_EXPORT_PASSED',
             full_state_checkpoint=ck, finite_cpu_tensors=tensor_count[0],
             adam_step_min=min(steps), adam_step_max=max(steps),
             saved_pending_evaluation_counter=pending_eval,
             no_pending_evaluation_remaining=True,
             native_vae_eval_steps=10, runtime_validation_batch_primitives=4,
             inference=dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path)),
             trained_model_sha256=module_sha(model), exact_inference_reload=True,
             cpu_model_loads=2, cpu_full_state_loads=1, cuda_rng_restored=False,
             corpus_receipt_sha256=v['spec']['corpus_receipt_sha256'],
             data_acceptance_sha256=v['spec']['data_acceptance_sha256'],
             strict_audit_receipt_sha256=v['spec']['strict_audit_receipt_sha256'],
             output_writers_closed=True)
    save()
    print(json.dumps(r), flush=True)
except BaseException as e:
    r.update(status='FAILED_FILTERED15000_VAE_CPU_EXPORT', error=repr(e), traceback=traceback.format_exc())
    save()
    raise

"""New CPU-only integrity/export entry for the user's stopped100k LDM prefix.

Disabled until actual closed control evidence is bound by a separate activation.
No original Runtime, manager, CUDA RNG restore, data arrays, optimizer update,
or claim of semantic quality. All full states and failed outputs are retained.
"""
import argparse
import importlib.metadata
import json
import os
import pathlib
import shutil
import sys
import time
import traceback
from ldm100k_prefix_completion import ROOT, read, sha, owned, stopped_ldm100k, validate_state_metadata, verify_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    a = p.parse_args()
    assert a.run.isascii() and a.run.isidentifier() and os.getuid() == 1013
    assert set(os.sched_getaffinity(0)) == {0, 1, 2, 3}
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    sp = ROOT / 'configs/ldm100k-prefix-export-spec.json'
    s = read(sp)
    assert s['schema'] == 'textop.ldm100k_prefix_cpu_export.v1' and s['export_authorized']
    assert sha(__file__) == s['executor_sha256']
    for n, h in s['bound_sources'].items():
        assert sha(ROOT / n) == h, n
    base = read(ROOT / 'configs/paper69-env-spec.json')
    assert sys.version.split()[0] == base['python_version']
    assert {k: importlib.metadata.version(k) for k in base['required_versions']} == base['required_versions']
    v = stopped_ldm100k(s['ldm_training'])
    out = ROOT / 'ldm100k_user_handoff_20261028' / a.run
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    r = dict(status='STARTING_LDM100K_PREFIX_CPU_EXPORT', run=a.run,
             source_sha256=sha(__file__), spec_sha256=sha(sp), output_writers_closed=False,
             control_receipt_sha256=v['control_receipt_sha256'],
             source_full_state_preserved=True, new_gpu_contexts=0, new_sdk_tasks=0,
             training_optimizer_updates_performed=0, semantic_quality_accepted=False,
             original300k_completion_claimed=False)
    started = time.monotonic()

    def save():
        r['elapsed_seconds'] = time.monotonic() - started
        temp = out / 'receipt.partial'
        temp.write_text(json.dumps(r, indent=2, allow_nan=False))
        temp.replace(out / 'receipt.json')

    save()
    try:
        import torch
        import numpy as np
        from generator_training_state import equal, module_sha
        torch.set_num_threads(4)
        assert not torch.cuda.is_initialized()
        ck = v['checkpoint']
        fingerprint = verify_checkpoint(ck, v['policy']['checkpoint_allowed_root'])
        state = torch.load(ck['path'], map_location='cpu')
        audit = validate_state_metadata(state, ck, v['expected_state_binding'])
        count = [0]

        def finite(x):
            if torch.is_tensor(x):
                assert x.device.type == 'cpu' and bool(torch.isfinite(x).all())
                count[0] += 1
            elif isinstance(x, dict):
                for value in x.values():
                    finite(value)
            elif isinstance(x, (list, tuple)):
                for value in x:
                    finite(value)
            elif isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number):
                assert np.isfinite(x).all()
            elif isinstance(x, float):
                import math
                assert math.isfinite(x)

        finite(state)
        groups = state['optimizer']['param_groups']
        ids = [key for group in groups for key in group['params']]
        assert ids and len(ids) == len(set(ids))
        assert set(ids) == set(state['optimizer']['state'])
        adam_steps = [int(value['step']) for value in state['optimizer']['state'].values()]
        assert min(adam_steps) == max(adam_steps) == ck['actual_optimizer_updates']
        assert state['rng']['generators'].keys() == {'train', 'val'}
        assert all(k in state['rng'] for k in ['python', 'numpy', 'torch', 'cuda'])
        sys.path.insert(0, str(ROOT / 'upstream/TextOpRobotMDAR'))
        from robotmdar.model.mld_denoiser import DenoiserTransformer

        def new_model():
            return DenoiserTransformer(h_dim=512, ff_size=1024, num_layers=8, num_heads=4,
                                      clip_dim=512, history_shape=(2, 69), noise_shape=(1, 128),
                                      cond_mask_prob=.1).cpu().eval()

        model = new_model()
        model.load_state_dict(state['denoiser'], strict=True)
        assert equal(model.state_dict(), state['denoiser'])
        assert len(ids) == len([x for x in model.parameters() if x.requires_grad])
        parent = owned(s['export_dir'], pathlib.Path(s['export_allowed_root']))
        path = parent / 'ldm100000_inference.private.pth'
        partial = path.with_suffix('.partial')
        assert not path.exists() and not partial.exists()
        assert shutil.disk_usage(parent).free >= s['reserve_bytes'] + s['log_allowance_bytes'] + s['export_upper_bound_bytes']
        payload = dict(schema='textop.ldm100k_prefix_inference.v1', denoiser=model.state_dict(),
                       completed_primitive_attempts=100000, actual_optimizer_updates=ck['actual_optimizer_updates'],
                       source_full_state_sha256=ck['sha256'], control_receipt_sha256=v['control_receipt_sha256'],
                       source_numeric_binding_sha256=v['policy']['original_runtime_binding_sha256'],
                       trained_denoiser_sha256=module_sha(model),
                       native_schedule_primitive_attempts=300000, selected_user_endpoint=100000,
                       full_native_schedule_completed=False, semantic_quality_accepted=False,
                       actual_checkpoint_history_counters=audit['actual_checkpoint_history_counters'],
                       vae_inference_sha256=v['policy']['vae_inference_sha256'],
                       corpus_receipt_sha256=v['policy']['corpus_receipt_sha256'],
                       architecture=v['source_spec']['ldm'])
        with partial.open('xb') as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        assert partial.stat().st_size <= s['export_upper_bound_bytes']
        partial.replace(path)
        directory = os.open(str(parent), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        loaded = torch.load(path, map_location='cpu')
        assert equal(payload, loaded) and 'optimizer' not in loaded and 'rng' not in loaded
        fresh = new_model()
        fresh.load_state_dict(loaded['denoiser'], strict=True)
        assert module_sha(fresh) == module_sha(model)
        # A small deterministic CPU inference check uses synthetic inputs only.
        # It is not a heldout corpus, semantic metric or physical acceptance.
        probe = dict(history_motion_normalized=torch.zeros(2, 2, 69),
                     text_embedding=torch.zeros(2, 512))
        with torch.no_grad():
            for uncond in [False, True]:
                y = dict(probe, uncond=uncond)
                x = torch.zeros(2, 1, 128)
                t = torch.tensor([0, 4], dtype=torch.long)
                first = model(x, t, y=y)
                second = fresh(x, t, y=y)
                assert first.shape == (2, 1, 128) and bool(torch.isfinite(first).all())
                assert torch.equal(first, second)
        assert sha(ck['path']) == ck['sha256'] and not torch.cuda.is_initialized()
        assert verify_checkpoint(ck, v['policy']['checkpoint_allowed_root']) == fingerprint
        r.update(status='USER_LDM100000_PREFIX_CPU_EXPORT_PASSED', selected_checkpoint=ck,
                 metadata_audit=audit, finite_cpu_tensors=count[0], adam_step_min=min(adam_steps),
                 adam_step_max=max(adam_steps), exact_inference_reload=True,
                 trained_denoiser_sha256=module_sha(model), cpu_model_loads=2, cpu_full_state_loads=1,
                 synthetic_cpu_reload_probe=dict(batch=2, conditional_and_unconditional=True,
                                                forward_calls=4, exact=True, heldout_semantics=False),
                 cuda_rng_restored=False, inference=dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path)),
                 original_interrupted_exit_retained=True, full_native_schedule_completed=False,
                 output_writers_closed=True)
        save()
        print(json.dumps(r), flush=True)
    except BaseException as error:
        r.update(status='FAILED_LDM100K_PREFIX_CPU_EXPORT', error=repr(error), traceback=traceback.format_exc())
        save()
        raise


if __name__ == '__main__':
    main()

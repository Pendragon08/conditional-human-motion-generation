"""Independent read-only CPU fullstate/Adam/inference verification after export.

No optimizer step, original manager/Runtime, datasets, CUDA RNG restoration or
export overwrite. Synthetic forward equality is an integrity check only.
"""
import argparse
import importlib.metadata
import json
import os
import sys
import time
import traceback
from ldm100k_export_acceptance import ROOT, read, sha, accepted_export, verify_checkpoint
from ldm100k_prefix_completion import validate_state_metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    a = p.parse_args()
    assert a.run == 'user_ldm100k_prefix_document_v1' and os.getuid() == 1013
    assert set(os.sched_getaffinity(0)) == {0, 1, 2, 3}
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    sp = ROOT / 'configs/ldm100k-prefix-verify-spec.json'
    s = read(sp)
    assert s['schema'] == 'textop.ldm100k_prefix_cpu_verification.v1' and s['verify_authorized']
    assert sha(__file__) == s['executor_sha256']
    for n, h in s['bound_sources'].items():
        assert sha(ROOT / n) == h, n
    base = read(ROOT / 'configs/paper69-env-spec.json')
    assert sys.version.split()[0] == base['python_version']
    assert {k: importlib.metadata.version(k) for k in base['required_versions']} == base['required_versions']
    accepted = accepted_export(s['export_binding'])
    v = accepted['stopped']
    ck, artifact = v['checkpoint'], accepted['inference']
    out = ROOT / 'ldm100k_user_handoff_20261028' / a.run
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    started = time.monotonic()
    r = dict(status='STARTING_LDM100K_FRESH_CPU_VERIFICATION', run=a.run,
             source_sha256=sha(__file__), spec_sha256=sha(sp), output_writers_closed=False,
             export_receipt_sha256=accepted['receipt_sha256'], selected_checkpoint=ck, inference=artifact,
             new_gpu_contexts=0, new_sdk_tasks=0, training_optimizer_updates_performed=0,
             original300k_completion_claimed=False, semantic_quality_accepted=False,
             inference_or_fullstate_modified=False)

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
        before_ck = verify_checkpoint(ck, v['policy']['checkpoint_allowed_root'])
        before_inf = verify_checkpoint(artifact, accepted['export_spec']['export_allowed_root'])
        full = torch.load(ck['path'], map_location='cpu')
        payload = torch.load(artifact['path'], map_location='cpu')
        metadata = validate_state_metadata(full, ck, v['expected_state_binding'])
        assert payload['schema'] == 'textop.ldm100k_prefix_inference.v1'
        assert payload['completed_primitive_attempts'] == payload['selected_user_endpoint'] == 100000
        assert payload['native_schedule_primitive_attempts'] == 300000
        assert payload['full_native_schedule_completed'] is False and payload['semantic_quality_accepted'] is False
        assert 'optimizer' not in payload and 'rng' not in payload
        assert payload['source_full_state_sha256'] == ck['sha256']
        assert payload['actual_optimizer_updates'] == ck['actual_optimizer_updates']
        assert payload['control_receipt_sha256'] == v['control_receipt_sha256']
        assert payload['source_numeric_binding_sha256'] == v['policy']['original_runtime_binding_sha256']
        assert payload['corpus_receipt_sha256'] == v['policy']['corpus_receipt_sha256']
        assert payload['vae_inference_sha256'] == v['policy']['vae_inference_sha256']
        assert payload['architecture'] == v['source_spec']['ldm']
        assert payload['actual_checkpoint_history_counters'] == metadata['actual_checkpoint_history_counters']
        assert equal(full['denoiser'], payload['denoiser'])
        count = [0]

        def finite(value):
            if torch.is_tensor(value):
                assert value.device.type == 'cpu' and bool(torch.isfinite(value).all())
                count[0] += 1
            elif isinstance(value, dict):
                for x in value.values():
                    finite(x)
            elif isinstance(value, (tuple, list)):
                for x in value:
                    finite(x)
            elif isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
                assert np.isfinite(value).all()
            elif isinstance(value, float):
                import math
                assert math.isfinite(value)

        finite(full)
        finite(payload)
        sys.path.insert(0, str(ROOT / 'upstream/TextOpRobotMDAR'))
        from robotmdar.model.mld_denoiser import DenoiserTransformer

        def create_model():
            return DenoiserTransformer(h_dim=512, ff_size=1024, num_layers=8, num_heads=4,
                                      clip_dim=512, history_shape=(2, 69), noise_shape=(1, 128),
                                      cond_mask_prob=.1).cpu().eval()

        source_model, exported_model = create_model(), create_model()
        source_model.load_state_dict(full['denoiser'], strict=True)
        exported_model.load_state_dict(payload['denoiser'], strict=True)
        model_hash = module_sha(source_model)
        assert model_hash == module_sha(exported_model) == payload['trained_denoiser_sha256']
        assert model_hash == accepted['receipt']['trained_denoiser_sha256']
        adam = torch.optim.AdamW(source_model.parameters(), lr=1e-4)
        adam.load_state_dict(full['optimizer'])
        assert equal(adam.state_dict(), full['optimizer'])
        steps = [int(x['step']) for x in adam.state_dict()['state'].values()]
        assert steps and min(steps) == max(steps) == ck['actual_optimizer_updates']
        before_model = module_sha(source_model)
        # Fixed nonzero CPU synthetic conditioning exercises both masking
        # branches; it is not a text embedding from the heldout corpus.
        x = torch.linspace(-.5, .5, 256).reshape(2, 1, 128)
        history = torch.linspace(-.2, .2, 276).reshape(2, 2, 69)
        text = torch.linspace(-.1, .1, 1024).reshape(2, 512)
        t = torch.tensor([0, 4], dtype=torch.long)
        with torch.no_grad():
            for uncond in [False, True]:
                y = dict(history_motion_normalized=history, text_embedding=text, uncond=uncond)
                a1 = source_model(x, t, y=y)
                a2 = exported_model(x, t, y=y)
                assert a1.shape == (2, 1, 128) and bool(torch.isfinite(a1).all())
                assert torch.equal(a1, a2)
        assert module_sha(source_model) == before_model and equal(adam.state_dict(), full['optimizer'])
        assert verify_checkpoint(ck, v['policy']['checkpoint_allowed_root']) == before_ck
        assert verify_checkpoint(artifact, accepted['export_spec']['export_allowed_root']) == before_inf
        assert not torch.cuda.is_initialized()
        r.update(status='USER_LDM100K_PREFIX_FRESH_CPU_VERIFIED', output_writers_closed=True,
                 finite_cpu_tensors=count[0], fullstate_and_exported_denoiser_exact=True,
                 cpu_adam_restore_exact=True, adam_step_min=min(steps), adam_step_max=max(steps),
                 primitive_clock=metadata['primitive_clock'], metadata_audit=metadata,
                 trained_denoiser_sha256=model_hash, synthetic_cpu_forward_calls=4,
                 synthetic_cpu_forward_exact=True, no_heldout_semantics_claimed=True,
                 cuda_rng_restored=False, actual_full_state_loads=1, actual_inference_loads=1,
                 source_and_inference_fingerprints_unchanged=True)
        save()
        print(json.dumps(r), flush=True)
    except BaseException as error:
        r.update(status='FAILED_LDM100K_PREFIX_FRESH_CPU_VERIFICATION', error=repr(error), traceback=traceback.format_exc())
        save()
        raise


if __name__ == '__main__':
    main()

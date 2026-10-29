"""CPU fullstate audit and frozen inference export/independent process reload.

Never updates an optimizer or restores CUDA RNG. Synthetic probes establish
serialization equality, not heldout retrieval quality.
"""
import argparse
import copy
import json
import os
import pathlib
import random
import shutil
import sys
import time
import traceback
from tmr100_trained_acceptance import (ROOT, HOME, PARENT, owned, read, sha, fingerprint,
    accepted_training, accepted_frozen_export, validate_fullstate_metadata)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    args = parser.parse_args()
    sp = owned(args.request)
    spec = read(sp)
    assert os.getuid() == 1013 and sha(__file__) == spec['executor_sha256']
    assert spec['authorized'] is True and spec['role'] in ['export', 'verify']
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and set(os.sched_getaffinity(0)) == {0, 1, 2, 3}
    for name, value in spec['bound_sources'].items():
        assert sha(owned(ROOT/name)) == value
    train = accepted_training(spec['training_binding'])
    export = accepted_frozen_export(spec['export_binding']) if spec['role'] == 'verify' else None
    out = owned(PARENT) / spec['run']
    out.mkdir(mode=0o700, exist_ok=False)
    start = time.monotonic()
    rec = dict(run=spec['run'], role=spec['role'], status='AUDITING_TMR100_PREFIX_ON_CPU',
        source_sha256=sha(__file__), spec_sha256=sha(sp), output_writers_closed=False,
        training_binding=train['binding'], training_optimizer_updates_performed=0,
        new_cuda_contexts=0, cuda_rng_restored=False, semantic_metrics_accepted=False)

    def persist():
        rec['elapsed_seconds'] = time.monotonic() - start
        tmp = out/'receipt.partial'
        with tmp.open('w') as stream:
            json.dump(rec, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        tmp.replace(out/'receipt.json')

    persist()
    try:
        item = train['spec']['inputs']['text_receipt']
        text_receipt = owned(item['path'])
        assert sha(text_receipt) == item['sha256']
        sys.path.insert(0, str(owned(read(text_receipt)['runtime_dependency_dir'])))
        import numpy as np
        import torch
        from tmr69d_core import make_model, Sampler
        from generator_training_state import equal, module_sha
        torch.set_num_threads(4)
        assert not torch.cuda.is_initialized() and torch.cuda.device_count() == 0
        torch.use_deterministic_algorithms(True)
        selected = train['binding']['selected_checkpoint']
        before = fingerprint(selected['path'], home=True)
        assert before['sha256'] == selected['sha256'] and before['bytes'] == selected['bytes']
        state = torch.load(owned(selected['path'], HOME), map_location='cpu')
        metadata = validate_fullstate_metadata(state, train['spec'])
        finite_tensors = 0

        def inspect(value):
            nonlocal finite_tensors
            if torch.is_tensor(value):
                assert value.device.type == 'cpu'
                assert torch.isfinite(value).all()
                finite_tensors += 1
            elif isinstance(value, dict):
                for child in value.values():
                    inspect(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    inspect(child)
            elif isinstance(value, np.ndarray):
                assert np.isfinite(value).all()

        inspect(state)
        saved_rng = state['all_rng']
        random.Random().setstate(saved_rng['python'])
        np.random.RandomState().set_state(saved_rng['numpy'])
        for value in [saved_rng['torch']] + saved_rng['cuda']:
            assert value.dtype == torch.uint8 and value.ndim == 1 and value.numel() > 0
        torch.Generator(device='cpu').set_state(saved_rng['torch'])
        # Saved CUDA byte states are inspected without applying them on CPU.
        sampler = Sampler(32462, train['spec']['seed'] + 1)
        sampler.restore(state['sampler'])
        assert equal(sampler.state(), state['sampler'])
        next_ids = sampler.take(32)
        assert len(next_ids) == 32 and sampler.epoch == 100 and sampler.cursor == 32
        sampler.restore(state['sampler'])
        assert equal(sampler.state(), state['sampler'])
        random.seed(20261103)
        np.random.seed(20261103)
        torch.manual_seed(20261103)
        model, optimizer, ast_audit = make_model(owned(train['spec']['upstream_model_dir']),
            train['spec']['upstream_model_sha256'], 'cpu')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        assert equal(model.state_dict(), state['model']) and equal(optimizer.state_dict(), state['optimizer'])
        assert len(optimizer.state) == len(list(model.parameters()))
        for param, row in optimizer.state.items():
            assert set(row) == {'step', 'exp_avg', 'exp_avg_sq'} and int(row['step']) == 101500
            assert row['step'].numel() == 1
            assert row['exp_avg'].shape == row['exp_avg_sq'].shape == param.shape
        for group in optimizer.param_groups:
            assert group['lr'] == .0001 and group['weight_decay'] == .01
        selected_model_sha256 = module_sha(model)
        model.eval().requires_grad_(False)

        def probe(module):
            outputs = {}
            for modality, dim, lengths in [('motion', 69, [3, 9]), ('text', 768, [3, 7])]:
                length = max(lengths)
                x = torch.linspace(-.75, .8, 2*length*dim, dtype=torch.float32).reshape(2, length, dim)
                mask = torch.arange(length)[None, :] < torch.tensor(lengths)[:, None]
                x = x.masked_fill(~mask[..., None], 0)
                with torch.inference_mode():
                    z = module.encode(dict(x=x, mask=mask), modality=modality, sample_mean=True)
                assert z.shape == (2, 256) and torch.isfinite(z).all() and bool((z != 0).any())
                outputs[modality] = z.detach().clone()
            return outputs

        before_rng = torch.get_rng_state().clone()
        expected = probe(model)
        assert torch.equal(before_rng, torch.get_rng_state()), 'Mean encoding in eval must consume no RNG'
        inference_path = pathlib.Path(spec['inference_path'])
        if spec['role'] == 'export':
            folder = owned(inference_path.parent, HOME)
            assert not inference_path.exists() and not inference_path.with_suffix('.partial').exists()
            required = spec['reserve_bytes'] + spec['home_log_allowance_bytes'] + spec['inference_upper_bytes']
            assert shutil.disk_usage(folder).free >= required
            payload = dict(schema='textop.user_tmr100_inference.v1', model=copy.deepcopy(model.state_dict()),
                model_sha256=module_sha(model), training_binding=train['binding'],
                upstream_model_sha256=train['spec']['upstream_model_sha256'],
                encoding=dict(sample_mean=True, raw_latent_dimensions=256, feature_dimensions=69,
                    feature_fps=50, token_dimensions=768, per_sample_L2_normalization=False),
                provenance='independent_TRAIN_only_robot_TMR_not_author_released_weights',
                semantic_metrics_accepted=False)
            partial = inference_path.with_suffix('.partial')
            with partial.open('xb') as stream:
                os.chmod(partial, 0o600)
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            assert 0 < partial.stat().st_size <= spec['inference_upper_bytes']
            loaded = torch.load(partial, map_location='cpu')
            assert equal(loaded, payload)
            assert shutil.disk_usage(folder).free >= spec['reserve_bytes'] + spec['home_log_allowance_bytes']
            partial.rename(inference_path)
            descriptor = os.open(str(folder), os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        else:
            assert str(inference_path) == export['binding']['inference']['path']
            loaded = torch.load(owned(inference_path, HOME), map_location='cpu')
        assert loaded['schema'] == 'textop.user_tmr100_inference.v1'
        assert loaded['training_binding'] == train['binding']
        assert loaded['model_sha256'] == module_sha(model) and equal(loaded['model'], state['model'])
        assert loaded['upstream_model_sha256'] == train['spec']['upstream_model_sha256']
        assert loaded['encoding'] == dict(sample_mean=True, raw_latent_dimensions=256, feature_dimensions=69,
            feature_fps=50, token_dimensions=768, per_sample_L2_normalization=False)
        assert loaded['semantic_metrics_accepted'] is False
        independent = copy.deepcopy(model)
        independent.load_state_dict(loaded['model'], strict=True)
        assert module_sha(independent) == selected_model_sha256
        actual = probe(independent)
        assert equal(actual, expected) and torch.equal(before_rng, torch.get_rng_state())
        assert fingerprint(selected['path'], home=True) == before
        assert accepted_training(spec['training_binding'])['binding'] == train['binding']
        infer_fp = fingerprint(inference_path, home=True)
        item = {key: infer_fp[key] for key in ['path', 'bytes', 'sha256']}
        if export:
            assert item == export['binding']['inference']
            assert rec['training_binding'] == export['receipt']['training_binding']
        rec.update(status='TMR100_TRAINED_FROZEN_CPU_EXPORTED' if spec['role'] == 'export' else
            'TMR100_TRAINED_FROZEN_FRESH_CPU_VERIFIED', output_writers_closed=True, inference=item,
            full_model_and_cpu_adamw_restored_exact=True, independent_inference_reload_exact=True,
            synthetic_mean_forward_exact=True, original_fullstate_unchanged=True,
            selected_checkpoint_fingerprint_before_and_after=before, finite_cpu_tensors=finite_tensors,
            fullstate_metadata_audit=metadata, original_class_AST_audit=ast_audit,
            trained_model_sha256=module_sha(model), optimizer_step_min=101500, optimizer_step_max=101500,
            synthetic_probe=dict(motion_lengths=[3,9], text_lengths=[3,7], latent_dimensions=256,
                sample_mean=True, raw_features=True, exact=True, heldout_semantic_quality=False),
            export_binding=spec.get('export_binding'))
        assert not torch.cuda.is_initialized()
        persist()
    except BaseException as error:
        rec.update(status='FAILED_TMR69D_FROZEN_CPU_ALL_OUTPUTS_RETAINED', error=repr(error), traceback=traceback.format_exc())
        persist()
        raise


if __name__ == '__main__':
    main()

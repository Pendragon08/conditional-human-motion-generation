"""Bounded paired heldout inference entry; disabled until real input/witness acceptance."""
import argparse
import hashlib
import json
import math
import os
import pathlib
import shutil
import time
import traceback
from ldm100k_verified_input import ROOT, read, sha, owned
from ldm100k_scope_acceptance import accepted_scope
from ldm100k_generation_protocol import SEEDS, MODES, pair_output_upper_bytes


def atomic_json(path, value):
    raw = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    tmp = path.with_name(path.name + '.partial')
    with tmp.open('xb') as f:
        os.chmod(tmp, 0o600)
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--mode', choices=['witness', 'diagnostic'], required=True)
    a = p.parse_args()
    assert os.getuid() == 1013 and set(os.sched_getaffinity(0)) == {0, 1, 2, 3}
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '1'
    from ldm100k_generation_acceptance import RUNS
    assert a.run in {'witness': RUNS,
                     'diagnostic': ['user_ldm100k_diagnostic32_v1']}[a.mode]
    sp = owned(ROOT / 'configs/ldm100k-generation-spec.json')
    s = read(sp)
    assert s['schema'] == 'textop.ldm100k_heldout_generation.v1'
    assert s['witnesses_authorized'] if a.mode == 'witness' else s['generation_authorized']
    assert sha(__file__) == s['executor_sha256']
    for name, digest in s['bound_sources'].items():
        assert sha(ROOT / name) == digest, name
    actual = accepted_scope(s['scope_binding'])
    pairs = [(seg, seed) for seg in actual['scope']['segments'] for seed in SEEDS]
    if a.mode == 'witness':
        pairs = pairs[:1]
    else:
        # Diagnostic authorization must bind two closed real GPU inference
        # witnesses and the actual fresh document; no synthetic weights.
        from ldm100k_generation_acceptance import accepted_generation_witnesses
        accepted_generation_witnesses(s['witness_bindings'])
    budget = sum(pair_output_upper_bytes(seg['future_frames']) for seg, _ in pairs)
    assert budget == s['accepted_output_upper_bytes'][a.mode]
    assert shutil.disk_usage(ROOT).free >= s['reserve_bytes'] + s['log_allowance_bytes'] + budget
    out = ROOT / 'ldm100k_user_handoff_20261028' / a.run
    out.mkdir(mode=0o700, exist_ok=False)
    started = time.monotonic()
    receipt = dict(status='STARTING_LDM100K_HELDOUT_GENERATION', run=a.run, mode=a.mode,
        spec_sha256=sha(sp), source_sha256=sha(__file__), output_writers_closed=False,
        scope_receipt_sha256=actual['receipt_sha256'], selected_ldm_attempts=100000,
        planned_pairs=len(pairs), planned_trials=len(pairs) * 2, completed_pairs=0,
        completed_trials=0, failed_pairs=0, unattempted_pairs=len(pairs),
        declared_validation_families=32, full_validation_generation_claimed=False,
        training_optimizer_updates=0, future_posterior_encoded=False,
        semantic_metrics_computed=False, physical_quality_accepted=False,
        original300k_completion_claimed=False)

    def save():
        receipt['elapsed_seconds'] = time.monotonic() - started
        atomic_json(out / 'receipt.json', receipt)

    save()
    try:
        import importlib.metadata
        import sys
        base = read(ROOT / 'configs/paper69-env-spec.json')
        assert sys.version.split()[0] == base['python_version']
        assert {k: importlib.metadata.version(k) for k in base['required_versions']} == base['required_versions']
        from ldm100k_diagnostic_runtime import Runtime
        runtime = Runtime(actual['verified'], actual['source_index'])
        receipt.update(trained_denoiser_sha256=runtime.model_hash, frozen_vae_sha256=runtime.vae_hash,
                       actual_inference_model_loads=2, new_gpu_contexts=1)
        used, npz_used = 0, 0
        with (out / 'pairs.private.jsonl').open('x', encoding='utf-8') as journal:
            os.chmod(out / 'pairs.private.jsonl', 0o600)
            for position, (segment, seed) in enumerate(pairs):
                assert time.monotonic() - started < s['timeout_seconds'][a.mode]
                remaining = sum(pair_output_upper_bytes(x['future_frames']) for x, _ in pairs[position:])
                assert shutil.disk_usage(ROOT).free >= s['reserve_bytes'] + s['log_allowance_bytes'] + remaining
                pairdir = out / (segment['segment_id'] + '_' + str(seed))
                pairdir.mkdir(mode=0o700, exist_ok=False)
                record = dict(segment_id=segment['segment_id'], seed=seed, mode_artifacts={},
                              status='PAIR_STARTED', future_frames=segment['future_frames'])
                journal.write(json.dumps(record) + '\n')
                journal.flush()
                os.fsync(journal.fileno())
                journal_start = journal.tell()

                def mode_closed(mode, arrays, noise_trace):
                    nonlocal npz_used
                    assert mode in MODES and math.isfinite(arrays['normalized_huber'])
                    path = pairdir / (mode + '.private.npz')
                    partial = path.with_suffix('.partial')
                    with partial.open('xb') as f:
                        os.chmod(partial, 0o600)
                        runtime.np.savez(f, **{k: v for k, v in arrays.items() if k != 'normalized_huber'})
                        f.flush()
                        os.fsync(f.fileno())
                    partial.replace(path)
                    npz_used += path.stat().st_size
                    item = dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path),
                                normalized_huber=arrays['normalized_huber'],
                                actual_primitive_calls=len(noise_trace),
                                array_sha256={k: hashlib.sha256(v.tobytes()).hexdigest()
                                              for k, v in arrays.items() if k != 'normalized_huber'},
                                array_shapes={k: list(v.shape) for k, v in arrays.items()
                                              if k != 'normalized_huber'})
                    record['mode_artifacts'][mode] = item
                    atomic_json(pairdir / (mode + '_receipt.private.json'), item)
                    receipt['completed_trials'] += 1

                try:
                    _, probe = runtime.paired(segment, seed, on_mode=mode_closed)
                    assert set(record['mode_artifacts']) == set(MODES)
                    record.update(status='PAIR_FINISHED_SAME_NOISE_VERIFIED', numeric_probe=probe)
                    receipt['completed_pairs'] += 1
                    if a.mode == 'witness':
                        receipt['witness_pair_arrays'] = {mode: {k: item[k] for k in ['array_sha256', 'array_shapes']}
                                                         for mode, item in record['mode_artifacts'].items()}
                        receipt['witness_pair_probe_sha256'] = hashlib.sha256(json.dumps(
                            probe, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                except (AssertionError, RuntimeError, ValueError) as error:
                    # A failure is retained independently of all successful
                    # seeds. Never select a replacement or relabel it physics.
                    record.update(status='PAIR_FAILED_OUTPUTS_RETAINED', error=repr(error),
                                  traceback=traceback.format_exc(), physical_failure_claimed=False)
                    receipt['failed_pairs'] += 1
                receipt['unattempted_pairs'] -= 1
                atomic_json(pairdir / 'pair_receipt.private.json', record)
                journal.write(json.dumps(record) + '\n')
                journal.flush()
                os.fsync(journal.fileno())
                used += sum(path.stat().st_size for path in pairdir.iterdir() if path.is_file())
                used += journal.tell() - journal_start + 1024  # includes the initial intent line
                assert used <= budget
                save()
        runtime.verify_unchanged()
        receipt.update(status='USER_LDM100K_GENERATION_WITNESS_FINISHED' if a.mode == 'witness'
                       else 'USER_LDM100K_DIAGNOSTIC32_GENERATION_FINISHED',
                       private_journal_sha256=sha(out / 'pairs.private.jsonl'),
                       private_output_npz_bytes=npz_used, conservative_accounted_output_bytes=used,
                       model_and_source_bytes_unchanged=True,
                       verified_source_count=len(runtime.verified_source_hashes), output_writers_closed=True)
        save()
        print(json.dumps(receipt), flush=True)
        if a.mode == 'witness':
            assert receipt['failed_pairs'] == 0 and receipt['completed_pairs'] == 1
    except BaseException as error:
        receipt.update(status='FAILED_LDM100K_HELDOUT_GENERATION', error=repr(error),
                       traceback=traceback.format_exc())
        save()
        raise


if __name__ == '__main__':
    main()

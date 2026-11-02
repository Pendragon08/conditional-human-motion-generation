"""Independent CPU audit and paired descriptive statistics of all declared outputs.

No Torch, neural model, simulator or dynamics. Private rows keep provenance for
the next actual FK/physics job; normalized Huber is not a semantic metric.
"""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import time
import zipfile
import numpy as np
from ldm100k_verified_input import ROOT, owned, read, sha, closed_job
from ldm100k_generation_protocol import SEEDS, MODES, validate_diagnostic_scope


def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        os.chmod(path, 0o600)
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def huber(x, y):
    d = np.abs(x.astype(np.float64) - y.astype(np.float64))
    return float(np.where(d <= 1, .5 * d * d, d - .5).mean())


def aggregate(rows):
    assert rows
    families = sorted({x['family_sha256'] for x in rows})
    by_mode = {}
    for m in MODES:
        v = [x[m] for x in rows]
        by_mode[m] = dict(trial_mean=float(np.mean(v)),
            frame_weighted_mean=float(np.average(v, weights=[x['future_frames'] for x in rows])),
            family_equal_mean=float(np.mean([np.mean([x[m] for x in rows if x['family_sha256'] == f]) for f in families])))
    delta = [x[MODES[0]] - x[MODES[1]] for x in rows]
    return dict(pairs=len(rows), families=len(families), normalized_huber=by_mode,
                conditional_minus_unconditional_pair_mean=float(np.mean(delta)),
                conditional_lower_count=sum(x < 0 for x in delta),
                conditional_higher_count=sum(x > 0 for x in delta), exact_tie_count=sum(x == 0 for x in delta))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--request', required=True)
    a = p.parse_args()
    assert os.getuid() == 1013 and set(os.sched_getaffinity(0)) == {0, 1, 2, 3}
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    request_path = owned(a.request)
    s = read(request_path)
    assert sha(__file__) == s['executor_sha256']
    started = time.monotonic()
    sp = owned(ROOT / 'configs/ldm100k-generation-spec.json')
    assert sha(sp) == s['generation_spec_sha256']
    spec = read(sp)
    for name, digest in spec['bound_sources'].items():
        assert sha(owned(ROOT / name)) == digest, name
    origin = owned(ROOT / 'ldm100k_user_handoff_20261028/user_ldm100k_diagnostic32_v1')
    rp = owned(origin / 'receipt.json')
    assert sha(rp) == s['generation_receipt_sha256']
    rec = read(rp)
    assert rec['status'] == 'USER_LDM100K_DIAGNOSTIC32_GENERATION_FINISHED' and rec['output_writers_closed']
    closure = closed_job('ldm100k_generation_' + rec['run'], rp, sha(rp), spec['supervisor_sha256'])
    assert rec['failed_pairs'] == rec['unattempted_pairs'] == 0
    srp = owned(ROOT / 'ldm100k_user_handoff_20261028' / spec['scope_binding']['run'] / 'receipt.json')
    assert sha(srp) == spec['scope_binding']['receipt_sha256']
    scope_receipt = read(srp)
    item = scope_receipt['artifacts']['diagnostic_scope.private.json']
    scope_path = owned(item['path'])
    assert sha(scope_path) == item['sha256'] and scope_path.stat().st_size == item['bytes']
    scope = read(scope_path)
    validate_diagnostic_scope(scope)
    segments = {x['segment_id']: x for x in scope['segments']}
    expected = {(segment, seed) for segment in segments for seed in SEEDS}
    parent = owned(s['output_root'], pathlib.Path(s['allowed_root']))
    out = parent / s['run']
    out.mkdir(mode=0o700, exist_ok=False)
    assert shutil.disk_usage(out).free >= 2 * 1024**3 + s['metadata_upper_bytes'] + 128 * 1024**2
    # Recover the real common qpos joint order from bound prepared metadata,
    # not an assumed Isaac ordering. No original motion arrays are loaded.
    csp = owned(ROOT / 'configs/filtered15000-corpus-spec.json')
    assert sha(csp) == s['corpus_spec_sha256']
    cs = read(csp)
    selected = {x['canonical_source'] for x in segments.values()}
    names, prepared_hashes = {}, {}
    for manifest_name in cs['prepared_manifests']:
        mp = pathlib.Path(manifest_name)
        allowed = [root for root in [ROOT, pathlib.Path('/dev/shm/ouyanghao_textop_20261025')] if root in mp.parents]
        assert len(allowed) == 1
        mp = owned(mp, allowed[0])
        assert sha(mp) == cs['prepared_manifest_sha256'][manifest_name]
        prepared_hashes[manifest_name] = sha(mp)
        with mp.open(encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                if row['canonical_source'] in selected:
                    names[row['canonical_source']] = row['joint_names']
    assert set(names) == selected
    joint_names = next(iter(names.values()))
    assert len(joint_names) == len(set(joint_names)) == 29 and all(x == joint_names for x in names.values())
    jp = owned(origin / 'pairs.private.jsonl')
    assert sha(jp) == rec['private_journal_sha256']
    journal_before = jp.stat()
    seen, starts, rows, physical = set(), set(), [], []
    checked_arrays = total_bytes = 0
    contact_outside = {m: 0 for m in MODES}
    contact_count = {m: 0 for m in MODES}
    contact_range = {m: [float('inf'), float('-inf')] for m in MODES}
    max_quat_error = 0.0
    with jp.open(encoding='utf-8') as f:
        for line in f:
            pair = json.loads(line)
            key = pair['segment_id'], pair['seed']
            assert key in expected
            if pair['status'] == 'PAIR_STARTED':
                assert key not in starts and key not in seen
                starts.add(key)
                continue
            assert key in starts and key not in seen
            assert pair['status'] == 'PAIR_FINISHED_SAME_NOISE_VERIFIED'
            seen.add(key)
            seg = segments[key[0]]
            n = seg['future_frames']
            assert n == pair['future_frames'] and 0 < n <= 200
            full = ((n + 7) // 8) * 8
            probe = pair['numeric_probe']
            assert probe['same_noise_trace_exact'] and not probe['future_posterior_encoded']
            assert probe['feature_history_after_initial_generated_only'] and not probe['contact_or_dof_clipping_applied']
            assert probe['diffusion_steps'] == probe['cfg'] == 5
            assert set(pair['mode_artifacts']) == set(MODES)
            row = dict(segment_id=key[0], seed=key[1], future_frames=n,
                       family_sha256=hashlib.sha256(seg['canonical_source'].encode()).hexdigest())
            prior = None
            for mode in MODES:
                item = pair['mode_artifacts'][mode]
                path = owned(item['path'])
                before = path.stat()
                assert path.parent.parent == origin and path.name == mode + '.private.npz'
                assert before.st_size == item['bytes'] and sha(path) == item['sha256']
                shapes = dict(generated_normalized_future=[full, 69], reference_normalized_future=[n, 69],
                    generated_qpos=[n, 36], generated_contact=[n, 2], initial_reference_history=[2, 69])
                assert item['array_shapes'] == shapes and set(item['array_sha256']) == set(shapes)
                with zipfile.ZipFile(path) as z:
                    assert z.testzip() is None and len(z.infolist()) == 5
                    assert set(z.namelist()) == {k + '.npy' for k in shapes}
                    assert all(x.compress_type == zipfile.ZIP_STORED and x.file_size < 1024**2 for x in z.infolist())
                with np.load(path, allow_pickle=False) as data:
                    arrays = {k: data[k] for k in shapes}
                for k, array in arrays.items():
                    assert list(array.shape) == shapes[k] and array.dtype == np.dtype('<f4')
                    assert array.flags.c_contiguous and np.isfinite(array).all()
                    assert hashlib.sha256(array.tobytes()).hexdigest() == item['array_sha256'][k]
                    checked_arrays += 1
                value = huber(arrays['generated_normalized_future'][:n], arrays['reference_normalized_future'])
                assert np.isclose(value, item['normalized_huber'], rtol=1e-5, atol=1e-6)
                row[mode] = value
                q = arrays['generated_qpos']
                error = float(np.max(np.abs(np.sum(q[:, 3:7].astype(np.float64)**2, axis=1) - 1)))
                assert error < 2e-3
                max_quat_error = max(max_quat_error, error)
                contact = arrays['generated_contact']
                contact_outside[mode] += int(((contact < 0) | (contact > 1)).sum())
                contact_count[mode] += int(contact.size)
                contact_range[mode][0] = min(contact_range[mode][0], float(contact.min()))
                contact_range[mode][1] = max(contact_range[mode][1], float(contact.max()))
                current = tuple(item['array_sha256'][k] for k in ['reference_normalized_future', 'initial_reference_history'])
                assert prior is None or current == prior
                prior = current
                after = path.stat()
                assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
                total_bytes += item['bytes']
                physical.append(dict(segment_id=key[0], seed=key[1], mode=mode, frames=n, fps=50,
                    generated_npz=str(path), generated_npz_sha256=item['sha256'], generated_npz_bytes=item['bytes'],
                    qpos_key='generated_qpos', qpos_sha256=item['array_sha256']['generated_qpos'],
                    qpos_shape=shapes['generated_qpos'], joint_names=joint_names,
                    source_kind='REAL_LDM100000_GENERATED_SCORED_FUTURE', future_tail_excluded=True,
                    physical_filter_passed=False, transition_applicable=n >= 2))
            rows.append(row)
    assert seen == starts == expected and len(rows) == rec['completed_pairs'] == 940
    assert len(physical) == rec['completed_trials'] == 1880 and total_bytes == rec['private_output_npz_bytes']
    assert sha(jp) == rec['private_journal_sha256'] and jp.stat().st_mtime_ns == journal_before.st_mtime_ns
    assert sha(rp) == s['generation_receipt_sha256'] and sha(sp) == s['generation_spec_sha256']
    for name, digest in prepared_hashes.items():
        assert sha(name) == digest
    for name, content in [('paired_results.private.json', rows), ('physical_inputs.private.json', physical)]:
        write(out / name, content)
    result = dict(status='USER_LDM100K_DIAGNOSTIC32_CPU_SUMMARY_FINISHED', run=s['run'],
        source_sha256=sha(__file__), request_sha256=sha(request_path), generation_receipt_sha256=sha(rp),
        generation_closure=closure, generation_spec_sha256=sha(sp), private_journal_sha256=sha(jp),
        actual_pairs=940, actual_trials=1880, actual_segments=len(segments), actual_families=len(selected),
        actual_array_payloads_checked=checked_arrays, actual_npz_bytes_checked=total_bytes,
        crc_float32_finite_shape_data_sha_checks_passed=True, paired_reference_and_noise_verified=True,
        statistics=aggregate(rows), per_seed={str(seed): aggregate([x for x in rows if x['seed'] == seed]) for seed in SEEDS},
        raw_unclamped_contact=dict(outside_unit_interval_values=contact_outside, values=contact_count, ranges=contact_range),
        maximum_quaternion_squared_norm_error=max_quat_error,
        physical_transition_applicable_trials=sum(x['transition_applicable'] for x in physical),
        no_transition_trials=sum(not x['transition_applicable'] for x in physical),
        private_artifacts={name: dict(path=str(out / name), bytes=(out / name).stat().st_size, sha256=sha(out / name))
                           for name in ['paired_results.private.json', 'physical_inputs.private.json']},
        semantic_metrics_computed=False, physical_quality_accepted=False, full_validation_generation_claimed=False,
        neural_model_loads=0, new_gpu_contexts=0, training_optimizer_updates=0, output_writers_closed=True,
        elapsed_seconds=time.monotonic() - started,
        interpretation='Huber is normalized feature reconstruction; raw contact is descriptive, neither establishes semantic or physical quality.')
    write(out / 'receipt.json', result)
    fd = os.open(str(out), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()

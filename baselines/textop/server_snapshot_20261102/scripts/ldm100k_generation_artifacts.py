"""Independent stdlib audit of the tiny real witness NPZ outputs; no Torch/model."""
import ast
import hashlib
import io
import math
import struct
import zipfile
from ldm100k_verified_input import read, sha, owned

KEYS = {'generated_normalized_future', 'reference_normalized_future', 'generated_qpos',
        'generated_contact', 'initial_reference_history'}


def check_npy(raw, expected_shape, expected_sha):
    assert raw[:6] == b'\x93NUMPY' and raw[6] in [1, 2, 3] and raw[7] == 0
    offset = 10 if raw[6] == 1 else 12
    size = int.from_bytes(raw[8:offset], 'little')
    assert 0 < size <= 4096 and offset + size <= len(raw)
    header = ast.literal_eval(raw[offset:offset + size].decode('utf-8').strip())
    assert set(header) == {'descr', 'fortran_order', 'shape'}
    assert header['descr'] == '<f4' and header['fortran_order'] is False
    assert list(header['shape']) == expected_shape
    assert len(expected_shape) == 2 and all(type(x) is int and x > 0 for x in expected_shape)
    data = raw[offset + size:]
    assert len(data) == expected_shape[0] * expected_shape[1] * 4
    assert hashlib.sha256(data).hexdigest() == expected_sha
    values = [v[0] for v in struct.iter_unpack('<f', data)]
    assert all(math.isfinite(x) for x in values)
    return values


def audit_witness_artifacts(out, receipt):
    out = owned(out)
    directories = [p for p in out.iterdir() if p.is_dir()]
    assert len(directories) == 1
    pair = owned(directories[0])
    pr = read(owned(pair / 'pair_receipt.private.json'))
    assert pr['status'] == 'PAIR_FINISHED_SAME_NOISE_VERIFIED'
    probe = pr['numeric_probe']
    assert probe['same_noise_trace_exact'] and probe['future_posterior_encoded'] is False
    assert probe['feature_history_after_initial_generated_only']
    assert probe['semantic_metrics_computed'] is probe['physical_quality_accepted'] is False
    assert probe['diffusion_steps'] == probe['cfg'] == 5
    canonical = hashlib.sha256(__import__('json').dumps(probe, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert canonical == receipt['witness_pair_probe_sha256']
    assert set(pr['mode_artifacts']) == {'conditional_cfg5', 'unconditional'}
    checked = 0
    for mode, item in pr['mode_artifacts'].items():
        p = owned(item['path'])
        assert p.parent == pair and p.name == mode + '.private.npz'
        assert p.stat().st_size == item['bytes'] <= 1024**2 and sha(p) == item['sha256']
        assert set(item['array_sha256']) == set(item['array_shapes']) == KEYS
        assert {k: item[k] for k in ['array_sha256', 'array_shapes']} == receipt['witness_pair_arrays'][mode]
        data = {}
        with zipfile.ZipFile(p) as archive:
            assert len(archive.infolist()) == len(KEYS)
            assert set(archive.namelist()) == {k + '.npy' for k in KEYS}
            assert archive.testzip() is None
            for key in KEYS:
                info = archive.getinfo(key + '.npy')
                assert info.file_size <= 1024**2 and info.compress_type == zipfile.ZIP_STORED
                data[key] = check_npy(archive.read(info), item['array_shapes'][key], item['array_sha256'][key])
                checked += 1
        future = pr['future_frames']
        assert 0 < future <= 200
        full = ((future + 7) // 8) * 8
        assert item['array_shapes'] == dict(generated_normalized_future=[full, 69],
            reference_normalized_future=[future, 69], generated_qpos=[future, 36],
            generated_contact=[future, 2], initial_reference_history=[2, 69])
        for frame in range(future):
            q = data['generated_qpos'][frame * 36 + 3:frame * 36 + 7]
            assert abs(sum(x * x for x in q) - 1) < 2e-3
        differences = [abs(x - y) for x, y in zip(data['generated_normalized_future'][:future * 69],
                                                 data['reference_normalized_future'])]
        huber = sum(.5 * x * x if x <= 1 else x - .5 for x in differences) / len(differences)
        assert math.isclose(huber, item['normalized_huber'], rel_tol=1e-5, abs_tol=1e-6)
    return dict(real_numpy_array_payloads_checked=checked, complete_npz_sha_verified=True,
                numpy_headers_shapes_float32_finite_verified=True, generated_quaternion_norm_verified=True,
                normalized_huber_recomputed=True, same_noise_probe_sha_verified=True,
                model_loads=0, gpu_contexts=0, semantic_or_physical_quality_accepted=False)


def audit_diagnostic_journal(out, receipt, spec):
    from ldm100k_verified_input import ROOT
    from ldm100k_generation_protocol import SEEDS, MODES
    out = owned(out)
    rp = owned(ROOT / 'ldm100k_user_handoff_20261028' / spec['scope_binding']['run'] / 'receipt.json')
    assert sha(rp) == spec['scope_binding']['receipt_sha256']
    scope_receipt = read(rp)
    item = scope_receipt['artifacts']['diagnostic_scope.private.json']
    path = owned(item['path'])
    assert path.stat().st_size == item['bytes'] <= 128 * 1024**2 and sha(path) == item['sha256']
    plan = read(path)
    expected = {(x['segment_id'], seed) for x in plan['segments'] for seed in SEEDS}
    assert receipt['planned_pairs'] == len(expected) and receipt['planned_trials'] == len(expected) * 2
    journal = owned(out / 'pairs.private.jsonl')
    assert sha(journal) == receipt['private_journal_sha256']
    starts, ends, counts = set(), set(), dict(pairs=0, failed=0, trials=0, npz_bytes=0)
    with journal.open(encoding='utf-8') as stream:
        for line in stream:
            r = __import__('json').loads(line)
            key = (r['segment_id'], r['seed'])
            assert key in expected
            if r['status'] == 'PAIR_STARTED':
                assert key not in starts and key not in ends
                starts.add(key)
                continue
            assert key in starts and key not in ends
            ends.add(key)
            assert r['status'] in ['PAIR_FINISHED_SAME_NOISE_VERIFIED', 'PAIR_FAILED_OUTPUTS_RETAINED']
            counts['pairs' if r['status'] == 'PAIR_FINISHED_SAME_NOISE_VERIFIED' else 'failed'] += 1
            if r['status'] == 'PAIR_FINISHED_SAME_NOISE_VERIFIED':
                assert set(r['mode_artifacts']) == set(MODES) and r['numeric_probe']['same_noise_trace_exact']
            for mode, artifact in r['mode_artifacts'].items():
                assert mode in MODES
                q = owned(artifact['path'])
                assert q.parent.parent == out and q.name == mode + '.private.npz'
                assert q.stat().st_size == artifact['bytes'] and sha(q) == artifact['sha256']
                counts['trials'] += 1
                counts['npz_bytes'] += artifact['bytes']
    assert starts == ends == expected
    assert counts['pairs'] == receipt['completed_pairs'] and counts['failed'] == receipt['failed_pairs']
    assert counts['trials'] == receipt['completed_trials'] and receipt['unattempted_pairs'] == 0
    assert counts['npz_bytes'] == receipt['private_output_npz_bytes']
    return dict(complete_declared_pair_coverage_verified=True, journal_and_npz_bytes_sha_verified=True,
                completed_pairs=counts['pairs'], failed_pairs=counts['failed'], completed_trials=counts['trials'],
                decoded_numpy_array_payloads=0, new_model_loads=0, new_gpu_contexts=0,
                semantic_or_physical_quality_accepted=False)

"""Read-only fail-closed certification of the already completed V33 cache.

Anchors are from committed independent witness and terminal summary. This is
not an efficacy run, threshold revision, new simulator invocation or V33 rerun.
No inherited launcher is executed. Licensed arrays remain on the server.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

ANCHORS = {
    'DOCUMENT_WITNESS_AUDIT.json': '01d1a30494ef80325d28055b4c5911786072d30a5abb25bbe172a8c170b19cab',
    'runtime_witness.json': '7b66498cc709ccf349a730e3e2a3e1b7a301b9fdab3edba1ca93bb6919dcba2c',
    'diagnostic_summary.json': 'f9c72f1225261a06602110edd4d4852fa365eb94cc4f5728722e15d027e04f98',
}


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def bound(path, expected, bindings):
    actual = sha(path)
    if actual != expected:
        raise ValueError('Checksum mismatch: ' + str(path))
    bindings[str(path)] = actual


def self_check():
    # Isolated fixtures only: never alter a real cache to test its protection.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'fixture'
        path.write_bytes(b'frozen')
        expected = sha(path)
        bound(path, expected, {})
        for altered in (b'changed threshold', b'changed witness', b'changed output'):
            path.write_bytes(altered)
            try:
                bound(path, expected, {})
            except ValueError:
                pass
            else:
                raise AssertionError('Tampered fixture was accepted')
        try:
            bound(Path(directory) / 'missing', expected, {})
        except FileNotFoundError:
            pass
        else:
            raise AssertionError('Missing fixture was accepted')
    print('MOCAP_CACHE_BINDING_NUMERIC_PASS')


def verify(root):
    import numpy as np
    from physics_guided_metric_v23_20261012 import metrics
    assert np.__version__ == '1.23.5', 'Use the recording numerical runtime'
    base = root / 'results/mocap_reference_control_v33_20261012'
    c1 = root / 'results/condition_feedback_formal_v28_20261012'
    bindings = {}
    for name, expected in ANCHORS.items():
        bound(base / name, expected, bindings)
    witness = read(base / 'DOCUMENT_WITNESS_AUDIT.json')
    summary = read(base / 'diagnostic_summary.json')
    assert witness['verdict'] == 'PASS' and witness['check_invocations'] == 1
    assert read(base / 'queue_state.json')['status'] == 'COMPLETE_ORIGINAL_REFERENCE_DIAGNOSTIC'
    for name, expected in witness['input_hashes'].items():
        bound(base / name, expected, bindings)
    for name in ('source_manifest.json', 'frozen_thresholds.json'):
        bound(c1 / name, witness['input_hashes'][name], bindings)
    version = read(base / 'versions.json')
    for name, expected in witness['source_hashes'].items():
        assert version['files'][name] == expected
        bound(root / name, expected, bindings)
    for name, expected in witness['external_input_hashes'].items():
        bound(Path(name), expected, bindings)
    threshold = read(base / 'frozen_thresholds.json')['thresholds']
    batches = {}
    for split, label, count in (('validation', 'val_mocap_replay_s11', 72),
                                ('test', 'test_mocap_replay_s11', 144)):
        path = base / 'raw' / label
        for name, expected in summary[split]['output_hashes'].items():
            bound(path / name, expected, bindings)
        if split == 'validation':
            for name, expected in witness['output_hashes'].items():
                bound(path / name, expected, bindings)
        lineage = read(path / 'lineage.json')
        for name, expected in lineage['outputs'].items():
            bound(path / name, expected, bindings)
        request = lineage['request']
        for key, name in (('manifest', 'source_manifest.json'),
                          ('thresholds', 'frozen_thresholds.json'), ('versions', 'versions.json')):
            assert request[key] == witness['input_hashes'][name]
        bound(root / 'configs/physics_guided_20261011.json', request['config'], bindings)
        metadata = read(path / 'metadata.json')
        receipt = read(path / 'receipt.json')
        bound(path / 'trajectories.npz', receipt['trajectories_sha256'], bindings)
        bound(path / 'dataset_reference.npz', metadata['dataset_reference_sha256'], bindings)
        bound(path / 'sensor_diagnostic.npz', metadata['sensor_diagnostic_sha256'], bindings)
        assert receipt['status'] == 'complete' and metadata['generation_batch_calls'] == 0
        assert metadata['initial_reset_gate_pass']
        reset = metadata['manual_reset_vs_first_action_state_max_difference']
        assert len(reset) == 7 and all(value == 0 for value in reset.values())
        measured = read(path / 'metrics.json')
        assert len(measured) == count
        with np.load(path / 'trajectories.npz', allow_pickle=False) as replay, \
                np.load(path / 'dataset_reference.npz', allow_pickle=False) as reference:
            assert np.array_equal(replay['reference_frame'], np.arange(120))
            assert np.array_equal(reference['original_goal'], np.asarray(metadata['original_goal'], np.float32))
            canonical = reference['reference'] - reference['reference'][:, :, 0:1] + reference['reference'][:, :, 0:1]
            assert np.allclose(replay['reference'][replay['valid']], canonical[replay['valid']], rtol=0, atol=1e-12)
            for index, stored in enumerate(measured):
                recomputed = metrics(replay, index, metadata, threshold)
                assert recomputed == stored, 'Frozen metrics differ: ' + stored['case_id']
        batches[label] = dict(attempts=count, all_metric_rows_exactly_recomputed=True,
            joint_pass=sum(row['joint_pass'] for row in measured), reset_consistent=True,
            frozen_reference_clock=True, generation_calls=0)
    result = dict(status='PASS', time=time.time(), source_sha256=sha(Path(__file__)),
        input_bindings=bindings, anchors=ANCHORS, batches=batches,
        simulation_calls=0, generation_calls=0, thresholds_changed=False,
        old_scores_changed=False, innovation_confirmed=False, overleaf_results_sync=False,
        scope='Read-only strict V33 reuse certificate; no new efficacy or contact truth')
    out = root / 'results/mocap_cache_binding_v36_20261012'
    out.mkdir(exist_ok=True)
    target = out / ('certificate_%d.json' % time.time_ns())
    with target.open('x', encoding='utf8') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps({'status': result['status'], 'batches': batches, 'path': str(target)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        verify(args.root.resolve())

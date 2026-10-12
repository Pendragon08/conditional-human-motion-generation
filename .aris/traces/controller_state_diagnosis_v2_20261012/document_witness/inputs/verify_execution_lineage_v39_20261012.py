"""Fail-closed, read-only execution-time lineage guard for completed caches.

No retrospective certification, repair, legacy certifier call or simulator run.
Terminal V28 source/results stay unchanged; future queues must invoke this
guard before accepting a receipt. A receipt alone is insufficient evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
ANCHORS = {
    'versions.json': 'afb2193a00d15a4c9e2fe8addbab5a1968ac122f59d80278670b7f9aa3d9421a',
    'runtime_witness.json': 'ce07574de3dde3d800b4b6baa567f60b80131f6290fcc45558b6f1c74a9a6468',
    'DOCUMENT_WITNESS_AUDIT.json': 'ede8df766bf96092ed6ef7a62159c07024a0e90f857c5ad38414cb25440099a3',
    'summary.json': 'b1628ec89e0de61cb07952e8cbb62354ba31fa657c664434a5de2d46b2068509',
}


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def guard_batch(path, expected):
    # These files must preexist before any cache verifier may mutate a sidecar.
    lineage = read(path / 'lineage.json')
    if lineage.get('certification') != 'bound_request_at_execution':
        raise ValueError('No execution-time binding; preserve incomplete/retrospective attempt')
    if lineage['request'] != expected:
        raise ValueError('Cached request differs from the execution request')
    required = {'metadata.json', 'metrics.json', 'trajectories.npz'}
    if set(lineage['outputs']) != required:
        raise ValueError('Incomplete execution-time output binding')
    for name, digest in lineage['outputs'].items():
        if sha(path / name) != digest:
            raise ValueError('Bound output changed: ' + name)
    receipt = read(path / 'receipt.json')
    if receipt['status'] != 'complete' or receipt['trajectories_sha256'] != lineage['outputs']['trajectories.npz']:
        raise ValueError('Receipt is incomplete or disagrees with the bound trajectory')
    return dict(lineage_sha256=sha(path / 'lineage.json'), receipt_sha256=sha(path / 'receipt.json'),
                request=expected, output_hashes=lineage['outputs'], certification=lineage['certification'])


def self_check():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        expected = dict(mode='fixture', seed=11)
        for name, data in (('metadata.json', b'{}'), ('metrics.json', b'[]'), ('trajectories.npz', b'explicit synthetic bytes')):
            (path / name).write_bytes(data)
        outputs = {name: sha(path / name) for name in ('metadata.json', 'metrics.json', 'trajectories.npz')}
        receipt = dict(status='complete', trajectories_sha256=outputs['trajectories.npz'])
        (path / 'receipt.json').write_text(json.dumps(receipt), encoding='utf8')
        def store(lineage):
            (path / 'lineage.json').write_text(json.dumps(lineage), encoding='utf8')
        def rejected(call):
            try:
                call()
            except (ValueError, FileNotFoundError):
                return
            raise AssertionError('Invalid fixture cache accepted')
        rejected(lambda: guard_batch(path, expected))
        good = dict(certification='bound_request_at_execution', request=expected, outputs=outputs)
        store(dict(good, certification='retrospective_exact_array_and_metric_verification'))
        rejected(lambda: guard_batch(path, expected))
        store(good)
        guard_batch(path, expected)
        rejected(lambda: guard_batch(path, dict(expected, seed=12)))
        store(dict(good, outputs={'metadata.json': outputs['metadata.json']}))
        rejected(lambda: guard_batch(path, expected))
        store(good)
        (path / 'metrics.json').write_bytes(b'["tampered"]')
        rejected(lambda: guard_batch(path, expected))
    print('EXECUTION_LINEAGE_STRICT_FIXTURES_PASS')


def main(root):
    base = root / 'results/condition_feedback_formal_v28_20261012'
    for name, digest in ANCHORS.items():
        if sha(base / name) != digest:
            raise ValueError('Frozen witness/version/summary changed: ' + name)
    version = read(base / 'versions.json')
    for name, digest in version['files'].items():
        if sha(root / name) != digest:
            raise ValueError('Frozen source changed: ' + name)
    # Verify external assets with the already completed strict V36 bindings.
    certificates = sorted((root / 'results/mocap_cache_binding_v36_20261012').glob('certificate_*.json'))
    certificate = read(certificates[-1])
    if certificate['status'] != 'PASS':
        raise ValueError('Missing current strict input certificate')
    for name, digest in certificate['input_bindings'].items():
        if sha(Path(name)) != digest:
            raise ValueError('V36 input binding changed: ' + name)
    ledger_labels = {row['label'] for row in read(base / 'budget_ledger.json') if 'label' in row}
    entries = []
    batches = sorted((base / 'raw').iterdir())
    assert len(batches) == 74
    for path in batches:
        expected = read(base / 'requests' / (path.name + '.json'))
        assert expected['versions'] == ANCHORS['versions.json']
        for key, name in (('manifest', 'source_manifest.json'), ('thresholds', 'frozen_thresholds.json')):
            assert expected[key] == sha(base / name)
        assert path.name in ledger_labels, 'Missing charged attempt'
        entries.append(dict(label=path.name, **guard_batch(path, expected)))
    result = dict(status='PASS', time=time.time(), source_sha256=sha(Path(__file__)),
        anchors=ANCHORS, external_input_certificate_sha256=sha(certificates[-1]), entries=entries,
        checked_batches=len(entries), retroactive_lineages_accepted=0,
        simulation_calls=0, generation_calls=0, thresholds_changed=False, historical_results_changed=False,
        old_launchers_unchanged=True, innovation_confirmed=False, overleaf_results_sync=False,
        scope='Current execution-time lineage and witness binding; legacy resume control remains unsuitable for future unguarded use')
    out = root / 'results/execution_lineage_guard_v39_20261012'
    out.mkdir(exist_ok=True)
    target = out / ('certificate_%d.json' % time.time_ns())
    with target.open('x', encoding='utf8') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps({'status': 'PASS', 'checked_batches': len(entries), 'path': str(target)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        main(args.root.resolve())

"""Read-only publication binding; never alters historical scores or acceptance.

Server mode checks all bound raw bytes. Local mode checks public artifacts only
and explicitly does not certify server-only arrays. This is not an integrity
verdict or an efficacy test.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/controller_state_diagnosis_v2_20261012'
STATES = ('multitask', 't2m', 'unfinetuned')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def require(value, message):
    if not value:
        raise ValueError(message)


def verify(server=False):
    analysis = read(OUT / 'analysis.json')
    versions = read(OUT / 'versions.json')
    witness = read(OUT / 'DOCUMENT_WITNESS.json')
    terminal = read(OUT / 'EXECUTOR_TERMINAL_OBSERVATION.json')
    require(witness['status'] == 'PASS', 'Document execution witness missing')
    require(witness['versions_sha256'] == sha(OUT / 'versions.json'), 'Witness versions changed')
    require(witness['runtime_witness_sha256'] == sha(OUT / 'runtime_witness.json'), 'Runtime witness changed')
    require(terminal['actual_parent_exit_code'] == 0, 'No observed successful parent termination')
    require(analysis['same_algorithm_different_training_states'] and not analysis['candidate_efficacy_test'],
            'Training states must remain an engineering diagnosis')
    checked = {}
    unavailable = []
    for name, digest in versions['files'].items():
        require(sha(ROOT / name) == digest, 'Frozen source differs: ' + name)
        checked[name] = digest
    for name, digest in analysis['input_hashes'].items():
        path = ROOT / name
        if path.suffix == '.npz' and not server:
            unavailable.append(name)
            continue
        require(path.is_file() and sha(path) == digest, 'Analysis input differs: ' + name)
        checked[name] = digest
    for state in STATES:
        base = OUT / 'raw' / state
        rows, meta = read(base / 'metrics.json'), read(base / 'metadata.json')
        lineage, receipt = read(base / 'lineage.json'), read(base / 'receipt.json')
        request = read(OUT / 'requests' / (state + '.json'))
        require(lineage['request'] == request, 'Execution request differs: ' + state)
        require(len(rows) == len(meta['cases']) == receipt['cases'] == 72, 'Incomplete denominator: ' + state)
        require([r['case_id'] for r in rows] == [c['case_id'] for c in meta['cases']], 'Case order changed')
        require(meta['generation_batch_calls'] == 0, 'Fixed replay silently regenerated')
        require(meta['initial_reset_gate_pass'], 'Recorded reset gate failed')
        initial_name = str((base / 'initial_states.npz').relative_to(ROOT)).replace('\\', '/')
        initial_digest = analysis['input_hashes'][initial_name]
        require(meta['initial_state_archive_sha256'] == initial_digest, 'Initial archive metadata binding differs')
        for name, digest in lineage['outputs'].items():
            bound_name = str((base / name).relative_to(ROOT)).replace('\\', '/')
            require(analysis['input_hashes'][bound_name] == digest, 'Lineage/analysis digest differs')
        counts = {k: sum(bool(r[k]) for r in rows) for k in
                  ('completed', 'condition_pass', 'physical_pass', 'tracking_pass', 'joint_pass')}
        require(counts == analysis['reports'][state]['counts'], 'Published counts differ')
    return dict(status='PUBLICATION_BINDINGS_CHECKED', raw_array_bytes_verified=server,
                server_only_arrays_not_verified_here=unavailable, checked_hashes=checked,
                analysis_sha256=sha(OUT / 'analysis.json'), source_sha256=sha(Path(__file__)),
                candidate_efficacy_test=False, scores_changed=False,
                scope='Deterministic publication binding only; independent audit verdict remains separate')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--server', action='store_true')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    result = verify(a.server)
    if a.output:
        with a.output.open('x', encoding='utf8') as f:
            json.dump(result, f, indent=2)
        print('PUBLICATION_BINDINGS_CHECKED', 'server_raw_bytes=' + str(a.server))
    else:
        print(json.dumps(result, indent=2))

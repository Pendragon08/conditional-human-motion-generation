"""CPU-only paired analysis; no change to scores or experiment inputs.

The 21 conceptual batches include one carried, already charged candidate.
New ledger must contain exactly 20 successful actual calls. Validation only.
"""
import hashlib
import json
from pathlib import Path
import time
from verify_execution_lineage_v39_20261012 import guard_batch, sha
from physics_guided_lineage import motion_hash

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/task_matched_feedback_continue_20261012'
OLD = ROOT / 'results/task_matched_feedback_v2_20261012'
DEST = ROOT / 'results/task_matched_feedback_analysis_20261012'
PARENT = ROOT / 'scripts/analyze_paired_feedback_v51_20261012.py'


def load(path):
    return json.loads(path.read_bytes())


if __name__ == '__main__':
    if DEST.exists():
        raise RuntimeError('Analysis attempt retained; do not overwrite')
    if sha(PARENT) != '8f99e1901ca091003dc431f902c525eaa877877736982447311a677278be8eec':
        raise RuntimeError('Parent analysis changed')
    summary, v = load(OUT / 'summary.json'), load(OUT / 'versions.json')
    config = load(ROOT / 'configs/physics_guided_20261011.json')
    assert config['bootstrap_repetitions'] == 5000 and config['min_gain_percentage_points'] == 10
    for name, digest in v['files'].items():
        assert sha(ROOT / name) == digest
    for name, key in [('source_manifest.json', 'source_manifest_sha256'), ('frozen_thresholds.json', 'thresholds_sha256')]:
        assert sha(OUT / name) == v[key]
    for name, digest in v['carried_first_hashes'].items():
        assert sha(ROOT / name) == digest
    ledger = load(OUT / 'budget_ledger.json')
    assert len(ledger) == 20 and len({r['label'] for r in ledger}) == 20
    assert all(r['exit_code'] == 0 for r in ledger)
    assert sum(r['seconds'] for r in ledger) <= 3600
    assert {p.name for p in (OUT / 'raw').iterdir() if p.is_dir()} == {r['label'] for r in ledger}
    checks = {}
    bindings = {}
    for seed in (11, 12, 13):
        labels = ['val_shared_s%d_r0' % seed]
        labels += ['val_%s_s%d_r%d' % (arm, seed, ordinal) for ordinal in (1, 2) for arm in ('feedback', 'random')]
        labels += ['val_%s_replay_s%d' % (arm, seed) for arm in ('feedback', 'random')]
        for label in labels:
            location = OLD if label == 'val_shared_s11_r0' else OUT
            path = location / 'raw' / label
            wanted = location / 'requests' / (label + '.json')
            req, execution_version = load(wanted), load(location / 'versions.json')
            assert req['manifest'] == v['source_manifest_sha256']
            assert req['thresholds'] == v['thresholds_sha256']
            assert req['versions'] == sha(location / 'versions.json')
            assert req['split'] == 'val' and req['controller_state'] == 't2m'
            if location == OUT:
                assert next(r for r in ledger if r['label'] == label)['request_sha256'] == sha(wanted)
            checks[label] = guard_batch(path, req)
            meta, data, weights = [load(path / n) for n in ('metadata.json', 'generation_input_binding.json', 'loaded_checkpoint_binding.json')]
            assert data['exact_loaded_dataset_bytes'] == v['external_dataset_files']
            assert data['loaded_correction_object_sha256'] == req.get('correction')
            assert data['request_sha256'] == weights['request_sha256'] == sha(wanted)
            assert weights['versions_sha256'] == sha(location / 'versions.json')
            assert sorted(weights['actual_loaded_weight_bytes'].values()) == sorted([v['generator_weights_sha256'], v['controller_states']['t2m']['sha256']])
            assert weights['controller_state'] == 't2m'
            assert meta['initial_reset_gate_pass'] and not any(meta['initial_rigid_cache_consistency'].values())
            assert sha(path / 'initial_states.npz') == meta['initial_state_archive_sha256']
            assert sha(path / 'observation_trace.npz') == meta['observation_trace_sha256']
            for file in path.iterdir():
                if file.is_file():
                    bindings[str(file.relative_to(ROOT))] = sha(file)
            bindings[str(wanted.relative_to(ROOT))] = sha(wanted)
        for arm in ('feedback', 'random'):
            name = '%s_s%d' % (arm, seed)
            selected = OUT / 'selected' / (name + '.json')
            lineage = load(OUT / 'selected' / (name + '.lineage.json'))
            assert sha(selected) == lineage['selection_json']
            replay_request = load(OUT / 'requests' / ('val_%s_replay_s%d.json' % (arm, seed)))
            assert motion_hash(OUT / 'selected' / (name + '.npz')) == lineage['selected_motion'] == replay_request['replay_motion']
            candidates = [OLD / 'raw/val_shared_s11_r0' if seed == 11 else OUT / 'raw' / ('val_shared_s%d_r0' % seed)]
            candidates += [OUT / 'raw' / ('val_%s_s%d_r%d' % (arm, seed, i)) for i in (1, 2)]
            assert len(lineage['candidate_outputs']) == 3
            for path, recorded in zip(candidates, lineage['candidate_outputs']):
                assert recorded == load(path / 'lineage.json')['outputs']
                for field, digest in recorded.items():
                    assert sha(path / field) == digest
    certificate = dict(status='CURRENT_EXECUTION_BINDINGS_MATCH', batches=checks,
        raw_input_hashes=bindings, original_measurement_seconds=v['original_measurement_seconds'],
        new_batch_count=20, conceptual_construction_batches=15, conceptual_replay_batches=6,
        carried_candidate='val_shared_s11_r0', carried_candidate_reexecuted=False,
        source_sha256=sha(Path(__file__)), parent_sha256=sha(PARENT),
        versions_sha256=sha(OUT / 'versions.json'), summary_sha256=sha(OUT / 'summary.json'),
        bootstrap_repetitions=5000, scope='24 internal validation sources; no independent-test/G1/downstream claim',
        generation_calls=0, simulation_calls=0, innovation_confirmed=False, time=time.time())
    DEST.mkdir()
    certificate_path = DEST / 'execution_binding_certificate.json'
    certificate_path.write_text(json.dumps(certificate, indent=2), encoding='utf8')
    source = PARENT.read_text(encoding='utf8')
    substitutions = {
        "'results/paired_feedback_validation_v51_20261012'": "'results/task_matched_feedback_continue_20261012'",
        "'results/paired_feedback_analysis_v53_20261012'": "'results/task_matched_feedback_analysis_20261012'",
        'len(ledger) == 21 and len({x[\'label\'] for x in ledger}) == 21': 'len(ledger) == 20 and len({x[\'label\'] for x in ledger}) == 20',
        'size=(10000, 24)': 'size=(5000, 24)',
        'bootstrap_resamples=10000': 'bootstrap_resamples=5000',
        'bound = {}': "bound = {str(certificate_path.relative_to(ROOT)): sha(certificate_path), str(PARENT.relative_to(ROOT)): sha(PARENT), str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}",
    }
    for old, new in substitutions.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    exec(compile(source, str(Path(__file__)), 'exec'), globals())

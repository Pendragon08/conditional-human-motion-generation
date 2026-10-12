"""New paired C1 validation epoch with task-matched official frozen tracker."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import run_paired_feedback_validation_v51_20261012 as prior
import controller_state_diagnosis_20261012 as states
from verify_execution_lineage_v39_20261012 import read, sha, guard_batch
from physics_guided_com_lineage_v24_20261012 import request, object_hash

ROOT = prior.ROOT
OUT = ROOT / 'results/task_matched_feedback_20261012'
DOC = ROOT / 'docs/task_matched_feedback_20261012.md'
BASE = ROOT / 'results/controller_state_diagnosis_v2_20261012'
STATE = 't2m'
prior.OUT = OUT


def save(path, value, exclusive=False):
    with path.open('x' if exclusive else 'w', encoding='utf8') as f:
        json.dump(value, f, indent=2)


def validate():
    if sys.flags.optimize:
        raise RuntimeError('Do not disable inherited integrity gates')
    v = read(OUT / 'versions.json')
    for name, digest in v['files'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError('Frozen source differs: ' + name)
    for name, key in [('source_manifest.json', 'source_manifest_sha256'), ('frozen_thresholds.json', 'thresholds_sha256')]:
        if sha(OUT / name) != v[key]:
            raise RuntimeError('Frozen inputs differ')
    for state in v['controller_states'].values():
        if sha(ROOT / state['path']) != state['sha256']:
            raise RuntimeError('Frozen controller differs')
    if sha(ROOT / v['generator_path']) != v['generator_weights_sha256']:
        raise RuntimeError('Frozen generator differs')
    for path, digest in v['external_dataset_files'].items():
        if sha(Path(path)) != digest:
            raise RuntimeError('External dataset differs')
    upstream = ROOT / 'third_party/CLoSD'
    if subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip() != v['upstream_commit']:
        raise RuntimeError('Upstream changed')
    if subprocess.check_output(['git', '-C', str(upstream), 'diff', '--name-only', 'HEAD', '--', 'closd'], text=True).strip():
        raise RuntimeError('Tracked upstream code changed')
    return v


def prepare():
    if OUT.exists():
        validate()
        return
    v = read(BASE / 'versions.json')
    for name, digest in v['files'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError('Prior bound source differs')
    audit = read(BASE / 'EXPERIMENT_AUDIT.json')
    verdict = str(audit.get('overall_verdict', audit.get('verdict'))).upper()
    if verdict not in ('PASS', 'WARN'):
        raise RuntimeError('Prior diagnosis integrity audit has not completed')
    OUT.mkdir(); (OUT / 'raw').mkdir(); (OUT / 'requests').mkdir()
    for name in ('source_manifest.json', 'frozen_thresholds.json'):
        shutil.copyfile(BASE / name, OUT / name)
    manifest = read(OUT / 'source_manifest.json')
    paths = {manifest['mean'], manifest['std']} | {r['cache_file'] for r in manifest['rows'] if r['split'] == 'val'}
    v['external_dataset_files'] = {str(Path(p).resolve()): sha(Path(p)) for p in paths}
    for p in (Path(__file__), ROOT / 'scripts/task_matched_feedback_collect_20261012.py',
              ROOT / 'scripts/run_paired_feedback_validation_v51_20261012.py', DOC):
        v['files'][str(p.relative_to(ROOT))] = sha(p)
    v.update(source_manifest_sha256=sha(OUT / 'source_manifest.json'),
             thresholds_sha256=sha(OUT / 'frozen_thresholds.json'),
             previous_audit_sha256=sha(BASE / 'EXPERIMENT_AUDIT.json'),
             scope='Independent task-matched C1 internal validation; no historical results pooled',
             controller_states={STATE: v['controller_states'][STATE]},
             seeds=[11, 12, 13], candidates_per_method=3, max_new_seconds=1800,
             inputs_per_method=216, measurement_batches=3, total_unique_batches=23,
             candidate_efficacy='validation_only', confirmed_innovation=False, overleaf_results_sync=False)
    save(OUT / 'versions.json', v, True)
    save(OUT / 'budget_ledger.json', [], True)
    validate()
    print('TASK_MATCHED_FEEDBACK_PREPARED', flush=True)


def batch(label, mode, seed, ordinal=0, correction=None, replay=None, stage='comparison'):
    v = validate()
    options = dict(noise_round=ordinal)
    if correction: options['correction_file'] = correction
    if replay: options['replay_file'] = replay
    expected = request(ROOT, OUT, 'val', mode, seed, ordinal, options)
    expected.update(controller_state=STATE, controller_weights_sha256=v['controller_states'][STATE]['sha256'])
    path = OUT / 'raw' / label
    wanted = OUT / 'requests' / (label + '.json')
    ledger = read(OUT / 'budget_ledger.json')
    if path.exists():
        if not any(r['label'] == label and r['exit_code'] == 0 for r in ledger):
            raise RuntimeError('Failed/incomplete invocation retained; no retry')
        if read(wanted) != expected:
            raise RuntimeError('Cannot resume different request')
        guard_batch(path, expected)
        verify_batch(path, wanted, expected, v)
        return path
    snapshots = sorted((ROOT / 'results/round_resource_status_20261012').glob('snapshot_*.json'))
    resource = read(snapshots[-1]); dirs = [ROOT / d for d in resource['directories']]
    if OUT not in dirs: dirs.append(OUT)
    rows = [r for d in dirs if (d / 'budget_ledger.json').exists() for r in read(d / 'budget_ledger.json')]
    stage_cap = 4 if stage == 'measurement' else 12
    timeout = min(180, 1800 - sum(r['seconds'] for r in ledger),
                  72 * 3600 - sum(r['seconds'] for r in rows),
                  stage_cap * 3600 - sum(r['seconds'] for r in rows if r['stage'] == stage))
    if timeout < 60 or shutil.disk_usage('/data').free < 20 * 1024**3:
        raise RuntimeError('Resource gate unmet')
    if sum(p.stat().st_size for d in dirs for p in d.rglob('*') if p.is_file()) + resource['archive_bytes'] >= 15 * 1024**3:
        raise RuntimeError('Output cap reached')
    prior.runtime.OUT = OUT
    resources = prior.runtime.resource_guard()
    save(wanted, expected, True)
    command = [str(prior.runtime.PYTHON), '-u', str(ROOT / 'scripts/task_matched_feedback_collect_20261012.py'),
               '--manifest', str(OUT / 'source_manifest.json'), '--split', 'val', '--mode', mode,
               '--seed', str(seed), '--round', str(ordinal), '--noise-round', str(ordinal),
               '--output', str(path), '--thresholds', str(OUT / 'frozen_thresholds.json'),
               '--request-file', str(wanted), '--execution-versions', str(OUT / 'versions.json'),
               '--controller-state', STATE]
    if correction: command += ['--correction-file', str(correction)]
    if replay: command += ['--replay-file', str(replay)]
    save(OUT / 'state.json', dict(status='RUNNING', label=label, resources=resources, pid=os.getpid()))
    start, code = time.monotonic(), -1
    try:
        with (OUT / (label + '.log')).open('x') as log:
            code = subprocess.run(command, cwd=ROOT / 'third_party/CLoSD', env=prior.runtime.environment(),
                                  stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode
    finally:
        ledger.append(dict(label=label, stage=stage, seconds=time.monotonic() - start,
                           exit_code=code, request_sha256=sha(wanted)))
        save(OUT / 'budget_ledger.json', ledger)
    if code != 0: raise RuntimeError('Failed charged invocation retained')
    guard_batch(path, expected)
    verify_batch(path, wanted, expected, v)
    return path


def verify_batch(path, wanted, expected, v):
    meta, input_binding = read(path / 'metadata.json'), read(path / 'generation_input_binding.json')
    weights = read(path / 'loaded_checkpoint_binding.json')
    if input_binding['exact_loaded_dataset_bytes'] != v['external_dataset_files']:
        raise RuntimeError('Actual dataset bytes mismatch')
    if input_binding['loaded_correction_object_sha256'] != expected.get('correction'):
        raise RuntimeError('Actual correction mismatch')
    if input_binding['request_sha256'] != sha(wanted) or weights['request_sha256'] != sha(wanted):
        raise RuntimeError('Child request mismatch')
    if weights['versions_sha256'] != sha(OUT / 'versions.json') or weights['controller_state'] != STATE:
        raise RuntimeError('Child checkpoint mismatch')
    if sorted(weights['actual_loaded_weight_bytes'].values()) != sorted([v['generator_weights_sha256'], v['controller_states'][STATE]['sha256']]):
        raise RuntimeError('Actual loaded checkpoint bytes mismatch')
    if sha(path / 'initial_states.npz') != meta['initial_state_archive_sha256']:
        raise RuntimeError('Initial archive binding mismatch')
    if sha(path / 'observation_trace.npz') != meta['observation_trace_sha256']:
        raise RuntimeError('Observation archive binding mismatch')
    if not meta['initial_reset_gate_pass'] or any(meta['initial_rigid_cache_consistency'].values()):
        raise RuntimeError('Reset consistency failed')


def check():
    import physics_guided_queue as helpers
    first = batch('val_shared_s11_r0', 'native', 11, stage='measurement')
    zero = helpers.correction(OUT, first, 'witness_zero', no_feedback=True)
    repeated = batch('witness_zero_s11', 'native', 11, correction=zero, stage='measurement')
    same = {}
    for name in ('trajectories.npz', 'observation_trace.npz', 'initial_states.npz'):
        with np.load(first / name, allow_pickle=False) as a, np.load(repeated / name, allow_pickle=False) as b:
            if set(a.files) != set(b.files): raise RuntimeError('Repeat archive fields differ')
            for key in a.files: same[name + ':' + key] = bool(np.array_equal(a[key], b[key]))
    if not all(same.values()): raise RuntimeError('Same input repeat exported state/trajectory differs')
    reference = helpers.select(OUT, [first], 'witness_raw', raw=True)
    replay = batch('witness_raw_replay_s11', 'replay', 11, replay=reference, stage='measurement')
    with np.load(reference, allow_pickle=False) as selected, np.load(replay / 'trajectories.npz', allow_pickle=False) as z:
        wanted = selected['reference']; canonical = (wanted - wanted[:, :, 0:1, :]) + wanted[:, :, 0:1, :]
        if not np.allclose(z['reference'][z['valid']], canonical[z['valid']], atol=1e-12, rtol=0):
            raise RuntimeError('Fixed replay not supplied at same time')
    save(OUT / 'runtime_witness.json', dict(status='PASS', exact_same_input_arrays=same,
         versions_sha256=sha(OUT / 'versions.json'), actual_modes=['native', 'native_zero_correction', 'replay'],
         cases_per_batch=72, hidden_physx_identity_established=False, candidate_efficacy_test=False), True)
    print('TASK_MATCHED_FEEDBACK_CHECK_PASS', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); g = p.add_mutually_exclusive_group(required=True)
    for mode in ('prepare', 'check', 'run'): g.add_argument('--' + mode, action='store_true')
    a = p.parse_args()
    prepare()
    prior.prepare = validate
    prior.batch = batch
    if a.prepare: pass
    elif a.check:
        with (OUT / 'queue.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            check()
    else:
        witness = read(OUT / 'DOCUMENT_WITNESS.json')
        if witness['status'] != 'PASS' or witness['versions_sha256'] != sha(OUT / 'versions.json'):
            raise RuntimeError('Independent task-matched witness required')
        prior.main(False)
        print('TASK_MATCHED_FEEDBACK_VALIDATION_COMPLETE', flush=True)

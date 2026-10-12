"""Corrected-interface validation pilot; equal budget, paired per-round noise.

New epoch, not a rewrite or pooled extension of the negative V28 experiment.
No test sources, robot training or new thresholds. Inspect validation first.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import run_reset_alignment_checks_v4 as runtime
from physics_guided_com_lineage_v24_20261012 import request
from verify_execution_lineage_v39_20261012 import guard_batch, read, sha, ANCHORS

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/condition_feedback_formal_v28_20261012'
OUT = ROOT / 'results/paired_feedback_validation_v51_20261012'
DOC = ROOT / 'docs/paired_feedback_validation_v51_20261012.md'


def save(path, data):
    path.write_text(json.dumps(data, indent=2), encoding='utf8')


def prepare():
    for name, digest in ANCHORS.items():
        assert sha(BASE / name) == digest
    version = read(BASE / 'versions.json')
    assert all(sha(ROOT / name) == digest for name, digest in version['files'].items())
    calibration = ROOT / 'results/surface_proxy_calibration_v52_20261012'
    assert read(calibration / 'first_attempt/summary.json')['status'] == 'PASS_ANALYTICAL_CONTROLS'
    witness = read(calibration / 'DOCUMENT_WITNESS.json')
    assert witness['status'] == 'PASS', 'Independent analytical document witness required'
    assert read(ROOT / 'results/reference_velocity_v43_20261012/DOCUMENT_WITNESS.json')['status'] == 'PASS'
    files = ('scripts/run_paired_feedback_validation_v51_20261012.py',
             'scripts/reference_velocity_collect_v43_20261012.py',
             'scripts/reference_observation_collect_v42_20261012.py',
             'scripts/run_reference_observation_v42_20261012.py',
             'scripts/compare_reference_observation_v42_20261012.py',
             'scripts/verify_execution_lineage_v39_20261012.py',
             'scripts/physics_guided_queue.py', str(DOC.relative_to(ROOT)))
    version['files'].update({name: sha(ROOT / name) for name in files})
    version.update(role='Validation-only paired-noise feedback pilot after V43 interface repair',
        calibration_summary_sha256=sha(calibration / 'first_attempt/summary.json'),
        calibration_witness_sha256=sha(calibration / 'DOCUMENT_WITNESS.json'),
        velocity_witness_sha256=sha(ROOT / 'results/reference_velocity_v43_20261012/DOCUMENT_WITNESS.json'),
        source_manifest_sha256=sha(BASE / 'source_manifest.json'),
        thresholds_sha256=sha(BASE / 'frozen_thresholds.json'),
        seeds=[11, 12, 13], inputs_per_seed=72, sources=24, nominal_candidates_per_method=3,
        construction_batches=15, independent_replay_batches=6,
        pairing='Actual-prefix mode for both methods; identical noise for each candidate ordinal',
        prior_protocol_differences='Unlike historical V28, candidate ordinal changes noise for both arms; no old/new scores pooled. Equal-budget random and corrected condition arms share ordinal0. Only validation, no claim of full ten-arm test.',
        confirmatory=False, candidate_efficacy='validation_pilot_only', innovation_confirmed=False,
        overleaf_results_sync=False)
    if OUT.exists():
        assert read(OUT / 'versions.json') == version
        return
    OUT.mkdir(); (OUT / 'raw').mkdir(); (OUT / 'requests').mkdir()
    for name in ('source_manifest.json', 'frozen_thresholds.json'):
        shutil.copyfile(BASE / name, OUT / name)
    save(OUT / 'versions.json', version); save(OUT / 'budget_ledger.json', [])


def batch(label, mode, seed, ordinal=0, correction=None, replay=None, stage='comparison'):
    options = dict(noise_round=ordinal)
    if correction: options['correction_file'] = correction
    if replay: options['replay_file'] = replay
    expected = request(ROOT, OUT, 'val', mode, seed, ordinal, options)
    path = OUT / 'raw' / label; wanted = OUT / 'requests' / (label + '.json')
    ledger = read(OUT / 'budget_ledger.json')
    if path.exists():
        assert read(wanted) == expected
        guard_batch(path, expected)
        assert any(x['label'] == label and x['exit_code'] == 0 for x in ledger)
        assert sha(path / 'observation_trace.npz') == read(path / 'metadata.json')['observation_trace_sha256']
        return path
    snapshots = sorted((ROOT / 'results/round_resource_status_20261012').glob('snapshot_*.json'))
    prior = read(snapshots[-1]); dirs = [ROOT / x for x in prior['directories']]
    if OUT not in dirs: dirs.append(OUT)
    rows = [x for d in dirs if (d / 'budget_ledger.json').exists() for x in read(d / 'budget_ledger.json')]
    spent = sum(x['seconds'] for x in rows)
    stage_spent = sum(x['seconds'] for x in rows if x['stage'] == stage)
    own_spent = sum(x['seconds'] for x in ledger)
    stage_cap = (4 if stage == 'measurement' else 12) * 3600
    timeout = int(min(600, 72 * 3600 - spent, stage_cap - stage_spent, 3600 - own_spent))
    assert timeout >= 60, 'Budget exhausted; no extension'
    assert shutil.disk_usage('/data').free >= 20 * 1024**3
    assert sum(p.stat().st_size for d in dirs for p in d.rglob('*') if p.is_file()) + prior['archive_bytes'] < 15 * 1024**3
    runtime.OUT = OUT; resources = runtime.resource_guard()
    save(wanted, expected)
    command = [str(runtime.PYTHON), '-u', str(ROOT / 'scripts/reference_velocity_collect_v43_20261012.py'),
        '--manifest', str(OUT / 'source_manifest.json'), '--split', 'val', '--mode', mode,
        '--seed', str(seed), '--round', str(ordinal), '--noise-round', str(ordinal),
        '--output', str(path), '--thresholds', str(OUT / 'frozen_thresholds.json'), '--request-file', str(wanted)]
    if correction: command += ['--correction-file', str(correction)]
    if replay: command += ['--replay-file', str(replay)]
    save(OUT / 'state.json', dict(status='RUNNING', label=label, resources=resources, pid=os.getpid()))
    begin = time.monotonic(); code = -1
    try:
        with (OUT / (label + '.log')).open('x') as log:
            code = subprocess.run(command, cwd=ROOT / 'third_party/CLoSD', env=runtime.environment(),
                                  stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode
    finally:
        ledger.append(dict(label=label, stage=stage, seconds=time.monotonic() - begin,
                           exit_code=code, request_sha256=sha(wanted)))
        save(OUT / 'budget_ledger.json', ledger)
    assert code == 0, 'Failure retained, no automatic retry'
    guard_batch(path, expected)
    assert sha(path / 'observation_trace.npz') == read(path / 'metadata.json')['observation_trace_sha256']
    return path


def counts(path):
    rows = read(path / 'metrics.json')
    return {key: sum(bool(x[key]) for x in rows) for key in
            ('completed', 'condition_pass', 'physical_pass', 'tracking_pass', 'joint_pass')}


def main(check):
    import numpy as np
    import physics_guided_queue as helpers
    from compare_reference_observation_v42_20261012 import compare
    assert np.__version__ == '1.23.5'
    prepare()
    with (OUT / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if check:
                first = batch('val_shared_s11_r0', 'native', 11, stage='measurement')
                prior = ROOT / 'results/native_reference_velocity_v46_20261012/raw/val_native_s11'
                matched = {}
                for name in ('trajectories.npz', 'observation_trace.npz', 'initial_states.npz'):
                    with np.load(first / name, allow_pickle=False) as current, np.load(prior / name, allow_pickle=False) as old:
                        assert current.files == old.files
                        for field in current.files:
                            matched[name + ':' + field] = bool(np.array_equal(current[field], old[field]))
                assert all(matched.values())
                save(OUT / 'runtime_witness.json', dict(status='PASS', exact_arrays=matched,
                     old_control_unchanged=True, generation_noise_same=read(first / 'metadata.json')['generation_noise_sha256'] == read(prior / 'metadata.json')['generation_noise_sha256']))
                save(OUT / 'state.json', dict(status='WITNESS_COMPLETE_PENDING_INDEPENDENT_REVIEW'))
                print('PAIRED_FEEDBACK_VALIDATION_DOCUMENT_CHECK_PASS', flush=True)
                return
            assert read(OUT / 'runtime_witness.json')['status'] == 'PASS'
            independent = read(OUT / 'DOCUMENT_WITNESS.json')
            assert independent['status'] == 'PASS' and independent['runtime_witness_sha256'] == sha(OUT / 'runtime_witness.json')
            reports = {}; replay_records = {}
            for seed in (11, 12, 13):
                first = batch('val_shared_s%d_r0' % seed, 'native', seed,
                              stage='measurement' if seed == 11 else 'comparison')
                arms = dict(feedback=[first], random=[first])
                for ordinal in (1, 2):
                    feedback = helpers.correction(OUT, arms['feedback'][-1], 's%d_r%d' % (seed, ordinal))
                    repaired = batch('val_feedback_s%d_r%d' % (seed, ordinal), 'native', seed, ordinal, correction=feedback)
                    random = batch('val_random_s%d_r%d' % (seed, ordinal), 'native', seed, ordinal)
                    assert read(repaired / 'metadata.json')['generation_noise_sha256'] == read(random / 'metadata.json')['generation_noise_sha256']
                    arms['feedback'].append(repaired); arms['random'].append(random)
                for arm, paths in arms.items():
                    selected = helpers.select(OUT, paths, '%s_s%d' % (arm, seed))
                    replay = batch('val_%s_replay_s%d' % (arm, seed), 'replay', seed, replay=selected)
                    label = '%s_s%d' % (arm, seed)
                    reports[label] = dict(construction=[counts(p) for p in paths], replay=counts(replay),
                        candidates_per_input=3, cases=72, source_ids=sorted({r['source_id'] for r in read(replay / 'metrics.json')}))
                    replay_records[label] = guard_batch(replay, read(OUT / 'requests' / (replay.name + '.json')))
                save(OUT / 'progress.json', dict(completed_seed=seed, reports=reports, innovation_confirmed=False))
            total = {arm: {key: sum(reports['%s_s%d' % (arm, seed)]['replay'][key] for seed in (11, 12, 13))
                          for key in ('completed', 'condition_pass', 'physical_pass', 'tracking_pass', 'joint_pass')}
                     for arm in ('feedback', 'random')}
            floor = all(v['joint_pass'] == 0 for v in total.values())
            save(OUT / 'summary.json', dict(status='COMPLETE_VALIDATION_QUALITY_FLOOR' if floor else 'COMPLETE_VALIDATION_REQUIRES_FULL_ANALYSIS',
                reports=reports, totals=total, inputs_per_method=216, sources=24, seeds=[11, 12, 13],
                generation_candidate_budget_per_method=3, actual_construct_batches=15, replay_batches=6,
                replay_records=replay_records, candidate_efficacy='validation_only', confirmed_innovation=False,
                c2_quality_floor_prerequisite_met=not floor, thresholds_changed=False, old_scores_unchanged=True,
                test_or_g1_or_downstream_run=False, overleaf_results_sync=False))
            save(OUT / 'state.json', dict(status='COMPLETE_VALIDATION_QUALITY_FLOOR' if floor else 'COMPLETE_VALIDATION_REQUIRES_FULL_ANALYSIS'))
            print('PAIRED_FEEDBACK_VALIDATION_COMPLETE', json.dumps(total), flush=True)
        except Exception as error:
            save(OUT / 'state.json', dict(status='FAILED_RETAINED', error=repr(error), pid=os.getpid()))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true'); group.add_argument('--run', action='store_true')
    args = parser.parse_args()
    main(args.check)

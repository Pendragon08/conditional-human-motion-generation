"""Small validation-only mechanism probe after reset repair, not efficacy test."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import run_reset_alignment_checks_v4 as runtime
from physics_guided_queue import correction, select, load
from physics_guided_lineage import request, certify_cached

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/condition_feedback_probe_20261011'
FIX = ROOT / 'results/reset_alignment_v4_20261011'
OLD = ROOT / 'results/physics_guided_20261011'
CONFIG = json.loads((ROOT / 'configs/physics_guided_20261011.json').read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')


def prepare():
    if OUT.exists():
        version = json.loads((OUT / 'versions.json').read_text())
        assert all(runtime.sha(ROOT / name) == digest for name, digest in version['files'].items())
        assert runtime.sha(OUT / 'source_manifest.json') == version['manifest_sha256']
        return
    diagnosis = json.loads((FIX / 'measurement_repair_summary.json').read_text())
    assert diagnosis['coordinate_gate_pass'] and not diagnosis['coordinate_errors']
    assert diagnosis['attempts'] == 216
    OUT.mkdir()
    shutil.copyfile(FIX / 'measurement_manifest.json', OUT / 'source_manifest.json')
    shutil.copyfile(OLD / 'frozen_thresholds.json', OUT / 'frozen_thresholds.json')
    version = json.loads((OLD / 'versions.json').read_text())
    files = ['scripts/reset_aligned_collect_v4.py', 'scripts/physics_guided_collect.py',
        'scripts/physics_guided_core.py', 'scripts/physics_guided_queue.py',
        'scripts/physics_guided_lineage.py', 'scripts/r3_closd_isolated_entry.py',
        'scripts/run_reset_alignment_checks_v4.py', 'scripts/run_condition_feedback_probe_20261011.py',
        'configs/physics_guided_20261011.json', 'configs/env-closd-baseline-20261010.json',
        'docs/condition_feedback_probe_20261011.md']
    version['files'] = {name: runtime.sha(ROOT / name) for name in files}
    version.update(manifest_sha256=runtime.sha(OUT / 'source_manifest.json'),
        measurement_repair_sha256=runtime.sha(FIX / 'measurement_repair_summary.json'),
        role='validation-only exploratory mechanism probe; not a formal test or novelty claim',
        deviation='Four validation sources, 12 inputs, seed11 only. Formal 432-input comparison is deferred at the non-discriminating physical/joint gate.',
        expected_construction_batches=9, expected_independent_replay_batches=4,
        evaluation_inputs_per_arm=12, all_attempts_retained=True, thresholds_unchanged=True,
        generator_frozen=True, tracker_frozen=True, overleaf_results_sync=False)
    save(OUT / 'versions.json', version)
    save(OUT / 'budget_ledger.json', [])
    (OUT / 'raw').mkdir()
    (OUT / 'requests').mkdir()


def batch(label, mode, round_id=0, **kwargs):
    destination = OUT / 'raw' / label
    ledger = json.loads((OUT / 'budget_ledger.json').read_text())
    wanted = request(ROOT, OUT, 'val', mode, 11, round_id, kwargs)
    if (destination / 'receipt.json').exists():
        assert any(row['label'] == label and row['exit_code'] == 0 for row in ledger)
        certify_cached(ROOT, OUT, destination, wanted, kwargs, CONFIG)
        return destination
    assert not destination.exists(), 'Incomplete attempt preserved; no silent retry'
    previous = runtime.ledger_before() + json.loads((FIX / 'budget_ledger.json').read_text())
    remaining = min(600, 72 * 3600 - sum(row['seconds'] for row in previous + ledger),
                    12 * 3600 - sum(row['seconds'] for row in previous + ledger if row['stage'] == 'comparison'))
    assert remaining > 60
    assert shutil.disk_usage('/data').free >= 20 * 1024 ** 3
    directories = runtime.PREVIOUS + [FIX, OUT]
    assert sum(path.stat().st_size for directory in directories for path in directory.rglob('*') if path.is_file()) < 15 * 1024 ** 3
    runtime.OUT = OUT
    resources = runtime.resource_guard()
    request_path = OUT / 'requests' / (label + '.json')
    save(request_path, wanted)
    command = [str(runtime.PYTHON), '-u', str(ROOT / 'scripts/reset_aligned_collect_v4.py'),
        '--manifest', str(OUT / 'source_manifest.json'), '--output', str(destination),
        '--split', 'val', '--mode', mode, '--seed', '11', '--round', str(round_id),
        '--thresholds', str(OUT / 'frozen_thresholds.json'), '--request-file', str(request_path)]
    for key, value in kwargs.items():
        command.extend(['--' + key.replace('_', '-'), str(value)])
    save(OUT / 'state.json', dict(status='RUNNING', label=label, resources=resources, time=time.time()))
    start = time.monotonic()
    with (OUT / (label + '.log')).open('w') as stream:
        try:
            process = subprocess.run(command, cwd=ROOT / 'third_party/CLoSD', env=runtime.environment(),
                stdout=stream, stderr=subprocess.STDOUT, timeout=remaining)
            code = process.returncode
        except subprocess.TimeoutExpired:
            code = 124
    ledger.append(dict(stage='comparison', label=label, seconds=time.monotonic() - start, exit_code=code))
    save(OUT / 'budget_ledger.json', ledger)
    if code:
        raise RuntimeError('Probe failure retained: ' + label)
    meta = certify_cached(ROOT, OUT, destination, wanted, kwargs, CONFIG)
    assert meta['initial_reset_gate_pass']
    assert all(value == 0 for value in meta['manual_reset_vs_first_action_state_max_difference'].values())
    print('CONDITION_PROBE_BATCH_PASS', label, flush=True)
    return destination


def analyze(arms):
    summary = {}
    packages = {}
    for arm, path in arms.items():
        meta, rows, arrays = load(Path(path))
        choices = json.loads((OUT / 'selected' / (arm + '_s11.json')).read_text())['choices']
        for row, choice in zip(rows, choices):
            row['joint_pass'] = bool(row['joint_pass'] and choice['construction_complete'])
        packages[arm] = rows
        summary[arm] = dict(attempts=len(rows),
            **{key: sum(row.get(key, False) for row in rows) for key in
               ('completed', 'condition_pass', 'physical_pass', 'tracking_pass', 'joint_pass')},
            **{key: float(np.mean([row[key] for row in rows if row.get(key) is not None])) for key in
               ('endpoint_error_m', 'heading_error_rad', 'endpoint_time_error_s', 'tracking_mean_m',
                'surface_slip_proxy_fraction', 'penetration_fraction')})
        arrays.close()
    proposed = packages['proposed']
    rng = np.random.RandomState(20261011)
    differences = {}
    for arm in ('native_best', 'no_feedback', 'shuffled'):
        control = {row['case_id']: row for row in packages[arm]}
        sources = sorted({row['source_id'] for row in proposed})
        clusters = [np.mean([int(row['joint_pass']) - int(control[row['case_id']]['joint_pass'])
                           for row in proposed if row['source_id'] == source]) for source in sources]
        bootstrap = rng.choice(clusters, size=(5000, len(clusters)), replace=True).mean(axis=1)
        differences[arm] = dict(joint_gain_pp=100 * float(np.mean(clusters)),
            source_cluster_ci95_pp=(100 * np.quantile(bootstrap, [.025, .975])).tolist(),
            endpoint_error_reduction_m=summary[arm]['endpoint_error_m'] - summary['proposed']['endpoint_error_m'])
    ledger = json.loads((OUT / 'budget_ledger.json').read_text())
    numeric = all(value['joint_gain_pp'] >= 10 and value['source_cluster_ci95_pp'][0] > 0 for value in differences.values())
    result = dict(status='COMPLETE_EXPLORATORY_PROBE', summary=summary, differences=differences,
        numeric_pilot_gate=numeric, novelty='not_established',
        innovation_status='not_supported_by_joint_quality_probe' if not numeric else 'requires_full_independent_test_and_semantic_guards',
        confirmatory=False, sources=4, offsets=3, generation_seeds=[11],
        candidates_per_input_per_arm=3, reference_replays=48, construction_attempts=108,
        physical_quality_floor=all(value['physical_pass'] == 0 for value in summary.values()),
        qualification='A diagnostic on preselected validation sources and one seed. Not a 432-input test, not three-seed efficacy, and not evidence that the mechanism is universally ineffective. Physical proxy discrimination remains unresolved.',
        thresholds_changed=False, semantics_independently_verified=False,
        g1='not_run', downstream='not_run', overleaf_results_sync=False,
        charged_gpu_hours=sum(row['seconds'] for row in ledger) / 3600)
    save(OUT / 'probe_summary.json', result)
    print('CONDITION_PROBE_COMPLETE', json.dumps(result), flush=True)


def main(check):
    import fcntl
    prepare()
    with (OUT / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            first = batch('val_native_s11_r0', 'native')
            if check:
                save(OUT / 'state.json', dict(status='WITNESS_PASS', time=time.time()))
                print('CONDITION_PROBE_DOCUMENT_CHECK_PASS', flush=True)
                return
            random = [first] + [batch('val_native_s11_r%d' % ordinal, 'native', ordinal) for ordinal in (1, 2)]
            candidates = {'native_best': random}
            for arm in ('proposed', 'no_feedback', 'shuffled'):
                paths = [first]
                for ordinal in (1, 2):
                    feedback = correction(OUT, paths[-1], arm + '_s11_r%d' % ordinal,
                        no_feedback=arm == 'no_feedback', shuffled=arm == 'shuffled')
                    paths.append(batch('val_%s_s11_r%d' % (arm, ordinal), 'native', ordinal,
                        noise_round=0, correction_file=feedback))
                candidates[arm] = paths
            arms = {}
            for arm, paths in candidates.items():
                reference = select(OUT, paths, arm + '_s11')
                arms[arm] = str(batch('val_eval_%s_s11' % arm, 'replay', replay_file=reference))
                save(OUT / 'arms.json', arms)
            analyze(arms)
            save(OUT / 'state.json', dict(status='COMPLETE_EXPLORATORY_PROBE', time=time.time()))
        except Exception as error:
            save(OUT / 'state.json', dict(status='FAILED_RETAINED', error=repr(error), time=time.time()))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true')
    group.add_argument('--run', action='store_true')
    main(parser.parse_args().check)

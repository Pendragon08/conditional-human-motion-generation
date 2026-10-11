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
from physics_guided_numeric_lineage import request, certify_cached

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/contact_cpu_validation_v12_20261012'
V11=ROOT/'results/contact_cpu_export_v11_20261012'
PROBE = ROOT / 'results/condition_feedback_probe_v2_20261011'
FAILED = ROOT / 'results/condition_feedback_probe_20261011'
FIX = ROOT / 'results/reset_alignment_v4_20261011'
OLD = ROOT / 'results/physics_guided_20261011'
V5 = ROOT / 'results/contact_sampling_v5_20261011'
V6 = ROOT / 'results/contact_sampling_v6_20261011'
V7 = ROOT / 'results/contact_sampling_v7_20261011'
V8 = ROOT / 'results/contact_sampling_v8_20261011'
BOX = ROOT / 'results/contact_force_fixture_20261011'
V9 = ROOT / 'results/contact_sensor_v9_20261011'
V10 = ROOT / 'results/contact_sensor_v10_20261011'
BOX2 = ROOT / 'results/contact_force_fixture_v2_20261011'
BOX3 = ROOT / 'results/contact_force_fixture_v3_20261012'
CONFIG = json.loads((ROOT / 'configs/physics_guided_20261011.json').read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')


def prepare():
    if OUT.exists():
        version = json.loads((OUT / 'versions.json').read_text())
        assert all(runtime.sha(ROOT / name) == digest for name, digest in version['files'].items())
        assert runtime.sha(OUT / 'source_manifest.json') == version['manifest_sha256']
        return
    assert json.loads((BOX3/'repeat_comparison.json').read_text())['cpu_export_hypothesis_supported']
    diagnosis = json.loads((FIX / 'measurement_repair_summary.json').read_text())
    assert diagnosis['coordinate_gate_pass'] and not diagnosis['coordinate_errors']
    assert diagnosis['attempts'] == 216
    OUT.mkdir()
    assert json.loads((V11/'sampling_summary.json').read_text())['force_sampling_repeatable_at_1mN']
    manifest=json.loads((OLD/'source_manifest.json').read_text())
    manifest['rows']=[row for row in manifest['rows'] if row['split']=='val']
    assert len(manifest['rows'])==24
    save(OUT/'source_manifest.json',manifest)
    shutil.copyfile(OLD / 'frozen_thresholds.json', OUT / 'frozen_thresholds.json')
    version = json.loads((OLD / 'versions.json').read_text())
    files = ['scripts/reset_aligned_collect_v4.py', 'scripts/physics_guided_collect.py',
        'scripts/physics_guided_core.py', 'scripts/physics_guided_queue.py',
        'scripts/physics_guided_lineage.py', 'scripts/r3_closd_isolated_entry.py',
        'scripts/run_reset_alignment_checks_v4.py', 'scripts/run_contact_cpu_validation_v12_20261012.py', 'scripts/contact_cpu_export_collect_v11.py', 'scripts/contact_sampling_collect_v5.py', 'scripts/contact_sampling_collect_v8.py', 'scripts/contact_sensor_collect_v10.py', 'scripts/closd_cpu_export_bridge.py',
        'scripts/metric_numeric_equivalence.py', 'scripts/physics_guided_numeric_lineage.py',
        'configs/physics_guided_20261011.json', 'configs/env-closd-baseline-20261010.json',
        'docs/contact_cpu_validation_v12_20261012.md']
    for key,path in [('generator_weights_sha256',ROOT/'third_party/CLoSD/closd/diffusion_planner/save/DiP_multi-target_10steps_context20_predict40/model000300000.pt'),('controller_weights_sha256',ROOT/'third_party/CLoSD/output/CLoSD/CLoSD_multitask_finetune/Humanoid.pth')]:
        assert runtime.sha(path)==version[key]
    version['protocol_config_sha256']=runtime.sha(ROOT/'configs/physics_guided_20261011.json')
    version['files'] = {name: runtime.sha(ROOT / name) for name in files}
    version.update(manifest_sha256=runtime.sha(OUT / 'source_manifest.json'),
        measurement_repair_sha256=runtime.sha(FIX / 'measurement_repair_summary.json'),
        role='CPU export/CUDA mirror measurement diagnostic; GPU PhysX and frozen models; no efficacy, metric or threshold change',
        deviation='Full24 validation sources,72inputs per run, planned/native each3generationseeds. New export route, all methods must be rerun; not pooled with GPU-export results.',
        numeric_equivalence_scope='Only float64 roundoff; exact types, classifications, keys and all lineage bindings remain required',
        prior_failed_probe_state_sha256=runtime.sha(FAILED / 'state.json'),
        expected_construction_batches=6, expected_independent_replay_batches=0,
        evaluation_inputs_per_arm=72, all_attempts_retained=True, thresholds_unchanged=True,
        generator_frozen=True, tracker_frozen=True, overleaf_results_sync=False)
    save(OUT / 'versions.json', version)
    save(OUT / 'budget_ledger.json', [])
    (OUT / 'raw').mkdir()
    (OUT / 'requests').mkdir()


def batch(label, mode, seed=11, round_id=0, **kwargs):
    destination = OUT / 'raw' / label
    ledger = json.loads((OUT / 'budget_ledger.json').read_text())
    wanted = request(ROOT, OUT, 'val', mode, seed, round_id, kwargs)
    if (destination / 'receipt.json').exists():
        assert any(row['label'] == label and row['exit_code'] == 0 for row in ledger)
        certify_cached(ROOT, OUT, destination, wanted, kwargs, CONFIG)
        return destination
    assert not destination.exists(), 'Incomplete attempt preserved; no silent retry'
    previous = runtime.ledger_before() + json.loads((FIX / 'budget_ledger.json').read_text()) + json.loads((FAILED / 'budget_ledger.json').read_text()) + json.loads((PROBE / 'budget_ledger.json').read_text()) + json.loads((V5 / 'budget_ledger.json').read_text()) + json.loads((V6 / 'budget_ledger.json').read_text()) + json.loads((V7 / 'budget_ledger.json').read_text()) + json.loads((V8 / 'budget_ledger.json').read_text()) + json.loads((BOX / 'budget_ledger.json').read_text()) + json.loads((V9 / 'budget_ledger.json').read_text()) + json.loads((V10 / 'budget_ledger.json').read_text()) + json.loads((BOX2 / 'budget_ledger.json').read_text()) + json.loads((BOX3 / 'budget_ledger.json').read_text()) + json.loads((V11/'budget_ledger.json').read_text())
    remaining = min(600, 72 * 3600 - sum(row['seconds'] for row in previous + ledger),
                    4 * 3600 - sum(row['seconds'] for row in previous + ledger if row['stage'] == 'measurement'))
    assert remaining > 60
    assert shutil.disk_usage('/data').free >= 20 * 1024 ** 3
    directories = runtime.PREVIOUS + [FIX, FAILED, PROBE, V5, V6, V7, V8, BOX, V9, V10, BOX2, BOX3, V11, OUT]
    assert sum(path.stat().st_size for directory in directories for path in directory.rglob('*') if path.is_file()) < 15 * 1024 ** 3
    runtime.OUT = OUT
    resources = runtime.resource_guard()
    request_path = OUT / 'requests' / (label + '.json')
    save(request_path, wanted)
    command = [str(runtime.PYTHON), '-u', str(ROOT / 'scripts/contact_cpu_export_collect_v11.py'),
        '--manifest', str(OUT / 'source_manifest.json'), '--output', str(destination),
        '--split', 'val', '--mode', mode, '--seed', str(seed), '--round', str(round_id),
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
    ledger.append(dict(stage='measurement', label=label, seconds=time.monotonic() - start, exit_code=code))
    save(OUT / 'budget_ledger.json', ledger)
    if code:
        raise RuntimeError('Probe failure retained: ' + label)
    meta = certify_cached(ROOT, OUT, destination, wanted, kwargs, CONFIG)
    assert meta['completion_fence_calls'] >= 120
    assert 'fetch_results(True)' in meta['completion_fence_protocol']
    assert meta['passive_sensor_bodies'] == ['L_Ankle','L_Toe','R_Ankle','R_Toe']
    assert meta['policy_observation_mode'] == 1
    assert not meta['sensor_metrics_used_for_selection']
    assert runtime.sha(destination/'sensor_diagnostic.npz') == meta['sensor_diagnostic_sha256']
    assert not meta['sim_params']['use_gpu_pipeline'] and meta['sim_params']['physx_use_gpu']
    assert all(row['native_device']=='cpu' and row['policy_device'].startswith('cuda') for row in meta['bridge_buffers'].values())
    assert meta['initial_reset_gate_pass']
    assert all(value == 0 for value in meta['manual_reset_vs_first_action_state_max_difference'].values())
    print('CONTACT_SAMPLING_BATCH_PASS', label, flush=True)
    return destination


def compare():
    entries=[]
    for seed in (11,12,13):
        for mode in ('planned','native'):
            path=OUT/'raw'/('val_%s_s%d'%(mode,seed))
            rows=json.loads((path/'metrics.json').read_text())
            meta=json.loads((path/'metadata.json').read_text())
            t=json.loads((OUT/'frozen_thresholds.json').read_text())['thresholds']
            assert len(rows)==72
            entries.append(dict(seed=seed,mode=mode,inputs=len(rows),
                **{key:sum(row.get(key,False) for row in rows) for key in ('completed','condition_pass','physical_pass','tracking_pass','joint_pass')},
                slip_proxy_pass=sum(row.get('surface_slip_proxy_fraction') is not None and row['surface_slip_proxy_fraction']<=t['sliding_fraction'] for row in rows),
                penetration_pass=sum(row.get('penetration_fraction',1)<=t['penetration_fraction'] for row in rows),
                continuity_pass=sum(row.get('jerk_p95_m_s3') is not None and row['jerk_p95_m_s3']<=t['continuity_jerk_p95_m_s3'] for row in rows),
                original_goal_sha256=meta['original_goal_sha256'],noise_sha256=meta['generation_noise_sha256'],
                reset_gate_pass=meta['initial_reset_gate_pass'],metadata_sha256=runtime.sha(path/'metadata.json'),metrics_sha256=runtime.sha(path/'metrics.json')))
    assert len({x['original_goal_sha256'] for x in entries})==1
    for seed in (11,12,13): assert len({x['noise_sha256'] for x in entries if x['seed']==seed})==1
    floor=all(x['physical_pass']==0 for x in entries)
    result=dict(entries=entries,attempts=432,full_quality_floor=floor,
        decision='STOP_CONDITION_FEEDBACK_ROUTE_PENDING_PHYSICAL_QUALITY_DIAGNOSIS' if floor else 'REVIEW_CALIBRATION_BEFORE_FORMAL_COMPARISON',
        thresholds_changed=False,confirmatory=False,innovation_confirmed=False,overleaf_results_sync=False,
        qualification='New fixed CPU-export setting; no pooling with old route. Does not constitute innovation efficacy, known-good humanoid quality discrimination or downstream training.')
    save(OUT/'validation_summary.json',result)
    print('CPU_EXPORT_FULL_VALIDATION',json.dumps(result),flush=True)


def main(check):
    import fcntl
    prepare()
    with (OUT/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            batch('val_planned_s11','planned',11)
            if check:
                save(OUT/'state.json',dict(status='WITNESS_PASS',time=time.time()))
                print('CPU_EXPORT_VALIDATION_DOCUMENT_CHECK_PASS',flush=True)
                return
            for seed in (11,12,13):
                for mode in ('planned','native'):batch('val_%s_s%d'%(mode,seed),mode,seed)
            compare()
            save(OUT/'state.json',dict(status='COMPLETE_FULL_VALIDATION',time=time.time()))
        except Exception as error:
            save(OUT/'state.json',dict(status='FAILED_RETAINED',error=repr(error),time=time.time()))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true')
    group.add_argument('--run', action='store_true')
    main(parser.parse_args().check)

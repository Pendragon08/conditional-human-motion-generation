"""Warm-runtime reset measurement; resumable only by verified completed batches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'results/physics_guided_20261011'
OUT = ROOT / 'results/reset_alignment_v4_20261011'
RUNTIME = Path('/data/code/ouyanghao/oyh/humanoid-combat-motion')
PYTHON = RUNTIME / '.venv-pbhc-training/bin/python'
PREVIOUS = [OLD] + [ROOT / ('results/' + name) for name in (
    'reset_alignment_20261011', 'reset_alignment_v2_20261011', 'reset_alignment_v3_20261011')]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def ledger_before():
    return [row for directory in PREVIOUS if (directory / 'budget_ledger.json').exists()
            for row in json.loads((directory / 'budget_ledger.json').read_text())]


def resource_guard():
    """Allow the pre-existing viewer contexts, never kill or displace workloads."""
    last = None
    for attempt in range(5):
        gpu = subprocess.check_output(['nvidia-smi', '--id=0',
            '--query-gpu=memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True).strip().split(',')
        used, total, utilization = map(int, gpu)
        last = dict(memory_used_mib=used, memory_total_mib=total, utilization_percent=utilization)
        if total - used >= 24576 and utilization <= 5:
            return last
        save(OUT / 'state.json', dict(status='WAITING_RESOURCE', snapshot=last, attempt=attempt, time=time.time()))
        if attempt < 4:
            time.sleep(15)
    raise RuntimeError('GPU0 is active; existing workloads preserved: ' + repr(last))


def environment():
    env = os.environ.copy()
    env.update(PYTHONPATH=str(RUNTIME / 'envs/closd-r3/site-packages') + ':' + str(ROOT / 'third_party/CLoSD') + ':' + str(ROOT / 'third_party/CLoSD/closd'),
        HF_HOME=str(RUNTIME / 'cache/closd_hf'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        TORCH_EXTENSIONS_DIR=str(RUNTIME / 'cache/closd-r3-torch-extensions'), WANDB_MODE='disabled', WANDB_SILENT='true')
    env['PATH'] = str(RUNTIME / 'envs/closd-r3/site-packages/ninja/data/bin') + ':' + env['PATH']
    for key in ('R3_CLOSD_SCENE_UDP', 'R3_CLOSD_WEB_FULL_SCENE', 'R3_CLOSD_TELEMETRY_UDP'):
        env.pop(key, None)
    return env


def prepare():
    if OUT.exists():
        previous = json.loads((OUT / 'versions_before_execution.json').read_text())
        assert all(sha(path) == value for path, value in previous['hashes'].items()), 'Frozen measurement source changed'
        return
    assert json.loads((OLD / 'queue_state.json').read_text())['status'] == 'COMPLETE_GATE_NOT_MET'
    OUT.mkdir()
    source = json.loads((OLD / 'source_manifest.json').read_text())
    rows = []
    for family in ('jump', 'dance'):
        eligible = [r for r in source['rows'] if r['split'] == 'val' and r['family'] == family]
        assert len(eligible) == 12
        rows.extend(eligible[:2])
    source['rows'] = rows
    save(OUT / 'measurement_manifest.json', source)
    original = json.loads((OLD / 'versions.json').read_text())
    weights = dict(generator_weights_sha256=ROOT / 'third_party/CLoSD/closd/diffusion_planner/save/DiP_multi-target_10steps_context20_predict40/model000300000.pt',
                   controller_weights_sha256=ROOT / 'third_party/CLoSD/output/CLoSD/CLoSD_multitask_finetune/Humanoid.pth')
    for key, path in weights.items():
        assert sha(path) == original[key], 'Frozen model changed'
    assert all(sha(path) == digest for path, digest in original['external_motion_inputs'].items())
    source_root = ROOT / 'third_party/CLoSD'
    commit = subprocess.check_output(['git', '-C', str(source_root), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == original['upstream_commit']
    assert subprocess.check_output(['git', '-C', str(source_root), 'diff', '--name-only', 'HEAD', '--', 'closd'], text=True) == ''
    files = [ROOT / ('scripts/' + name) for name in (
        'reset_aligned_collect_v4.py', 'run_reset_alignment_checks_v4.py', 'analyze_reset_alignment_v4.py',
        'physics_guided_collect.py', 'physics_guided_core.py', 'physics_guided_lineage.py', 'r3_closd_isolated_entry.py')]
    files += [ROOT / 'configs/physics_guided_20261011.json', ROOT / 'configs/env-closd-baseline-20261010.json',
              OLD / 'source_manifest.json', OLD / 'frozen_thresholds.json', OUT / 'measurement_manifest.json']
    save(OUT / 'versions_before_execution.json', dict(time=time.time(), hashes={str(path): sha(path) for path in files},
        upstream_commit=commit, external_motion_inputs=original['external_motion_inputs'],
        **{key: sha(path) for key, path in weights.items()},
        reset_fix='Remove only collector refresh after official reset; preserve authoritative upstream reset cache, physics step and player initialization.',
        scope='Seven exported kinematic tensors and official reset caches; does not certify hidden PhysX state.',
        total_expected_attempts=216, environment_seed=11, generation_seeds=[11, 12, 13],
        selections='First two validation sources per family in manifest order; offsets -0.15,0,+0.15m; then all 24 validation sources.',
        output_policy='Initial full-state NPZ and licensed trajectories stay server-local; no Overleaf result writes.',
        prior_charged_seconds=sum(row['seconds'] for row in ledger_before())))
    save(OUT / 'budget_ledger.json', [])


def run_batch(label, mode, seed, manifest):
    destination = OUT / label
    ledger = json.loads((OUT / 'budget_ledger.json').read_text())
    if (destination / 'receipt.json').exists():
        assert any(row['label'] == label and row['exit_code'] == 0 for row in ledger)
        lineage = json.loads((destination / 'lineage.json').read_text())
        request = json.loads((OUT / (label + '_execution_request.json')).read_text())
        assert lineage['request'] == request
        assert all(sha(destination / name) == digest for name, digest in lineage['outputs'].items())
        assert request['version_hash'] == sha(OUT / 'versions_before_execution.json')
        return
    assert not destination.exists(), 'Failed attempt retained; create a new version, not an overwrite'
    previous = ledger_before()
    assert sum(row['seconds'] for row in previous + ledger) < 72 * 3600
    measurement_seconds = sum(row['seconds'] for row in previous + ledger if row['stage'] == 'measurement')
    remaining = min(600, 4 * 3600 - measurement_seconds)
    assert remaining > 60, 'Measurement budget exhausted'
    assert shutil.disk_usage('/data').free >= 20 * 1024 ** 3
    size = sum(path.stat().st_size for directory in PREVIOUS + [OUT] for path in directory.rglob('*') if path.is_file())
    assert size < 15 * 1024 ** 3
    resources = resource_guard()
    request_path = OUT / (label + '_execution_request.json')
    command = [str(PYTHON), '-u', str(ROOT / 'scripts/reset_aligned_collect_v4.py'),
        '--manifest', str(manifest), '--output', str(destination), '--split', 'val', '--mode', mode,
        '--seed', str(seed), '--thresholds', str(OLD / 'frozen_thresholds.json'), '--request-file', str(request_path)]
    save(request_path, dict(command=command, version_hash=sha(OUT / 'versions_before_execution.json'),
        mode=mode, seed=seed, round=0, manifest=sha(manifest), config=sha(ROOT / 'configs/physics_guided_20261011.json'),
        thresholds=sha(OLD / 'frozen_thresholds.json'), time=time.time()))
    save(OUT / 'state.json', dict(status='RUNNING', label=label, resources=resources, time=time.time()))
    start = time.monotonic()
    with (OUT / (label + '.log')).open('w') as stream:
        try:
            result = subprocess.run(command, cwd=ROOT / 'third_party/CLoSD', env=environment(),
                stdout=stream, stderr=subprocess.STDOUT, timeout=remaining)
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            exit_code = 124
    ledger.append(dict(stage='measurement', label=label, seconds=time.monotonic() - start, exit_code=exit_code))
    save(OUT / 'budget_ledger.json', ledger)
    if exit_code:
        raise RuntimeError('Measurement failed; all logs retained: ' + label)
    meta = json.loads((destination / 'metadata.json').read_text())
    assert meta['initial_reset_gate_pass'] and meta['initial_actual_reference_xy_max_m'] < 0.001
    assert sha(destination / 'initial_states.npz') == meta['initial_state_archive_sha256']
    print('RESET_V4_BATCH_PASS', label, flush=True)


def main(check):
    import fcntl
    prepare()
    with (OUT / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            manifest = OUT / 'measurement_manifest.json'
            jobs = [(mode + '_s' + str(seed), mode, seed, manifest)
                    for seed in (11, 12, 13) for mode in ('planned', 'native')]
            if check:
                jobs = jobs[:1]
            else:
                jobs += [('full_val_' + mode + '_s11', mode, 11, OLD / 'source_manifest.json') for mode in ('planned', 'native')]
            for args in jobs:
                run_batch(*args)
            status = 'WITNESS_BATCH_PASS_REQUIRES_CROSS_MODE_CHECK' if check else 'COLLECTION_COMPLETE_REQUIRES_INDEPENDENT_CHECK'
            save(OUT / 'state.json', dict(status=status, time=time.time()))
            if not check:
                subprocess.run([str(PYTHON), str(ROOT / 'scripts/analyze_reset_alignment_v4.py'), '--output', str(OUT)], check=True)
            print('RESET_V4_DOCUMENT_CHECK_PASS' if check else 'RESET_V4_COLLECTION_COMPLETE', flush=True)
        except Exception as error:
            save(OUT / 'state.json', dict(status='FAILED_RETAINED', error=repr(error), time=time.time()))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true')
    group.add_argument('--run', action='store_true')
    main(parser.parse_args().check)

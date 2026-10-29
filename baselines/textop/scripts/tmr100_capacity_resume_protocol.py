"""Strict provenance for a separate100-epoch resume; stdlib only."""
import json
import hashlib
import os
import pathlib
from ldm100k_verified_input import ROOT, owned, read, sha

PARENT = ROOT / 'evaluator_tmr_20261029'
HOME = pathlib.Path('/home/ouyanghao')
BASE_SPEC = ROOT / 'configs/tmr69d-evaluator-train-spec.json'
BASE_SHA = '2196a765dbaa61ba7265b9d798f99fe5c094a502fd34bd47869f0fe141360b13'
EMERGENCY = HOME / 'textop-tmr69d-evaluator-20261029/checkpoints/tmr_emergency.pth'
EMERGENCY_SHA = '8169a44260d0e238505e3a42c8cc859b214482e102d35241e054931d470f5d05'
RUNS = {role: 'tmr100_capacity_resume_' + role + '_v1' for role in ['author', 'document', 'train']}


def runtime_digest(spec):
    keys = ['bound_sources', 'executor_sha256', 'supervisor_sha256', 'original_failure_files', 'cpu_audit',
            'target_epochs', 'target_optimizer_updates', 'initial_optimizer_updates', 'remaining_optimizer_updates',
            'checkpoint_dir', 'checkpoint_upper_bytes', 'reserve_bytes', 'log_allowance_bytes']
    payload = {key: spec[key] for key in keys}
    payload.update(base_spec_sha256=BASE_SHA, original_emergency_sha256=EMERGENCY_SHA)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def workers():
    result = []
    names = ['train_tmr69d_evaluator.py', 'supervise_tmr69d_evaluator_training.py',
             'watch_tmr100_endpoint.py', 'supervise_tmr100_endpoint.py',
             'run_tmr100_capacity_resume.py', 'supervise_tmr100_capacity_resume.py']
    for proc in pathlib.Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid != 1013:
                continue
            argv = (proc / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            if any(str(ROOT / 'scripts' / name) in argv for name in names):
                st = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
                result.append(dict(pid=int(proc.name), start_ticks=st[19], state=st[0], argv=argv))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return result


def origin(spec):
    assert spec['runtime_binding_sha256'] == runtime_digest(spec)
    assert sha(owned(BASE_SPEC)) == BASE_SHA
    base = read(BASE_SPEC)
    assert sha(ROOT / 'scripts/train_tmr69d_evaluator.py') == base['executor_sha256']
    assert sha(ROOT / 'scripts/supervise_tmr69d_evaluator_training.py') == base['supervisor_sha256']
    assert spec['target_epochs'] == 100 and spec['target_optimizer_updates'] == 101500
    assert spec['initial_optimizer_updates'] == 83703 and spec['remaining_optimizer_updates'] == 17797
    for name, value in spec['bound_sources'].items():
        assert sha(owned(ROOT / name)) == value, name
    for name, value in base['bound_sources'].items():
        assert sha(owned(ROOT / name)) == value, name
    for path, value in spec['original_failure_files'].items():
        assert sha(owned(path)) == value
    old = read(PARENT / 'tmr69d_evaluator500_train_v1/receipt.json')
    assert old['status'] == 'FAILED_TMR69D_FORMAL_EVALUATOR_ALL_STATES_RETAINED'
    assert old['error'] == "RuntimeError('Owned TMR evaluator interrupted with signal 15')"
    assert old['formal_optimizer_updates'] == 83703 and old['attempted_optimizer_updates'] == 83704
    assert old['completed_epochs'] == 82
    item = old['emergency_checkpoint']
    ck = owned(EMERGENCY, HOME)
    assert item['path'] == str(ck) and item['sha256'] == EMERGENCY_SHA
    assert ck.stat().st_size == item['bytes'] == 192951813 and ck.stat().st_nlink == 1
    assert sha(ck) == EMERGENCY_SHA and item['exact_fullstate_deserialization'] and item['pending_validation'] is False
    audit = spec['cpu_audit']
    assert sha(owned(audit['path'])) == audit['sha256'] == '9c1188f26fba1355d39d9601246268865b7026d749cfd2e90de98c8c81d06d4d'
    ar = read(audit['path'])
    assert ar['status'] == 'TMR100_CAPACITY_EMERGENCY_CPU_METADATA_FINITE_VERIFIED'
    assert ar['output_writers_closed'] and ar['optimizer_updates'] == 83703 and ar['completed_epochs'] == 82
    assert ar['actual_optimizer_updates'] == ar['new_cuda_contexts'] == 0
    return base, item


def check_metadata(state, base):
    updates, completed = state['optimizer_updates'], state['completed_epochs']
    assert 83703 <= updates <= 101500 and 82 <= completed <= 100
    assert state['request_path'] == str(BASE_SPEC) and state['request_sha256'] == BASE_SHA
    assert state['runtime_binding_sha256'] == base['runtime_binding_sha256']
    assert state['input_signature_sha256'] == base['accepted_input_signature_sha256']
    ss = state['sampler']
    assert ss['count'] == 32462 and ss['steps'] == updates
    assert type(state['pending_validation']) is bool
    if state['pending_validation']:
        assert ss['cursor'] == ss['count'] and ss['epoch'] == completed
        assert updates == (completed + 1) * 1015
    elif ss['cursor'] == ss['count']:
        assert ss['epoch'] + 1 == completed and updates == completed * 1015
    else:
        assert ss['epoch'] == completed and 0 < ss['cursor'] < ss['count'] and ss['cursor'] % 32 == 0
        assert updates == completed * 1015 + ss['cursor'] // 32
    assert set(state['all_rng']) == {'python', 'numpy', 'torch', 'cuda'} and len(state['all_rng']['cuda']) == 1


def closed_job(role):
    run = RUNS[role]
    log = owned(PARENT / (run + '_supervision'))
    rp = owned(PARENT / run / 'receipt.json')
    paths = dict(receipt=rp, exit=log / 'exit.json', process=log / 'process_exit.json',
                 resources=log / 'resources.jsonl', outer=ROOT / 'receipts' / ('tmr100_capacity_' + run + '_outer_exit.json'))
    paths = {k: owned(v) for k, v in paths.items()}
    rec, ex, pr, outer = [read(paths[k]) for k in ['receipt', 'exit', 'process', 'outer']]
    assert outer['actual_returncode'] == 0
    for item in [ex, pr]:
        assert item['supervision_passed'] and item['child_reaped'] and item['child_exit_code'] == 0
        assert not item['timed_out'] and item['supervisor_error'] is None and not item['sample_errors']
        assert item['shutdown_actions'] == [] and not item.get('cleanup_error')
    assert ex['receipt_sha256'] == sha(rp) and ex['resource_sha256'] == sha(paths['resources'])
    assert ex['process_exit_sha256'] == sha(paths['process'])
    request = owned(ex['request_path'])
    s = read(request)
    assert sha(request) == ex['request_sha256'] == rec['request_sha256']
    assert ex['source_sha256'] == s['supervisor_sha256'] and rec['source_sha256'] == s['executor_sha256']
    assert rec['runtime_binding_sha256'] == s['runtime_binding_sha256'] and rec['output_writers_closed']
    rows = [json.loads(x) for x in paths['resources'].read_text().splitlines()]
    assert len(rows) == ex['samples_written'] == pr['samples_written'] and len(rows) >= 3
    assert rows[-1]['phase'] == 'after_exit' and rows[-1]['owned_group_processes'] == []
    return dict(receipt=rec, spec=s, sha256={k: sha(v) for k, v in paths.items()}, request_path=str(request), request_sha256=sha(request))


def accepted_witnesses(document=None):
    a, d = closed_job('author'), closed_job('document')
    for value in [a, d]:
        r = value['receipt']
        assert r['status'] == 'TMR100_CAPACITY_RESUME_REAL_GPU_WITNESS_VERIFIED'
        assert r['actual_diagnostic_optimizer_calls'] == 4 and r['selected_logical_diagnostic_steps'] == 3
        assert r['actual_new_formal_optimizer_updates'] == 0 and r['final_restored_optimizer_updates'] == 83703
        assert r['exact_source_model_adamw_rng_sampler_nextbatch_restored']
        assert r['exact_third_step_loss_fullstate_replay'] and r['source_inputs_and_checkpoint_unchanged']
    for key in ['runtime_binding_sha256', 'actual_input_signature_sha256', 'source_restored_content_sha256',
                'initial_next_batch_sha256', 'replayed_content_sha256', 'diagnostic_losses', 'diagnostic_batch_sha256',
                'source_restored_model_sha256']:
        assert a['receipt'][key] == d['receipt'][key], key
    if document is not None:
        assert document['status'] == 'TMR100_CAPACITY_RESUME_FRESH_DOCUMENT_VERIFIED'
        assert document['run'] == RUNS['document'] and document['actual_exit_code'] == 0 and document['findings'] == []
        for key in ['source_sha256', 'runtime_binding_sha256', 'source_restored_content_sha256',
                    'initial_next_batch_sha256', 'replayed_content_sha256']:
            assert document[key] == d['receipt'][key]
        assert document['actual_closure_sha256'] == d['sha256']
    return dict(author=a['sha256'], document=d['sha256'], runtime_binding_sha256=d['receipt']['runtime_binding_sha256'])

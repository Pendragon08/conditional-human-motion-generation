"""Closed 500-epoch evaluator provenance. Standard library only, no models."""
import os
import pathlib
from filtered15000_vae_completion import ROOT, read, sha, owned

TRAIN_RUN = 'tmr69d_evaluator500_train_v1'
PARENT = ROOT / 'evaluator_tmr_20261029'
TRAIN_SPEC = ROOT / 'configs/tmr69d-evaluator-train-spec.json'
TRAIN_SPEC_SHA = '2196a765dbaa61ba7265b9d798f99fe5c094a502fd34bd47869f0fe141360b13'
TRAIN_ENTRY_SHA = '04014d8f2e657e9c48827edd7fae817c3f01362d97dcdbd3a0e3dea092c604c2'
TRAIN_RUNTIME_SHA = 'c89a006b8730db6f767e0e3542506a3b9be928aaa12679f2ae6a1b0e8c4b7030'
HOME = pathlib.Path('/home/ouyanghao')
CHECKPOINT_DIR = HOME / 'textop-tmr69d-evaluator-20261029/checkpoints'


def digest(value):
    assert isinstance(value, str) and len(value) == 64
    assert all(x in '0123456789abcdef' for x in value)
    return value


def validate_training_receipt(rec, spec):
    assert rec['run'] == spec['run'] == TRAIN_RUN
    assert spec['formal_training_authorized'] is True
    assert spec['epochs'] == 500 and spec['expected_optimizer_updates'] == 507500
    assert rec['status'] == 'TMR69D_FORMAL_EVALUATOR_500_EPOCHS_FINISHED'
    assert rec['output_writers_closed'] is True
    assert rec['source_sha256'] == spec['executor_sha256'] == TRAIN_ENTRY_SHA
    assert rec['request_sha256'] == TRAIN_SPEC_SHA
    assert spec['runtime_binding_sha256'] == TRAIN_RUNTIME_SHA
    assert rec['formal_optimizer_updates'] == rec['attempted_optimizer_updates'] == 507500
    assert rec['completed_epochs'] == 500 and rec['skipped_updates'] == 0
    for name in ['generator_or_teacher_training_updates', 'original_validation_sources_used_for_gradient',
                 'internal_validation_optimizer_updates', 'actual_original_validation_source_reads']:
        assert type(rec[name]) is int and rec[name] == 0
    for name in ['exact_final_model_adamw_all_rng_sampler_nextbatch_disk_reload',
                 'model_sources_and_training_inputs_unchanged', 'original_class_method_AST_unchanged']:
        assert rec[name] is True
    assert rec['inference_export_and_heldout_semantic_metrics_accepted'] is False
    assert rec['diagnostic_weights_used_for_initialization'] is False
    assert rec['actual_input_signature_sha256'] == spec['accepted_input_signature_sha256']
    assert rec['initial_model_sha256'] != spec['diagnostic_initial_model_sha256']
    digest(rec['final_model_sha256'])
    checkpoints = rec['checkpoints']
    assert len(checkpoints) == 2
    for item, epoch in zip(checkpoints, [100, 500]):
        assert item['epoch'] == epoch and item['optimizer_updates'] == epoch * 1015
        assert item['path'] == str(CHECKPOINT_DIR / ('tmr_epoch_' + str(epoch) + '.pth'))
        assert type(item['bytes']) is int and 0 < item['bytes'] <= 201326592
        digest(item['sha256'])
        assert item['pending_validation'] is False and item['emergency'] is False
        assert item['exact_fullstate_deserialization'] is True
        assert item['production_model_adamw_rng_sampler_nextbatch_reloaded'] is True
    assert rec['selected_checkpoint'] == checkpoints[-1]
    return checkpoints[-1]


def validate_fullstate_metadata(state, spec):
    assert state['optimizer_updates'] == 507500 and state['completed_epochs'] == 500
    assert state['pending_validation'] is False
    assert state['request_path'] == str(TRAIN_SPEC) and state['request_sha256'] == TRAIN_SPEC_SHA
    assert state['runtime_binding_sha256'] == TRAIN_RUNTIME_SHA
    assert state['input_signature_sha256'] == spec['accepted_input_signature_sha256']
    sampler = state['sampler']
    assert sampler['count'] == sampler['cursor'] == 32462
    assert sampler['epoch'] == 499 and sampler['steps'] == 507500
    assert set(state['all_rng']) == {'python', 'numpy', 'torch', 'cuda'}
    assert isinstance(state['all_rng']['cuda'], list) and len(state['all_rng']['cuda']) == 1
    digest(state['next_batch_sha256'])
    assert isinstance(state['model'], dict) and state['model']
    assert isinstance(state['optimizer'], dict) and state['optimizer']['state']
    return dict(completed_epochs=500, optimizer_updates=507500, sampler_epoch=499,
                sampler_cursor=32462, pending_validation=False, saved_cuda_rng_devices=1,
                cuda_rng_restored=False, actual_new_optimizer_updates=0)


def fingerprint(path, home=False):
    p = owned(path, HOME if home else ROOT)
    assert p.is_file()
    value = p.stat()
    return dict(path=str(p), bytes=value.st_size, sha256=sha(p), uid=value.st_uid,
                device=value.st_dev, inode=value.st_ino, mtime_ns=value.st_mtime_ns)


def closed_evaluator_job(run, receipt_path, receipt_sha, supervisor_sha, request_sha, training=False):
    """Actual supervisor resources live in evaluator/run_supervision, not old LDM logs."""
    assert run.isascii() and run.isidentifier()
    parent = owned(PARENT)
    rp = owned(receipt_path)
    assert rp == parent / run / 'receipt.json' and sha(rp) == receipt_sha
    log = owned(parent / (run + '_supervision'))
    prefix = ('tmr_evaluator_train_' if training else 'tmr_frozen_cpu_') + run
    paths = dict(exit=log/'exit.json', process=log/'process_exit.json',
                 resources=log/'resources.jsonl', outer=ROOT/'receipts'/(prefix+'_outer_exit.json'))
    paths = {k: owned(p) for k, p in paths.items()}
    ex, pr, outer = [read(paths[k]) for k in ['exit', 'process', 'outer']]
    assert type(outer['actual_returncode']) is int and outer['actual_returncode'] == 0
    for proof in [ex, pr]:
        assert proof['supervision_passed'] is True and proof['child_reaped'] is True
        assert proof['child_exit_code'] == 0 and proof['timed_out'] is False
        assert proof['supervisor_error'] is None and proof['sample_errors'] == []
        assert proof['shutdown_actions'] == [] and not proof.get('cleanup_error')
    assert ex['receipt_sha256'] == receipt_sha and ex['source_sha256'] == supervisor_sha
    assert ex['request_sha256'] == request_sha
    assert ex['process_exit_sha256'] == sha(paths['process'])
    assert ex['resource_sha256'] == sha(paths['resources'])
    lines = paths['resources'].read_text().splitlines()
    assert len(lines) == ex['samples_written'] == pr['samples_written'] and len(lines) >= 3
    samples = [__import__('json').loads(line) for line in lines]
    assert samples[0]['phase'] == 'before_launch' and samples[-1]['phase'] == 'after_exit'
    assert samples[-1]['owned_group_processes'] == []
    for proc in pathlib.Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid != 1013:
                continue
            argv = (proc/'cmdline').read_bytes().decode().split('\0')
            entry = ['train_tmr69d_evaluator.py', 'supervise_tmr69d_evaluator_training.py'] if training else [
                'run_tmr69d_frozen_cpu.py', 'supervise_tmr69d_frozen_cpu.py']
            references_run = False
            if '--request' in argv:
                request_path = pathlib.Path(argv[argv.index('--request')+1])
                if request_path.is_file() and (ROOT in request_path.parents):
                    references_run = read(owned(request_path)).get('run') == run
            assert not (references_run and any(str(ROOT/'scripts'/name) in argv for name in entry)), argv
            if training:
                assert not any(str(ROOT/'scripts'/name) in argv for name in entry), argv
        except (ProcessLookupError, FileNotFoundError):
            pass
    return {name+'_sha256': sha(path) for name, path in paths.items()}


def accepted_training(binding=None):
    sp = owned(TRAIN_SPEC)
    assert sha(sp) == TRAIN_SPEC_SHA
    spec = read(sp)
    assert sha(owned(ROOT/'scripts/train_tmr69d_evaluator.py')) == spec['executor_sha256']
    assert sha(owned(ROOT/'scripts/supervise_tmr69d_evaluator_training.py')) == spec['supervisor_sha256']
    for name, value in spec['bound_sources'].items():
        assert sha(owned(ROOT/name)) == value, name
    for name in ['scope_receipt', 'numeric_receipt', 'text_receipt', 'import_receipt']:
        item = spec['inputs'][name]
        path = owned(item['path'])
        assert sha(path) == item['sha256'] and read(path)['output_writers_closed'] is True
    for name, value in spec['upstream_model_sha256'].items():
        assert sha(owned(pathlib.Path(spec['upstream_model_dir'])/(name+'.py'))) == value
    rp = owned(PARENT/TRAIN_RUN/'receipt.json')
    rec = read(rp)
    ck = validate_training_receipt(rec, spec)
    closure = closed_evaluator_job(TRAIN_RUN, rp, sha(rp), spec['supervisor_sha256'], sha(sp), training=True)
    for item in rec['checkpoints']:
        fp = fingerprint(item['path'], home=True)
        assert fp['bytes'] == item['bytes'] and fp['sha256'] == item['sha256']
    result = dict(run=TRAIN_RUN, spec_sha256=sha(sp), receipt_sha256=sha(rp), closure=closure,
                  selected_checkpoint=ck, final_model_sha256=rec['final_model_sha256'],
                  input_signature=rec['actual_input_signature'],
                  input_signature_sha256=rec['actual_input_signature_sha256'],
                  runtime_binding_sha256=TRAIN_RUNTIME_SHA)
    if binding is not None:
        assert binding == result, 'Frozen input binding changed'
    return dict(binding=result, spec=spec, receipt=rec)


def accepted_frozen_export(binding):
    sp = owned(ROOT/'configs/tmr69d-frozen-export-spec.json')
    assert sha(sp) == binding['spec_sha256']
    spec = read(sp)
    assert spec['authorized'] is True and spec['role'] == 'export'
    assert sha(owned(ROOT/'scripts/run_tmr69d_frozen_cpu.py')) == spec['executor_sha256']
    assert sha(owned(ROOT/'scripts/supervise_tmr69d_frozen_cpu.py')) == spec['supervisor_sha256']
    assert binding['run'] == spec['run'] == 'tmr69d_frozen_export_v1'
    train = accepted_training(spec['training_binding'])
    for name, value in spec['bound_sources'].items():
        assert sha(owned(ROOT/name)) == value
    rp = owned(PARENT/spec['run']/'receipt.json')
    assert sha(rp) == binding['receipt_sha256']
    rec = read(rp)
    assert rec['status'] == 'TMR69D_TRAINED_FROZEN_CPU_EXPORTED' and rec['output_writers_closed'] is True
    assert rec['training_binding'] == train['binding'] and rec['spec_sha256'] == sha(sp)
    assert rec['source_sha256'] == spec['executor_sha256']
    assert rec['full_model_and_cpu_adamw_restored_exact'] is True
    assert rec['independent_inference_reload_exact'] is True and rec['synthetic_mean_forward_exact'] is True
    assert rec['original_fullstate_unchanged'] is True and rec['cuda_rng_restored'] is False
    assert rec['training_optimizer_updates_performed'] == rec['new_cuda_contexts'] == 0
    assert rec['semantic_metrics_accepted'] is False
    item = rec['inference']
    assert item['path'] == spec['inference_path'] and 0 < item['bytes'] <= spec['inference_upper_bytes']
    fp = fingerprint(item['path'], home=True)
    assert fp['bytes'] == item['bytes'] and fp['sha256'] == item['sha256']
    closure = closed_evaluator_job(spec['run'], rp, sha(rp), spec['supervisor_sha256'], sha(sp))
    assert binding == dict(run=spec['run'], spec_sha256=sha(sp), receipt_sha256=sha(rp),
                           closure=closure, inference=item)
    return dict(binding=binding, receipt=rec, spec=spec, training=train)

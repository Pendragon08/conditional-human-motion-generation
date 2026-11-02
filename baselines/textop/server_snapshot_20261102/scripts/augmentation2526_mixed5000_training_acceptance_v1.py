"""Draft metadata gate for the unique completed mixed-source PPO run.

No model is loaded here. Do not call while training is active, and do not
confuse checkpoint file hashing with an independent full-state content audit.
"""
import json
import os
import pathlib

from filtered15000_vae_completion import ROOT, owned, read, sha

RUN = 'user_augmentation2526_mixed_ppo_train_v1'
REQUEST_SHA = 'b3652704ef2ce6bd4c5b2f433de37d17329b4a57b1ec1d1891fdd880b0ad04f9'
ENTRY_SHA = '150cf5d13d59a08f1d595e8b71b318284f22e418e5abac44121d26bb091f2d53'
SUPERVISOR_SHA = 'bd799982b0b59cf2c98cd3d1316b5c34b715529bbd0f924951800240de0cefc1'


def accepted_mixed_training(binding=None):
    if os.getuid() != 1013:
        raise ValueError('unexpected owner')
    parent = owned(ROOT / 'step34_20261029')
    receipts = owned(ROOT / 'receipts')
    out, log = owned(parent / RUN), owned(parent / (RUN + '_supervision'))
    paths = dict(request=receipts / (RUN + '_request.json'), receipt=out / 'receipt.json',
                 exit=log / 'exit.json', process=log / 'process_exit.json',
                 outer=receipts / (RUN + '_outer_exit.json'), resources=log / 'resources.jsonl')
    hashes = {key: sha(owned(path)) for key, path in paths.items()}
    if hashes['request'] != REQUEST_SHA:
        raise ValueError('unexpected training request')
    request, receipt, ex, process, outer = (read(paths[x]) for x in ('request', 'receipt', 'exit', 'process', 'outer'))
    driver = read(owned(receipts / (RUN + '_driver.json')))
    supervisor = read(owned(receipts / (RUN + '_supervisor.json')))
    if driver['request_sha256'] != REQUEST_SHA or supervisor['request_sha256'] != REQUEST_SHA:
        raise ValueError('training process binding mismatch')
    if outer['child_pid'] != supervisor['pid'] or process['supervisor_pid'] != supervisor['pid']:
        raise ValueError('training supervisor identity mismatch')
    for pid, start in ((driver['driver_pid'], driver['driver_start_ticks']),
                       (supervisor['pid'], supervisor['start_ticks'])):
        try:
            if (pathlib.Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19] == str(start):
                raise ValueError('training process still alive')
        except (FileNotFoundError, ProcessLookupError):
            pass
    for proc in pathlib.Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid == 1013 and str(paths['request']) in (proc / 'cmdline').read_bytes().decode().split('\0'):
                raise ValueError('training worker still alive')
        except (FileNotFoundError, ProcessLookupError):
            pass
    if outer['actual_returncode'] != 0:
        raise ValueError('training outer exit failed')
    for proof in (ex, process):
        if not (proof['supervision_passed'] is True and proof['child_reaped'] is True and
                proof['child_exit_code'] == 0 and proof['timed_out'] is False and not proof['sample_errors']):
            raise ValueError('training process failed')
    if (ex['request_sha256'] != REQUEST_SHA or ex['receipt_sha256'] != hashes['receipt'] or
            ex['process_exit_sha256'] != hashes['process'] or ex['resource_sha256'] != hashes['resources'] or
            ex['source_sha256'] != SUPERVISOR_SHA):
        raise ValueError('training closure mismatch')
    last = json.loads(paths['resources'].read_text().splitlines()[-1])
    if last['phase'] != 'after_exit' or last['owned_group_processes'] or last['owned_group_gpu_memory']:
        raise ValueError('training resources not closed')
    if (request['run'] != RUN or request['role'] != 'train' or request['executor_sha256'] != ENTRY_SHA or
            request['supervisor_sha256'] != SUPERVISOR_SHA or request['author_checkpoint'] is not None or
            request['policy']['training_authorized'] is not True or request['policy']['witnesses_authorized'] is not False):
        raise ValueError('unexpected training configuration')
    for group in ('bound_sources', 'external_sources'):
        for name, digest in request[group].items():
            path = ROOT / name
            if path.resolve() != path or path.lstat().st_uid != 1013 or sha(path) != digest:
                raise ValueError('training source changed')
    if receipt['status'] != 'STAGE4_MIXED_TRAIN_ONE_STAGE_PPO_FINISHED' or not receipt['output_writers_closed'] or not receipt['inputs_unchanged']:
        raise ValueError('training receipt incomplete')
    if receipt['request_sha256'] != REQUEST_SHA or receipt['source_sha256'] != ENTRY_SHA or receipt['input_library_binding'] != request['library_binding']:
        raise ValueError('training input provenance mismatch')
    if (receipt['formal_completed_iterations'] != 5000 or receipt['formal_optimizer_updates'] != 100000 or
            receipt['actual_optimizer_calls_in_process'] != 100000 or receipt['optimizer_attempts_in_process'] != 100000):
        raise ValueError('training target not reached')
    if (receipt['actual_original_train_clips'] != 5551 or receipt['generated_training_clips'] != 2526 or
            receipt['original_VAL_motions_in_training'] != 0 or receipt['original_reference_environment_steps'] <= 0 or
            receipt['generated_reference_environment_steps'] <= 0):
        raise ValueError('mixed training exposure not established')
    if receipt['old_teacher_training_updates'] != 0 or receipt['VAE_updates'] != 0 or receipt['LDM_updates'] != 0:
        raise ValueError('unexpected frozen model updates')
    checkpoints = receipt['checkpoints']
    names = ['completed_' + str(i) + '.private.pth' for i in (1000, 2000, 3000, 4000, 5000)]
    if len(checkpoints) != 5 or [pathlib.Path(x['path']).name for x in checkpoints] != names:
        raise ValueError('training checkpoints incomplete')
    for item, name in zip(checkpoints, names):
        path = owned(item['path'])
        if (path != out / name or path.stat().st_size != item['bytes'] or
                not 0 < item['bytes'] <= request['state_slot_bytes'] or sha(path) != item['sha256'] or
                not isinstance(item['content_sha256'], str) or len(item['content_sha256']) != 64):
            raise ValueError('training checkpoint bytes mismatch')
    actual = dict(kind='MIXED_PPO5000_FULLSTATE_CONTENT_UNAUDITED', run=RUN,
                  request_sha256=REQUEST_SHA, closure_sha256=hashes,
                  input_library_binding=request['library_binding'], checkpoint=checkpoints[4],
                  fullstate_content_independently_audited=False)
    if binding is not None and binding != actual:
        raise ValueError('training binding changed')
    return dict(binding=actual, request=request, receipt=receipt)

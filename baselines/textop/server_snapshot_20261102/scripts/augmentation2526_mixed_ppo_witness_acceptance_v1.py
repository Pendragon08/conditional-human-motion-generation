"""Draft gate for real mixed PPO diagnostic closures; no model is loaded.

Not deployable until both diagnostics exist and the full request/launch flow
has been independently checked against their real receipts.
"""
import json
import os
import pathlib

from filtered15000_vae_completion import ROOT, owned, read, sha


def accepted_mixed_ppo_witnesses(binding):
    if os.getuid() != 1013 or set(binding) != {'library_binding', 'author', 'document'}:
        raise ValueError('invalid mixed witness binding')
    parent = owned(ROOT / 'step34_20261029')
    receipts = owned(ROOT / 'receipts')
    reports = {}
    for role in ('author', 'document'):
        run = 'user_augmentation2526_mixed_ppo_' + role + '_v1'
        out = owned(parent / run)
        log = owned(parent / (run + '_supervision'))
        paths = dict(request=receipts / (run + '_request.json'),
                     receipt=out / 'receipt.json', exit=log / 'exit.json',
                     process=log / 'process_exit.json',
                     outer=receipts / (run + '_outer_exit.json'),
                     resources=log / 'resources.jsonl')
        hashes = {key: sha(owned(path)) for key, path in paths.items()}
        if hashes != binding[role]:
            raise ValueError('mixed witness evidence changed')
        request, receipt, exit_, process, outer = (read(paths[x]) for x in
                                                   ('request', 'receipt', 'exit', 'process', 'outer'))
        driver = read(owned(receipts / (run + '_driver.json')))
        supervisor = read(owned(receipts / (run + '_supervisor.json')))
        if driver['request_sha256'] != hashes['request'] or supervisor['request_sha256'] != hashes['request']:
            raise ValueError('mixed witness identity mismatch')
        if outer['child_pid'] != supervisor['pid'] or process['supervisor_pid'] != supervisor['pid']:
            raise ValueError('mixed witness supervisor mismatch')
        for pid, start in ((driver['driver_pid'], driver['driver_start_ticks']),
                           (supervisor['pid'], supervisor['start_ticks'])):
            try:
                if (pathlib.Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19] == str(start):
                    raise ValueError('mixed witness process remains live')
            except (FileNotFoundError, ProcessLookupError):
                pass
        for proc in pathlib.Path('/proc').iterdir():
            if not proc.name.isdigit() or int(proc.name) == os.getpid():
                continue
            try:
                if proc.stat().st_uid == 1013 and str(paths['request']) in (proc / 'cmdline').read_bytes().decode().split('\0'):
                    raise ValueError('mixed witness worker remains live')
            except (FileNotFoundError, ProcessLookupError):
                pass
        if outer['actual_returncode'] != 0:
            raise ValueError('mixed witness outer exit')
        for proof in (exit_, process):
            if not (proof['supervision_passed'] is True and proof['child_reaped'] is True and
                    proof['child_exit_code'] == 0 and proof['timed_out'] is False and not proof['sample_errors']):
                raise ValueError('mixed witness process failed')
        if (exit_['request_sha256'] != hashes['request'] or exit_['receipt_sha256'] != hashes['receipt'] or
                exit_['process_exit_sha256'] != hashes['process'] or exit_['resource_sha256'] != hashes['resources']):
            raise ValueError('mixed witness closure mismatch')
        last = json.loads(paths['resources'].read_text().splitlines()[-1])
        if last['phase'] != 'after_exit' or last['owned_group_processes'] or last['owned_group_gpu_memory']:
            raise ValueError('mixed witness resources not closed')
        if request['role'] != role or request['run'] != run or request['library_binding'] != binding['library_binding']:
            raise ValueError('mixed witness input mismatch')
        if receipt['status'] != 'STAGE4_MIXED_REAL_PPO_RECOVERY_WITNESS_FINISHED' or not receipt['output_writers_closed'] or not receipt['inputs_unchanged']:
            raise ValueError('mixed witness receipt incomplete')
        if receipt['request_sha256'] != hashes['request'] or receipt['input_library_binding'] != binding['library_binding']:
            raise ValueError('mixed witness request mismatch')
        if receipt['original_VAL_motions_in_training'] != 0 or receipt['actual_original_train_clips'] != 5551 or receipt['generated_training_clips'] != 2526:
            raise ValueError('mixed witness source counts mismatch')
        if receipt['original_reference_environment_steps'] <= 0 or receipt['generated_reference_environment_steps'] <= 0:
            raise ValueError('both source kinds must be sampled')
        if receipt['formal_updates'] != 0 or receipt['formal_optimizer_updates'] != 0 or receipt['diagnostic_logical_iterations'] != 2:
            raise ValueError('mixed witness iteration mismatch')
        if not receipt['recorded_rollout_replay']['real_recorded_rollout_exact_replay'] or not receipt['exact_model_Adam_normalizers_all_RNG_ADS_restore_verified']:
            raise ValueError('mixed witness recovery not verified')
        for item in receipt['checkpoints']:
            path = owned(item['path'])
            if path.parent != out or path.stat().st_size != item['bytes'] or not 0 < item['bytes'] <= request['state_slot_bytes'] or sha(path) != item['sha256']:
                raise ValueError('mixed witness checkpoint mismatch')
        reports[role] = (request, receipt)
    author_request, author = reports['author']
    document_request, document = reports['document']
    if author['diagnostic_optimizer_calls'] != 80 or document['diagnostic_optimizer_calls'] != 60:
        raise ValueError('mixed witness optimizer count mismatch')
    if len(author['checkpoints']) != 2 or len(document['checkpoints']) != 1:
        raise ValueError('mixed witness checkpoint count mismatch')
    if not document['cross_process_author_checkpoint_restored'] or document['source_optimizer_updates'] != 20:
        raise ValueError('mixed witness cross-process recovery missing')
    if document_request['author_checkpoint'] != author['checkpoints'][0] or document['author_checkpoint_sha256'] != author['checkpoints'][0]['sha256']:
        raise ValueError('mixed witness author checkpoint mismatch')
    if document['author_restored_fullstate_content_sha256'] != author['checkpoints'][0]['content_sha256']:
        raise ValueError('mixed witness restored state content mismatch')
    if author['initialized_model_sha256'] != document['initialized_model_sha256']:
        raise ValueError('mixed witness fresh initialization mismatch')
    if (author_request['executor_sha256'] != document_request['executor_sha256'] or
            author_request['supervisor_sha256'] != document_request['supervisor_sha256'] or
            author_request['bound_sources'] != document_request['bound_sources'] or
            author_request['external_sources'] != document_request['external_sources']):
        raise ValueError('mixed witness sources differ')
    return binding

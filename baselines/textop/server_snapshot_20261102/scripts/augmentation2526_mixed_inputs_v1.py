"""Draft real-source acceptance for mixed original/generated TRAIN PPO.

This is not deployed or executable until the generated terminal audit has
actually succeeded. The caller must hold the shared model lock.
"""
import json
import os
import pathlib

from filtered15000_vae_completion import ROOT, owned, read, sha
from step4_train_only_ppo_inputs import accepted_library
from augmentation2526_integrity_fk_inputs_v1 import accepted_generated_fk_inputs
from augmentation2526_mixed_policy_contract_v1 import validate_records

ORIGINAL_BINDING = dict(
    receipt_sha256='375b16c3b8d919561e59cea8830ab071f2a128f39bf76363474cf0c37eb5ddde',
    closure_sha256=dict(exit='1f03226b2a46f3d7d6f36bab243507c256b975e29fbc909544033623e1d85fd5',
                        process='b5834b6ce7d0309011a28460052bec1c7418a91135e02681834b390b5c5cad15',
                        outer='4ce907daf27d679864b769f47613c84e61f6e35e48a697680126aa3a388e2dc0'),
    manifest_sha256='8d43f71c03609466b0af2c0bdd43d714c12db7e240a6aaaecfe0d5d2b9f6a6aa')


def accepted_mixed_library(binding):
    assert os.getuid() == 1013 and set(binding) == {'original', 'generated_audit'}
    assert binding['original'] == ORIGINAL_BINDING
    original, original_rows = accepted_library(binding['original'])
    parent = owned(ROOT / 'step34_20261029')
    run = 'user_augmentation2526_fk_terminal_audit_v1'
    out = owned(parent / run)
    log = owned(parent / (run + '_supervision'))
    paths = dict(request=ROOT / 'receipts' / (run + '_request.json'),
                 receipt=out / 'receipt.json', exit=log / 'exit.json',
                 process=log / 'process_exit.json', outer=ROOT / 'receipts' / (run + '_outer_exit.json'),
                 resources=log / 'resources.jsonl', report=out / 'audit.private.json')
    actual = {key: sha(owned(path)) for key, path in paths.items()}
    assert actual == binding['generated_audit']
    request, receipt, ex, process, outer = (read(paths[x]) for x in ('request', 'receipt', 'exit', 'process', 'outer'))
    driver = read(ROOT / 'receipts' / (run + '_driver.json'))
    supervisor = read(ROOT / 'receipts' / (run + '_supervisor.json'))
    assert driver['request_sha256'] == supervisor['request_sha256'] == actual['request']
    assert outer['child_pid'] == supervisor['pid'] == process['supervisor_pid']
    for pid, start in ((driver['driver_pid'], driver['driver_start_ticks']),
                       (supervisor['pid'], supervisor['start_ticks'])):
        process_path = pathlib.Path('/proc') / str(pid)
        try:
            assert (process_path / 'stat').read_text().rsplit(')', 1)[1].split()[19] != str(start)
        except (FileNotFoundError, ProcessLookupError):
            pass
    assert request['schema'] == 'textop.augmentation2526_fk_terminal_audit.v1' and request['authorized'] is True
    assert request['run'] == run and receipt['run'] == run
    assert receipt['status'] == 'GENERATED_TRAIN_FK_TERMINAL_AUDIT_FINISHED' and receipt['output_writers_closed'] is True
    assert receipt['clips'] == 2526 and receipt['frames'] == 2668946 and receipt['report_sha256'] == actual['report']
    assert receipt['request_sha256'] == ex['request_sha256'] == actual['request']
    assert outer['actual_returncode'] == 0
    for proof in (ex, process):
        assert proof['supervision_passed'] is True and proof['child_reaped'] is True
        assert proof['child_exit_code'] == 0 and proof['timed_out'] is False and proof['sample_errors'] == []
    assert ex['receipt_sha256'] == actual['receipt'] and ex['process_exit_sha256'] == actual['process']
    assert ex['resource_sha256'] == actual['resources'] and paths['report'].stat().st_size == receipt['report_bytes']
    last = json.loads(paths['resources'].read_text().splitlines()[-1])
    assert last['phase'] == 'after_exit' and last['owned_group_processes'] == last['owned_group_gpu_memory'] == []
    for proc in pathlib.Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid == 1013:
                assert str(paths['request']) not in (proc / 'cmdline').read_bytes().decode().split('\0')
        except (FileNotFoundError, ProcessLookupError):
            pass
    audit = read(paths['report'])
    assert audit['status'] == 'GENERATED_TRAIN_FK_LIBRARY_ARRAYS_AUDITED'
    assert audit['run'] == 'user_augmentation2526_fk_library_v1'
    assert audit['request_sha256'] == '32d59893d9d688beaf51ae475e84e3964fc89df1d566a9c339078f88b2fd11be'
    assert audit['source_split'] == 'TRAIN' and audit['original_VAL_motions'] == 0
    assert audit['clips'] == 2526 and audit['frames'] == 2668946 and audit['control_intervals'] == 2666420
    integrity_binding = dict(integrity_report_sha256='0bb020c4182227be0c119d305bd9d8f679c5fac9f9f801087667a375528d8394',
                             production_request_sha256='419011b4909eea18dfe8eb7a698b8591d0b52d7261d5612875eea941c69ec806')
    metadata = accepted_generated_fk_inputs(integrity_binding)
    plan = metadata['plan']['inputs']
    assert len(audit['items']) == len(plan) == 2526
    originals = []
    for row in original_rows:
        originals.append(dict(kind='original_TRAIN', source_split='train', file=row['file'],
                              sha256=row['sha256'], frames=row['frames']))
    generated = []
    for index, (item, clip) in enumerate(zip(audit['items'], plan)):
        assert item['index'] == index and item['frames'] == clip['frames']
        assert item['source_family'] == clip['source_family'] and item['source_sha256'] == clip['source_sha256']
        path = owned(item['file'])
        assert path.parent == parent / 'user_augmentation2526_fk_library_v1'
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
        generated.append(dict(kind='generated_TRAIN', source_split='train', file=str(path),
                              sha256=item['sha256'], frames=item['frames'], generated_index=index))
    assert validate_records(originals, generated)['total_clips'] == 8077
    files = [x['file'] for x in originals] + [x['file'] for x in generated]
    accepted = dict(status='STAGE4_ORIGINAL_AND_GENERATED_TRAIN_LIBRARY_ACCEPTED',
                    actual_original_train_clips=5551, actual_original_train_families=5361,
                    actual_generated_clips=2526, original_VAL_motions_in_training=0,
                    numerical_failures_unaccepted=0, motion_files=files,
                    generated_start_index=5551, generated_audit_binding=actual,
                    final_generator_model_hash_verified=False)
    return accepted, originals + generated

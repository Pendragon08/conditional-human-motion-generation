"""Draft metadata acceptance of two independently compared real FK witnesses.

The comparison result must be authenticated by the launcher. This function
revalidates its immutable server-side inputs without loading a model or SDK.
"""
import json
import os
import pathlib

from filtered15000_vae_completion import ROOT, owned, read, sha


def accepted_fk_witnesses(binding):
    assert os.getuid() == 1013 and binding['status'] == 'TWO_FK_WITNESSES_ARRAYS_EQUAL'
    assert binding['frames'] == 1057 and binding['control_intervals'] == 1056
    assert binding['full_library_converted'] is False and binding['physical_tracking_accepted'] is False
    assert len(binding['witnesses']) == 2
    parent = owned(ROOT / 'step34_20261029')
    for row, role, run in zip(binding['witnesses'], ('author', 'document'),
                              ('user_augmentation2526_fk_author_v2', 'user_augmentation2526_fk_document_v3')):
        assert row['role'] == role
        out = owned(parent / run)
        log = owned(parent / (run + '_supervision'))
        paths = dict(request=ROOT / 'receipts' / (run + '_request.json'),
                     receipt=out / 'receipt.json', exit=log / 'exit.json',
                     process=log / 'process_exit.json', outer=ROOT / 'receipts' / (run + '_outer_exit.json'),
                     resource=log / 'resources.jsonl', manifest=out / 'references.private.jsonl')
        for key, path in paths.items():
            assert sha(owned(path)) == row[key + '_sha256']
        rec, request, ex, process, outer = (read(paths[k]) for k in ('receipt', 'request', 'exit', 'process', 'outer'))
        assert rec['role'] == role and rec['status'] == 'AUGMENTATION_TRAIN_ISAAC_REFERENCES_FINISHED'
        assert rec['output_writers_closed'] and rec['exported_clips'] == 1 and rec['exported_frames'] == 1057
        assert outer['actual_returncode'] == 0
        for proof in (ex, process):
            assert proof['supervision_passed'] and proof['child_reaped'] and proof['child_exit_code'] == 0
            assert not proof['timed_out'] and proof['sample_errors'] == []
        assert ex['receipt_sha256'] == row['receipt_sha256'] and ex['process_exit_sha256'] == row['process_sha256']
        assert ex['resource_sha256'] == row['resource_sha256'] and request['urdf_sha256'] == row['asset_sha256']
        assert rec['joint_names'] == row['joint_names'] and rec['body_names'] == row['body_names']
        lines = owned(paths['manifest']).read_text().splitlines()
        assert len(lines) == 1
        item = json.loads(lines[0])
        artifact = owned(item['file'])
        assert artifact.parent == out and sha(artifact) == row['artifact_sha256'] == item['sha256']
        assert sha(owned(item['source_file'])) == row['source_sha256'] == item['source_sha256']
        assert item['array_sha256'] == row['array_sha256']
        for proc in pathlib.Path('/proc').iterdir():
            if not proc.name.isdigit() or int(proc.name) == os.getpid():
                continue
            try:
                if proc.stat().st_uid == 1013:
                    assert str(paths['request']) not in (proc / 'cmdline').read_bytes().decode().split('\0')
            except (FileNotFoundError, ProcessLookupError):
                pass
    a, b = binding['witnesses']
    assert a['source_sha256'] == b['source_sha256'] and a['asset_sha256'] == b['asset_sha256']
    assert a['array_sha256'] == b['array_sha256'] and a['joint_names'] == b['joint_names'] and a['body_names'] == b['body_names']
    return binding

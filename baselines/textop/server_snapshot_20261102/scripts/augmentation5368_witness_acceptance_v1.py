"""Bind two actual closed augmentation witnesses before production.

Call only while the shared model lock is held. This reads the two completed
private numerical outputs with the standard library; it does not load a model.
Independent document-agent evidence must be bound separately by the launcher.
"""
import ast
import hashlib
import json
import math
import struct
import zipfile

from filtered15000_vae_completion import ROOT, owned, read, sha


def _arrays(path, item):
    result = {}
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {'qpos_29.npy', 'contact.npy', 'fps.npy'}
        for filename in archive.namelist():
            raw = archive.read(filename)
            assert raw[:6] == b'\x93NUMPY' and raw[6:8] in (b'\x01\x00', b'\x02\x00')
            width = 2 if raw[6] == 1 else 4
            length = struct.unpack('<H' if width == 2 else '<I', raw[8:8+width])[0]
            header = ast.literal_eval(raw[8+width:8+width+length].decode('latin1').strip())
            data = raw[8+width+length:]
            assert header['fortran_order'] is False
            key = filename[:-4]
            n = item['proof']['actual_pose_frames']
            expected = {'qpos_29': ((n, 36), '<f4'), 'contact': ((n, 2), '<f4'), 'fps': ((), '<i8')}[key]
            assert (header['shape'], header['descr']) == expected
            count = 1
            for dim in header['shape']:
                count *= dim
            assert len(data) == count * (8 if key == 'fps' else 4)
            if key == 'fps':
                assert struct.unpack('<q', data)[0] == 50
            else:
                assert all(math.isfinite(value[0]) for value in struct.iter_unpack('<f', data))
            result[key] = hashlib.sha256(data).hexdigest()
            assert result[key] == item['array_sha256'][key]
    return result


def accepted_witnesses(binding=None):
    parent = owned(ROOT / 'step34_20261029')
    witnesses = []
    for role in ('author', 'document'):
        run = 'user_augmentation5368_' + role + '_v1'
        out = owned(parent / run)
        log = owned(parent / (run + '_supervision'))
        prefix = 'augmentation5368_' + run
        request = owned(ROOT / 'receipts' / (prefix + '_request.json'))
        rec = read(owned(out / 'receipt.json'))
        ex = read(owned(log / 'exit.json'))
        proc = read(owned(log / 'process_exit.json'))
        outer_path = owned(ROOT / 'receipts' / (prefix + '_outer_exit.json'))
        assert read(outer_path)['actual_returncode'] == 0
        assert rec['run'] == run and rec['role'] == role and rec['status'] == 'TRAIN_AUGMENTATION_FINISHED'
        assert rec['completed_clips'] == 1 and rec['output_writers_closed'] is True
        assert rec['real_generated_reference'] is True and rec['training_optimizer_updates'] == 0
        for proof in (ex, proc):
            assert proof['supervision_passed'] is True and proof['child_reaped'] is True
            assert proof['child_exit_code'] == 0 and proof['timed_out'] is False and proof['sample_errors'] == []
        assert rec['request_sha256'] == sha(request) == ex['request_sha256']
        assert ex['receipt_sha256'] == sha(out / 'receipt.json')
        assert ex['process_exit_sha256'] == sha(log / 'process_exit.json')
        assert ex['resource_sha256'] == sha(log / 'resources.jsonl')
        last = json.loads((log / 'resources.jsonl').read_text().splitlines()[-1])
        assert last['phase'] == 'after_exit' and last['owned_group_processes'] == last['owned_group_gpu_memory'] == []
        journal = owned(out / 'clips.private.jsonl')
        assert sha(journal) == rec['journal_sha256']
        rows = [json.loads(x) for x in journal.read_text().splitlines()]
        assert len(rows) == 2 and rows[0]['status'] == 'STARTED' and rows[1]['status'] == 'COMPLETED'
        item = rows[1]
        path = owned(item['file'])
        assert path.parent == out and path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
        witnesses.append(dict(role=role, receipt_sha256=sha(out / 'receipt.json'), request_sha256=sha(request),
            exit_sha256=sha(log / 'exit.json'), process_sha256=sha(log / 'process_exit.json'),
            outer_sha256=sha(outer_path), journal_sha256=sha(journal), artifact_sha256=item['sha256'],
            array_sha256=_arrays(path, item), proof=item['proof'],
            generator_model_sha256=rec['generator_model_sha256'], vae_model_sha256=rec['vae_model_sha256']))
    a, b = witnesses
    assert a['array_sha256'] == b['array_sha256'] and a['proof'] == b['proof']
    assert a['generator_model_sha256'] == b['generator_model_sha256'] and a['vae_model_sha256'] == b['vae_model_sha256']
    proof = a['proof']
    assert proof['actual_pose_frames'] == proof['control_intervals'] + 1 and proof['generated_terminal_pose_included'] is True
    assert proof['parameter_updates'] == 0 and proof['teacher_loaded'] is False and proof['future_posterior_encoded'] is False
    actual = dict(status='TWO_REAL_AUGMENTATION_WITNESSES_NUMERICALLY_EQUAL', witnesses=witnesses,
                  arrays_finite=True, real_model_samples=2, optimizer_updates=0)
    if binding is not None:
        assert binding == actual
    return actual

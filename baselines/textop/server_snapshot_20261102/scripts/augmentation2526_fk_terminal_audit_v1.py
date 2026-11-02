"""Draft terminal audit for all generated TRAIN FK arrays.

This module is not deployed or authorized. Its caller must independently hold
the shared lock and verify process closure. No model or simulator is loaded.
"""
import ast
import hashlib
import json
import math
import os
import pathlib
import struct
import time
import zipfile

from filtered15000_vae_completion import ROOT, owned, read, sha
from augmentation2526_integrity_fk_inputs_v1 import accepted_generated_fk_inputs
from augmentation2526_fk_witness_acceptance_v1 import accepted_fk_witnesses

RUN = 'user_augmentation2526_fk_library_v1'
REQUEST_SHA = '32d59893d9d688beaf51ae475e84e3964fc89df1d566a9c339078f88b2fd11be'


def canonical_array(raw, expected_shape, expected_dtype, expected_digest):
    if raw[:6] != b'\x93NUMPY' or raw[6:8] not in (b'\x01\x00', b'\x02\x00'):
        raise ValueError('invalid NPY header')
    width = 2 if raw[6] == 1 else 4
    length = struct.unpack('<H' if width == 2 else '<I', raw[8:8 + width])[0]
    header = ast.literal_eval(raw[8 + width:8 + width + length].decode('latin1').strip())
    data = raw[8 + width + length:]
    if header['shape'] != expected_shape or type(header['fortran_order']) is not bool:
        raise ValueError('invalid shape or storage order')
    formats = {'<f4': ('f', 4, 'float32'), '<f8': ('d', 8, 'float64'), '<i8': ('q', 8, 'int64')}
    if header['descr'] not in formats:
        raise ValueError('invalid dtype')
    fmt, itemsize, dtype = formats[header['descr']]
    if dtype != expected_dtype:
        raise ValueError('dtype differs from manifest')
    count = 1
    for dim in expected_shape:
        count *= dim
    if len(data) != count * itemsize:
        raise ValueError('incorrect array byte count')
    if header['fortran_order']:
        strides = []
        stride = 1
        for dim in expected_shape:
            strides.append(stride)
            stride *= dim
        result = bytearray(len(data))
        for index in range(count):
            rest = index
            coords = [0] * len(expected_shape)
            for axis in range(len(expected_shape) - 1, -1, -1):
                coords[axis] = rest % expected_shape[axis]
                rest //= expected_shape[axis]
            src = sum(coords[axis] * strides[axis] for axis in range(len(expected_shape)))
            result[index * itemsize:(index + 1) * itemsize] = data[src * itemsize:(src + 1) * itemsize]
        data = bytes(result)
    if hashlib.sha256(data).hexdigest() != expected_digest:
        raise ValueError('array hash mismatch')
    return data, fmt


def audit_library(max_seconds=7200):
    if os.getuid() != 1013:
        raise ValueError('unexpected user')
    parent = owned(ROOT / 'step34_20261029')
    out, log = owned(parent / RUN), owned(parent / (RUN + '_supervision'))
    request = owned(ROOT / 'receipts' / (RUN + '_request.json'))
    outer = owned(ROOT / 'receipts' / (RUN + '_outer_exit.json'))
    if sha(request) != REQUEST_SHA or read(outer)['actual_returncode'] != 0:
        raise ValueError('wrong request or outer exit')
    spec = read(request)
    receipt, ex, process = read(out / 'receipt.json'), read(log / 'exit.json'), read(log / 'process_exit.json')
    driver = read(ROOT / 'receipts' / (RUN + '_driver.json'))
    supervisor = read(ROOT / 'receipts' / (RUN + '_supervisor.json'))
    if driver['request_sha256'] != REQUEST_SHA or supervisor['request_sha256'] != REQUEST_SHA:
        raise ValueError('producer process binding mismatch')
    if read(outer)['child_pid'] != supervisor['pid'] or process['supervisor_pid'] != supervisor['pid']:
        raise ValueError('producer supervisor identity mismatch')
    for pid, start in ((driver['driver_pid'], driver['driver_start_ticks']),
                       (supervisor['pid'], supervisor['start_ticks'])):
        proc_path = pathlib.Path('/proc') / str(pid)
        try:
            if (proc_path / 'stat').read_text().rsplit(')', 1)[1].split()[19] == str(start):
                raise ValueError('producer driver or supervisor still alive')
        except (FileNotFoundError, ProcessLookupError):
            pass
    if receipt['status'] != 'AUGMENTATION_TRAIN_ISAAC_REFERENCES_FINISHED' or receipt['output_writers_closed'] is not True:
        raise ValueError('library did not close successfully')
    if (receipt['exported_clips'], receipt['exported_frames'], receipt['control_intervals']) != (2526, 2668946, 2666420):
        raise ValueError('wrong library coverage')
    for proof in (ex, process):
        if not (proof['supervision_passed'] is True and proof['child_reaped'] is True and proof['child_exit_code'] == 0 and not proof['timed_out'] and proof['sample_errors'] == []):
            raise ValueError('supervised process failed')
    if ex['request_sha256'] != REQUEST_SHA or ex['receipt_sha256'] != sha(out / 'receipt.json') or ex['process_exit_sha256'] != sha(log / 'process_exit.json') or ex['resource_sha256'] != sha(log / 'resources.jsonl'):
        raise ValueError('supervisor closure mismatch')
    last = json.loads((log / 'resources.jsonl').read_text().splitlines()[-1])
    if last['phase'] != 'after_exit' or last['owned_group_processes'] or last['owned_group_gpu_memory']:
        raise ValueError('processes remain after exit')
    for proc in pathlib.Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid == 1013 and str(request) in (proc / 'cmdline').read_bytes().decode().split('\0'):
                raise ValueError('producer still running')
        except (FileNotFoundError, ProcessLookupError):
            pass
    actual = accepted_generated_fk_inputs(spec['audit_binding'])
    if actual['binding'] != spec['input_binding'] or accepted_fk_witnesses(spec['witness_binding']) != spec['witness_binding']:
        raise ValueError('input or witness binding mismatch')
    if receipt['run'] != RUN or receipt['role'] != 'library' or receipt['request_sha256'] != REQUEST_SHA or receipt['source_sha256'] != spec['executor_sha256'] or receipt['input_binding'] != spec['input_binding']:
        raise ValueError('producer identity or source mismatch')
    if receipt['generated_training_clips'] != 2526 or receipt['original_VAL_motions_in_training'] != 0:
        raise ValueError('producer training split mismatch')
    if receipt['joint_names'] != actual['original_joint_names'] or receipt['body_names'] != actual['original_body_names']:
        raise ValueError('robot joint or body order changed')
    witness_manifest = owned(parent / 'user_augmentation2526_fk_author_v2/references.private.jsonl')
    if sha(witness_manifest) != spec['witness_binding']['witnesses'][0]['manifest_sha256']:
        raise ValueError('witness manifest changed')
    witness_rows = witness_manifest.read_text().splitlines()
    if len(witness_rows) != 1:
        raise ValueError('invalid witness manifest')
    witness_dtypes = json.loads(witness_rows[0])['array_dtypes']
    manifest = owned(out / 'references.private.jsonl')
    manifest_before = manifest.stat()
    manifest_ident = (manifest_before.st_dev, manifest_before.st_ino, manifest_before.st_size,
                      manifest_before.st_mtime_ns, manifest_before.st_ctime_ns)
    manifest_sha = sha(manifest)
    if manifest_sha != receipt['private_manifest']['sha256']:
        raise ValueError('manifest mismatch')
    rows = [json.loads(x) for x in manifest.read_text().splitlines()]
    clips = actual['plan']['inputs']
    if len(rows) != len(clips) or len(rows) != 2526:
        raise ValueError('incomplete manifest')
    started = time.monotonic()
    files = set()
    items = []
    for index, (row, clip) in enumerate(zip(rows, clips)):
        if time.monotonic() - started > max_seconds:
            raise TimeoutError('terminal audit exceeded bound')
        for key, value in clip.items():
            if row.get(key) != value:
                raise ValueError('FK row differs from input plan: ' + key)
        path = owned(row['file'])
        if path.parent != out or path in files:
            raise ValueError('invalid or repeated output')
        files.add(path)
        before = path.stat()
        ident = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        if before.st_size != row['bytes'] or before.st_size > clip['frames'] * 2152 + 8192 or sha(path) != row['sha256']:
            raise ValueError('FK output size or file hash mismatch')
        frames = clip['frames']
        shapes = {'fps': (1,), 'joint_pos': (frames, 29), 'joint_vel': (frames, 29),
                  'body_pos_w': (frames, 30, 3), 'body_quat_w': (frames, 30, 4),
                  'body_lin_vel_w': (frames, 30, 3), 'body_ang_vel_w': (frames, 30, 3)}
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None or set(archive.namelist()) != {k + '.npy' for k in shapes}:
                raise ValueError('invalid NPZ entries or CRC')
            for name, shape in shapes.items():
                if row['array_shapes'][name] != list(shape):
                    raise ValueError('manifest shape mismatch')
                if row['array_dtypes'][name] != witness_dtypes[name] or (name == 'fps' and witness_dtypes[name] != 'int64') or (name != 'fps' and witness_dtypes[name] not in ('float32', 'float64')):
                    raise ValueError('dtype differs from verified witness')
                data, fmt = canonical_array(archive.read(name + '.npy'), shape, row['array_dtypes'][name], row['array_sha256'][name])
                if name == 'fps':
                    if struct.unpack('<q', data)[0] != 50:
                        raise ValueError('wrong fps')
                elif not all(math.isfinite(x[0]) for x in struct.iter_unpack('<' + fmt, data)):
                    raise ValueError('nonfinite FK data')
                if name == 'body_quat_w':
                    for q in struct.iter_unpack('<4' + fmt, data):
                        if abs(math.sqrt(sum(x * x for x in q)) - 1) >= 1e-4:
                            raise ValueError('invalid body quaternion')
        after = path.stat()
        if ident != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) or sha(path) != row['sha256']:
            raise ValueError('FK output changed during audit')
        final = path.stat()
        if ident != (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns, final.st_ctime_ns):
            raise ValueError('FK output changed after audit')
        source = owned(clip['source_file'])
        source_before = source.stat()
        source_ident = (source_before.st_dev, source_before.st_ino, source_before.st_size,
                        source_before.st_mtime_ns, source_before.st_ctime_ns)
        if source_before.st_size != clip['source_bytes'] or sha(source) != clip['source_sha256']:
            raise ValueError('generated source changed')
        with zipfile.ZipFile(source) as archive:
            qpos, fmt = canonical_array(archive.read('qpos_29.npy'), (frames, 36), 'float32', clip['qpos_sha256'])
            if fmt != 'f':
                raise ValueError('unexpected generated qpos format')
        order = [clip['joint_names'].index(name) for name in receipt['joint_names']]
        digest = hashlib.sha256()
        for frame in range(frames):
            offset = frame * 36 * 4 + 7 * 4
            for joint in order:
                digest.update(qpos[offset + joint * 4:offset + (joint + 1) * 4])
        if digest.hexdigest() != row['array_sha256']['joint_pos']:
            raise ValueError('FK joint positions differ from generated qpos')
        source_after = source.stat()
        if source_ident != (source_after.st_dev, source_after.st_ino, source_after.st_size,
                            source_after.st_mtime_ns, source_after.st_ctime_ns) or sha(source) != clip['source_sha256']:
            raise ValueError('generated source changed during audit')
        source_final = source.stat()
        if source_ident != (source_final.st_dev, source_final.st_ino, source_final.st_size,
                            source_final.st_mtime_ns, source_final.st_ctime_ns):
            raise ValueError('generated source changed after audit')
        items.append(dict(index=index, file=str(path), bytes=row['bytes'], sha256=row['sha256'],
                          frames=frames, source_family=clip['source_family'], source_sha256=clip['source_sha256'],
                          array_sha256=row['array_sha256']))
    manifest_after = manifest.stat()
    if manifest_ident != (manifest_after.st_dev, manifest_after.st_ino, manifest_after.st_size,
                          manifest_after.st_mtime_ns, manifest_after.st_ctime_ns) or sha(manifest) != manifest_sha:
        raise ValueError('manifest changed during audit')
    return dict(status='GENERATED_TRAIN_FK_LIBRARY_ARRAYS_AUDITED', run=RUN, request_sha256=REQUEST_SHA,
                receipt_sha256=sha(out / 'receipt.json'), exit_sha256=sha(log / 'exit.json'),
                process_sha256=sha(log / 'process_exit.json'), outer_sha256=sha(outer),
                resource_sha256=sha(log / 'resources.jsonl'), manifest_sha256=manifest_sha,
                clips=2526, frames=2668946, control_intervals=2666420, items=items,
                source_split='TRAIN', original_VAL_motions=0, model_loads=0, sdk_loads=0,
                physical_tracking_accepted=False, final_generator_model_hash_verified=False)

"""Reconstruct selected complete GRAB human/object timelines at declared source FPS.

Exploratory source-space data only. No text, contact, clock, robot or physical
quality is accepted. The restricted reader is still not a general safe parser.
"""
import argparse
import contextlib
import hashlib
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import smplx
import torch
import trimesh
from grab_restricted_reader import read_grab_field

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
MODEL = Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead')
REGISTRY = ROOT / 'grab_source_registry_20261030/source_registry.jsonl'
REGISTRY_SHA = '27d74b72fec830afa01cc05f788d921ef3d42fc1115aad08565d26604cdec8e4'
STAGE = ROOT / 'grab_mesh_stage_20261030'
FIELDS = ('global_orient','body_pose','left_hand_pose','right_hand_pose','jaw_pose','leye_pose','reye_pose','expression','transl')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def guard():
    if shutil.disk_usage(ROOT).free < 8*1024**3 + 256*1024**2:
        raise RuntimeError('disk safety threshold reached')


def asset(ref, expected):
    relative = PurePosixPath(ref)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('invalid asset path')
    target = (STAGE / ref).resolve(strict=True)
    if not target.is_relative_to(STAGE.resolve()) or sha(target) != expected:
        raise ValueError('asset hash mismatch')
    return target


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    guard()
    if sha(REGISTRY) != REGISTRY_SHA:
        raise ValueError('registry changed')
    selection_hash = sha(a.selection)
    selection = json.loads(a.selection.read_text(encoding='utf8'))
    requested = selection['members']
    if not isinstance(requested, list) or not 1 <= len(requested) <= 20 or len(set(requested)) != len(requested):
        raise ValueError('selection must contain 1..20 unique source members')
    registry = {r['member']: r for r in map(json.loads, REGISTRY.read_text().splitlines())}
    if not set(requested).issubset(registry):
        raise ValueError('selection not in source registry')
    a.out.mkdir(exist_ok=False)
    archive_cache = {}
    failures = 0
    with (a.out/'records.jsonl').open('x', encoding='utf8', buffering=1) as records:
        for number, member in enumerate(requested):
            guard()
            row = registry[member]
            result = {'member': member, 'selection_index': number, 'source_member_sha256': row['member_sha256'],
                      'training_ready': False}
            try:
                n = row['frames']
                if not 120 <= n <= 12000 or row['member_bytes'] > 8_000_000:
                    raise ValueError('bounded batch selection exceeded')
                archive = ROOT/'data/GRAB'/row['archive']
                if row['archive'] not in archive_cache:
                    archive_cache[row['archive']] = sha(archive)
                if archive_cache[row['archive']] != row['archive_sha256']:
                    raise ValueError('archive hash mismatch')
                with zipfile.ZipFile(archive) as z:
                    info = z.getinfo(member)
                    if info.file_size != row['member_bytes'] or info.file_size > 8_000_000:
                        raise ValueError('member size mismatch')
                    raw = z.read(info)
                    if hashlib.sha256(raw).hexdigest() != row['member_sha256']:
                        raise ValueError('source member hash mismatch')
                    body = read_grab_field(z, member, 'body')
                    obj = read_grab_field(z, member, 'object')
                    with np.load(io.BytesIO(raw), allow_pickle=False) as f:
                        fps = float(f['framerate'].item())
                        frame_count = int(f['n_frames'].item())
                        gender = str(f['gender'].item())
                        subject = str(f['sbj_id'].item())
                        components = int(f['n_comps'].item())
                if (fps != row['fps_declared'] or frame_count != n or gender != row['gender']
                    or subject != row['subject'] or components != 24):
                    raise ValueError('metadata mismatch')
                if body['vtemp'] != row['template'] or obj['object_mesh'] != row['object_mesh']:
                    raise ValueError('asset name mismatch')
                template = asset(row['template'], row['template_sha256'])
                asset(row['object_mesh'], row['object_mesh_sha256'])
                mesh = trimesh.load(str(template), process=False)
                vertices = np.asarray(mesh.vertices)
                if vertices.shape != (10475, 3) or not np.isfinite(vertices).all():
                    raise ValueError('invalid subject template')
                all_params = {k: np.asarray(body['params'][k]) for k in FIELDS}
                if any(v.shape[0] != n or not np.isfinite(v).all() for v in all_params.values()):
                    raise ValueError('invalid body parameter timeline')
                trans = np.asarray(obj['params']['transl'])
                orient = np.asarray(obj['params']['global_orient'])
                if trans.shape != (n,3) or orient.shape != (n,3) or not np.isfinite(trans).all() or not np.isfinite(orient).all():
                    raise ValueError('invalid object timeline')
                batch = 32
                with contextlib.redirect_stdout(sys.stderr):
                    model = smplx.create(str(MODEL), model_type='smplx', gender=gender, num_betas=10,
                        num_pca_comps=components, batch_size=batch, use_pca=True, ext='npz').cpu()
                    if not np.array_equal(np.asarray(mesh.faces), np.asarray(model.faces)):
                        raise ValueError('template topology mismatch')
                    model.v_template.copy_(torch.as_tensor(vertices, dtype=model.v_template.dtype))
                    pieces = []
                    with torch.no_grad():
                        for start in range(0, n, batch):
                            guard()
                            indices = np.minimum(np.arange(start,start+batch),n-1)
                            inputs = {k:torch.as_tensor(v[indices],dtype=torch.float32) for k,v in all_params.items()}
                            pieces.append(model(**inputs,return_verts=False).joints[:min(batch,n-start),:22].cpu().numpy())
                human = np.concatenate(pieces)
                if human.shape != (n,22,3) or not np.isfinite(human).all():
                    raise ValueError('invalid human reconstruction')
                target = a.out / f'{number:03d}.npz'
                with target.open('xb') as f:
                    np.savez_compressed(f, human_joints=human, source_frames=np.arange(n),
                        declared_fps=np.array(fps), object_transl=trans, object_orient_rotvec=orient)
                result.update({'status':'reconstructed_unreviewed', 'artifact':target.name, 'artifact_sha256':sha(target),
                    'frames':n, 'declared_fps':fps, 'archive_sha256':row['archive_sha256'],
                    'template_sha256':row['template_sha256'], 'object_mesh_sha256':row['object_mesh_sha256'],
                    'model_sha256':sha(MODEL/'smplx'/('SMPLX_'+gender.upper()+'.npz'))})
                failures = 0
            except Exception as exc:
                result.update({'status':'failed', 'error':f'{type(exc).__name__}: {exc}'})
                failures += 1
            records.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+'\n')
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if failures >= 3:
                raise RuntimeError('three consecutive sample failures')
    if sha(a.selection) != selection_hash or sha(REGISTRY) != REGISTRY_SHA:
        raise RuntimeError('inputs changed during reconstruction')
    out = {'selection_sha256':selection_hash, 'registry_sha256':REGISTRY_SHA,
           'records_sha256':sha(a.out/'records.jsonl'), 'script_sha256':sha(Path(__file__)),
           'requested':len(requested), 'training_ready':False,
           'limitations':['Full source frames reconstructed with approximate locked-head SMPL-X.',
               'Declared FPS is not verified capture time; no downsampling or robot retargeting.',
               'Contact fields deliberately not exported; source, semantics and physics unaccepted.']}
    with (a.out/'report.json').open('x', encoding='utf8') as f:
        json.dump(out, f, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()

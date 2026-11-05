"""Small S2 object-contact geometry diagnostic; not dataset acceptance."""
import contextlib
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

import numpy as np
import torch
import smplx
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from grab_restricted_reader import read_grab_field

BASE = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
ARCHIVE = BASE / 'data/GRAB/grab__s2.zip'
STAGE = BASE / 'grab_mesh_stage_20261030'
MODEL = Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def summary(x):
    if not len(x):
        return None
    return {'count': int(len(x)), 'mean_m': float(np.mean(x)),
            'median_m': float(np.median(x)), 'p95_m': float(np.percentile(x, 95)),
            'max_m': float(np.max(x))}


def main():
    torch.set_num_threads(1)
    expected = 'd77eb7d13cfb3e3d0b81f08ccf9d8d79320fa6541352e6e65173c56d4b01b9b6'
    if digest(ARCHIVE) != expected:
        raise ValueError('unexpected archive')
    members = ('s2/airplane_fly_1.npz', 's2/apple_eat_1.npz', 's2/camera_browse_1.npz')
    result = []
    with zipfile.ZipFile(ARCHIVE) as z:
        for member in members:
            body = read_grab_field(z, member, 'body')
            obj = read_grab_field(z, member, 'object')
            contact = read_grab_field(z, member, 'contact')
            with np.load(io.BytesIO(z.read(member)), allow_pickle=False) as meta:
                gender = str(meta['gender'].item())
                pca = int(meta['n_comps'].item())
                nframes = int(meta['n_frames'].item())
            if gender != 'male' or pca != 24 or body['vtemp'] != 'tools/subject_meshes/male/s2.ply':
                raise ValueError('unexpected subject metadata')
            ref = PurePosixPath(obj['object_mesh'])
            if ref.is_absolute() or '..' in ref.parts or ref.parts[:2] != ('tools', 'object_meshes'):
                raise ValueError('unsafe mesh path')
            template, mesh = STAGE / body['vtemp'], STAGE / str(ref)
            body_mesh = trimesh.load(str(template), process=False)
            object_mesh = trimesh.load(str(mesh), process=False)
            v_body, v_obj = np.asarray(body_mesh.vertices), np.asarray(object_mesh.vertices)
            labels = np.asarray(contact['object'])
            if labels.shape != (nframes, len(v_obj)) or labels.dtype.kind not in 'biu':
                raise ValueError('contact/mesh shape mismatch')
            if v_body.shape != (10475, 3) or not np.isfinite(v_obj).all():
                raise ValueError('invalid vertices')
            # Predetermined quantiles among frames containing at least 50 labeled vertices.
            candidates = np.flatnonzero((labels != 0).sum(axis=1) >= 50)
            if len(candidates) < 5:
                raise ValueError('not enough contact-bearing frames')
            frames = candidates[np.linspace(0, len(candidates)-1, 5, dtype=int)]
            keys = ('global_orient','body_pose','left_hand_pose','right_hand_pose','jaw_pose',
                    'leye_pose','reye_pose','expression','transl')
            tensors = {k: torch.as_tensor(np.asarray(body['params'][k])[frames], dtype=torch.float32) for k in keys}
            with contextlib.redirect_stdout(sys.stderr):
                model = smplx.create(str(MODEL), model_type='smplx', gender=gender, num_betas=10,
                                     num_pca_comps=pca, batch_size=len(frames), use_pca=True, ext='npz')
                if not np.array_equal(np.asarray(body_mesh.faces), np.asarray(model.faces)):
                    raise ValueError('template face indexing differs')
                model.v_template.copy_(torch.as_tensor(v_body, dtype=model.v_template.dtype))
                with torch.no_grad():
                    human = model(**tensors, return_verts=True).vertices.detach().cpu().numpy()
            orient = np.asarray(obj['params']['global_orient'])[frames]
            trans = np.asarray(obj['params']['transl'])[frames]
            if human.shape != (5, 10475, 3) or orient.shape != (5, 3) or trans.shape != (5, 3):
                raise ValueError('invalid outputs')
            frame_results = []
            for i, frame in enumerate(frames):
                world = Rotation.from_rotvec(orient[i]).inv().apply(v_obj) + trans[i]
                distance = cKDTree(human[i]).query(world, workers=1)[0]
                positive = labels[frame] != 0
                frame_results.append({'frame': int(frame), 'time_s_assuming_120fps': float(frame / 120.),
                                      'positive_contact_vertices_to_nearest_body_vertex': summary(distance[positive]),
                                      'unlabeled_vertices_to_nearest_body_vertex': summary(distance[~positive])})
            result.append({'member': member, 'object_mesh': str(ref), 'object_mesh_sha256': digest(mesh),
                           'template_sha256': digest(template), 'frames': frame_results})
            print('completed', member, flush=True)
    output = Path(__file__).with_name('report.json')
    with output.open('x', encoding='utf-8') as f:
        json.dump({'archive_sha256': expected, 'model_sha256': digest(MODEL/'smplx/SMPLX_MALE.npz'),
                   'script_sha256': digest(__file__), 'samples': result,
                   'contact_semantics_source': 'https://github.com/otaheri/GRAB',
                   'warning': 'Only three S2 sequences and five contact-bearing frames each. Distances are to nearest body VERTEX, not surface, forces or ground truth; official labels can include enclosed intersecting vertices and depend on other criteria. Existing locked-head SMPL-X may differ from original model; mesh counting and distances do not prove index ordering, calibration, dynamics or training readiness.'}, f, allow_nan=False, indent=2)


if __name__ == '__main__':
    main()

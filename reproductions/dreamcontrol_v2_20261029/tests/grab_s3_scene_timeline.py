"""Source apple/table geometry on original timestamps; no forces or dynamics."""
import hashlib, io, json, sys, zipfile
from pathlib import Path, PurePosixPath
import mujoco
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
sys.path.insert(0, str(ROOT / 'grab_s3_human_source_20261030'))
from grab_restricted_reader import read_grab_field, read_object_npy

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def load_table(archive, member):
    with zipfile.ZipFile(archive) as z:
        entry = z.getinfo(member)
        if entry.file_size > 256 * 1024 * 1024:
            raise ValueError('oversized source')
        with zipfile.ZipFile(io.BytesIO(z.read(entry))) as inner:
            entry = inner.getinfo('table.npy')
            if entry.file_size > 128 * 1024 * 1024:
                raise ValueError('oversized table')
            return read_object_npy(inner.read(entry))

def mesh_for(ref):
    ref = PurePosixPath(ref)
    if ref.is_absolute() or '..' in ref.parts or ref.parts[:2] != ('tools', 'object_meshes'):
        raise ValueError('invalid mesh path')
    path = ROOT / 'grab_mesh_stage_20261030' / str(ref)
    mesh = trimesh.load(str(path), process=False)
    return path, mesh, mesh.convex_hull

def main():
    source = ROOT / 'grab_s3_human_source_20261030'
    meta = json.loads((source / 'report.json').read_text())
    archive = ROOT / 'data/GRAB/grab__s3.zip'
    if sha(archive) != meta['archive_sha256']:
        raise ValueError('archive hash mismatch')
    with zipfile.ZipFile(archive) as z:
        obj = read_grab_field(z, meta['source_member'], 'object')
    table = load_table(archive, meta['source_member'])
    paths, originals, hulls = {}, {}, {}
    for name, ref in (('apple', obj['object_mesh']), ('table', table['table_mesh'])):
        paths[name], originals[name], hulls[name] = mesh_for(ref)
    model_spec = mujoco.MjSpec()
    for name in ('apple', 'table'):
        mesh = model_spec.add_mesh(name=name + '_hull')
        mesh.uservert = np.asarray(hulls[name].vertices, np.float64).reshape(-1).tolist()
        mesh.userface = np.asarray(hulls[name].faces, np.int32).reshape(-1).tolist()
        body = model_spec.worldbody.add_body(name=name, mocap=True)
        body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name+'_hull', contype=1, conaffinity=1)
    model = model_spec.compile()
    data = mujoco.MjData(model)
    gids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in hulls}
    bids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in hulls}
    if min(*gids.values(), *bids.values()) < 0:
        raise ValueError('missing body/geom')
    frames = np.arange(264, 382)
    transforms = {}
    for n, entry in (('apple', obj), ('table', table)):
        trans = np.asarray(entry['params']['transl'], dtype=float)
        orient = np.asarray(entry['params']['global_orient'], dtype=float)
        if trans.shape != orient.shape or trans.ndim != 2 or trans.shape[1] != 3 or len(trans) <= frames[-1]:
            raise ValueError('incompatible source timeline for '+n)
        if not (np.isfinite(trans).all() and np.isfinite(orient).all()):
            raise ValueError('nonfinite source transforms')
        transforms[n] = (trans, orient)
    max_transform_error = {n: 0. for n in hulls}
    rows = []
    for frame in frames:
        for n in hulls:
            trans, orient = transforms[n]
            q = Rotation.from_rotvec(orient[frame]).inv().as_quat()
            mid = int(model.body_mocapid[bids[n]])
            if mid < 0: raise ValueError('expected mocap')
            data.mocap_pos[mid] = trans[frame]
            data.mocap_quat[mid] = q[[3, 0, 1, 2]]
        mujoco.mj_forward(model, data)
        if frame in (frames[0], frames[len(frames)//2], frames[-1]):
            for n in hulls:
                gid = gids[n]; meshid = int(model.geom_dataid[gid])
                a = int(model.mesh_vertadr[meshid]); count = int(model.mesh_vertnum[meshid])
                actual = data.geom_xpos[gid] + model.mesh_vert[a:a+count] @ data.geom_xmat[gid].reshape(3,3).T
                trans, orient = transforms[n]
                expected = Rotation.from_rotvec(orient[frame]).inv().apply(hulls[n].vertices) + trans[frame]
                err = max(float(cKDTree(actual).query(expected)[0].max()), float(cKDTree(expected).query(actual)[0].max()))
                max_transform_error[n] = max(max_transform_error[n], err)
                if err > 1e-4: raise ValueError('mesh world transform mismatch for '+n)
        dist = float(mujoco.mj_geomDistance(model, data, gids['apple'], gids['table'], 2., None))
        boxes = {}
        for n in hulls:
            trans, orient = transforms[n]
            world = Rotation.from_rotvec(orient[frame]).inv().apply(hulls[n].vertices) + trans[frame]
            boxes[n] = (world.min(axis=0), world.max(axis=0))
        lower = np.maximum(0., np.maximum(boxes['apple'][0] - boxes['table'][1],
                                          boxes['table'][0] - boxes['apple'][1]))
        lower = float(np.linalg.norm(lower))
        rows.append({'frame': int(frame), 'source_time_s': float(frame / 120),
                     'signed_distance_m': dist, 'aabb_distance_lower_bound_m': lower,
                     'violates_aabb_lower_bound': bool(dist + 1e-6 < lower)})
    rowpath = Path(__file__).with_name('frames.jsonl')
    with rowpath.open('x', encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row, allow_nan=False)+'\n')
    distances = np.array([r['signed_distance_m'] for r in rows])
    result = {'archive_sha256': sha(archive), 'source_member': meta['source_member'],
              'script_sha256': sha(Path(__file__)), 'frames_sha256': sha(rowpath),
              'meshes': {n: {'sha256': sha(paths[n]), 'source_vertices': len(originals[n].vertices),
                             'hull_vertices': len(hulls[n].vertices), 'source_convex': bool(originals[n].is_convex)} for n in hulls},
              'mujoco_version': mujoco.__version__, 'frames': len(rows), 'fps': 120,
              'frame_range': [int(frames[0]), int(frames[-1])], 'transform_max_error_m': max_transform_error,
              'distance_m': {'min': float(distances.min()), 'median': float(np.median(distances)),
                             'max': float(distances.max()), 'nonpositive_frames': int(np.sum(distances <= 0)), 'cap': 2.},
              'aabb_lower_bound_violations': [r for r in rows if r['violates_aabb_lower_bound']],
              'limitations': 'Source apple and table convex hulls at 120 Hz original poses; signed static geometry only. No robot, no contact force, no dynamic object or measured physical ground. Convex hulls approximate meshes; not a grasp, training, Isaac Lab, or paper acceptance test.'}
    with Path(__file__).with_name('report.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result), flush=True)

if __name__ == '__main__': main()

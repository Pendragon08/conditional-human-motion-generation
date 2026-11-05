"""Read-only metadata inspection; table coordinates are not yet a scene validation."""
import hashlib, io, json, sys, zipfile
from pathlib import Path, PurePosixPath
import numpy as np
import trimesh

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
sys.path.insert(0, str(ROOT / 'grab_s3_human_source_20261030'))
from grab_restricted_reader import read_object_npy

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    source = ROOT / 'grab_s3_human_source_20261030'
    report = json.loads((source / 'report.json').read_text())
    archive = ROOT / 'data/GRAB/grab__s3.zip'
    if sha(archive) != report['archive_sha256']:
        raise ValueError('source archive hash differs')
    with zipfile.ZipFile(archive) as outer:
        info = outer.getinfo(report['source_member'])
        if info.file_size > 256 * 1024 * 1024:
            raise ValueError('source archive member too large')
        with zipfile.ZipFile(io.BytesIO(outer.read(info))) as inner:
            info = inner.getinfo('table.npy')
            if info.file_size > 128 * 1024 * 1024:
                raise ValueError('table metadata too large')
            table = read_object_npy(inner.read(info))
    ref = PurePosixPath(table['table_mesh'])
    if ref.is_absolute() or '..' in ref.parts or ref.parts[:2] != ('tools', 'object_meshes'):
        raise ValueError('unsafe table mesh path')
    mesh_path = ROOT / 'grab_mesh_stage_20261030' / str(ref)
    mesh = trimesh.load(str(mesh_path), process=False)
    vertices = np.asarray(mesh.vertices)
    p = table['params']
    result = {'archive_sha256': sha(archive), 'source_member': report['source_member'],
              'table_mesh': str(ref), 'table_mesh_sha256': sha(mesh_path),
              'mesh_vertices': len(vertices), 'mesh_faces': len(mesh.faces),
              'mesh_bounds': mesh.bounds.tolist(), 'mesh_is_watertight': bool(mesh.is_watertight),
              'mesh_vertex_rank': int(np.linalg.matrix_rank(vertices - vertices.mean(axis=0))),
              'params': {}, 'script_sha256': sha(Path(__file__))}
    for name, value in p.items():
        a = np.asarray(value)
        if not np.issubdtype(a.dtype, np.number) or not np.all(np.isfinite(a)):
            raise ValueError('non-numeric or non-finite table parameter')
        result['params'][name] = {'shape': list(a.shape), 'min': float(a.min()), 'max': float(a.max()),
                                   'first': a.reshape(-1, a.shape[-1])[0].tolist()}
    with Path(__file__).with_name('report.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()

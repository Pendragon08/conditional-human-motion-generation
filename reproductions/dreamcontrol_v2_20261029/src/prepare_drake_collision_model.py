"""Create a separate OBJ-based URDF for exploratory Drake collision checks.

Drake 1.26 proximity queries reject STL meshes. This converts collision meshes
to OBJ without processing and verifies vertices, triangle indices and bounds;
visual elements are removed. It never modifies the original URDF or STL files.
This does not make Drake collision filters equivalent to Isaac Lab's.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def prepare(source, destination):
    source=Path(source).resolve(strict=True); destination=Path(destination)
    destination.mkdir(parents=False,exist_ok=False)
    original=source.parent.resolve(); tree=ET.parse(source); root=tree.getroot(); converted={}
    for link in root.findall('link'):
        for visual in link.findall('visual'): link.remove(visual)
        for collision in link.findall('collision'):
            mesh_node=collision.find('geometry/mesh')
            if mesh_node is None: continue
            relative=Path(mesh_node.attrib['filename'])
            if relative.is_absolute(): raise ValueError('expected relative source mesh')
            path=(original/relative).resolve(strict=True)
            if not path.is_relative_to(original) or path.suffix.lower()!='.stl':
                raise ValueError('collision mesh must be an STL under the source model directory')
            key=str(path)
            if key not in converted:
                target=destination/(hashlib.sha256(key.encode()).hexdigest()[:16]+'.obj')
                mesh=trimesh.load_mesh(path,process=False)
                if not isinstance(mesh,trimesh.Trimesh) or len(mesh.faces)==0: raise ValueError('invalid collision mesh')
                with target.open('x',encoding='utf-8') as f: f.write(mesh.export(file_type='obj',digits=12))
                reloaded=trimesh.load_mesh(target,process=False)
                if (not isinstance(reloaded,trimesh.Trimesh) or mesh.vertices.shape!=reloaded.vertices.shape or
                    mesh.faces.shape!=reloaded.faces.shape or not np.array_equal(mesh.faces,reloaded.faces) or
                    not np.allclose(mesh.vertices,reloaded.vertices,rtol=0,atol=1e-10)):
                    raise ValueError(f'converted mesh changed triangle geometry: {path.name}')
                converted[key]={'source':str(path),'source_sha256':sha256(path),'obj':str(target.resolve()),
                                'obj_sha256':sha256(target),'vertices':len(mesh.vertices),'triangles':len(mesh.faces)}
            mesh_node.set('filename',converted[key]['obj'])
    urdf=destination/'robot_collision_obj.urdf'
    with urdf.open('xb') as f: tree.write(f,encoding='utf-8',xml_declaration=True)
    report={'source_urdf':str(source),'source_urdf_sha256':sha256(source),'derived_urdf':str(urdf.resolve()),
            'derived_urdf_sha256':sha256(urdf),'converted_meshes':list(converted.values()),
            'warning':'Collision-only URDF, triangle geometry checked numerically, Drake and Isaac filtering/physics may differ.'}
    with (destination/'manifest.json').open('x',encoding='utf-8') as f: json.dump(report,f,indent=2)
    return urdf

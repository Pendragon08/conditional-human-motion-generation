"""Independent Sdf/XML mass and parent-chain foot transform cross-check."""
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

BASE = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1')
USD = BASE / 'g1_27dof_with_hands_min_collisions_flat_white.usd'
URDF = BASE / 'g1_27dof.urdf'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
r = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
assert sha(USD) == r['usd_sha256'] and sha(URDF) == r['urdf_sha256']
layer = Sdf.Layer.FindOrOpen(str(USD)); root = ET.parse(URDF).getroot(); stage = Usd.Stage.Open(str(USD))
actual_bodies = {str(p.GetPath()) for p in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.RigidBodyAPI)}
assert actual_bodies == {v['path'] for v in r['usd_bodies'].values()}
u = {x.get('name'): (float(x.find('inertial/mass').get('value')) if x.find('inertial/mass') is not None else None) for x in root.findall('link')}
assert u == r['urdf_links']
usd = {}
for name, item in r['usd_bodies'].items():
    spec = layer.GetPrimAtPath(item['path']); assert spec is not None
    value = spec.attributes['physics:mass'].default
    assert abs(value-item['mass_kg']) < 1e-12
    usd[name] = float(value)
assert abs(sum(usd.values())-r['usd_mass_sum_kg']) < 1e-10
assert abs(sum(v for v in u.values() if v is not None)-r['urdf_declared_mass_sum_kg']) < 1e-10
max_error = 0.0
for side, pairs in r['feet'].items():
    parent = '/g1_29dof_with_hand_rev_1_0/' + side + '_ankle_roll_link'
    urdf_collisions = root.find("link[@name='%s_ankle_roll_link']" % side).findall('collision')
    remaining = {(tuple(float(v) for v in c.find('origin').get('xyz').split()), float(c.find('geometry/sphere').get('radius'))) for c in urdf_collisions}
    for pair in pairs:
        entry = pair['urdf']; key = (tuple(entry['center_m']),entry['radius_m']); assert key in remaining; remaining.remove(key)
        prim = stage.GetPrimAtPath(pair['usd']['path']); matrix = Gf.Matrix4d(1)
        assert UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        assert abs(float(UsdGeom.Sphere(prim).GetRadiusAttr().Get())-pair['usd']['radius_m']) < 1e-12
        while str(prim.GetPath()) != parent:
            x = UsdGeom.Xformable(prim); assert not x.GetResetXformStack()
            matrix = matrix * x.GetLocalTransformation(); prim = prim.GetParent(); assert prim
        center = matrix.Transform(Gf.Vec3d(0))
        axes = [matrix.TransformDir(Gf.Vec3d(*v)) for v in ((1,0,0),(0,1,0),(0,0,1))]
        assert max(abs(float(v.GetLength())-1) for v in axes) < 1e-7
        err = max(abs(center[i]-pair['usd']['center_m'][i]) for i in range(3)); max_error = max(max_error,err)
        assert err < 1e-10
    assert not remaining
out = {'status': 'passed', 'source_report_sha256': sha(Path(sys.argv[1])), 'usd_sha256': sha(USD), 'urdf_sha256': sha(URDF), 'usd_rigid_body_masses_checked': len(usd), 'urdf_link_masses_checked': len(u), 'foot_spheres_checked': 8, 'max_parent_chain_center_error_m': max_error, 'audit_scope': 'Sdf authored masses and separate local parent-chain USD sphere transforms; no PhysX/contact inference.'}
Path(sys.argv[2]).write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')

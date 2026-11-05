"""Read-only asset inventory; static USD values do not establish PhysX behavior."""
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from scipy.optimize import linear_sum_assignment

BASE = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1')
USD = BASE / 'g1_27dof_with_hands_min_collisions_flat_white.usd'
URDF = BASE / 'g1_27dof.urdf'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

stage = Usd.Stage.Open(str(USD))
root = ET.parse(URDF).getroot()
assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
assert UsdPhysics.GetStageKilogramsPerUnit(stage) == 1.0
urdf = {link.get('name'): (float(link.find('inertial/mass').get('value')) if link.find('inertial/mass') is not None else None) for link in root.findall('link')}
body_prims = [p for p in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
assert len({p.GetName() for p in body_prims}) == len(body_prims)
usd = {p.GetName(): {'path': str(p.GetPath()), 'mass_kg': UsdPhysics.MassAPI(p).GetMassAttr().Get(), 'authored': UsdPhysics.MassAPI(p).GetMassAttr().HasAuthoredValueOpinion()} for p in body_prims}
assert all(v['authored'] and v['mass_kg'] > 0 for v in usd.values())
cache = UsdGeom.XformCache()
feet = {}
for side in ('left', 'right'):
    name = side + '_ankle_roll_link'
    link = stage.GetPrimAtPath('/g1_29dof_with_hand_rev_1_0/' + name)
    inv = cache.GetLocalToWorldTransform(link).GetInverse()
    spheres = []
    for p in Usd.PrimRange(link, Usd.TraverseInstanceProxies()):
        if not p.HasAPI(UsdPhysics.CollisionAPI):
            continue
        assert p.GetTypeName() == 'Sphere' and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()
        transform = cache.GetLocalToWorldTransform(p) * inv
        basis = [transform.TransformDir(Gf.Vec3d(*axis)) for axis in ((1,0,0),(0,1,0),(0,0,1))]
        assert max(abs(float(v.GetLength()) - 1) for v in basis) < 1e-7
        assert max(abs(float(basis[i] * basis[j])) for i in range(3) for j in range(i)) < 1e-7
        spheres.append({'path': str(p.GetPath()), 'center_m': list(transform.Transform(Gf.Vec3d(0))), 'radius_m': float(UsdGeom.Sphere(p).GetRadiusAttr().Get())})
    target = []
    for c in root.find("link[@name='%s']" % name).findall('collision'):
        target.append({'center_m': [float(v) for v in c.find('origin').get('xyz').split()], 'radius_m': float(c.find('geometry/sphere').get('radius'))})
    assert len(target) == len(spheres) == 4
    cost = np.array([[np.linalg.norm(np.array(a['center_m'])-b['center_m']) for b in spheres] for a in target])
    rows, cols = linear_sum_assignment(cost)
    feet[side] = [{'urdf': target[i], 'usd': spheres[j], 'center_error_m': float(cost[i,j]), 'radius_error_m': abs(target[i]['radius_m']-spheres[j]['radius_m'])} for i,j in zip(rows,cols)]
report = {'usd_sha256': digest(USD), 'urdf_sha256': digest(URDF), 'meters_per_unit': 1.0, 'kilograms_per_unit': 1.0, 'usd_bodies': usd, 'urdf_links': urdf, 'usd_mass_sum_kg': sum(x['mass_kg'] for x in usd.values()), 'urdf_declared_mass_sum_kg': sum(x for x in urdf.values() if x is not None), 'feet': feet, 'limitations': ['Static declared masses; USD and URDF include different links, fixed-link treatment and hands.', 'Absent URDF inertials are reported as null, not treated as verified zero mass.', 'No runtime PhysX, collision filtering, contact or dynamic equivalence validated.', 'Foot shape comparison assumes unscaled orthonormal local transformations.']}
out = Path(sys.argv[1]); out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')

"""Create an isolated 29-body-DoF + two hands USD from the pinned old asset.

Only the two previously fixed waist joints change to URDF-axis revolutes.
The source file is never edited. Static schema checks are not PhysX validation.
"""
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from pxr import Gf, Usd, UsdPhysics

R = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
D = R / 'stage3_pick29_20261103'
ROBOT = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROBOT / 'g1_27dof_with_hands_min_collisions_flat_white.usd'
    urdf = ROBOT / 'g1_29dof.urdf'
    expected = '64b9760776de44c3c5b3045693e4115a0e8ca6505b59094bdd0e8db8a64aa2c6'
    if sha(source) != expected or sha(urdf) != '572b54991d8dccd6d3e091570788fb307bd0877b6a062426583f0a17fb56cc95':
        raise ValueError('pinned source assets changed')
    out = D / 'assets/g1_29dof_with_hands.usd'
    if out.exists():
        raise FileExistsError(out)
    stage = Usd.Stage.Open(str(source))
    stage.Flatten().Export(str(out))
    stage = Usd.Stage.Open(str(out))
    nodes = {j.attrib['name']: j for j in ET.parse(urdf).getroot().findall('joint')}
    changes = []
    for name in ('waist_roll_joint', 'waist_pitch_joint'):
        prims = [p for p in stage.Traverse() if p.GetName() == name]
        if len(prims) != 1 or not prims[0].IsA(UsdPhysics.FixedJoint):
            raise ValueError('expected one old fixed waist joint: ' + name)
        prim = prims[0]
        old = UsdPhysics.Joint(prim)
        r0 = old.GetLocalRot0Attr().Get()
        r1 = old.GetLocalRot1Attr().Get()
        node = nodes[name]
        rpy = [float(x) for x in node.find('origin').attrib.get('rpy', '0 0 0').split()]
        if any(abs(x) > 1e-9 for x in rpy):
            raise ValueError('explicit nonzero URDF origin rotation handling required')
        desired = Gf.Quatf(1., Gf.Vec3f(0, 0, 0))
        correction = r0.GetInverse() * desired
        joint = UsdPhysics.RevoluteJoint.Define(stage, prim.GetPath())
        joint.CreateLocalRot0Attr(desired)
        joint.CreateLocalRot1Attr(r1 * correction)
        axis = [float(x) for x in node.find('axis').attrib['xyz'].split()]
        indices = [i for i, x in enumerate(axis) if abs(x - 1) < 1e-9]
        if len(indices) != 1 or sum(abs(x) for x in axis) != 1:
            raise ValueError('positive unit URDF axis required')
        joint.CreateAxisAttr(('X', 'Y', 'Z')[indices[0]])
        limit = node.find('limit').attrib
        joint.CreateLowerLimitAttr(math.degrees(float(limit['lower'])))
        joint.CreateUpperLimitAttr(math.degrees(float(limit['upper'])))
        drive = UsdPhysics.DriveAPI.Apply(prim, 'angular')
        drive.CreateTypeAttr('force')
        drive.CreateStiffnessAttr(200.)
        drive.CreateDampingAttr(5.)
        drive.CreateMaxForceAttr(float(limit['effort']))
        drive.CreateTargetPositionAttr(0.)
        drive.CreateTargetVelocityAttr(0.)
        changes.append({'joint': name, 'axis': joint.GetAxisAttr().Get(),
                        'limits_rad': [float(limit['lower']), float(limit['upper'])],
                        'drive_max_force': float(limit['effort']),
                        'old_local_rot0': str(r0), 'old_local_rot1': str(r1),
                        'new_local_rot0': str(desired), 'new_local_rot1': str(r1 * correction)})
    stage.GetRootLayer().Save()
    check = Usd.Stage.Open(str(out))
    revolute = [p.GetName() for p in check.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    body = [n for n in revolute if not n.startswith(('left_hand', 'right_hand'))]
    hands = [n for n in revolute if n.startswith(('left_hand', 'right_hand'))]
    if len(body) != 29 or len(hands) != 14 or len(set(revolute)) != 43 or sha(source) != expected:
        raise ValueError('isolated asset joint/schema count or unchanged source check failed')
    report = {'source_usd': str(source), 'source_sha256': expected, 'urdf_sha256': sha(urdf),
              'output_usd': str(out), 'output_sha256': sha(out), 'body_joint_count': len(body),
              'hand_articulation_joints': len(hands), 'policy_action_dimension': 31,
              'body_joints': body, 'hand_joints': hands, 'changes': changes,
              'source_unchanged': True, 'PhysX_spawn_validated': False,
              'scope': 'isolated 29DoF adaptation; joint axes/limits taken from pinned 29DoF URDF, runtime motors need verification'}
    with (D / 'reports/asset29_static.json').open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

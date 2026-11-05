"""Diagnose capsule self-collision distances on two neutral poses; not trajectory acceptance."""
import json
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import pyroki as pk
import yourdfpy

D=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/stage3_pick29_20261103')
u=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf')
urdf=yourdfpy.URDF.load(str(u),load_meshes=False,load_collision_meshes=True)
robot=pk.Robot.from_urdf(urdf);collision=pk.collision.RobotCollision.from_urdf(urdf)
names=list(robot.joints.actuated_names)
q=np.zeros(29,dtype=np.float32)
for side in ['left','right']:
    for part,value in [('hip_pitch',-.2),('knee',.42),('ankle_pitch',-.23),('elbow',.87),('shoulder_pitch',.35)]:
        q[names.index(side+'_'+part+'_joint')]=value
q[names.index('left_shoulder_roll_joint')]=.16;q[names.index('right_shoulder_roll_joint')]=-.16
results=[]
for label,angles in [('zero',np.zeros(29,dtype=np.float32)),('default_standing',q)]:
    distances=np.asarray(collision.compute_self_collision_distance(robot,jnp.asarray(angles)))
    pair_i=np.asarray(collision.active_idx_i);pair_j=np.asarray(collision.active_idx_j)
    worst=[{'links':[collision.link_names[int(pair_i[k])],collision.link_names[int(pair_j[k])]],
            'distance_m':float(distances[k])} for k in np.argsort(distances)[:12]]
    results.append({'pose':label,'minimum_distance_m':float(distances.min()),'worst_pairs':worst})
report={'neutral_pose_capsule_diagnostic':results,
        'scope':'diagnostic only; intersections can include mechanically adjacent or merged fixed links; no source or generated motion filtered here'}
with (D/'reports/collision_neutral_diagnostic.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report))

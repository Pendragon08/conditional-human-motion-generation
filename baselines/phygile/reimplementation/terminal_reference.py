"""Named nominal G1 standing suffix, independent of a dataset's first pose."""
from pathlib import Path
import json,re
import numpy as np
from codec262 import quat_matrix,rotvec_matrix,matrix_quat
from physics_prefix import Motion


def nominal_standing(codec,prefix,config_path):
    config=json.loads(Path(config_path).read_text(encoding='utf-8'))
    q=np.full(29,float(config['unlisted_joint_position']))
    for pattern,value in config['joint_positions'].items():
        indices=[i for i,n in enumerate(codec.joint_names) if re.fullmatch(pattern,n)]
        if not indices:raise ValueError('Unknown standing joint pattern: '+pattern)
        q[indices]=float(value)
    if not np.isfinite(q).all() or (q<codec.low).any() or (q>codec.high).any():
        raise ValueError('Nominal standing joints violate limits')
    state=np.zeros((1,71),dtype=np.float64)
    state[0,:2]=prefix.states[-1,:2];state[0,2]=float(config['root_height_m'])
    if not .4<state[0,2]<1.2:raise ValueError('Invalid standing root height')
    initial=quat_matrix(prefix.states[-1,3:7])
    yaw=np.arctan2(initial[1,0],initial[0,0])
    state[0,3:7]=matrix_quat(rotvec_matrix([0,0,yaw]));state[0,13:42]=q
    positions,_=codec.fk(state[:,:3],state[:,3:7],state[:,13:42])
    bodies=np.stack([positions[n] for n in prefix.body_names],axis=1)
    return Motion(state,bodies,prefix.fps,prefix.state_schema,prefix.body_names),config

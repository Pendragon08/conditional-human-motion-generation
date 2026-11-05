"""Apply previously validated time brackets to translation and axis-angle poses.

Source/reference frame identity and caption timing remain separate checks.
Output is an experimental preprocessing artifact, not an accepted dataset.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def _slerp_quaternion(a, b, alpha):
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    dot=np.sum(a*b,axis=-1,keepdims=True)
    b=np.where(dot<0,-b,b); dot=np.clip(np.abs(dot),0.0,1.0)
    alpha=np.asarray(alpha,dtype=np.float64).reshape((-1,1,1))
    angle=np.arccos(dot); sine=np.sin(angle)
    near=sine<1e-8
    divisor=np.where(near,1.0,sine)
    a_weight=np.sin((1-alpha)*angle)/divisor
    b_weight=np.sin(alpha*angle)/divisor
    result=np.where(near,(1-alpha)*a+alpha*b,a_weight*a+b_weight*b)
    norm=np.linalg.norm(result,axis=-1,keepdims=True)
    if not np.isfinite(norm).all() or np.any(norm<1e-12):raise ValueError('invalid quaternion interpolation')
    return result/norm


def apply_grid(translations, rotvecs, grid):
    translation=np.asarray(translations,dtype=np.float64)
    rotation=np.asarray(rotvecs,dtype=np.float64)
    if translation.ndim!=2 or translation.shape[1]!=3 or rotation.ndim!=2 or rotation.shape[1]%3:
        raise ValueError('expected translations (frames,3) and axis angles (frames,3*joints)')
    if len(translation)!=len(rotation) or not np.isfinite(translation).all() or not np.isfinite(rotation).all():
        raise ValueError('source frame mismatch or nonfinite')
    weights=grid['weights']
    left=np.asarray([w['left_frame'] for w in weights],dtype=np.int64)
    right=np.asarray([w['right_frame'] for w in weights],dtype=np.int64)
    alpha=np.asarray([w['alpha'] for w in weights],dtype=np.float64)
    if len(left)==0 or (left<0).any() or (right>=len(translation)).any() or (right<left).any() or (alpha<0).any() or (alpha>1).any():
        raise ValueError('invalid time brackets')
    positions=translation[left]*(1-alpha[:,None])+translation[right]*alpha[:,None]
    joints=rotation.shape[1]//3
    q=Rotation.from_rotvec(rotation.reshape(-1,3)).as_quat().reshape(len(rotation),joints,4)
    out_q=_slerp_quaternion(q[left],q[right],alpha)
    out_rot=Rotation.from_quat(out_q.reshape(-1,4)).as_rotvec().reshape(len(weights),joints*3)
    return positions,out_rot

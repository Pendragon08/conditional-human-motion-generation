"""Optional continuous-quaternion 263D variant, not author-v2 preprocessing.

The pinned HumanML encoder computes its root angular channel from smoothed IK
quaternions without sign alignment. Equivalent quaternion sign flips can turn
a small physical step into a wrong signed angular increment. This experimental
variant aligns signs before deriving that one channel; all others are preserved.
"""
import numpy as np
import torch
from data_loaders.humanml.common.quaternion import qinv_np,qmul_np
from data_loaders.humanml.common.skeleton import Skeleton
from data_loaders.humanml.utils.paramUtil import t2m_raw_offsets
from feature263 import CHAINS,FACE,canonicalize_z_up,encode

def align_signs(q):
    q=np.asarray(q,dtype=np.float64)
    if q.ndim!=2 or q.shape[1]!=4 or len(q)<2 or not np.isfinite(q).all():
        raise ValueError('finite quaternion sequence required')
    if not np.allclose(np.linalg.norm(q,axis=1),1.,atol=1e-5,rtol=0):
        raise ValueError('unit quaternions required')
    dots=np.sum(q[1:]*q[:-1],axis=1)
    signs=np.r_[1.,np.cumprod(np.where(dots<0,-1.,1.))]
    return q*signs[:,None]

def encode_signfixed(points_z_up_world,*,foot_velocity_squared_threshold):
    features=encode(points_z_up_world,foot_velocity_squared_threshold=foot_velocity_squared_threshold)
    canonical=canonicalize_z_up(points_z_up_world)
    skel=Skeleton(torch.as_tensor(t2m_raw_offsets,dtype=torch.float32),CHAINS,'cpu')
    root=skel.inverse_kinematics_np(canonical.copy(),FACE,smooth_forward=True)[:,0].copy()
    original_velocity=qmul_np(root[1:],qinv_np(root[:-1]))
    original=np.arcsin(np.clip(original_velocity[:,2],-1.,1.)).astype(np.float32)
    if not np.array_equal(original,features[:,0]):
        raise ValueError('pinned encoder root channel did not reproduce')
    aligned=align_signs(root)
    velocity=qmul_np(aligned[1:],qinv_np(aligned[:-1]))
    fixed=features.copy();fixed[:,0]=np.arcsin(np.clip(velocity[:,2],-1.,1.)).astype(np.float32)
    if not np.isfinite(fixed).all():raise ValueError('nonfinite sign-fixed output')
    return fixed

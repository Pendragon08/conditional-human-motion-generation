"""HumanML3D's published joint-coordinate transform and mirror operation.

This transforms joint positions, not axis-angle poses or robot joint angles.
Inputs use AMASS/SMPL Z-up Cartesian joint coordinates; HumanML axes are Y-up.
"""
import numpy as np

BODY_LEFT=(1,4,7,10,13,16,18,20)
BODY_RIGHT=(2,5,8,11,14,17,19,21)
HAND_LEFT=(22,23,24,34,35,36,25,26,27,31,32,33,28,29,30)
HAND_RIGHT=(43,44,45,46,47,48,40,41,42,37,38,39,49,50,51)

def _validate(joints):
    arr=np.asarray(joints)
    if arr.ndim<2 or arr.shape[-1]!=3 or (arr.shape[-2]!=22 and arr.shape[-2]<52):
        raise ValueError('expected joint positions with 22 or at least 52 joints')
    if not np.issubdtype(arr.dtype,np.floating) or not np.isfinite(arr).all():
        raise ValueError('joint positions must be finite floats')
    return arr

def amass_to_humanml(joints):
    """Convert SMPL joint axes and apply official non-HumanAct12 initial x flip."""
    a=_validate(joints)
    out=a[..., (0,2,1)].copy()
    out[...,0]*=-1
    return out

def mirror_humanml(joints):
    """Reflect HumanML x and exchange left/right joints, preserving time."""
    a=_validate(joints)
    out=a.copy();out[...,0]*=-1
    left=BODY_LEFT+(HAND_LEFT if a.shape[-2]>=52 else ())
    right=BODY_RIGHT+(HAND_RIGHT if a.shape[-2]>=52 else ())
    out[...,left,:]=a[...,right,:]*np.array([-1,1,1],dtype=a.dtype)
    out[...,right,:]=a[...,left,:]*np.array([-1,1,1],dtype=a.dtype)
    return out

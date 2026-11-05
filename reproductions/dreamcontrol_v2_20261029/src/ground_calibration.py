"""Fit a ground plane only from explicitly identified ground-surface observations.

An ordinary body or foot marker is not a ground measurement. This isolated
component validates numerical calibration geometry, not the truth of the
provenance, dynamics, contact or overall training quality.
"""
from __future__ import annotations
import numpy as np


def fit_ground_surface(points, labels, *, observation_kind, source_to_robot, source_up,
                       stationary_tolerance_m, plane_tolerance_m, min_normal_alignment):
    if observation_kind!='fixed_ground_surface_points':
        raise ValueError('ground calibration requires fixed ground surface observations; body markers are insufficient')
    p=np.asarray(points,dtype=np.float64); transform=np.asarray(source_to_robot,dtype=np.float64)
    up=np.asarray(source_up,dtype=np.float64)
    if (p.ndim!=3 or p.shape[0]<2 or p.shape[1]<3 or p.shape[2]!=3 or not np.isfinite(p).all() or
        len(labels)!=p.shape[1] or any(not isinstance(s,str) or not s.strip() for s in labels) or len(set(labels))!=len(labels)):
        raise ValueError('finite, uniquely labelled ground observations are required')
    if transform.shape!=(4,4) or not np.isfinite(transform).all() or not np.allclose(transform[3],[0,0,0,1],rtol=0,atol=1e-10):
        raise ValueError('invalid source-to-robot transform')
    rot=transform[:3,:3]
    if not np.allclose(rot.T@rot,np.eye(3),rtol=0,atol=1e-8) or abs(np.linalg.det(rot)-1)>1e-8:
        raise ValueError('source-to-robot rotation must be proper and orthonormal')
    if up.shape!=(3,) or not np.isfinite(up).all() or abs(np.linalg.norm(up)-1)>1e-8:
        raise ValueError('source_up must be a unit vector')
    for value,name in ((stationary_tolerance_m,'stationary_tolerance_m'),(plane_tolerance_m,'plane_tolerance_m')):
        if isinstance(value,bool) or not np.isfinite(value) or value<=0:raise ValueError(name+' must be explicit and positive')
    if (isinstance(min_normal_alignment,bool) or not np.isfinite(min_normal_alignment) or
        not 0<min_normal_alignment<=1):raise ValueError('min_normal_alignment must be in (0,1]')
    centers=p.mean(axis=0)
    stationary=float(np.linalg.norm(p-centers[None,:,:],axis=-1).max())
    if stationary>stationary_tolerance_m:raise ValueError('ground observations are not stationary')
    centered=centers-centers.mean(axis=0)
    _,s,vh=np.linalg.svd(centered,full_matrices=False)
    if len(s)<2 or s[0]<1e-8 or s[1]<max(1e-8,s[0]*1e-6):raise ValueError('ground observations are collinear or coincident')
    normal=vh[-1]
    if normal@up<0:normal=-normal
    alignment=float(normal@up)
    if alignment<min_normal_alignment:raise ValueError('calibration plane inconsistent with documented up')
    origin=centers.mean(axis=0); residual=float(np.abs((p-origin)@normal).max())
    if residual>plane_tolerance_m:raise ValueError('surface observations exceed plane residual tolerance')
    robot_normal=rot@normal; robot_origin=rot@origin+transform[:3,3]
    return {'normal_source':normal.tolist(),'origin_source':origin.tolist(),
            'normal_robot':robot_normal.tolist(),'origin_robot':robot_origin.tolist(),
            'offset_robot':float(-robot_normal@robot_origin),'max_stationary_displacement_m':stationary,
            'max_plane_residual_m':residual,'normal_up_alignment':alignment,'source_points':len(labels),
            'frames':p.shape[0],'training_ready':False,
            'limitations':['A declared observation kind and labels do not independently prove measurement provenance.',
                           'Tolerance values are caller-selected engineering parameters, not DreamControl-v2 criteria.',
                           'This establishes no foot contact, dynamics or overall training acceptance.']}

"""Kinematic foot-sphere diagnostic against a caller-specified flat plane.

Uses actual URDF foot collision spheres and world rotations. The flat plane,
contact tolerance and source-derived contact labels are explicit assumptions,
not measurements of terrain or physics. Does not prove dynamic feasibility.
"""
from __future__ import annotations
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from g1_fk import URDFFK

FOOT_LINKS=('left_ankle_roll_link','right_ankle_roll_link')


def foot_spheres(urdf):
    root=ET.parse(Path(urdf)).getroot()
    out=[]
    for name in FOOT_LINKS:
        node=next((x for x in root.findall('link') if x.get('name')==name),None)
        if node is None: raise ValueError(f'missing foot link {name}')
        items=[]
        for collision in node.findall('collision'):
            geometry=collision.find('geometry'); sphere=geometry.find('sphere') if geometry is not None else None
            if sphere is None or len(geometry)!=1: raise ValueError(f'{name} collision is not a sphere')
            origin=collision.find('origin'); xyz=np.fromstring(origin.get('xyz','0 0 0') if origin is not None else '0 0 0',sep=' ')
            radius=float(sphere.get('radius','nan'))
            if xyz.shape!=(3,) or not np.isfinite(xyz).all() or not np.isfinite(radius) or radius<=0:
                raise ValueError('invalid foot collision sphere')
            items.append((xyz,radius))
        if not items: raise ValueError(f'no foot collision spheres for {name}')
        out.append(items)
    if len(out[0])!=len(out[1]): raise ValueError('different sphere counts on feet')
    return np.array([[item[0] for item in foot] for foot in out]),np.array([[item[1] for item in foot] for foot in out])


def sphere_centers(fk, root_pos, root_quat, joint_pos, joint_names, offsets):
    pos,rot=fk.forward(root_pos,root_quat,joint_pos,joint_names,links=FOOT_LINKS,return_rotations=True)
    offsets=np.asarray(offsets,dtype=float)
    if offsets.ndim!=3 or offsets.shape[0]!=2 or offsets.shape[2]!=3 or not np.isfinite(offsets).all():
        raise ValueError('expected finite local offsets (2,K,3)')
    return pos[:,:,None,:]+np.einsum('tfij,fkj->tfki',rot,offsets)


def compare(source, target, radii, *, fps, plane_z, height_tolerance, max_vertical_speed):
    source=np.asarray(source,dtype=float); target=np.asarray(target,dtype=float); radii=np.asarray(radii,dtype=float)
    if source.shape!=target.shape or source.ndim!=4 or source.shape[1]!=2 or source.shape[-1]!=3 or len(source)<2:
        raise ValueError('expected source and target sphere centers (T,2,K,3)')
    if radii.shape!=source.shape[1:3] or not np.isfinite(source).all() or not np.isfinite(target).all() or not np.isfinite(radii).all() or np.any(radii<=0):
        raise ValueError('invalid centers or radii')
    if not all(np.isfinite(x) for x in (fps,plane_z,height_tolerance,max_vertical_speed)) or fps<=0 or height_tolerance<=0 or max_vertical_speed<=0:
        raise ValueError('invalid plane or contact parameters')
    source_clear=source[...,2]-radii-plane_z
    target_clear=target[...,2]-radii-plane_z
    source_vel=np.diff(source,axis=0)*fps; target_vel=np.diff(target,axis=0)*fps
    stance=(np.abs(source_clear[1:])<=height_tolerance)&(np.abs(source_vel[...,2])<=max_vertical_speed)
    source_tangent=np.linalg.norm(source_vel[...,:2],axis=-1)
    target_tangent=np.linalg.norm(target_vel[...,:2],axis=-1)
    return {'source_min_clearance_m':float(source_clear.min()),'target_min_clearance_m':float(target_clear.min()),
            'source_max_penetration_m':float(np.maximum(-source_clear,0).max()),
            'target_max_penetration_m':float(np.maximum(-target_clear,0).max()),
            'source_penetrating_sphere_frame_fraction':float(np.mean(source_clear<0)),
            'target_penetrating_sphere_frame_fraction':float(np.mean(target_clear<0)),
            'source_contact_proxy_samples':int(stance.sum()),
            'source_contact_proxy_mean_tangent_speed_mps':float(source_tangent[stance].mean()) if stance.any() else None,
            'target_contact_proxy_mean_tangent_speed_mps':float(target_tangent[stance].mean()) if stance.any() else None,
            'contact_proxy_mean_center_error_m':float(np.linalg.norm(target[1:]-source[1:],axis=-1)[stance].mean()) if stance.any() else None,
            'assumptions':{'ground_plane_z_m':plane_z,'fps':fps,'height_tolerance_m':height_tolerance,
                           'max_source_vertical_speed_mps':max_vertical_speed},
            'warning':'Assumed flat plane and source-derived labels; no terrain, true contact, collision response or dynamics validation.'}

"""Shared measurements and bounded feedback; SI units, fixed external goals.

V21: linear velocity is at COM, proven by V19 isolated-body calibration.
Surface slip remains a simulator/geometry proxy, not contact-point truth.
No COM support-polygon test is imposed on dynamic or airborne phases.
"""
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def wrap_angle(x):
    return (x + np.pi) % (2*np.pi) - np.pi


def heading(positions, names):
    across = (positions[..., names.index('R_Hip'), :] - positions[..., names.index('L_Hip'), :]
              + positions[..., names.index('R_Shoulder'), :] - positions[..., names.index('L_Shoulder'), :])
    forward = np.cross(np.array([0., 0., 1.]), across)
    return np.arctan2(forward[..., 1], forward[..., 0])


def bounded_update(original, proposed, error, step, total):
    error = np.asarray(error, dtype=np.float64)
    norm = np.linalg.norm(error, axis=-1, keepdims=True)
    update = error * np.minimum(1., step / np.maximum(norm, 1e-12))
    result = proposed + update
    residual = result - original
    norm = np.linalg.norm(residual, axis=-1, keepdims=True)
    return original + residual * np.minimum(1., total / np.maximum(norm, 1e-12))


def rotate_xyzw(q, v):
    u, w = q[..., :3], q[..., 3:]
    return v + 2*np.cross(u, np.cross(u, v) + w*v)


def foot_geometry(xml_path):
    """Read collision primitives in body-local coordinates, not ankle centers."""
    result = {}
    for body in ET.parse(xml_path).getroot().iter('body'):
        name = body.attrib.get('name')
        if name not in ('L_Ankle','L_Toe','R_Ankle','R_Toe'):
            continue
        points, radii = [], []
        for geom in body.findall('geom'):
            typ = geom.attrib.get('type', 'sphere')
            size = np.fromstring(geom.attrib.get('size','0'),sep=' ')
            pos = np.fromstring(geom.attrib.get('pos','0 0 0'),sep=' ')
            if typ in ('capsule','cylinder') and 'fromto' in geom.attrib:
                ends = np.fromstring(geom.attrib['fromto'],sep=' ').reshape(2,3)
                points.extend(ends.tolist()); radii.extend([float(size[0])]*2)
            elif typ == 'sphere':
                points.append(pos.tolist()); radii.append(float(size[0]))
            elif typ == 'box' and len(size)==3:
                for x in (-1,1):
                    for y in (-1,1):
                        for z in (-1,1):
                            points.append((pos + size*np.array([x,y,z])).tolist()); radii.append(0.)
            else:
                raise ValueError('Unsupported foot geometry: '+repr(geom.attrib))
        if points:
            result[name] = {'points':points,'radii':radii}
    assert len(result)==4, 'Cannot silently use body centers as collision surfaces'
    return result


def surface_metrics(pos, quat, linvel, angvel, force, names, geometry, thresholds):
    speeds, support, heights = [], [], []
    for name, geom in geometry.items():
        i=names.index(name)
        points=np.array(geom['points'],dtype=float)
        radii=np.array(geom['radii'],dtype=float)
        rotated=rotate_xyzw(quat[:,i,None,:],points[None])
        world=pos[:,i,None,:]+rotated
        world[:,:,2]-=radii[None]
        closest=np.argmin(world[:,:,2],axis=1)
        lowest=world[np.arange(len(pos)),closest]
        # Velocity of the instantaneous closest surface material point.
        lever=lowest-pos[:,i]
        assert 'com_local' in geom, 'V21 requires measured native body COM, never a zero fallback'
        com=rotate_xyzw(quat[:,i],np.asarray(geom['com_local'],dtype=float))
        velocity=linvel[:,i]+np.cross(angvel[:,i],lever-com)
        supported=(force[:,i,2]>thresholds['contact_force_n']) & (lowest[:,2]<thresholds['surface_near_ground_m'])
        speeds.append(np.linalg.norm(velocity[:,:2],axis=1))
        support.append(supported); heights.append(lowest[:,2])
    speeds=np.stack(speeds,axis=1); support=np.stack(support,axis=1)
    heights=np.stack(heights,axis=1)
    count=int(support.sum())
    return dict(
        surface_slip_proxy_mean_m_s=float(speeds[support].mean()) if count else None,
        surface_slip_proxy_fraction=float((speeds[support]>thresholds['sliding_speed_m_s']).mean()) if count else None,
        support_observations=count,
        penetration_fraction=float((heights < -thresholds['penetration_depth_m']).any(axis=1).mean()),
        penetration_depth_p95_m=float(np.quantile(np.maximum(-heights,0),.95)),
        airborne_fraction=float((~support.any(axis=1)).mean()),
        contact_metric_type='COM velocity corrected body-force and collision-primitive surface proxy')


def metrics(arrays, index, metadata, thresholds):
    valid=arrays['valid'][:,index]
    length=int(valid.sum())
    expected=int(metadata['episode_steps'])
    names=metadata['names']; dt=metadata['dt']
    target=np.asarray(metadata['original_goal'][index])
    joint=names.index(metadata['cases'][index]['goal_joint_body'])
    goal_heading=float(metadata['original_heading'][index])
    result={k:metadata['cases'][index][k] for k in ('case_id','source_id','clip_id','family','offset_m')}
    result.update(length=length,terminated=bool(arrays['terminated'][index]),
                  completed=length==expected and not bool(arrays['terminated'][index]),
                  original_goal=target.tolist(),original_heading=goal_heading,
                  seed=metadata['seed'],mode=metadata['mode'],round=metadata['round'])
    if not length:
        result.update(joint_pass=False,status='no_valid_steps'); return result
    pos=arrays['actual'][valid,index]; ref=arrays['reference'][valid,index]
    distances=np.linalg.norm(pos[:,joint]-target,axis=-1)
    result['endpoint_error_m']=float(distances[-1])
    result['endpoint_error_vector']= (target-pos[-1,joint]).tolist()
    result['heading_error_rad']=float(abs(wrap_angle(heading(pos[-1],names)-goal_heading)))
    result['heading_error_signed_rad']=float(wrap_angle(goal_heading-heading(pos[-1],names)))
    # Endpoint event: among geometrically near-optimal times, choose closest
    # to the externally specified endpoint time. No changed target/time.
    pred_start=metadata['context_steps']
    if length>pred_start:
        d=distances[pred_start:]
        near=np.flatnonzero(d<=d.min()+0.001)+pred_start
        best=near[np.argmin(np.abs(near-(expected-1)))]
        result['endpoint_time_error_s']=float(abs(best-(expected-1))*dt)
    else:
        result['endpoint_time_error_s']=expected*dt
    # Prefix never excluded from completion/failure; quality uses generated part.
    start=min(pred_start,length-1)
    errors=np.linalg.norm(pos[start:]-ref[start:],axis=-1)
    result['tracking_mean_m']=float(errors.mean())
    result['tracking_p95_m']=float(np.quantile(errors.mean(axis=1),.95))
    v=np.diff(pos[start:],axis=0)/dt
    a=np.diff(v,axis=0)/dt; jerk=np.diff(a,axis=0)/dt
    result['jerk_p95_m_s3']=float(np.quantile(np.linalg.norm(jerk,axis=-1),.95)) if len(jerk) else None
    assert 'native_body_properties' in metadata, 'Native COM must be recorded per body and environment'
    geometry={name:dict(geom,com_local=metadata['native_body_properties'][index][names.index(name)]['com'])
              for name,geom in metadata['foot_geometry'].items()}
    result.update(surface_metrics(pos[start:],arrays['quaternion'][valid,index][start:],
                  arrays['linvel'][valid,index][start:],arrays['angvel'][valid,index][start:],
                  arrays['contact_force'][valid,index][start:],names,geometry,thresholds))
    conditions=(result['endpoint_error_m']<=thresholds['goal_position_m'] and
                result['heading_error_rad']<=thresholds['heading_rad'] and
                result['endpoint_time_error_s']<=thresholds['goal_time_s']+1e-6)
    physical=(result['penetration_fraction']<=thresholds['penetration_fraction'] and
              result['surface_slip_proxy_fraction'] is not None and
              result['surface_slip_proxy_fraction']<=thresholds['sliding_fraction'] and
              result['jerk_p95_m_s3'] is not None and
              result['jerk_p95_m_s3']<=thresholds['continuity_jerk_p95_m_s3'])
    tracking=(result['tracking_mean_m']<=thresholds['tracking_mean_m'] and
              result['tracking_p95_m']<=thresholds['tracking_p95_m'])
    result.update(condition_pass=bool(conditions),physical_pass=bool(physical),
                  tracking_pass=bool(tracking),joint_pass=bool(result['completed'] and conditions and physical and tracking),
                  status='measured',semantic_status='not_verified_by_independent_evaluator')
    return result


def rank_candidate(row, thresholds, tracking_only=False):
    if tracking_only:
        return (not row.get('completed',False),row.get('tracking_mean_m',float('inf')))
    # Same ranking for candidate repair and equal-budget random selection.
    return (not row.get('joint_pass',False),not row.get('completed',False),
            row.get('endpoint_error_m',float('inf'))/thresholds['goal_position_m'] +
            row.get('heading_error_rad',float('inf'))/thresholds['heading_rad'],
            row.get('tracking_mean_m',float('inf')))


def unit_witness():
    original=np.zeros((3,3)); error=np.array([[10,0,0],[0,-10,0],[0,0,0]])
    proposed=original.copy()
    for _ in range(10):
        prior=proposed.copy(); proposed=bounded_update(original,proposed,error,.05,.15)
        assert np.max(np.linalg.norm(proposed-prior,axis=-1))<=.05000001
        assert np.max(np.linalg.norm(proposed-original,axis=-1))<=.15000001
    q=np.array([[0,0,np.sin(np.pi/4),np.cos(np.pi/4)]])
    assert np.allclose(rotate_xyzw(q,np.array([[1.,0,0]])),[[0,1,0]],atol=1e-6)
    assert abs(wrap_angle(3*np.pi)-(-np.pi))<1e-8
    print('PHYSICS_GUIDED_CORE_WITNESS_PASS',flush=True)


if __name__=='__main__':
    unit_witness()

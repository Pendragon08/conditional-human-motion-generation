"""Recompute 27-joint trajectory diagnostics for a six-record preview.

Reports observed numeric limits and finite-difference motion, not a pass/fail
physical threshold. A joint trace that passes these checks is not necessarily
trainable; true source identity, contact and text semantics remain independent.
"""
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PREVIEW=ROOT/'six_candidate_technical_preview_20261031'
URDF=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
HERE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def joint_limits(path):
    limits={}
    for joint in ET.parse(path).getroot().findall('joint'):
        if joint.get('type')=='continuous':limits[joint.get('name')]=(-np.inf,np.inf)
        elif joint.get('type')=='revolute':
            node=joint.find('limit');limits[joint.get('name')]=(float(node.get('lower')),float(node.get('upper')))
    if len(limits)!=27:raise ValueError('expected 27 robot joints')
    return limits


def measure(q,t,names,limits):
    q=np.asarray(q,dtype=np.float64);t=np.asarray(t,dtype=np.float64)
    if q.shape!=(len(t),27) or len(t)<3 or not np.isfinite(q).all() or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('invalid trajectory')
    if len(set(names))!=27 or set(names)!=set(limits):raise ValueError('joint mapping mismatch')
    lo=np.array([limits[n][0] for n in names]);hi=np.array([limits[n][1] for n in names])
    outside=np.maximum(np.maximum(lo-q,q-hi),0);k,j=np.unravel_index(outside.argmax(),outside.shape)
    velocity=np.diff(q,axis=0)/np.diff(t)[:,None]
    dt=(t[2:]-t[:-2])/2.;accel=np.diff(velocity,axis=0)/dt[:,None]
    v_i,v_j=np.unravel_index(np.abs(velocity).argmax(),velocity.shape)
    return {'samples':len(t),'start_time_s':float(t[0]),'end_time_s':float(t[-1]),
            'max_limit_excess_rad':float(outside[k,j]),'max_limit_excess_joint':names[j],
            'max_limit_excess_row':int(k),'max_joint_speed_rad_s':float(abs(velocity[v_i,v_j])),
            'max_joint_speed_joint':names[v_j],'max_joint_speed_start_row':int(v_i),
            'max_joint_acceleration_rad_s2':float(np.abs(accel).max())}


def main():
    preview=PREVIEW/'experimental_manifest_not_for_training.json'
    check=json.loads((PREVIEW/'independent_audit.json').read_text())
    if sha(preview)!=check['manifest_sha256']:raise ValueError('preview changed')
    manifest=json.loads(preview.read_text());limits=joint_limits(URDF);cases=[]
    for rec in manifest['records']:
        mid=rec['sample_id']
        if mid=='000008':
            folder=ROOT/'pyroki_ssm_000008_global_20261031'
        elif mid.startswith('BML_'):
            folder=ROOT/'bml_joint_smoothness_20261031'/mid[4:]
        else:
            folder=ROOT/'ssm_three_global_20261031'/mid
        trajectory=folder/'trajectory.npz';solved=json.loads((folder/'report.json').read_text())
        if sha(trajectory)!=solved['artifact_sha256'] or sha(URDF)!=solved['urdf_sha256']:
            raise ValueError('trajectory or URDF changed')
        with np.load(trajectory,allow_pickle=False) as f:
            q=f['joint_pos'].copy();t=f['source_times'].copy();names=[str(x) for x in f['joint_names']]
        feature=PREVIEW/rec['feature_path']
        if sha(feature)!=rec['feature_sha256']:raise ValueError('feature changed')
        with np.load(feature,allow_pickle=False) as f:
            grid=(f['source_times'] if mid.startswith('BML_') else f['time_grid']).copy()
            if len(grid)!=len(f['features'])+1:raise ValueError('feature interval endpoints missing')
        if grid[0]<t[0]-1e-10 or grid[-1]>t[-1]+1e-10:raise ValueError('feature grid exceeds trajectory')
        out_q=np.column_stack([np.interp(grid,t,q[:,j]) for j in range(27)])
        cases.append({'sample_id':mid,'split':rec['split'],'trajectory_sha256':sha(trajectory),
                      'feature_sha256':sha(feature),'caption_accepted':rec.get('caption_accepted') is True,
                      'raw_trajectory':measure(q,t,names,limits),'feature_grid_linear_joints':measure(out_q,grid,names,limits),
                      'training_ready':False})
    result={'manifest_sha256':sha(preview),'urdf_sha256':sha(URDF),'cases':cases,'training_ready':False,
            'limitations':['Diagnostics use declared/reference-derived times and URDF limits, not independently verified FPS or hardware limits.',
                           'No numeric velocity/acceleration threshold implies physical acceptance.',
                           'SSM feature-grid joints are reconstructed by linear interpolation; no FK or feature re-encoding here.',
                           'Captions, source identity, physical feasibility and applicable license terms remain separate.']}
    with (HERE/'report.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'cases':len(cases),'unaccepted_captions':sum(not c['caption_accepted'] for c in cases),
                      'nonzero_raw_limit_excess':sum(c['raw_trajectory']['max_limit_excess_rad']>0 for c in cases),
                      'training_ready':False}),flush=True)


if __name__=='__main__':main()

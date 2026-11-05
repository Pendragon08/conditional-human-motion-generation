"""Recompute an explicitly conditional COM diagnostic during CPU preflight.

This does not prove physical feasibility, actual gravity, ground, timing, or
training eligibility. The model asset bundle must be pinned by a prior report.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def sampled_vertical_support(times, com, gravity_z):
    t=np.asarray(times,dtype=np.float64);p=np.asarray(com,dtype=np.float64)
    if t.ndim!=1 or len(t)<3 or p.shape!=(len(t),3) or not np.isfinite(t).all() or not np.isfinite(p).all() or not np.isfinite(gravity_z):
        raise ValueError('invalid timeline, COM, or gravity')
    dt=np.diff(t)
    if np.any(dt<=0) or not np.allclose(dt,dt[0],atol=1e-8,rtol=0):
        raise ValueError('expected an evenly sampled timeline')
    return np.diff(p[:,2],n=2)/(dt[0]*dt[0])-float(gravity_z)

def audit_vertical_support(records, *, state_root=None, state_manifest=None, scene=None,
                           asset_report=None, asset_report_sha256=None):
    keys=(state_root,state_manifest,scene,asset_report,asset_report_sha256)
    if all(x is None for x in keys):
        return {'status':'missing_vertical_diagnostic','training_ready':False}
    if any(x is None for x in keys):raise ValueError('vertical diagnostic requires state root, manifest, scene and pinned asset report')
    if not isinstance(records,list) or not records:raise ValueError('records required')
    import mujoco
    state_root=Path(state_root).resolve(strict=True);manifest=Path(state_manifest).resolve(strict=True)
    scene=Path(scene).resolve(strict=True);asset_report=Path(asset_report).resolve(strict=True)
    manifest_hash=digest(manifest)
    if digest(asset_report)!=asset_report_sha256:raise ValueError('asset report hash mismatch')
    assets=json.loads(asset_report.read_text(encoding='utf8'))
    if digest(scene)!=assets['scene_sha256']:raise ValueError('scene hash mismatch')
    model_xml=scene.parent/'g1_27dof_with_hand_rev_1_0.xml'
    if digest(model_xml)!=assets['model_xml_sha256']:raise ValueError('model XML hash mismatch')
    meshes=assets['model_meshes']
    if not isinstance(meshes,list) or not meshes:raise ValueError('mesh hashes required')
    for x in meshes:
        p=Path(x['path']).resolve(strict=True)
        if not p.is_relative_to(scene.parent) or digest(p)!=x['sha256']:raise ValueError('model mesh hash mismatch')
    rows=json.loads(manifest.read_text(encoding='utf8'))['records'];by_id={r['sample_id']:r for r in rows}
    wanted={r['sample_id'] for r in records}
    if len(by_id)!=len(rows) or len(wanted)!=len(records) or set(by_id)!=wanted:raise ValueError('state IDs differ from dataset')
    for r in records:
        if by_id[r['sample_id']]['feature_sha256']!=r['feature_sha256']:raise ValueError('state and feature hashes differ')
    model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model)
    root=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'floating_base_joint')
    if root<0 or model.jnt_type[root]!=mujoco.mjtJoint.mjJNT_FREE:raise ValueError('missing free base')
    rootbody=int(model.jnt_bodyid[root]);address=int(model.jnt_qposadr[root])
    bodies=[i for i in range(model.nbody) if model.body_rootid[i]==rootbody]
    mass=sum(float(model.body_mass[i]) for i in bodies)
    if mass<=0:raise ValueError('invalid robot mass')
    cases=[]
    for r in records:
        sid=r['sample_id'];state=by_id[sid];p=(state_root/state['state_path']).resolve(strict=True)
        if not p.is_relative_to(state_root) or digest(p)!=state['state_sha256']:raise ValueError('state file hash or path mismatch')
        with np.load(p,allow_pickle=False) as f:
            rootq=np.asarray(f['root_wxyz_xyz'],dtype=float);q=np.asarray(f['joint_pos'],dtype=float)
            t=np.asarray(f['sample_times'],dtype=float);names=[str(n) for n in f['joint_names']]
        if rootq.shape!=(len(t),7) or q.shape!=(len(t),len(names)) or len(set(names))!=len(names) or not np.isfinite(rootq).all() or not np.isfinite(q).all():raise ValueError('invalid state arrays')
        addresses=[]
        for name in names:
            j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
            if j<0:raise ValueError('missing joint '+name)
            addresses.append(int(model.jnt_qposadr[j]))
        com=[]
        for base,angles in zip(rootq,q):
            mujoco.mj_resetData(model,data);data.qpos[address:address+3]=base[4:];data.qpos[address+3:address+7]=base[:4]
            for addr,value in zip(addresses,angles):data.qpos[addr]=value
            mujoco.mj_forward(model,data)
            com.append([sum(float(model.body_mass[i])*float(data.xipos[i,k]) for i in bodies)/mass for k in range(3)])
        support=sampled_vertical_support(t,com,float(model.opt.gravity[2]))
        if digest(p)!=state['state_sha256']:raise RuntimeError('state changed during diagnostic')
        cases.append({'sample_id':sid,'state_sha256':state['state_sha256'],'frames':len(t),
                      'minimum_n_per_kg':float(support.min()),'negative_beyond_1e_6':int((support < -1e-6).sum())})
    if digest(manifest)!=manifest_hash or digest(asset_report)!=asset_report_sha256 or digest(scene)!=assets['scene_sha256']:
        raise RuntimeError('diagnostic inputs changed')
    return {'status':'computed_necessary_condition_only','state_manifest_sha256':manifest_hash,
            'scene_sha256':assets['scene_sha256'],'asset_report_sha256':asset_report_sha256,
            'model_mass_kg':mass,'gravity_z_m_s2':float(model.opt.gravity[2]),'cases':cases,'training_ready':False,
            'limitations':['Nominal sample times and old model gravity are assumptions.',
                           'No downward external force other than gravity is assumed.',
                           'Passing this necessary inequality does not prove contact, force feasibility or full dynamics.',
                           'No source, licensing or training acceptance.']}

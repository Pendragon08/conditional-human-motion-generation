"""Read official 29DoF motions and screen technical validity/declared joint limits.

No retargeting, no claim of physical tracking or caption correctness.
"""
import hashlib,io,json,xml.etree.ElementTree as ET,zipfile
from pathlib import Path
import numpy as np
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
U=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf')
names=('left_hip_pitch','left_hip_roll','left_hip_yaw','left_knee','left_ankle_pitch','left_ankle_roll','right_hip_pitch','right_hip_roll','right_hip_yaw','right_knee','right_ankle_pitch','right_ankle_roll','waist_yaw','waist_roll','waist_pitch','left_shoulder_pitch','left_shoulder_roll','left_shoulder_yaw','left_elbow','left_wrist_roll','left_wrist_pitch','left_wrist_yaw','right_shoulder_pitch','right_shoulder_roll','right_shoulder_yaw','right_elbow','right_wrist_roll','right_wrist_pitch','right_wrist_yaw')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 prev=json.loads((R/'omni_direct_waist_audit_20261103.json').read_text());known={(x['archive'],x['member']):x for x in prev['rows']}
 lim={j.get('name'):(float(j.find('limit').get('lower')),float(j.find('limit').get('upper'))) for j in ET.parse(U).getroot().findall('joint') if j.find('limit') is not None and j.get('type') in ('revolute','prismatic')}
 lower=np.array([lim[n+'_joint'][0] for n in names]);upper=np.array([lim[n+'_joint'][1] for n in names]);rows=[]
 for archive,digest in prev['archives_sha256'].items():
  path=R/'data/OmniRetarget_Dataset'/archive
  if sha(path)!=digest:raise ValueError('archive changed')
  with zipfile.ZipFile(path) as z:
   for info in z.infolist():
    key=(archive,info.filename)
    if key not in known:continue
    raw=z.read(info)
    if hashlib.sha256(raw).hexdigest()!=known[key]['member_sha256']:raise ValueError('member mismatch')
    with np.load(io.BytesIO(raw),allow_pickle=False) as f:q=np.asarray(f['qpos'],dtype=float)
    excess=np.maximum(np.maximum(lower-q[:,7:36],q[:,7:36]-upper),0)
    robot_error=float(np.max(np.abs(np.linalg.norm(q[:,:4],axis=1)-1)))
    object_error=float(np.max(np.abs(np.linalg.norm(q[:,36:40],axis=1)-1))) if q.shape[1]==43 else None
    rows.append({'archive':archive,'member':info.filename,'sha256':known[key]['member_sha256'],'joint_max_excess_rad':float(excess.max()),'joint_count_above_1e-6':int(np.count_nonzero(excess>1e-6)),'root_quaternion_max_error':robot_error,'object_quaternion_max_error':object_error,'technical_pass':bool(excess.max()<=1e-6 and robot_error<=1e-4 and (object_error is None or object_error<=1e-4))})
  if sha(path)!=digest:raise ValueError('archive changed')
 if len(rows)!=len(known) or len({(x['archive'],x['member']) for x in rows})!=len(known):raise ValueError('coverage mismatch')
 report={'trajectory_count':len(rows),'urdf_sha256':sha(U),'joint_order_witness':'reports/drake_joint_order_witness_20261030.json','source_audit_sha256':sha(R/'omni_direct_waist_audit_20261103.json'),'joint_limit_tolerance_rad':1e-6,'quaternion_norm_tolerance':1e-4,'counts':{'technical_pass':sum(x['technical_pass'] for x in rows),'joint_limit_failed':sum(x['joint_max_excess_rad']>1e-6 for x in rows),'root_quaternion_failed':sum(x['root_quaternion_max_error']>1e-4 for x in rows),'object_quaternion_failed':sum(x['object_quaternion_max_error'] is not None and x['object_quaternion_max_error']>1e-4 for x in rows)},'rows':rows,'training_ready':False,'limitations':['These are technical checks against one URDF, not actual physics or motion tracking.','No caption, license scope, real clock or split acceptance.']}
 with (R/'omni29_direct_kinematic_audit_20261103.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
 print(json.dumps(report['counts']))
if __name__=='__main__':main()

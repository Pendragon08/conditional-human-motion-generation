"""Read-only GMR full-action audit; compare source and prior partial run."""
import hashlib,json,sys,xml.etree.ElementTree as ET,zipfile
from pathlib import Path
import numpy as np
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
U=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf')
sys.path.insert(0,str(R/'grab_full_human_work_v10_20261103'))
from grab_restricted_reader import read_grab_field
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 folder=R/'gmr_grab29_full_airplane_20261103';p=folder/'gmr_source_rate.npz';report=json.loads((folder/'report.json').read_text()); archive=R/'data/GRAB/grab__s2.zip'
 if sha(p)!=report['output_sha256'] or sha(archive)!=report['archive_sha256']:raise ValueError('hash mismatch')
 with zipfile.ZipFile(archive) as z:
  raw=z.read(report['member'])
  if hashlib.sha256(raw).hexdigest()!=report['source_member_sha256']:raise ValueError('member mismatch')
  obj=read_grab_field(z,report['member'],'object')
 with np.load(p,allow_pickle=False) as f:
  frames=f['source_frames'];names=list(f['joint_names']);q=f['joint_pos'];root=f['root_wxyz_xyz'];o=f['object_transl'];rot=f['object_orient_rotvec'];human=f['human_joints']
 if not np.array_equal(frames,np.arange(1198)) or q.shape!=(1198,29) or len(set(names))!=29 or root.shape!=(1198,7) or human.shape!=(1198,22,3):raise ValueError('frame or shape mismatch')
 if not all(np.isfinite(x).all() for x in (q,root,o,rot,human)):raise ValueError('nonfinite')
 if not np.array_equal(o,np.asarray(obj['params']['transl'])) or not np.array_equal(rot,np.asarray(obj['params']['global_orient'])):raise ValueError('object source mismatch')
 limits={j.get('name'):(float(j.find('limit').get('lower')),float(j.find('limit').get('upper'))) for j in ET.parse(U).getroot().findall('joint') if j.get('type') in ('revolute','prismatic') and j.find('limit') is not None}
 lo=np.asarray([limits[n][0] for n in names]);hi=np.asarray([limits[n][1] for n in names]);excess=np.maximum(np.maximum(lo-q,q-hi),0)
 prior=R/'gmr_grab_29dof_smoke_20261103/gmr_source_rate.npz'
 with np.load(prior,allow_pickle=False) as f:
  if not np.array_equal(f['source_frames'],frames[540:660]) or list(f['joint_names'])!=names:raise ValueError('prior source mismatch')
  partial_joint_diff=float(np.max(np.abs(q[540:660]-f['joint_pos'])));partial_root_diff=float(np.max(np.linalg.norm(root[540:660,4:]-f['root_wxyz_xyz'][:,4:],axis=1)))
 out={'source_sha256':sha(archive),'member_sha256':report['source_member_sha256'],'result_sha256':sha(p),'urdf_sha256':sha(U),'frames':len(frames),'joint_count':len(names),'object_arrays_equal_source':True,'max_limit_excess_rad':float(excess.max()),'count_above_1e-6':int(np.count_nonzero(excess>1e-6)),'root_quaternion_max_error':float(np.max(np.abs(np.linalg.norm(root[:,:4],axis=1)-1))),'prior_partial_max_joint_difference_rad':partial_joint_diff,'prior_partial_max_root_position_difference_m':partial_root_diff,'limits':['Same-library GMR not independently rerun.','Partial sample begins with different IK state; difference is not proof of correctness.','No semantic or physical-tracking acceptance.'],'training_ready':False}
 with (folder/'audit.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
 print(json.dumps(out))
if __name__=='__main__':main()

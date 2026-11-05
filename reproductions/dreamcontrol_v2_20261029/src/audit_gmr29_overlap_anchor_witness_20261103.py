"""Independently re-read anchored and baseline 29DoF window evidence."""
import hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');D=R/'gmr29_overlap_anchor_witness_20261103';B=R/'gmr29_pyroki_full_airplane_20261103'
U=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 report=json.loads((D/'report.json').read_text());p=D/'result.npz';previous=B/'window_0135.npz';baseline=B/'window_0136.npz'
 if sha(p)!=report['result_sha256'] or sha(previous)!=report['previous_window_sha256'] or sha(baseline)!=report['baseline_window_sha256']:raise ValueError('input hash mismatch')
 with np.load(p,allow_pickle=False) as f:idx=f['indices'];q=f['joint_pos'];root=f['root_wxyz_xyz'];names=list(f['joint_names'])
 with np.load(previous,allow_pickle=False) as f:pi=f['indices'];pq=f['joint_pos'];pr=f['root_wxyz_xyz'];pn=list(f['joint_names'])
 with np.load(baseline,allow_pickle=False) as f:bi=f['indices'];bq=f['joint_pos'];br=f['root_wxyz_xyz']
 if names!=pn or not np.array_equal(idx,bi) or q.shape!=(7,29) or root.shape!=(7,7) or not np.isfinite(q).all() or not np.isfinite(root).all():raise ValueError('shape or mapping')
 common,ia,ib=np.intersect1d(pi,idx,return_indices=True)
 def diff(r,j):return {'joint_rad':float(np.max(np.abs(pq[ia]-j[ib]))),'root_m':float(np.max(np.linalg.norm(pr[ia,4:]-r[ib,4:],axis=1))),'root_angle_rad':float(np.max((Rotation.from_quat(pr[ia][:,[1,2,3,0]]).inv()*Rotation.from_quat(r[ib][:,[1,2,3,0]])).magnitude()))}
 before,after=diff(br,bq),diff(root,q)
 if any(abs(before[k]-report['baseline'][k])>1e-9 or abs(after[k]-report['anchored'][k])>1e-9 for k in before):raise ValueError('overlap metrics mismatch')
 limits={j.get('name'):(float(j.find('limit').get('lower')),float(j.find('limit').get('upper'))) for j in ET.parse(U).getroot().findall('joint') if j.get('type') in ('revolute','prismatic') and j.find('limit') is not None};lo=np.array([limits[n][0] for n in names]);hi=np.array([limits[n][1] for n in names]);excess=np.maximum(np.maximum(lo-q,q-hi),0)
 out={'result_sha256':sha(p),'previous_sha256':sha(previous),'baseline_sha256':sha(baseline),'urdf_sha256':sha(U),'overlap_frames':common.tolist(),'baseline':before,'anchored':after,'max_joint_limit_excess_rad':float(excess.max()),'values_above_1e-6':int(np.count_nonzero(excess>1e-6)),'quaternion_max_norm_error':float(np.max(np.abs(np.linalg.norm(root[:,:4],axis=1)-1))),'training_ready':False,'limitations':['One window re-read only; no independent re-solve.','No full action, source fidelity, real timing or dynamic tracking verified.','Overlap anchor is an experimental objective, not author-v2 implementation.']}
 with (D/'audit.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
 print(json.dumps(out))
if __name__=='__main__':main()

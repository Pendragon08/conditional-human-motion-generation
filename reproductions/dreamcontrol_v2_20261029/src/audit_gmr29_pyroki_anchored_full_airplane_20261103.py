"""Audit all 29DoF source-rate windows and timing; no physical-action filter."""
import hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');D=R/'gmr29_pyroki_anchored_full_airplane_20261103'
U=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 rep=json.loads((D/'report.json').read_text());rows=[json.loads(x) for x in (D/'windows.jsonl').read_text().splitlines()]
 if sha(D/'windows.jsonl')!=rep['rows_sha256'] or sha(U)!=rep['urdf_sha256'] or len(rows)!=rep['window_count']:raise ValueError('manifest mismatch')
 src=R/'gmr_grab29_full_airplane_20261103/gmr_source_rate.npz'
 if sha(src)!=rep['source_sha256']:raise ValueError('source mismatch')
 with np.load(src,allow_pickle=False) as f:frames=f['source_frames'];source_names=list(f['joint_names'])
 limits={j.get('name'):(float(j.find('limit').get('lower')),float(j.find('limit').get('upper'))) for j in ET.parse(U).getroot().findall('joint') if j.get('type') in ('revolute','prismatic') and j.find('limit') is not None}
 covered=set();prev=None;records=[];overlaps=[]
 for i,row in enumerate(rows):
  if row['window']!=i:raise ValueError('window order')
  path=D/row['file']
  if path.parent!=D or sha(path)!=row['sha256']:raise ValueError('artifact mismatch')
  with np.load(path,allow_pickle=False) as f:idx=f['indices'];sf=f['source_frames'];names=list(f['joint_names']);q=f['joint_pos'];root=f['root_wxyz_xyz']
  if idx.shape!=(7,) or not np.array_equal(idx,np.arange(row['start']+1,row['start']+8)) or not np.array_equal(sf,frames[idx]) or len(names)!=29 or set(names)!=set(source_names) or q.shape!=(7,29) or root.shape!=(7,7) or not np.isfinite(q).all() or not np.isfinite(root).all():raise ValueError('window data')
  lo=np.array([limits[n][0] for n in names]);hi=np.array([limits[n][1] for n in names]);exc=np.maximum(np.maximum(lo-q,q-hi),0);covered.update(idx.tolist())
  rowout={'window':i,'max_limit_excess_rad':float(exc.max()),'limit_values_above_1e-6':int(np.count_nonzero(exc>1e-6)),'quat_max_norm_error':float(np.max(np.abs(np.linalg.norm(root[:,:4],axis=1)-1)))}
  if prev is not None:
   common=np.intersect1d(prev['idx'],idx);a=np.searchsorted(prev['idx'],common);b=np.searchsorted(idx,common)
   qa=Rotation.from_quat(prev['root'][a][:,[1,2,3,0]]);qb=Rotation.from_quat(root[b][:,[1,2,3,0]])
   actual={'root_m':float(np.max(np.linalg.norm(prev['root'][a,4:]-root[b,4:],axis=1))),'root_angle_rad':float(np.max((qa.inv()*qb).magnitude())),'joint_rad':float(np.max(np.abs(prev['q'][a]-q[b])))}
   if any(abs(actual[k]-row['overlap'][k])>1e-9 for k in actual):raise ValueError('overlap mismatch')
   overlaps.append(actual)
  prev={'idx':idx,'q':q,'root':root};records.append(rowout)
 expected=set(range(1,len(frames)))
 if covered!=expected:raise ValueError('coverage mismatch')
 result={'report_sha256':sha(D/'report.json'),'source_sha256':sha(src),'windows':len(rows),'unique_covered_source_frames':len(covered),'missing_source_frame_0_by_design':True,'source_total_frames':len(frames),'max_joint_excess_rad':max(x['max_limit_excess_rad'] for x in records),'window_count_limit_excess_above_1e-6':sum(x['limit_values_above_1e-6']>0 for x in records),'max_quaternion_norm_error':max(x['quat_max_norm_error'] for x in records),'max_overlap_joint_rad':max(x['joint_rad'] for x in overlaps),'max_overlap_root_m':max(x['root_m'] for x in overlaps),'max_overlap_root_angle_rad':max(x['root_angle_rad'] for x in overlaps),'elapsed_s':rep['elapsed_s'],'first_window_s':rep['first_window_s'],'subsequent_median_window_s':rep['subsequent_median_window_s'],'mean_subsequent_window_s':float(np.mean([x['elapsed_s'] for x in rows[1:]])),'training_ready':False,'limitations':['Re-reads outputs; does not re-solve windows.','Overlaps are diagnostics, not an author threshold or physical-action filtering.','Source frame zero is missing, windows have not been stitched, and text/time remain unaccepted.']}
 with (D/'audit.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
 print(json.dumps(result))
if __name__=='__main__':main()

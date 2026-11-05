"""Audit artifact provenance and 29DoF timing; not semantic acceptance."""
import hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
O=R/'omni29_direct_text_motion_pilot_20261103'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 report=json.loads((O/'report.json').read_text());p=O/'state_features.npz';archive=R/'data/OmniRetarget_Dataset/robot-object.zip';anno=R/'omomo_text_pairing_20261101/omomo_text_anno.zip'
 if sha(p)!=report['output_sha256'] or sha(archive)!=report['archive_sha256'] or sha(anno)!=report['annotation_archive_sha256']:raise ValueError('hash mismatch')
 with zipfile.ZipFile(archive) as z:raw=z.read(report['source_member'])
 with zipfile.ZipFile(anno) as z:ann=z.read(report['annotation_member'])
 if hashlib.sha256(raw).hexdigest()!=report['source_member_sha256'] or hashlib.sha256(ann).hexdigest()!=report['annotation_sha256']:raise ValueError('member hash')
 if json.loads(ann)['sub3_largebox_011']!=report['caption']:raise ValueError('caption mismatch')
 with np.load(io.BytesIO(raw),allow_pickle=False) as f:source=f['qpos'];sfps=float(f['fps'])
 with np.load(p,allow_pickle=False) as f:q=f['qpos29_source_layout_20hz'];features=f['features'];points=f['points_z_up'];t=f['sample_times'];names=f['joint_names']
 if source.shape!=(192,43) or sfps!=30 or q.shape!=(128,43) or features.shape!=(127,263) or points.shape!=(128,22,3) or len(names)!=29:raise ValueError('shape')
 if not all(np.isfinite(x).all() for x in (q,features,points,t)) or not np.allclose(t,np.arange(128)/20.,atol=1e-12,rtol=0):raise ValueError('numeric or time mismatch')
 expected=source[0].copy();expected[0:4]/=np.linalg.norm(expected[0:4]);expected[36:40]/=np.linalg.norm(expected[36:40])
 if not np.allclose(q[0],expected,atol=1e-12,rtol=0):raise ValueError('first frame mismatch after unit normalization')
 if not np.allclose(np.linalg.norm(q[:,:4],axis=1),1,atol=1e-6,rtol=0) or not np.allclose(np.linalg.norm(q[:,36:40],axis=1),1,atol=1e-6,rtol=0):raise ValueError('quaternion norm')
 result={'artifact_sha256':sha(p),'source_member_sha256':hashlib.sha256(raw).hexdigest(),'annotation_sha256':hashlib.sha256(ann).hexdigest(),'caption':report['caption'],'states':128,'features':127,'source_fps_declared':30,'target_fps':20,'nominal_last_sample_s':float(t[-1]),'source_last_sample_s':float((len(source)-1)/sfps),'first_frame_match_after_quaternion_normalization':True,'first_frame_normalization_max_change':float(np.max(np.abs(q[0]-source[0]))),'waist_max_abs_rad':float(np.max(np.abs(q[:,[20,21]]))),'training_ready':False,'limitations':['Hash/shape/timing checks do not prove full resampling or re-encode independently.','Filename and caption hash do not prove semantic or recorded source identity.','Prior review used a 27DoF preview, not this 29DoF sequence.','Source actual time and license scope remain unresolved; no physical filtering.']}
 with (O/'audit_v2.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
 print(json.dumps(result))
if __name__=='__main__':main()

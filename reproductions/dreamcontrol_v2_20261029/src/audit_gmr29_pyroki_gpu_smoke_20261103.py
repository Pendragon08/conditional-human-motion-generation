"""Independently re-read the GMR-warm-start smoke output; no physical claims."""
import hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
S=R/'gmr_grab_29dof_smoke_20261103/gmr_source_rate.npz'
O=R/'gmr29_pyroki_gpu_smoke_20261103'
H=R/'gmr_pyroki_initialization_20261103/retarget_helpers/_utils.py'
U=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 r=json.loads((O/'report.json').read_text());p=O/'result.npz'
 if sha(S)!=r['source_sha256'] or sha(p)!=r['output_sha256'] or sha(H)!=r['helper_sha256'] or sha(U)!=r['urdf_sha256']:raise ValueError('input hash mismatch')
 with np.load(S,allow_pickle=False) as f:
  sn=list(f['joint_names']);sf=f['source_frames'];sq=f['joint_pos'];sr=f['root_wxyz_xyz']
 with np.load(p,allow_pickle=False) as f:
  names=list(f['joint_names']);frames=f['source_frames'];q=f['joint_pos'];root=f['root_wxyz_xyz'];iq=f['gmr_joint_pos'];ir=f['gmr_root_wxyz_xyz']
 if set(names)!=set(sn) or len(names)!=29 or not np.array_equal(frames,sf[1:8]):raise ValueError('joint or frame identity')
 if not np.array_equal(iq,sq[1:8][:,[sn.index(n) for n in names]]) or not np.array_equal(ir,sr[1:8]):raise ValueError('warm start artifact differs from source')
 if q.shape!=(7,29) or root.shape!=(7,7) or not np.isfinite(q).all() or not np.isfinite(root).all():raise ValueError('output shape or values')
 lim={j.get('name'):(float(j.find('limit').get('lower')),float(j.find('limit').get('upper'))) for j in ET.parse(U).getroot().findall('joint') if j.get('type') in ('revolute','prismatic') and j.find('limit') is not None}
 lo=np.array([lim[x][0] for x in names]);hi=np.array([lim[x][1] for x in names]);excess=np.maximum(np.maximum(lo-q,q-hi),0)
 init_excess=np.maximum(np.maximum(lo-iq,iq-hi),0);normerr=float(np.max(np.abs(np.linalg.norm(root[:,:4],axis=1)-1)))
 report={'source_sha256':sha(S),'result_sha256':sha(p),'helper_sha256':sha(H),'urdf_sha256':sha(U),'frames':frames.tolist(),'joint_count':len(names),'initial_values_match_source_exactly':True,'initial_max_limit_excess_rad':float(init_excess.max()),'output_max_limit_excess_rad':float(excess.max()),'output_limit_violations_above_1e-6':int(np.count_nonzero(excess>1e-6)),'quaternion_max_norm_error':normerr,'max_joint_change_rad':float(np.max(np.abs(q-iq))),'max_root_translation_change_m':float(np.max(np.linalg.norm(root[:,4:]-ir[:,4:],axis=1))),'limitations':['Re-read of a single seven-frame result, not independent re-solving.','Initial arrays match stored GMR inputs; objective initialization is based on project helper code.','Source timing, tracking, contact, text, dynamics and full-clip consistency are not checked.'],'training_ready':False}
 with (O/'audit.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
 print(json.dumps(report))
if __name__=='__main__':main()

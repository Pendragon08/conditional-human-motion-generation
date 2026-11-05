"""Read official archives without extracting; measure 29-DoF waist activity.

Publisher qpos layout is bound to the separately checked Drake joint-order
witness. This is not a physical tracking or training-quality test.
"""
import hashlib, io, json, zipfile
from pathlib import Path
import numpy as np
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
A=R/'data/OmniRetarget_Dataset'
EXPECTED={'robot-object.zip':'93001f298aec587cd2b0a79aa5553d264cf2148e5397d4aa4bb767a9d58041b3','robot-terrain.zip':'f30bdc915287547dbb2225bdb4efe323dbfcc2237e2a4e82c7482dd05402aea6','robot-object-terrain.zip':'c3e2a7a3827182b23f5e6b72f496cda0448ee58c4ebe6f316bee46adc21c5505'}
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 out=R/'omni_direct_waist_audit_20261103.json'
 if out.exists(): raise FileExistsError(out)
 rows=[]; failures=[]
 for archive,digest in EXPECTED.items():
  path=A/archive
  if sha(path)!=digest: raise ValueError('archive hash mismatch '+archive)
  with zipfile.ZipFile(path) as z:
   for member in z.infolist():
    if member.is_dir() or not member.filename.endswith('.npz'): continue
    try:
     raw=z.read(member)
     with np.load(io.BytesIO(raw),allow_pickle=False) as f:
      q=np.asarray(f['qpos']); fps=float(np.asarray(f['fps']).reshape(()))
     if q.ndim!=2 or q.shape[1] not in (36,43) or len(q)<2 or not np.isfinite(q).all() or not np.isfinite(fps) or fps<=0:raise ValueError('invalid qpos/fps')
     # Publisher base is 7 positions, waist roll/pitch are indices 13,14 of 29 joints.
     w=np.asarray(q[:,[20,21]],dtype=np.float64)
     rows.append({'archive':archive,'member':member.filename,'member_sha256':hashlib.sha256(raw).hexdigest(),'frames':len(q),'fps_declared':fps,'qpos_width':q.shape[1],'max_abs_waist_rad':float(np.max(np.abs(w))),'exactly_zero_both':bool(np.all(w==0))})
    except Exception as e: failures.append({'archive':archive,'member':member.filename,'error':repr(e)})
  if sha(path)!=digest:raise ValueError('archive changed '+archive)
 values=np.array([r['max_abs_waist_rad'] for r in rows])
 bins={str(t):int(np.count_nonzero(values<=t)) for t in (0.,1e-6,1e-4,.001,.01,.05,.1)}
 result={'archives_sha256':EXPECTED,'source_revision':'135259130158f1cb1799e4ab9b6566c938285fc9','joint_order_witness':'reports/drake_joint_order_witness_20261030.json','trajectories':len(rows),'invalid_members':failures,'counts_at_or_below_rad':bins,'maximum_rad':float(values.max()) if len(values) else None,'rows':rows,'training_ready':False,'limitations':['Zero waist alone does not prove 29-DoF and target 27-DoF models have identical FK or collisions.','Nonzero waist requires a 29-DoF model, a documented conversion, or exclusion for a 27-DoF model.','This does not validate caption semantics, source clock or actual tracking.']}
 with out.open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
 print(json.dumps({'trajectories':len(rows),'invalid':len(failures),'counts_at_or_below_rad':bins,'max_rad':result['maximum_rad']}))
if __name__=='__main__': main()

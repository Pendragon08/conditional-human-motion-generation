"""Independent scalar-form source-frequency witness for three Omni variant comparisons."""
import hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
D=R/'omni29_augmentation_consistency_20261103'
M=R/'grab_omni_mixed29_full_preview_20261103/manifest.json'
Z=R/'data/OmniRetarget_Dataset/robot-object.zip'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def invrot(q,v):
 # q = wxyz, v = world vector. Explicit quaternion inverse rotation.
 q=np.asarray(q,float);q=q/np.linalg.norm(q); w=q[0];u=-q[1:]
 return v+2*np.cross(u,np.cross(u,v)+w*v)
def main():
 summary=json.loads((D/'report.json').read_text());rec=D/'records.jsonl'
 if sha(rec)!=summary['records_sha256'] or sha(M)!=summary['manifest_sha256']:raise ValueError('binding')
 comparisons=sorted((json.loads(x) for x in rec.read_text().splitlines()),key=lambda x:x['joint_max_abs_rad'])
 selected=[comparisons[0],comparisons[len(comparisons)//2],comparisons[-1]]
 rows={x['sample_id']:x for x in json.loads(M.read_text())['records']};results=[]
 with zipfile.ZipFile(Z) as z:
  for row in selected:
   pair=[]
   for sid in (row['original_id'],row['id']):
    r=rows[sid];raw=z.read(r['source_member'])
    if hashlib.sha256(raw).hexdigest()!=r['source_sha256']:raise ValueError('source member hash')
    with np.load(io.BytesIO(raw),allow_pickle=False) as f:q=np.asarray(f['qpos'],float);fps=float(f['fps'])
    pair.append((q,fps))
   a,b=pair[0][0],pair[1][0]
   if len(a)!=len(b) or pair[0][1]!=30 or pair[1][1]!=30:raise ValueError('source shape/fps')
   joint=np.abs(a[:,7:36]-b[:,7:36]);frame,col=np.unravel_index(np.argmax(joint),joint.shape)
   # Check every source frame independently using scalar quaternion arithmetic.
   distances=[]
   for k in range(len(a)):
    pa=invrot(a[k,:4],a[k,40:43]-a[k,4:7]);pb=invrot(b[k,:4],b[k,40:43]-b[k,4:7])
    distances.append(float(np.linalg.norm(pa-pb)))
   results.append({'variant':row['id'],'original':row['original_id'],'source_fps_nominal':30,'source_frames':len(a),
      'source_joint_max_abs_rad':float(joint.max()),'source_joint_max_frame':int(frame),'joint_index':int(col),
      'original_joint_rad':float(a[frame,7+col]),'variant_joint_rad':float(b[frame,7+col]),
      'source_object_root_max_m':max(distances),'preview_joint_max_abs_rad':row['joint_max_abs_rad']})
 result={'variant_records_sha256':sha(rec),'cases':results,'limits':['This confirms numerical variation in source files, not semantic mismatch, contact, or tracking quality.','No physical-action filtering applied.']}
 with (D/'source_witness.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
 print(json.dumps(result))
if __name__=='__main__':main()

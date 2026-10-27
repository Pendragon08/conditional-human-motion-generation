"""Verify each NPZ against local provenance; do not manufacture data."""
from pathlib import Path
import argparse, json, hashlib
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--root',default='data');p.add_argument('--output',default='results/data_inventory.json');args=p.parse_args()
out={'reference_type':'video-reconstructed, GMR-retargeted, FK-derived kinematic reference; not measured robot trajectory','splits':{}}
all_names=None
for split in ['train','test']:
 folder=Path(args.root)/split
 if not folder.exists():continue
 manifest=json.loads((folder/'manifest.json').read_text())
 records={x['output']:x for x in manifest['records']}
 checked=[]
 for f in sorted(folder.glob('*.npz')):
  digest=hashlib.sha256(f.read_bytes()).hexdigest()
  assert digest==records[f.name]['sha256'],f
  with np.load(f,allow_pickle=False) as data:
   names=(data['joint_names'].tolist(),data['body_names'].tolist())
   if all_names is None:all_names=names
   assert names==all_names,f
   n=data['joint_pos'].shape[0]
   for k in ['joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w']:
    assert data[k].shape[0]==n and np.isfinite(data[k]).all(),(f,k)
   assert np.allclose(np.linalg.norm(data['body_quat_w'],axis=-1),1,atol=1e-4),f
   fps=float(data['fps'].reshape(-1)[0]);assert fps==50
   checked.append({'file':f.name,'sha256':digest,'frames':n,'fps':fps,'duration_s':n/fps,'input_source_sha256':records[f.name]['source_sha256']})
 out['splits'][split]=checked
train={r['sha256'] for r in out['splits'].get('train',[])}
test={r['sha256'] for r in out['splits'].get('test',[])}
assert not train.intersection(test),'identical files across splits'
out['names']={'joints':all_names[0],'bodies':all_names[1]}
out['split_check']='no identical NPZ hashes; source-group independence must be checked separately'
dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'files':{k:len(v) for k,v in out['splits'].items()},'output':str(dest)}))

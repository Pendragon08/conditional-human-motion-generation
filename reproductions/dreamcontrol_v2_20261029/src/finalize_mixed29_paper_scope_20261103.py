"""Record completion of requested paper-scope DATASET construction only.

Decisions about model training, legal scope or independent semantic truth are
not made by this receipt. Published text is kept as weak source supervision.
"""
import hashlib,json
from pathlib import Path
from collections import Counter
import numpy as np
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
D=R/'grab_omni_29dof_weak_baseline_v1_20261103'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 out=D/'paper_scope_completion.json'
 if out.exists():raise FileExistsError(out)
 rep=json.loads((D/'report.json').read_text());pack=json.loads((D/'package_audit.json').read_text());feat=json.loads((D/'feature_audit.json').read_text())
 m=D/'manifest.json';s=D/'stats.npz'
 if sha(m)!=rep['manifest_sha256'] or sha(m)!=pack['manifest_sha256'] or sha(s)!=rep['stats_sha256'] or sha(s)!=pack['stats_sha256']:raise ValueError('package evidence mismatch')
 if not pack['technical_format_passed'] or not feat['technical_checks_passed'] or feat['exact_cross_split_duplicates']!=0:raise ValueError('technical audit incomplete')
 rows=json.loads(m.read_text())['records'];counts=Counter();texts=Counter();splits=Counter();states=0;maxn=0
 for r in rows:
  if r['fps']!=20.:raise ValueError('frame rate')
  p=R/r['state_path']
  if sha(p)!=r['state_sha256']:raise ValueError('state checksum')
  with np.load(p,allow_pickle=False) as f:
   key=next((k for k in ('state29_wxyz_xyz','qpos29_source_layout_20hz') if k in f),None)
   if key is None:raise ValueError('missing 29DoF states')
   q=np.asarray(f[key]);names=np.asarray(f['joint_names']);x=np.asarray(f['features']);times=np.asarray(f['sample_times'])
   if len(names)!=29 or len(set(names.tolist()))!=29 or q.shape[1] not in (36,43) or len(q)!=len(x)+1 or x.shape[1]!=263 or len(q)>200 or not np.isfinite(q).all() or not np.allclose(times,np.arange(len(q))/20.,atol=1e-10,rtol=0):raise ValueError('format')
  caps=r.get('candidate_captions',[r['caption']])
  if not isinstance(caps,list) or not caps or r['caption'] not in caps or any(not isinstance(x,str) or not x.strip() for x in caps):raise ValueError('captions')
  counts[r['dataset_source']]+=1;texts[r['dataset_source']]+=len(caps);splits[r['split']]+=1;states+=len(q);maxn=max(maxn,len(q))
 if dict(counts)!={'OmniRetarget/OMOMO':1826,'DexWeave/GRAB':1335} or dict(texts)!={'OmniRetarget/OMOMO':1826,'DexWeave/GRAB':4005}:raise ValueError('coverage')
 result={'scope':'dataset_construction_using_published_caption_pairs_and_existing_29DoF_robot_references',
  'paper_scope_dataset_construction_complete':True,'source_counts':dict(counts),'available_caption_pair_counts':dict(texts),
  'trajectory_records':len(rows),'available_text_motion_pairs':sum(texts.values()),'splits':dict(splits),
  'nominal_fps':20,'max_states_per_reference':maxn,'joint_count':29,'feature_dim':263,'keypoints':22,
  'state_frames':states,'train_feature_frames':feat['train_frames'],
  'manifest_sha256':sha(m),'stats_sha256':sha(s),'package_audit_sha256':sha(D/'package_audit.json'),'feature_audit_sha256':sha(D/'feature_audit.json'),
  'published_caption_supervision':'preserved_source_sequence_weak_labels','extra_semantic_screening_applied':False,
  'physical_action_screening_applied':False,'training_started':False,
  'limits':['This confirms construction and technical format only; independent semantic truth and data-use permissions are not certified.',
   'Caps/truncation and modified Omni references retain original source-sequence text as weak supervision.',
   '29DoF, DexWeave reuse, sign-fixed encoding and FIR quaternion filtering are documented adaptations rather than exact author implementation.',
   'The current loader uses the primary caption; all source alternatives are retained as candidate_captions.']}
 with out.open('x',encoding='utf8') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
 print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()

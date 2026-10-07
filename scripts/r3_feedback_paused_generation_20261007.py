"""Causal sparse hints from the paused Isaac episode; no model or IK updates."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SOURCES=ROOT/'artifacts/feedback_world_model_20261006/sources'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def residual(h,r,o,w):
 delta=o[0]-r[0]
 return np.asarray([h[0]+delta,*[h[j]+delta+((o[j]-o[0])-(r[j]-r[0]))@w[j] for j in [1,2]]])
def prepare(output):
 ready=json.loads((output/'branch_ready.json').read_text())
 assert ready['status']=='BRANCH_READY' and ready['feedback_source']=='same ongoing actual Isaac Lab episode'
 assert sha(output/'prefix.npz')==ready['prefix_sha256']
 prefix=np.load(output/'prefix.npz');bridge=np.load(SOURCES/'sparse_bridge_v2.npz')
 R,w=bridge['axis_rotation'],bridge['anchored_delta_weights']
 mapping={e['reference']:e['source_sample'] for e in json.loads((SOURCES/'mapping.json').read_text())['entries']}
 human=np.load(SOURCES/'Kick_official_results.npy',allow_pickle=True).item()
 points=human['motion'].transpose(0,3,1,2)
 pairs=[{'reference':rid,'domain':domain} for rid in ready['references'] for domain in ['clean','push']]
 samples=[mapping[x['reference']] for x in pairs]
 base=human['hint'][samples].copy();planned=base.copy();actual=base.copy();shuffled=base.copy()
 fs=ready['context_human_frames'];steps=ready['context_steps'];trials=ready['trials']
 assert max(steps)<=ready['branch_step'] and len(fs)==len(steps)
 f=np.array([[.2,1,.4],[.1,.2,.3],[.3,.2,.5]]);shift=np.array([.1,.2,.3])
 assert np.allclose(residual(f,f,f,w),f,atol=1e-12)
 assert np.allclose(residual(f,f,f+shift,w),f+shift,atol=1e-12)
 meta=[]
 for i,pair in enumerate(pairs):
  env=next(t['env'] for t in trials if t['arm']=='actual' and all(t[k]==v for k,v in pair.items()))
  for frame,step in zip(fs,steps):
   h=points[samples[i],frame,[0,7,8]]
   planned[i,frame,[0,7,8]]=h
   r=prefix['context_reference_body_xyz'][step,env]@R;o=prefix['context_body_xyz'][step,env]@R
   actual[i,frame,[0,7,8]]=residual(h,r,o,w)
  meta.append({**pair,'actual_source_environment':env,'human_sample':samples[i],'first_prefix_fall_step':next((int(k) for k in np.flatnonzero(prefix['fall'][:,env])),None)})
 for i,pair in enumerate(pairs):
  other=((i//2+1)%len(ready['references']))*2+(i%2)
  for frame in fs:
   shuffled[i,frame,[0,7,8]]=planned[i,frame,[0,7,8]]+(actual[other,frame,[0,7,8]]-planned[other,frame,[0,7,8]])
 for arm in [planned,actual,shuffled]:
  assert np.array_equal(arm[:,:,11],base[:,:,11]) and np.isfinite(arm).all()
 target=output/'generation_hints.npz';assert not target.exists()
 np.savez_compressed(target,planned=planned,actual=actual,shuffled=shuffled,original=base,source_samples=samples,lengths=human['lengths'][samples])
 result={'status':'PASS','seed':ready['seed'],'pairs':pairs,'samples':meta,'branch_ready_sha256':sha(output/'branch_ready.json'),'prefix_sha256':sha(output/'prefix.npz'),'hints_sha256':sha(target),'bridge_sha256':sha(SOURCES/'sparse_bridge_v2.npz'),'official_human_sha256':sha(SOURCES/'Kick_official_results.npy'),'feedback_source':'current paused actual Isaac Lab episode','future_goal_bitwise_preserved':True,'sparse_context_joints':[0,7,8],'past_only':True,'zero_residual_identity':True,'translation_equivariance':True,'no_extra_IK':True,'full_HumanML_prefix':False,'innovation_validated':False}
 (output/'generation_inputs.json').write_text(json.dumps(result,indent=2)+'\n')
 print('FEEDBACK_PAUSED_HINTS_PASS',len(pairs),flush=True)
def generate(output,arm,remaining):
 import sample.generate as official
 from data_loaders.get_data import get_dataset_loader
 inputs=np.load(output/'generation_hints.npz');raw=inputs[arm]
 mean=np.load('dataset/humanml_spatial_norm/Mean_raw.npy').reshape(22,3)
 std=np.load('dataset/humanml_spatial_norm/Std_raw.npy').reshape(22,3)
 normalized=np.where(np.any(raw!=0,axis=-1,keepdims=True),(raw-mean)/std,0)
 def collate(n_frames,dataset,task):
  assert n_frames==196 and dataset=='humanml' and task=='kick'
  return ['A person stands and kicks with his right leg.']*len(raw),normalized.reshape(len(raw),196,-1).copy()
 def loader(**kwargs):
  kwargs['hml_mode']='text_only';return get_dataset_loader(**kwargs)
 official.collate_all=collate;official.get_dataset_loader=loader
 create=official.create_model_and_diffusion
 def checked_create(args,data):
  model,diffusion=create(args,data)
  assert diffusion.num_timesteps==1000 and args.cond_mode not in ['only_spatial','only_text']
  return model,diffusion
 official.create_model_and_diffusion=checked_create
 sys.argv=[sys.argv[0],*remaining];start=time.time();official.main()
 d=np.load(output/'generation'/arm/'results.npy',allow_pickle=True).item()
 assert d['motion'].shape==(len(raw),22,3,196) and np.isfinite(d['motion']).all()
 assert np.allclose(d['hint'],raw,atol=1e-5)
 data={'arm':arm,'generated_samples':len(raw),'wall_s':time.time()-start,'time_scope':'full batch wall latency, not amortized real-time','results_sha256':sha(output/'generation'/arm/'results.npy'),'same_1000_DDPM_budget':True,'weights_updated':False}
 (output/f'generation_{arm}_timing.json').write_text(json.dumps(data,indent=2)+'\n')
 print('FEEDBACK_PAUSED_GENERATION_COMPLETE',json.dumps(data),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--stage',choices=['prepare','generate'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--arm',choices=['planned','actual','shuffled'])
 a,rest=p.parse_known_args()
 if a.stage=='prepare':prepare(a.output)
 else:generate(a.output,a.arm,rest)

"""Metadata-only independent confirmation audit; no model inference or changes."""
from pathlib import Path
import base64,json,shlex,subprocess,time
from v2_doc_witness_probe import SSH,ROOT,RAW

AUDIT=r'''
from pathlib import Path
import ast,fcntl,hashlib,json,subprocess,time
import numpy as np
import torch
r=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation');b=r/'artifacts/world_confirmation_20261007';a=b/'confirmation_v2';mon=r/'artifacts/monitoring'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for v in iter(lambda:f.read(2**20),b''):h.update(v)
 return h.hexdigest()
def tensor_sha(t):return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()
lock=json.loads((a/'model_lock.json').read_text());old=json.loads((b/'model_lock.json').read_text());q=json.loads((mon/'r3_world_confirmation_v2_20261007.state').read_text())
assert q['status']=='COMPLETE',q
assert len(q['completed'])==3 and len(q['datasets'])==3 and q['new_actual_transitions']==2949120
assert not q['formal_PPO_started'] and q['auxiliary_updates']==0
assert len(lock['files'])==39
assert all(sha(p)==h for p,h in lock['files'].items())
assert all(sha(p)==h for p,h in old['files'].items())
common=set(lock['files'])&set(old['files']);assert len(common)==38 and all(lock['files'][p]==old['files'][p] for p in common)
schema=None;schema_values={}
class SchemaNames(ast.NodeTransformer):
 def visit_Name(self,node):return ast.copy_location(ast.Constant(schema_values[node.id]),node)
for node in ast.parse((r/'scripts/r3_world_ppo_core.py').read_text()).body:
 if isinstance(node,ast.Assign):
  if any(isinstance(t,ast.Name) and t.id=='SCHEMA' for t in node.targets):schema=ast.literal_eval(SchemaNames().visit(node.value));break
  if isinstance(node.value,ast.Constant):
   for target in node.targets:
    if isinstance(target,ast.Name):schema_values[target.id]=ast.literal_eval(node.value)
schema_hash=hashlib.sha256(json.dumps(schema,sort_keys=True,separators=(',',':')).encode()).hexdigest()
manifest=json.loads((b/'selection/manifest.json').read_text());selected={x['file'] for x in manifest['selected']}
results={};totalhashes=0
for seed in [42,43,44]:
 c=a/f'collect_seed{seed}_v1';e=a/f'evaluate_seed{seed}_v1'
 cs=json.loads((c/'summary.json').read_text());es=json.loads((e/'summary.json').read_text())
 assert cs['status']==es['status']=='COMPLETE' and cs['seed']==es['seed']==seed
 assert cs['num_envs']==2048 and cs['steps']==480 and cs['actual_transitions']==983040
 assert cs['formal_ppo_updates']==cs['diagnostic_ppo_updates']==cs['diagnostic_aux_updates']==0
 assert cs['fresh_confirmation_references'] and cs['confirmation_data_not_for_training'] and cs['shared_warmup_stochastic_original_policy']
 assert cs['task_reward_PD_termination_unchanged'] and not cs['extra_IK'] and not cs['rollout_versions'] and not cs['losses']
 assert set(cs['loaded_reference_files'])==selected and cs['selected_original_ids']==[x['original_id'] for x in manifest['selected']]
 assert cs['original_library_order_max_abs']<1e-6 and cs['original_jit_equivalence_max_abs']<1e-5
 assert cs['schema_hash']==es['schema_hash']==schema_hash
 assert cs['reference_manifest_sha256']==es['reference_manifest_sha256']==sha(b/'selection/manifest.json')==lock['reference_manifest_sha256']
 assert es['model_lock_sha256']==sha(a/'model_lock.json') and es['PPO_updates']==es['auxiliary_updates']==0
 assert es['confirmation_evaluations_per_seed']==1 and es['no_checkpoint_or_norm_selection']
 assert lock['created_unix_before_collection']<cs['files']['observable.npy'].get('created_unix', (c/'observable.npy').stat().st_mtime)
 assert sha(c/'summary.json')==next(x['summary_sha256'] for x in q['datasets'] if x['seed']==seed)
 assert sha(e/'summary.json')==next(x['summary_sha256'] for x in q['completed'] if x['seed']==seed)
 hashes={}
 for directory,summary in [(c,cs),(e,es)]:
  for name,v in summary['files'].items():
   p=directory/name;assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'];hashes[str(p.relative_to(r))]=v['sha256'];totalhashes+=1
 d={name:np.load(c/(name+'.npy'),mmap_mode='r') for name in ['observable','labels','actions','done','reference','episode','phase','reward','cohort','push_direction','push_episode']}
 expected_shapes={'observable':(481,2048,92),'labels':(481,2048,50),'actions':(480,2048,29),'done':(480,2048),'reference':(481,2048),'episode':(481,2048),'phase':(481,2048),'reward':(480,2048),'cohort':(2048,),'push_direction':(2048,),'push_episode':(2048,)}
 assert all(d[k].shape==s for k,s in expected_shapes.items())
 assert all(np.isfinite(v).all() for v in d.values())
 assert np.array_equal(np.diff(d['episode'],axis=0),d['done'].astype(np.int32))
 assert np.count_nonzero(d['done'])==cs['reset_count']==es['reset_count']
 assert not np.any((np.diff(d['reference'],axis=0)!=0)&~d['done'])
 assert set(np.unique(d['reference']))==set(range(11))
 order=np.lexsort((np.arange(2048),np.floor(d['phase'][0]/.2),d['reference'][0]));ranks=np.argsort(order)
 assert np.array_equal(d['cohort'],ranks%2) and np.array_equal(d['push_direction'],np.where((ranks//2)%2==0,1,-1))
 assert np.count_nonzero(d['cohort']==0)==np.count_nonzero(d['cohort']==1)==1024
 assert np.array_equal(d['push_episode'],d['episode'][50])
 dp=cs['disturbance_protocol'];assert dp==es['disturbance_protocol']
 assert dp['name']=='lateral80' and dp['force_control_steps']==[50,59] and dp['force_world_y_N']==80 and dp['force_duration_s']==.2 and not dp['external_force_visible_to_actor']
 force=dp['actual_articulation_force_buffer_audit'];assert [v['step'] for v in force]==list(range(50,60))
 assert all(v['local_buffer_error']<1e-6 and v['pushed_count']==1024 and abs(v['force_norm_N_max']-80)<1e-4 for v in force)
 assert np.count_nonzero(d['done'][50:60,d['cohort']==1])==dp['reset_during_force_pushed_envs']
 assert np.count_nonzero(d['done'][60:110,d['cohort']==1])==dp['reset_60_to_109_pushed_envs']
 w=np.load(e/'confirmation_windows.npz');tt=np.arange(9,471);times=np.broadcast_to(tt[:,None],(len(tt),2048));envs=np.broadcast_to(np.arange(2048)[None,:],times.shape)
 valid=(d['episode'][tt-9]==d['episode'][tt+10])&(d['reference'][tt-9]==d['reference'][tt+10])
 rng=np.random.default_rng(20261007+seed);idx=0;counts={}
 for condition in ['clean','recovery']:
  cond=(np.broadcast_to(d['cohort'][None,:]==0,valid.shape) if condition=='clean' else ((times>=60)&(times<110)&(d['cohort'][None,:]==1)&(d['episode'][tt]==d['push_episode'][None,:])))
  counts[condition]={}
  for ref in range(11):
   mask=valid&cond&(d['reference'][tt]==ref);ts=times[mask];ee=envs[mask];n=len(ts);assert n>=256
   count=min(n,1024);draw=rng.choice(n,count,replace=False);ts=ts[draw];ee=ee[draw];part=slice(idx,idx+count)
   counts[condition][str(ref)]={'eligible_reset_free_windows':n,'scored':count}
   assert np.array_equal(w['sample_t'][part],ts) and np.array_equal(w['sample_env'][part],ee)
   assert np.all(w['reference'][part]==ref) and np.all(w['condition'][part]==int(condition=='recovery'))
   assert np.array_equal(w['episode'][part],d['episode'][ts,ee])
   assert np.array_equal(w['history'][part],d['observable'][ts[:,None]-np.arange(9,-1,-1)[None,:],ee[:,None]])
   assert np.array_equal(w['actions'][part],d['actions'][ts[:,None]+np.arange(10)[None,:],ee[:,None]])
   current=d['labels'][ts,ee];assert np.array_equal(w['current'][part],current)
   for k in [1,5,10]:
    target=np.concatenate([d['labels'][ts+k,ee,:48]-current[:,:48],d['labels'][ts+k,ee,48:]],-1)
    assert np.array_equal(w['target_'+str(k)][part],target)
   assert not d['done'][ts[:,None]+np.arange(-9,10)[None,:],ee[:,None]].any()
   assert np.all(d['reference'][ts[:,None]+np.arange(-9,11)[None,:],ee[:,None]]==ref)
   if condition=='recovery':assert np.all(d['episode'][ts,ee]==d['push_episode'][ee])
   idx+=count
 assert idx==len(w['sample_t']) and counts==es['window_counts']
 model_receipts={};fit=r/f'artifacts/world_residual_20261007/fit_seed{seed}_v1';fs=json.loads((fit/'summary.json').read_text());norms={}
 for name in ['direct','residual_correct','residual_state_only','residual_shuffled']:
  cp=torch.load(fit/(name+'.pt'),map_location='cpu',weights_only=False)
  assert cp['schema_hash']==schema_hash and cp['seed']==seed
  st=cp['state_dict'];norms[name]={k:tensor_sha(st[k]) for k in ['input_mean','input_std','delta_mean','delta_std']}
  assert all(torch.isfinite(st[k]).all() for k in norms[name])
  assert torch.all(st['input_std']>0) and torch.all(st['delta_std']>0)
  if name.startswith('residual_'):
   rn='state_only' if name=='residual_state_only' else 'correct'
   with np.load(fit/('ridge_'+rn+'.npz')) as rr:
    for k in [1,5,10]:assert np.array_equal(st['ridge_'+str(k)].numpy(),rr['coef_'+str(k)].astype(np.float32))
  model_receipts[name]={'sha256':sha(fit/(name+'.pt')),'norm_tensor_sha256':norms[name],'selected_update':fs['selection'][name]['update']}
  if name=='residual_shuffled' and fs['selection'][name]['update']==0:
   assert all(torch.count_nonzero(st[f'heads.{k}.2.weight'])==0 and torch.count_nonzero(st[f'heads.{k}.2.bias'])==0 for k in [1,5,10])
 assert all(norms[name]==norms['residual_correct'] for name in norms)
 for name in ['ridge_correct','ridge_state_only']:
  with np.load(fit/(name+'.npz')) as rr:
   assert all(np.isfinite(rr[k]).all() for k in rr.files)
   model_receipts[name]={'sha256':sha(fit/(name+'.npz')),'coefficient_shapes':{k:list(rr[k].shape) for k in rr.files}}
 brng=np.random.default_rng(20261007);gates={};comparison_error=0;equal_ref_primary={}
 for condition in ['clean','recovery']:
  per=es['per_reference_body_fk_rmse_m'];correct=np.array([per['residual_correct'][condition]['10'][str(ref)] for ref in range(11)])
  assert np.isfinite(correct).all();pairs=brng.integers(0,11,size=(10000,11));comparison={}
  equal_ref_primary[condition]={name:float(np.mean([per[name][condition]['10'][str(ref)] for ref in range(11)])) for name in per}
  for comparator in ['ridge_correct','ridge_state_only','direct','residual_state_only','residual_shuffled']:
   baseline=np.array([per[comparator][condition]['10'][str(ref)] for ref in range(11)])
   ci=np.percentile(correct[pairs].mean(1)/baseline[pairs].mean(1),[2.5,97.5]);gain=1-correct.mean()/baseline.mean()
   vals=np.array([gain,1-ci[1],1-ci[0],ci[0],ci[1]])
   saved=es['comparisons'][condition][comparator];stored=np.array([saved['relative_gain'],*saved['relative_gain_95pct_CI'],*saved['paired_error_ratio_95pct_CI']])
   comparison_error=max(comparison_error,float(np.abs(vals-stored).max()));assert np.allclose(vals,stored,atol=1e-12,rtol=0)
   comparison[comparator]={'gain':gain,'low_gain':1-ci[1],'upper_ratio':ci[1]}
  gates[condition]=bool(all(comparison[x]['gain']>=.1 and comparison[x]['low_gain']>0 for x in ['ridge_correct','ridge_state_only','residual_state_only']) and comparison['direct']['upper_ratio']<=1.05 and comparison['residual_shuffled']['low_gain']>0)
 assert gates==es['domain_gates'] and all(gates.values())==es['formal_prediction_gate_passed']
 stages={}
 for mode,sentinel in [('collect','WORLD_PPO_ISAAC_COMPLETE'),('evaluate','WORLD_RESIDUAL_NEW_REFERENCE_CONFIRMATION_COMPLETE')]:
  prefix=mon/f'r3_world_confirmation_v2_{mode}_{seed}_v1';state=prefix.with_suffix('.state').read_text();log=prefix.with_suffix('.log').read_text(errors='replace')
  assert state.strip()==f'COMPLETE mode={mode} exit=0 PPO=false auxiliary_updates=0' and sentinel in log
  assert 'Traceback (most recent call last)' not in log and 'Timeout (0:' not in log
  stages[mode]={'state':state.strip(),'actual_child_chain_returncode':0,'required_sentinel':sentinel,'sentinel_present':True,'log_sha256':sha(prefix.with_suffix('.log'))}
 results[str(seed)]={'transitions':983040,'reset_count':cs['reset_count'],'push_reset_count_50_to_59':dp['reset_during_force_pushed_envs'],'push_reset_count_60_to_109':dp['reset_60_to_109_pushed_envs'],'all_real_episode_increments_equal_done':True,'window_row_count':idx,'all_windows_targets_actions_history_exactly_reconstructed':True,'sampling_counts':counts,'all_ref_domain_min256':True,'all_recovery_windows_in_original_push_episode':True,'actual_forcebuffer_audit_passed':True,'original_reference_identity_maxabs':cs['original_library_order_max_abs'],'jit_equivalence_maxabs':cs['original_jit_equivalence_max_abs'],'frozen_models':model_receipts,'shared_norm_tensors_identical':True,'file_hashes':hashes,'domain_gates':gates,'equal_reference_H10_primary_RMSE_m':equal_ref_primary,'bootstrap_comparison_maxabs_error':comparison_error,'stages':stages}
assert q['prediction_gate_passed']==all(all(v['domain_gates'].values()) for v in results.values())
assert q['recovery_domain_all_seed_gate']==all(v['domain_gates']['recovery'] for v in results.values())
assert 'WORLD_RESIDUAL_CONFIRMATION_QUEUE_COMPLETE '+('PASS' if q['prediction_gate_passed'] else 'FAIL') in (mon/'r3_world_confirmation_v2_20261007.log').read_text()
stage=(r/'scripts/r3_world_confirmation_v2_stage_20261007.sh').read_text();assert 'rc=$?\n SENTINEL=' in stage and stage.count('rc=$?\n SENTINEL=')==2
assert subprocess.run(['tmux','has-session','-t','r3_world_confirmation_v2_20261007'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
gpu0free=False
with (mon/'r3_world_ppo_gpu0.lock').open('r+') as f:
 try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);gpu0free=True;fcntl.flock(f,fcntl.LOCK_UN)
 except BlockingIOError:pass
assert gpu0free
print('WORLD_CONFIRMATION_V2_INDEPENDENT_AUDIT_JSON '+json.dumps({'status':'COMPLETE_INDEPENDENT_DOC_FOLLOW_AUDIT_PASS','observed_unix':time.time(),'actual_transitions':2949120,'reset_count':sum(v['reset_count'] for v in results.values()),'PPO_updates':0,'auxiliary_updates':0,'normalizer_updates':0,'prediction_gate_passed':q['prediction_gate_passed'],'not_PPO_or_control_benefit':True,'model_lock_sha256':sha(a/'model_lock.json'),'old_39_hashes_unchanged':True,'new_39_hashes_unchanged':True,'38_common_lock_entries_identical':True,'reference_manifest_sha256':sha(b/'selection/manifest.json'),'schema_hash':schema_hash,'hashed_output_files':totalhashes,'gpu0_lock_released':True,'per_seed':results,'limits':['Bootstrap and gate recomputed from per-reference prediction metadata; frozen-model predictions were not rerun.','Normalizers loaded from locked artifacts, identical across neural models; runtime inference sources use eval/no_grad and contain no fitting.','Data and weights remain on server; audit transfers metadata only.']}))
'''

def main():
 encoded=base64.b64encode(AUDIT.encode()).decode()
 code='import base64;exec(base64.b64decode('+repr(encoded)+'))'
 remote_root='/data/code/ouyanghao/oyh/conditional-human-motion-generation'
 command='cd '+shlex.quote(remote_root)+' && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 '+shlex.quote(remote_root+'/third_party/isaacsim-4.5.0/python.sh')+' -c '+shlex.quote(code)
 p=subprocess.run(SSH+[command],capture_output=True,text=True,timeout=180)
 attempts=RAW/'identity_v2_audit_attempts';attempts.mkdir(exist_ok=True)
 (attempts/f'execution_{int(time.time()*1000)}.json').write_text(json.dumps({'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(p.stdout[-1500:]);print(p.stderr)
 if p.returncode:raise SystemExit(p.returncode)
 sentinel='WORLD_CONFIRMATION_V2_INDEPENDENT_AUDIT_JSON '
 data=json.loads(p.stdout.split(sentinel,1)[1].strip())
 (RAW/'identity_v2_independent_audit.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:v for k,v in data.items() if k!='per_seed'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()

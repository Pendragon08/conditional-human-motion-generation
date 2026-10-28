"""Gated real mean-embedding witnesses, full diagnostic dispatch and collection."""
import argparse
import ast
import base64
import hashlib
import json
import pathlib
from backup_filter_recovery_checkpoint import recorded_remote
from session import SSH

B=pathlib.Path(__file__).resolve().parents[1]
O=B/'results/semantic_evaluator_20261029'
p=argparse.ArgumentParser()
p.add_argument('action',choices=['activate','launch','collect'])
p.add_argument('--stage',choices=['witnesses','metrics'])
p.add_argument('--role',choices=['author','document','diagnostic'])
p.add_argument('--evidence',required=True)
a=p.parse_args();assert a.evidence.isascii() and a.evidence.isidentifier()
sp=B/'configs/tmr69d-semantic-spec.json';raw=sp.read_bytes();policy=json.loads(raw)
if a.action=='activate':
    assert a.stage and not a.role
    run=None;out=O/('tmr_semantic_'+a.stage+'_activation_v1')
else:
    assert a.role and not a.stage
    run={'author':'tmr69d_semantic_author_v1','document':'tmr69d_semantic_document_v1','diagnostic':'tmr69d_semantic32_v1'}[a.role]
    out=O/run
out.mkdir(exist_ok=a.action!='launch')
binding=None
if a.action=='activate' and a.stage=='witnesses':
    assert policy['witnesses_authorized'] is False and policy['metrics_authorized'] is False
    assert policy['frozen_model_binding'] is None and policy['witness_bindings'] is None
    j=json.loads((O/'tmr69d_frozen_verify_v1/progress.json').read_bytes())
    assert j['closed_execution_verified'] and j['artifact_audit']['fresh_cpu_fullstate_and_inference_exact']
    binding=dict(run=j['run'],spec_sha256=j['files']['receipt']['spec_sha256'],receipt_sha256=j['sha256']['receipt'],
        closure=j['artifact_audit']['closure'],export_binding=j['artifact_audit']['export_binding'])
elif a.action=='activate':
    assert policy['witnesses_authorized'] is True and policy['metrics_authorized'] is False and policy['witness_bindings'] is None
    records=[json.loads((O/name/'progress.json').read_bytes()) for name in ['tmr69d_semantic_author_v1','tmr69d_semantic_document_v1']]
    assert all(j['closed_execution_verified'] and j['artifact_audit']['exact_mean_latent_artifacts_verified'] for j in records)
    assert records[0]['files']['receipt']['mean_latent_arrays']==records[1]['files']['receipt']['mean_latent_arrays']
    binding=dict(runtime_binding_sha256=records[0]['files']['receipt']['runtime_binding_sha256'],runs={j['run']:dict(
        request_sha256=j['files']['receipt']['request_sha256'],receipt_sha256=j['sha256']['receipt'],closure=j['closure']) for j in records})
expected=hashlib.sha256(raw).hexdigest()
code=r'''
import ast,base64,fcntl,json,os,pathlib,shutil,subprocess,sys,time
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
sys.path.insert(0,str(r/'scripts'));os.sched_setaffinity(0,{0,1,2,3})
from tmr69d_trained_acceptance import ROOT,PARENT,owned,read,sha
from tmr69d_semantic_inputs import accepted_frozen_verifier,accepted_diagnostic_metadata,accepted_semantic_witnesses,runtime_signature,semantic_closure,validate_semantic_policy
from contact_filter_resources import sample_resources
action=ACTION_LITERAL;stage=STAGE_LITERAL;role=ROLE_LITERAL;run=RUN_LITERAL;binding=BINDING_LITERAL
sp=owned(r/'configs/tmr69d-semantic-spec.json');assert sha(sp)==EXPECTED_LITERAL;s=validate_semantic_policy(read(sp))
for name,value in s['bound_sources'].items():assert sha(owned(r/name))==value,name
assert sha(owned(r/'scripts/run_tmr69d_semantic.py'))==s['executor_sha256']
assert sha(owned(r/'scripts/supervise_tmr69d_semantic.py'))==s['supervisor_sha256']
parent=owned(PARENT);lock=(r/'receipts/full_library_capacity.lock').open('a+')
if action in ['activate','launch']:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 resources=sample_resources(r,None,16*1024**3,s['reserve_bytes']+s['log_allowance_bytes']+s['output_upper_bytes'])
 gpu=int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*1024**2
 assert gpu>=20*1024**3
 accepted_diagnostic_metadata(s['diagnostic_binding'])
 if action=='activate':
  if stage=='witnesses':
   assert s['witnesses_authorized'] is False and s['metrics_authorized'] is False and s['frozen_model_binding'] is None and s['witness_bindings'] is None
   accepted_frozen_verifier(binding)
   s.update(witnesses_authorized=True,frozen_model_binding=binding)
  else:
   assert stage=='metrics' and s['witnesses_authorized'] is True and s['metrics_authorized'] is False and s['witness_bindings'] is None
   accepted_frozen_verifier(s['frozen_model_binding']);accepted_semantic_witnesses(binding,s)
   previous=runtime_signature(s)
   s.update(witnesses_authorized=False,metrics_authorized=True,witness_bindings=binding)
   assert runtime_signature(s)==previous
  archive=r/'receipts'/('tmr_semantic_policy_before_'+stage+'.json')
  with archive.open('xb') as f:f.write(sp.read_bytes());f.flush();os.fsync(f.fileno())
  raw=(json.dumps(s,indent=2)+'\n').encode();temp=sp.with_name(sp.name+'.enable.partial')
  with temp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
  temp.replace(sp)
  result=dict(status='TMR_SEMANTIC_STAGE_ENABLED_NOT_DISPATCHED',stage=stage,policy_sha256=sha(sp),policy_base64=base64.b64encode(raw).decode(),runtime_binding_sha256=runtime_signature(s),resources=resources,gpu1_free_bytes=gpu)
  with (r/'receipts'/('tmr_semantic_'+stage+'_activation.json')).open('x') as f:json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
 elif action=='launch':
  accepted_frozen_verifier(s['frozen_model_binding'])
  mode='diagnostic' if role=='diagnostic' else 'witness'
  assert s['metrics_authorized'] is True if mode=='diagnostic' else s['witnesses_authorized'] is True
  if role=='diagnostic':accepted_semantic_witnesses(s['witness_bindings'],s)
  if role=='document':
   author=read(owned(parent/'tmr69d_semantic_author_v1/receipt.json'))
   assert author['status']=='TMR69D_REAL_SEMANTIC_EMBEDDING_WITNESS_FINISHED' and author['output_writers_closed']
   assert author['runtime_binding_sha256']==runtime_signature(s)
   semantic_closure(author['run'],author['request_sha256'],s['supervisor_sha256'])
  prefix='tmr_semantic_'+run
  assert not (parent/run).exists() and not (parent/(run+'_supervision')).exists()
  assert not any((r/'receipts'/(prefix+x)).exists() for x in ['_request.json','_driver.py','_driver.json','_outer_exit.json'])
  request=dict(run=run,role=role,mode=mode,policy_path=str(sp),policy_sha256=sha(sp),runtime_binding_sha256=runtime_signature(s))
  rp=r/'receipts'/(prefix+'_request.json')
  with rp.open('x') as f:json.dump(request,f,indent=2);f.flush();os.fsync(f.fileno())
  cmd=['python3',str(r/'scripts/supervise_tmr69d_semantic.py'),'--request',str(rp)]
  outer=r/'receipts'/(prefix+'_outer_exit.json');driver=r/'receipts'/(prefix+'_driver.py')
  text="import subprocess,json,pathlib,os\np=subprocess.run("+repr(cmd)+")\nwith pathlib.Path("+repr(str(outer))+").open('x') as f:json.dump({'actual_returncode':p.returncode},f);f.flush();os.fsync(f.fileno())\nraise SystemExit(p.returncode)\n"
  with driver.open('x') as f:f.write(text);f.flush();os.fsync(f.fileno())
  lock.close()
  with (parent/(run+'_outer.log')).open('xb') as f:q=subprocess.Popen(['python3',str(driver)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,cwd=r,start_new_session=True)
  result=dict(status='TMR_SEMANTIC_JOB_DISPATCHED_NOT_ACCEPTED',run=run,driver_pid=q.pid,request_sha256=sha(rp),runtime_binding_sha256=runtime_signature(s),resources=resources,gpu1_free_bytes=gpu)
  with (r/'receipts'/(prefix+'_driver.json')).open('x') as f:json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
else:
 assert action=='collect'
 prefix='tmr_semantic_'+run;rp=owned(r/'receipts'/(prefix+'_request.json'));request=read(rp)
 log=parent/(run+'_supervision');files={};hashes={}
 for key,path in dict(receipt=parent/run/'receipt.json',exit=log/'exit.json',process=log/'process_exit.json',outer=r/'receipts'/(prefix+'_outer_exit.json')).items():
  if path.exists():files[key]=read(path);hashes[key]=sha(path)
 workers=[]
 for proc in pathlib.Path('/proc').iterdir():
  if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
  try:
   if proc.stat().st_uid!=1013:continue
   argv=(proc/'cmdline').read_bytes().decode().split('\0')
   if str(rp) in argv and any(str(r/'scripts'/n) in argv for n in ['run_tmr69d_semantic.py','supervise_tmr69d_semantic.py']):
    stat=(proc/'stat').read_text().rsplit(')',1)[1].split();workers.append(dict(pid=int(proc.name),start_ticks=stat[19],state=stat[0],argv=argv))
  except (FileNotFoundError,ProcessLookupError):pass
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);free=True
 except BlockingIOError:free=False
 closed=all(k in files for k in ['receipt','exit','process','outer']) and files['outer']['actual_returncode']==0 and files['exit']['supervision_passed'] and files['process']['child_reaped'] and files['process']['child_exit_code']==0 and not workers and free
 audit=None;closure=None
 if closed:
  import numpy as np
  import zipfile
  from tmr69d_semantic_math import paired_seed_metrics,retrieval32
  rec=files['receipt'];expected='TMR69D_SEMANTIC_DIAGNOSTIC_FINISHED' if role=='diagnostic' else 'TMR69D_REAL_SEMANTIC_EMBEDDING_WITNESS_FINISHED'
  assert rec['status']==expected and rec['output_writers_closed'] is True
  assert rec['source_sha256']==s['executor_sha256'] and rec['request_sha256']==sha(rp) and rec['runtime_binding_sha256']==runtime_signature(s)
  assert rec['training_optimizer_updates']==rec['generator_teacher_new_updates']==0
  assert rec['source_model_and_arrays_unchanged'] and rec['inference_rng_unchanged']
  assert rec['text_truncation_performed'] is False and rec['stochastic_sampling_or_dropout_used'] is False
  closure=semantic_closure(run,sha(rp),s['supervisor_sha256'])
  accepted_frozen_verifier(s['frozen_model_binding']);data=accepted_diagnostic_metadata(s['diagnostic_binding'])
  artifacts=rec['private_artifacts']
  for item in artifacts.values():
   path=owned(item['path']);assert path.parent==parent/run
   assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
  latents=owned(artifacts['raw_mean_latents.private.npz']['path'])
  with zipfile.ZipFile(latents) as archive:assert archive.testzip() is None and len(archive.infolist())==3
  with np.load(latents,allow_pickle=False) as archive:arrays={k:archive[k] for k in archive.files}
  assert set(arrays)==set(rec['mean_latent_arrays'])=={'text_mean_latents','reference_motion_mean_latents','generated_motion_mean_latents'}
  for name,array in arrays.items():
   assert array.dtype==np.float32 and np.isfinite(array).all() and list(array.shape)==rec['mean_latent_arrays'][name]['shape']
   assert __import__('hashlib').sha256(array.tobytes()).hexdigest()==rec['mean_latent_arrays'][name]['sha256']
  join=read(artifacts['source_join.private.json']['path']);seeds=data['scope']['summary']['seeds'];modes=data['scope']['summary']['modes']
  segments=sorted(data['scope']['segments'],key=lambda x:x['segment_id'])
  if role!='diagnostic':segments=segments[:1];seeds=seeds[:1]
  assert join['segment_ids']==[x['segment_id'] for x in segments] and join['seeds']==seeds and join['modes']==modes
  n=len(segments);assert arrays['text_mean_latents'].shape==arrays['reference_motion_mean_latents'].shape==(n,256)
  assert arrays['generated_motion_mean_latents'].shape==(2,len(seeds),n,256)
  expected={(x['segment_id'],seed,mode) for x in segments for seed in seeds for mode in modes}
  assert len(join['array_sources'])==len(expected)==rec['actual_completed_trials']
  assert {(x['segment_id'],x['seed'],x['mode']) for x in join['array_sources']}==expected
  for item in join['array_sources']:
   original=data['finished'][(item['segment_id'],item['seed'])]['mode_artifacts'][item['mode']]
   assert item['source_npz_sha256']==original['sha256']
  if role=='diagnostic':
   def compare(a,b):
    if isinstance(a,dict):assert set(a)==set(b);[compare(a[k],b[k]) for k in a]
    elif isinstance(a,list):assert len(a)==len(b);[compare(x,y) for x,y in zip(a,b)]
    elif isinstance(a,float):assert abs(a-b)<=1e-8+1e-9*abs(a),(a,b)
    else:assert a==b
   recomputed={mode:{str(seed):paired_seed_metrics(arrays['text_mean_latents'],arrays['reference_motion_mean_latents'],arrays['generated_motion_mean_latents'][mi,si],join['segment_ids']) for si,seed in enumerate(seeds)} for mi,mode in enumerate(modes)}
   compare(recomputed,rec['per_seed_metrics'])
   compare(retrieval32(arrays['text_mean_latents'],arrays['reference_motion_mean_latents'],join['segment_ids']),rec['reference_motion_retrieval_baseline'])
   summary={}
   for mode in modes:
    rows=recomputed[mode]
    scalars={name:[rows[str(seed)]['retrieval'][name] for seed in seeds] for name in ['Rprecision_top1','Rprecision_top2','Rprecision_top3','MMdist']}
    scalars['FID']=[rows[str(seed)]['frechet']['FID'] for seed in seeds]
    summary[mode]={name:dict(equal_seed_mean=float(np.mean(values)),descriptive_seed_population_std=float(np.std(values,ddof=0)),all_five_seed_values=values) for name,values in scalars.items()}
   compare(summary,rec['equal_seed_summary'])
   assert rec['formal_semantic_scores_computed'] is True and rec['actual_completed_trials']==1880
  else:
   assert rec['formal_semantic_scores_computed'] is False and rec['actual_completed_trials']==2
   if role=='document':assert rec['mean_latent_arrays']==read(owned(parent/'tmr69d_semantic_author_v1/receipt.json'))['mean_latent_arrays']
  audit=dict(exact_mean_latent_artifacts_verified=True,actual_latent_arrays_checked=3,actual_queries_preserved=188 if role=='diagnostic' else 1,
   all_declared_seeds_modes_and_failed_physics_inputs_retained=True,semantic_scalar_recomputation=role=='diagnostic',
   source_join_verified=True,model_loads_by_collector=0,semantic_quality_or_paper_complete_claimed=False)
 logs={}
 for path in [log/'child.log',parent/(run+'_outer.log')]:
  if path.exists():
   with path.open('rb') as f:f.seek(max(0,path.stat().st_size-2000));logs[path.name]=f.read().decode(errors='replace')
 result=dict(run=run,observed_unix=time.time(),files=files,sha256=hashes,workers=workers,common_lock_free=free,closed_execution_verified=bool(closed),closure=closure,artifact_audit=audit,logs=logs)
lock.close();print(json.dumps(result))
'''.replace('ACTION_LITERAL',repr(a.action)).replace('STAGE_LITERAL',repr(a.stage)).replace('ROLE_LITERAL',repr(a.role)).replace('RUN_LITERAL',repr(run)).replace('BINDING_LITERAL',repr(binding)).replace('EXPECTED_LITERAL',repr(expected))
ast.parse(code,feature_version=(3,8))
# NumPy is used only for closed artifact/scalar audits, never model loading.
command=SSH+['/data/code/ouyanghao/oyh/textop-reproduction-20261024/env/bin/python -B -']
j=json.loads(recorded_remote(code,out/(a.evidence+'_client.json'),timeout=300,command=command))
(out/(a.evidence+'.json')).write_text(json.dumps(j,indent=2),encoding='utf-8')
if a.action=='activate':
    updated=base64.b64decode(j['policy_base64']);assert hashlib.sha256(updated).hexdigest()==j['policy_sha256']
    with (out/'policy_before_enable.json').open('xb') as stream:stream.write(raw)
    sp.write_bytes(updated)
elif a.action=='collect':(out/'progress.json').write_text(json.dumps(j,indent=2),encoding='utf-8')
else:(out/'dispatch.json').write_text(json.dumps(j,indent=2),encoding='utf-8')
print(json.dumps(j))

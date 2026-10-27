"""Separate pre-filter8278 VAE200k; paper-sized model/schedule, explicitly different data filtering."""
import argparse,hashlib,importlib.metadata,json,os,pathlib,sys,time,traceback
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['witness','train'],required=True);p.add_argument('--run',required=True);p.add_argument('--resume');p.add_argument('--resume-sha');a=p.parse_args()
specpath=ROOT/'configs/generator8278-vae-spec.json';spec=json.loads(specpath.read_text());base=json.loads((ROOT/'configs/paper69-env-spec.json').read_text())
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert spec['schema']=='textop.generator8278_vae.v1' and a.run.isascii() and a.run.isidentifier()
assert os.getuid()==1013 and set(os.sched_getaffinity(0))=={0,1,2,3} and os.environ.get('CUDA_VISIBLE_DEVICES')=='1'
assert sys.version.split()[0]==base['python_version'] and {k:importlib.metadata.version(k) for k in base['required_versions']}==base['required_versions']
assert sha(__file__)==spec['entrypoint_sha256']
for n,h in spec['bound_sources'].items():assert sha(ROOT/n)==h
assert spec['vae']=={'stages':[100000,50000,50000],'batch_size':512,'learning_rate':.0001,'latent_shape':[1,128],'hidden':512,'ff':1024,'layers':9,'heads':4,'feature_dim':69,'history':2,'future':8,'primitives':4,'ema':False}
acceptance=ROOT/'receipts/generator8278_data_acceptance_v1.json';assert sha(acceptance)==spec['data_acceptance_sha256']
accepted=json.loads(acceptance.read_text());assert accepted['prefilter_training_data_accepted'] is True and accepted['paper_filtered_dataset_accepted'] is False
corpus=pathlib.Path(spec['corpus_dir']);meta=json.loads((corpus/'view_receipt.json').read_text());assert sha(corpus/'view_receipt.json')==spec['corpus_receipt_sha256']==accepted['corpus_receipt_sha256']
for n,item in meta['artifacts'].items():assert sha(corpus/n)==item['sha256']
assert spec['witness_authorized'] is True
if a.mode=='train':
 assert spec['train_authorized'] is True and len(spec['witnesses'])==2
 for w in spec['witnesses']:
  rp=ROOT/'generator8278_20261027/vae_runs'/w['run']/'receipt.json';ep=ROOT/'receipts'/('generator8278_vae_'+w['run']+'_exit.json');r=json.loads(rp.read_text());e=json.loads(ep.read_text())
  assert sha(rp)==w['receipt_sha256'] and sha(ep)==w['exit_sha256'] and e['supervision_passed'] is True and e['child_exit_code']==0
  assert r['status']=='GENERATOR8278_VAE_WITNESS_PASSED' and r['output_writers_closed'] is True and r['memory_witness']['model_and_adam_roundtrip_exact'] is True
out=ROOT/'generator8278_20261027/vae_runs'/a.run;out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
r={'status':'STARTING_GENERATOR8278_VAE','mode':a.mode,'quality_label':'PREFILTER8278_PAPER_SCHEDULE_NOT_FILTERED_PAPER_REPRODUCTION','full_paper_training':False,
   'optimizer_updates':0,'completed_primitive_attempts':0,'skipped_updates':0,'output_writers_closed':False,'source_sha256':sha(__file__),
   'spec_sha256':sha(specpath),'corpus_receipt_sha256':spec['corpus_receipt_sha256'],'data_acceptance_sha256':spec['data_acceptance_sha256']}
def save():
 r['elapsed_seconds']=time.monotonic()-started;tmp=out/'receipt.partial';tmp.write_text(json.dumps(r,indent=2,allow_nan=False));tmp.replace(out/'receipt.json')
save();ctx=None;checkpoints=[];at_closed_boundary=False;binding=None;dest=None;policy=None
try:
 import torch
 from generator8278_vae_runtime import Runtime,batch_sha
 from checkpoint_storage_guard import check
 from generator_training_state import save_manager,load_manager,module_sha,equal
 dest=pathlib.Path(spec['checkpoint_dir']);assert dest.parent.resolve()==dest.parent;dest.mkdir(parents=True,exist_ok=True)
 policy=spec['checkpoint_policy'];budget=check(dest,ROOT,policy,remaining_saves=5)
 ctx=Runtime(spec,base,out);r.update(status='RUNNING_GENERATOR8278_VAE_WITNESS' if a.mode=='witness' else 'RUNNING_GENERATOR8278_VAE_TRAINING',initial_storage_check=budget)
 if a.mode=='witness':
  r['memory_witness']=ctx.memory_witness();r.update(status='GENERATOR8278_VAE_WITNESS_PASSED',optimizer_updates=4,completed_primitive_attempts=4,skipped_updates=0,
    memory_only_diagnostic=True,all_rng_recursive_verified=True,training_checkpoint_created=False,weights_used_for_formal_training=False)
 else:
  numerical={k:spec[k] for k in ['vae','seed','corpus_receipt_sha256','data_acceptance_sha256','bound_sources','entrypoint_sha256','checkpoint_dir','checkpoint_policy']}
  binding={'stage':'vae','feature_dim':69,'dof':29,'quick_training':False,'formal_training':False,'paper_schedule':True,'prefilter_data':True,
           'quality_label':r['quality_label'],'numeric_sha256':hashlib.sha256(json.dumps(numerical,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
           'corpus_receipt_sha256':spec['corpus_receipt_sha256']}
  if a.resume:
   path=pathlib.Path(a.resume);assert path.parent==dest and a.resume_sha
   r['resume']=load_manager(ctx.manager,'vae',ctx.clock,ctx.generators,path,binding,a.resume_sha)
  else:assert not list(dest.glob('*.pth')),'Use explicit bound resume; do not overwrite existing run checkpoints'
  # Quality diagnostics must not advance resumed training/validation RNG streams.
  import random,numpy as np
  from generator8278_vae_runtime import restore_rng
  from generator_training_state import rng_state
  quality_rng=rng_state(ctx.generators);quality_seed=spec['seed']+99
  random.seed(quality_seed);np.random.seed(quality_seed);torch.manual_seed(quality_seed);torch.cuda.manual_seed_all(quality_seed)
  ctx.generators['val'].manual_seed(quality_seed);fixed_batch=ctx.batch('val');restore_rng(quality_rng,ctx.generators)
  assert equal(rng_state(ctx.generators),quality_rng)
  r['fixed_validation_batch_sha256']=batch_sha(fixed_batch);r['quality_baseline_manager_step']=ctx.manager.step;r['quality_batch_preserves_training_rng']=True
  initial={'teacher_forced':ctx.quality(fixed_batch,False),'self_rollout':ctx.quality(fixed_batch,True)};r['initial_quality']=initial;save()
  while ctx.manager:
   at_closed_boundary=False
   records=ctx.run_batch(ctx.batch('train'))
   while ctx.manager.should_eval():ctx.run_batch(ctx.batch('val'),is_eval=True)
   at_closed_boundary=True
   step=ctx.manager.step;r.update(optimizer_updates=ctx.clock.updates,completed_primitive_attempts=ctx.clock.attempts,skipped_updates=ctx.clock.skipped,last_train_batch=records)
   if step%100==0:
    remaining=sum(x>step for x in [50000,100000,150000,200000])+int(step in [50000,100000,150000,200000])+1
    r['last_storage_check']=check(dest,ROOT,policy,remaining_saves=remaining)
    save();print(json.dumps({'step':step,'updates':ctx.clock.updates,'skips':ctx.clock.skipped,'loss':records[-1]['loss']}),flush=True)
   if step in [50000,100000,150000,200000]:
    path=dest/('vae_state_'+str(step)+'.pth');assert not path.exists()
    remaining=sum(x>=step for x in [50000,100000,150000,200000])+1;check(dest,ROOT,policy,remaining_saves=remaining)
    ck=save_manager(ctx.manager,'vae',ctx.clock,ctx.generators,path,binding,policy['reserve_bytes']+policy['log_allowance_bytes'],policy['checkpoint_upper_bound_bytes'])
    checkpoints.append(ck);r['checkpoints']=checkpoints;assert ck['bytes']<=policy['checkpoint_upper_bound_bytes'],'Checkpoint larger than declared bound; retain it and stop to re-budget'
    r['latest_file_resume_verification']=load_manager(ctx.manager,'vae',ctx.clock,ctx.generators,path,binding,ck['sha256']);save()
  final={'teacher_forced':ctx.quality(fixed_batch,False),'self_rollout':ctx.quality(fixed_batch,True)}
  r['final_quality']=final;r['vae_quality_gate_passed']=all(final[k]['normalized_reconstruction_huber']<initial[k]['normalized_reconstruction_huber'] for k in final)
  assert ctx.clock.attempts==200000 and ctx.manager.step==200000 and not ctx.manager.should_eval()
  r.update(status='GENERATOR8278_VAE_200000_TRAINING_FINISHED',trained_model_sha256=module_sha(ctx.model),
    next_stage='Validate separate LDM300000 with released use_full_sample=True after realVAE200k and resources; no automatic reuse of quickLDM')
 r.update(peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),output_writers_closed=True);save();print(json.dumps(r),flush=True)
except BaseException as e:
 r.update(status='FAILED_GENERATOR8278_VAE',error=repr(e),traceback=traceback.format_exc(),checkpoints=checkpoints)
 if ctx is not None:r.update(optimizer_updates=ctx.clock.updates,completed_primitive_attempts=ctx.clock.attempts,skipped_updates=ctx.clock.skipped)
 if a.mode=='train' and ctx is not None and binding is not None and at_closed_boundary and ctx.manager.step>0 and not (checkpoints and checkpoints[-1]['manager_step']==ctx.manager.step):
  try:
   assert not list(dest.glob('emergency_*.pth')),'The reserved emergency slot is already retained; re-budget before another'
   check(dest,ROOT,policy,remaining_saves=1)
   r['emergency_checkpoint']=save_manager(ctx.manager,'vae',ctx.clock,ctx.generators,dest/('emergency_'+str(ctx.manager.step)+'.pth'),binding,
      policy['reserve_bytes']+policy['log_allowance_bytes'],policy['checkpoint_upper_bound_bytes'])
   assert r['emergency_checkpoint']['bytes']<=policy['checkpoint_upper_bound_bytes']
  except BaseException as save_error:r['emergency_save_refused_or_failed']=repr(save_error)
 else:r['emergency_save_scope']='No new file unless a complete four-primitive and validation boundary exists; prior checkpoints retained'
 save();raise
finally:
 if ctx is not None:ctx.close()

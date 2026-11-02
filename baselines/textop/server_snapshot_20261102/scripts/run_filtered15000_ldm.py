"""Fresh full LDM300k after real filtered VAE200k and CPU inference export."""
import argparse,hashlib,importlib.metadata,json,os,pathlib,shutil,sys,time,traceback
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['witness','train'],required=True);p.add_argument('--run',required=True);p.add_argument('--resume');p.add_argument('--resume-sha');a=p.parse_args()
sp=ROOT/'configs/filtered15000-ldm-spec.json';spec=json.loads(sp.read_text());base=json.loads((ROOT/'configs/paper69-env-spec.json').read_text())
from filtered15000_vae_completion import sha,completed_vae,closed_job
assert spec['schema']=='textop.filtered15000_full_ldm.v1' and a.run.isascii() and a.run.isidentifier()
assert os.getuid()==1013 and set(os.sched_getaffinity(0))=={0,1,2,3} and os.environ.get('CUDA_VISIBLE_DEVICES')=='1'
assert sys.version.split()[0]==base['python_version'] and {k:importlib.metadata.version(k) for k in base['required_versions']}==base['required_versions']
assert sha(__file__)==spec['entrypoint_sha256']
for n,h in spec['bound_sources'].items():assert sha(ROOT/n)==h,n
assert spec['ldm']=={'stages':[50000,50000,50000,150000],'batch_size':512,'learning_rate':.0001,'latent_shape':[1,128],'hidden':512,'ff':1024,'layers':8,'heads':4,'feature_dim':69,'history':2,'future':8,'primitives':4,'clip_dim':512,'cond_mask_prob':.1,'diffusion_steps':5,'noise_schedule':'cosine','prediction':'x0','cfg':5,'ema':False,'use_full_sample':True,'maximum_rollout_probability':.8,'static_pose':False,'posterior_variance':'FIXED_LARGE'}
acceptance=ROOT/'receipts/filtered15000_data_acceptance_v1.json';assert sha(acceptance)==spec['data_acceptance_sha256'];accepted=json.loads(acceptance.read_text())
assert accepted['actual_strict_filtered_training_data_accepted'] and not accepted['paper_two_stage_route_reproduced']
from filtered15000_inputs import load_selection
cs=json.loads((ROOT/'configs/filtered15000-corpus-spec.json').read_text());load_selection(cs)
assert spec['strict_audit_receipt_sha256']==cs['strict_audit_receipt_sha256']==accepted['strict_audit_receipt_sha256']
corpus=pathlib.Path(spec['corpus_dir']);meta=json.loads((corpus/'view_receipt.json').read_text());assert sha(corpus/'view_receipt.json')==spec['corpus_receipt_sha256']==accepted['corpus_receipt_sha256']
for n,item in meta['artifacts'].items():assert sha(corpus/n)==item['sha256']
v=completed_vae(spec['vae_training']);vr=v['receipt'];ck=v['checkpoint']
ep=spec['vae_export'];rp=pathlib.Path(ep['receipt_path']);er=json.loads(rp.read_text())
assert sha(rp)==ep['receipt_sha256'] and er['status']=='FILTERED15000_REAL_VAE200000_CPU_EXPORT_PASSED' and er['output_writers_closed']
assert er['exact_inference_reload'] and er['source_full_state_preserved'] and er['new_gpu_contexts']==0
assert er['vae_training_receipt_sha256']==v['receipt_sha256'] and er['full_state_checkpoint']==ck
exspec=json.loads((ROOT/'configs/filtered15000-vae-export-spec.json').read_text())
assert er['source_sha256']==exspec['executor_sha256'] and er['spec_sha256']==sha(ROOT/'configs/filtered15000-vae-export-spec.json')
closure=closed_job('filtered15000_vae_export_'+ep['run'],rp,ep['receipt_sha256'],exspec['supervisor_sha256'])
assert closure['exit_sha256']==ep['exit_sha256']
assert er['inference']==spec['vae_inference'];inference=er['inference']
assert sha(inference['path'])==inference['sha256'] and pathlib.Path(inference['path']).stat().st_size==inference['bytes']
assert inference['bytes']<=spec['vae_cache_upper_bound_bytes'] and er['trained_model_sha256']==vr['trained_model_sha256']
for k in ['corpus_receipt_sha256','data_acceptance_sha256','strict_audit_receipt_sha256']:assert spec[k]==er[k]==v['spec'][k]
numerical_keys=['ldm','seed','corpus_receipt_sha256','data_acceptance_sha256','strict_audit_receipt_sha256','vae_training','vae_export','vae_inference','bound_sources','entrypoint_sha256','checkpoint_dir','checkpoint_allowed_root','checkpoint_policy']
runtime_binding_sha256=hashlib.sha256(json.dumps({k:spec[k] for k in numerical_keys},sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert spec['witness_authorized']
if a.mode=='train':
 assert spec['train_authorized'] and len(spec['witnesses'])==2 and len({w['run'] for w in spec['witnesses']})==2
 for w in spec['witnesses']:
  wp=ROOT/'filtered15000_20261027/ldm_runs'/w['run']/'receipt.json';wr=json.loads(wp.read_text())
  proof=closed_job('filtered15000_ldm_'+w['run'],wp,w['receipt_sha256'],spec['supervisor_sha256'])
  assert proof['exit_sha256']==w['exit_sha256']
  assert wr['status']=='FILTERED15000_LDM_WITNESS_PASSED' and wr['output_writers_closed']
  assert wr['source_sha256']==spec['entrypoint_sha256'] and wr['runtime_binding_sha256']==runtime_binding_sha256
  assert wr['vae_checkpoint_sha256']==ck['sha256'] and wr['vae_inference_sha256']==inference['sha256']
  assert all(wr['memory_witness'][k] for k in ['model_and_adam_roundtrip_exact','all_rngs_restored_exactly','vae_frozen_and_unchanged'])
  assert wr['full_history_helper_witness']['actual_denoiser_forward_calls']==5 and wr['full_history_helper_witness']['formal_later_stage_training_reached'] is False
out=ROOT/'filtered15000_20261027/ldm_runs'/a.run;out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
r={'status':'STARTING_FILTERED15000_LDM','mode':a.mode,'quality_label':'DIRECT_STRICT15000_FILTERED_FULL_LDM300K','full_paper_training':False,'optimizer_updates':0,'completed_primitive_attempts':0,'skipped_updates':0,
 'output_writers_closed':False,'source_sha256':sha(__file__),'spec_sha256':sha(sp),'corpus_receipt_sha256':spec['corpus_receipt_sha256'],'data_acceptance_sha256':spec['data_acceptance_sha256'],
 'strict_audit_receipt_sha256':spec['strict_audit_receipt_sha256'],'runtime_binding_sha256':runtime_binding_sha256,'vae_checkpoint_sha256':ck['sha256'],'vae_inference_sha256':inference['sha256']}

def save():
 r['elapsed_seconds']=time.monotonic()-started;tmp=out/'receipt.partial';tmp.write_text(json.dumps(r,indent=2,allow_nan=False));tmp.replace(out/'receipt.json')

save();ctx=None;checkpoints=[];binding=None;at_closed_boundary=False;dest=pathlib.Path(spec['checkpoint_dir']);policy=spec['checkpoint_policy'];allowed=pathlib.Path(spec['checkpoint_allowed_root'])
try:
 import torch
 from filtered15000_ldm_runtime import Runtime
 from generator8278_vae_runtime import batch_sha,restore_rng
 from checkpoint_storage_guard import check
 from generator_training_state import save_manager,load_manager,module_sha,equal,rng_state
 assert allowed.is_dir() and dest.is_dir() and dest.resolve()==dest
 budget=check(dest,allowed,policy,remaining_saves=4);cache=dest/'vae.pth'
 if cache.exists():assert sha(cache)==inference['sha256']
 else:assert shutil.disk_usage(dest).free>=budget['required_bytes']+spec['vae_cache_upper_bound_bytes']
 ctx=Runtime(spec,base,out);assert sha(cache)==inference['sha256'] and cache.stat().st_size<=spec['vae_cache_upper_bound_bytes']
 assert ctx.frozen_vae_sha==vr['trained_model_sha256']
 r.update(status='RUNNING_FILTERED15000_LDM_WITNESS' if a.mode=='witness' else 'RUNNING_FILTERED15000_LDM_TRAINING',initial_storage_check=budget,vae_cache_bytes=cache.stat().st_size,vae_frozen_model_sha256=ctx.frozen_vae_sha)
 if a.mode=='witness':
  r['memory_witness']=ctx.memory_witness();r['full_history_helper_witness']=ctx.full_history_witness();r['five_step_cfg_sampling_witness']=ctx.sample_quality(ctx.batch('val'))
  assert module_sha(ctx.vae)==ctx.frozen_vae_sha
  r.update(status='FILTERED15000_LDM_WITNESS_PASSED',optimizer_updates=4,completed_primitive_attempts=4,skipped_updates=0,memory_only_diagnostic=True,training_checkpoint_created=False,weights_used_for_formal_training=False)
 else:
  binding={'stage':'ldm','feature_dim':69,'dof':29,'quick_training':False,'formal_training':True,'paper_schedule':True,'prefilter_data':False,'actual_strict_input':True,'paper_two_stage_route_reproduced':False,'quality_label':r['quality_label'],
   'numeric_sha256':runtime_binding_sha256,'corpus_receipt_sha256':spec['corpus_receipt_sha256'],'vae_checkpoint_sha256':ck['sha256'],'vae_inference_sha256':inference['sha256']}
  if a.resume:
   path=pathlib.Path(a.resume);assert path.parent==dest and a.resume_sha;r['resume']=load_manager(ctx.manager,'denoiser',ctx.clock,ctx.generators,path,binding,a.resume_sha);ctx.restore_auxiliary_counters()
  else:assert not list(dest.glob('ldm_state_*.pth')) and not list(dest.glob('emergency_*.pth')),'Explicit bound resume required'
  import random,numpy as np
  outer=rng_state(ctx.generators);quality_seed=spec['seed']+99;random.seed(quality_seed);np.random.seed(quality_seed);torch.manual_seed(quality_seed);torch.cuda.manual_seed_all(quality_seed);ctx.generators['val'].manual_seed(quality_seed)
  fixed_batch=ctx.batch('val');restore_rng(outer,ctx.generators);assert equal(rng_state(ctx.generators),outer)
  r.update(fixed_validation_batch_sha256=batch_sha(fixed_batch),quality_baseline_manager_step=ctx.manager.step,quality_batch_preserves_training_rng=True,initial_full_sampling_quality=ctx.sample_quality(fixed_batch));save()
  selected_rollouts=int(ctx.manager.extra.get('selected_rollout_history_primitives',0))
  while ctx.manager:
   at_closed_boundary=False;records=ctx.run_batch(ctx.batch('train'))
   selected_rollouts+=sum(int(x['rollout_history_selected']) for x in records);ctx.manager.extra['selected_rollout_history_primitives']=selected_rollouts
   while ctx.manager.should_eval():ctx.run_batch(ctx.batch('val'),is_eval=True)
   at_closed_boundary=True;step=ctx.manager.step
   r.update(optimizer_updates=ctx.clock.updates,completed_primitive_attempts=ctx.clock.attempts,skipped_updates=ctx.clock.skipped,last_train_batch=records,full_sampling_history_calls=ctx.full_history_calls,selected_rollout_history_primitives=selected_rollouts,actual_native_stage=ctx.manager.stage_idx)
   if step%100==0:
    r['last_storage_check']=check(dest,allowed,policy,remaining_saves=sum(x>=step for x in [100000,200000,300000])+1);save();print(json.dumps({'step':step,'updates':ctx.clock.updates,'skips':ctx.clock.skipped,'loss':records[-1]['loss'],'full_history_calls':ctx.full_history_calls}),flush=True)
   if step in [100000,200000,300000]:
    path=dest/('ldm_state_'+str(step)+'.pth');assert not path.exists();check(dest,allowed,policy,remaining_saves=sum(x>=step for x in [100000,200000,300000])+1)
    item=save_manager(ctx.manager,'denoiser',ctx.clock,ctx.generators,path,binding,policy['reserve_bytes']+policy['log_allowance_bytes'],policy['checkpoint_upper_bound_bytes']);checkpoints.append(item);r['checkpoints']=checkpoints
    assert item['bytes']<=policy['checkpoint_upper_bound_bytes'],'Retain oversized file, stop and re-budget'
    r['latest_file_resume_verification']=load_manager(ctx.manager,'denoiser',ctx.clock,ctx.generators,path,binding,item['sha256']);ctx.restore_auxiliary_counters()
    assert ctx.full_history_calls==ctx.manager.extra['full_sampling_history_calls'];save()
  r['final_full_sampling_quality']=ctx.sample_quality(fixed_batch);assert module_sha(ctx.vae)==ctx.frozen_vae_sha
  assert ctx.clock.attempts==300000 and ctx.manager.step==300000 and not ctx.manager.should_eval()
  assert ctx.full_history_calls>0 and selected_rollouts>0,'Actual later-stage full sampling and selected rollout history required'
  r.update(status='FILTERED15000_LDM_300000_TRAINING_FINISHED',trained_denoiser_sha256=module_sha(ctx.model),vae_frozen_and_unchanged=True,full_native_schedule_completed=True,text_semantics_accepted=False,
   next_stage='Heldout multi-seed text/unconditional controls, TMR and actual physical endpoints; training alone is not semantic or online quality acceptance')
 r.update(peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),output_writers_closed=True);save();print(json.dumps(r),flush=True)
except BaseException as e:
 r.update(status='FAILED_FILTERED15000_LDM',error=repr(e),traceback=traceback.format_exc(),checkpoints=checkpoints)
 if ctx is not None:r.update(optimizer_updates=ctx.clock.updates,completed_primitive_attempts=ctx.clock.attempts,skipped_updates=ctx.clock.skipped)
 if a.mode=='train' and ctx is not None and binding is not None and at_closed_boundary and ctx.manager.step>0 and not(checkpoints and checkpoints[-1]['manager_step']==ctx.manager.step):
  try:
   assert not list(dest.glob('emergency_*.pth'));check(dest,allowed,policy,remaining_saves=1)
   r['emergency_checkpoint']=save_manager(ctx.manager,'denoiser',ctx.clock,ctx.generators,dest/('emergency_'+str(ctx.manager.step)+'.pth'),binding,policy['reserve_bytes']+policy['log_allowance_bytes'],policy['checkpoint_upper_bound_bytes']);assert r['emergency_checkpoint']['bytes']<=policy['checkpoint_upper_bound_bytes']
  except BaseException as err:r['emergency_save_refused_or_failed']=repr(err)
 else:r['emergency_save_scope']='Only a closed four-primitive and validation boundary; all prior checkpoints retained'
 save();raise
finally:
 if ctx is not None:ctx.close()

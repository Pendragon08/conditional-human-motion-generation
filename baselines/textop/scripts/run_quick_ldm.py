"""Source-bound quick LDM3000 using completed VAE2000; no SDK or paper claim."""
import argparse,hashlib,importlib.metadata,json,os,pathlib,shutil,sys,time,traceback
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['witness','train'],required=True);p.add_argument('--run',required=True);p.add_argument('--resume');p.add_argument('--resume-sha');a=p.parse_args()
sp=ROOT/'configs/quick-ldm-runtime-spec.json';spec=json.loads(sp.read_text());base=json.loads((ROOT/'configs/paper69-env-spec.json').read_text())
from generator_training_state import sha
assert spec['schema']=='textop.quick_ldm_runtime.v1' and a.run.isascii() and a.run.isidentifier()
assert os.getuid()==1013 and set(os.sched_getaffinity(0))=={0,1,2,3} and os.environ.get('CUDA_VISIBLE_DEVICES')=='1'
assert sys.version.split()[0]==base['python_version'] and {k:importlib.metadata.version(k) for k in base['required_versions']}==base['required_versions']
assert sha(__file__)==spec['entrypoint_sha256']
for n,h in spec['bound_sources'].items():assert sha(ROOT/n)==h
assert spec['ldm']=={'stages':[1000,1000,1000],'batch_size':512,'learning_rate':.0001,'latent_shape':[1,128],'hidden':512,'ff':1024,'layers':8,'heads':4,'feature_dim':69,'history':2,'future':8,'primitives':4,'clip_dim':512,'cond_mask_prob':.1,'diffusion_steps':5,'noise_schedule':'cosine','prediction':'x0','cfg':5,'ema':False}
acceptance=ROOT/'receipts/quick69_data_acceptance_v1.json';assert sha(acceptance)==spec['data_acceptance_sha256'];accepted=json.loads(acceptance.read_text())
assert accepted['quick_corpus_accepted'] is True and accepted['full_paper_dataset_accepted'] is False
corpus=pathlib.Path(spec['corpus_dir']);meta=json.loads((corpus/'corpus_receipt.json').read_text());assert sha(corpus/'corpus_receipt.json')==spec['corpus_receipt_sha256']==accepted['corpus_receipt_sha256']
for n,item in meta['artifacts'].items():assert sha(corpus/n)==item['sha256']
v=spec['vae_training'];vp=pathlib.Path(v['receipt_path']);ve=pathlib.Path(v['exit_path']);vr=json.loads(vp.read_text());vx=json.loads(ve.read_text())
assert sha(vp)==v['receipt_sha256'] and sha(ve)==v['exit_sha256'] and vx['evaluation_receipt_sha256']==sha(vp)
assert vx['supervision_passed'] and vx['child_exit_code']==0 and vr['status']=='QUICK_VAE_2000_TRAINING_FINISHED' and vr['output_writers_closed'] and vr['vae_quality_gate_passed']
assert vr['completed_primitive_attempts']==2000 and vr['corpus_receipt_sha256']==spec['corpus_receipt_sha256'] and vr['latest_file_resume_verification']['rng_and_generator_restored_exactly']
ck=spec['vae_checkpoint'];assert ck in vr['checkpoints'] and ck['manager_step']==2000 and sha(ck['path'])==ck['sha256']
assert spec['witness_authorized']
if a.mode=='train':
 assert spec['train_authorized'] and len(spec['witnesses'])==2
 for w in spec['witnesses']:
  rp=ROOT/'quick_generator_20261026/ldm_runs'/w['run']/'receipt.json';ep=ROOT/'receipts'/('quick_ldm_'+w['run']+'_exit.json');wr=json.loads(rp.read_text());we=json.loads(ep.read_text())
  assert sha(rp)==w['receipt_sha256'] and sha(ep)==w['exit_sha256'] and we['evaluation_receipt_sha256']==sha(rp)
  assert we['supervision_passed'] and we['child_exit_code']==0 and wr['status']=='QUICK_LDM_WITNESS_PASSED' and wr['output_writers_closed']
  assert wr['source_sha256']==spec['entrypoint_sha256'] and wr['vae_checkpoint_sha256']==ck['sha256']
  assert wr['memory_witness']['model_and_adam_roundtrip_exact'] and wr['memory_witness']['all_rngs_restored_exactly'] and wr['memory_witness']['vae_frozen_and_unchanged']
out=ROOT/'quick_generator_20261026/ldm_runs'/a.run;out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
r={'status':'STARTING_QUICK_LDM','mode':a.mode,'quality_label':'QUICK_REPRODUCTION_NOT_PAPER_COMPLETE','full_paper_training':False,'optimizer_updates':0,'completed_primitive_attempts':0,'skipped_updates':0,
 'output_writers_closed':False,'source_sha256':sha(__file__),'spec_sha256':sha(sp),'corpus_receipt_sha256':spec['corpus_receipt_sha256'],'data_acceptance_sha256':spec['data_acceptance_sha256'],'vae_checkpoint_sha256':ck['sha256']}
def save():
 r['elapsed_seconds']=time.monotonic()-started;tmp=out/'receipt.partial';tmp.write_text(json.dumps(r,indent=2,allow_nan=False));tmp.replace(out/'receipt.json')
save();ctx=None;checkpoints=[];binding=None;at_closed_boundary=False;dest=pathlib.Path(spec['checkpoint_dir']);policy=spec['checkpoint_policy'];allowed=pathlib.Path(spec['checkpoint_allowed_root'])
try:
 import torch
 from quick_ldm_runtime import Runtime
 from quick_vae_runtime import batch_sha,restore_rng
 from checkpoint_storage_guard import check
 from generator_training_state import save_manager,load_manager,module_sha,equal,rng_state
 assert allowed.parent==pathlib.Path('/home/ouyanghao') and not allowed.is_symlink();allowed.mkdir(exist_ok=True);dest.mkdir(parents=True,exist_ok=True)
 budget=check(dest,allowed,policy,remaining_saves=4);cache=dest/'vae.pth'
 if cache.exists():assert sha(cache)==ck['sha256']
 else:assert shutil.disk_usage(dest).free>=budget['required_bytes']+spec['vae_cache_upper_bound_bytes']
 ctx=Runtime(spec,base,out);assert sha(cache)==ck['sha256'] and cache.stat().st_size<=spec['vae_cache_upper_bound_bytes']
 assert ctx.frozen_vae_sha==vr['trained_model_sha256']
 r.update(status='RUNNING_QUICK_LDM_WITNESS' if a.mode=='witness' else 'RUNNING_QUICK_LDM_TRAINING',initial_storage_check=budget,vae_cache_bytes=cache.stat().st_size,vae_frozen_model_sha256=ctx.frozen_vae_sha)
 if a.mode=='witness':
  r['memory_witness']=ctx.memory_witness();r['five_step_cfg_sampling_witness']=ctx.sample_quality(ctx.batch('val'))
  assert module_sha(ctx.vae)==ctx.frozen_vae_sha
  r.update(status='QUICK_LDM_WITNESS_PASSED',optimizer_updates=4,completed_primitive_attempts=4,skipped_updates=0,memory_only_diagnostic=True,training_checkpoint_created=False,weights_used_for_formal_training=False)
 else:
  numerical={k:spec[k] for k in ['ldm','seed','corpus_receipt_sha256','data_acceptance_sha256','vae_checkpoint','bound_sources','entrypoint_sha256','checkpoint_dir','checkpoint_policy']}
  binding={'stage':'ldm','feature_dim':69,'dof':29,'quick_training':True,'formal_training':False,'quality_label':r['quality_label'],
   'numeric_sha256':hashlib.sha256(json.dumps(numerical,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'corpus_receipt_sha256':spec['corpus_receipt_sha256'],'vae_checkpoint_sha256':ck['sha256']}
  if a.resume:
   path=pathlib.Path(a.resume);assert path.parent==dest and a.resume_sha;r['resume']=load_manager(ctx.manager,'denoiser',ctx.clock,ctx.generators,path,binding,a.resume_sha)
  else:assert not list(dest.glob('ldm_state_*.pth')) and not list(dest.glob('emergency_*.pth')),'Explicit bound resume required'
  import random,numpy as np
  outer=rng_state(ctx.generators);quality_seed=spec['seed']+99;random.seed(quality_seed);np.random.seed(quality_seed);torch.manual_seed(quality_seed);torch.cuda.manual_seed_all(quality_seed);ctx.generators['val'].manual_seed(quality_seed)
  fixed_batch=ctx.batch('val');restore_rng(outer,ctx.generators);assert equal(rng_state(ctx.generators),outer)
  r.update(fixed_validation_batch_sha256=batch_sha(fixed_batch),quality_baseline_manager_step=ctx.manager.step,quality_batch_preserves_training_rng=True,initial_full_sampling_quality=ctx.sample_quality(fixed_batch));save()
  while ctx.manager:
   at_closed_boundary=False;records=ctx.run_batch(ctx.batch('train'))
   while ctx.manager.should_eval():ctx.run_batch(ctx.batch('val'),is_eval=True)
   at_closed_boundary=True;step=ctx.manager.step;r.update(optimizer_updates=ctx.clock.updates,completed_primitive_attempts=ctx.clock.attempts,skipped_updates=ctx.clock.skipped,last_train_batch=records)
   if step%100==0:
    r['last_storage_check']=check(dest,allowed,policy,remaining_saves=sum(x>=step for x in [1000,2000,3000])+1);save();print(json.dumps({'step':step,'updates':ctx.clock.updates,'skips':ctx.clock.skipped,'loss':records[-1]['loss']}),flush=True)
   if step in [1000,2000,3000]:
    path=dest/('ldm_state_'+str(step)+'.pth');assert not path.exists();check(dest,allowed,policy,remaining_saves=sum(x>=step for x in [1000,2000,3000])+1)
    item=save_manager(ctx.manager,'denoiser',ctx.clock,ctx.generators,path,binding,policy['reserve_bytes']+policy['log_allowance_bytes'],policy['checkpoint_upper_bound_bytes']);checkpoints.append(item);r['checkpoints']=checkpoints
    assert item['bytes']<=policy['checkpoint_upper_bound_bytes'],'Retain oversized file, stop and re-budget'
    r['latest_file_resume_verification']=load_manager(ctx.manager,'denoiser',ctx.clock,ctx.generators,path,binding,item['sha256']);save()
  r['final_full_sampling_quality']=ctx.sample_quality(fixed_batch);assert module_sha(ctx.vae)==ctx.frozen_vae_sha
  assert ctx.clock.attempts==3000 and ctx.manager.step==3000 and not ctx.manager.should_eval()
  r.update(status='QUICK_LDM_3000_TRAINING_FINISHED',trained_denoiser_sha256=module_sha(ctx.model),vae_frozen_and_unchanged=True,next_stage='Independent actual text/autoregressive generation then physics execution; not yet a closed loop')
 r.update(peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),output_writers_closed=True);save();print(json.dumps(r),flush=True)
except BaseException as e:
 r.update(status='FAILED_QUICK_LDM',error=repr(e),traceback=traceback.format_exc(),checkpoints=checkpoints)
 if ctx is not None:r.update(optimizer_updates=ctx.clock.updates,completed_primitive_attempts=ctx.clock.attempts,skipped_updates=ctx.clock.skipped)
 if a.mode=='train' and ctx is not None and binding is not None and at_closed_boundary and ctx.manager.step>0 and not(checkpoints and checkpoints[-1]['manager_step']==ctx.manager.step):
  try:
   assert not list(dest.glob('emergency_*.pth'));check(dest,allowed,policy,remaining_saves=1)
   r['emergency_checkpoint']=save_manager(ctx.manager,'denoiser',ctx.clock,ctx.generators,dest/('emergency_'+str(ctx.manager.step)+'.pth'),binding,policy['reserve_bytes']+policy['log_allowance_bytes'],policy['checkpoint_upper_bound_bytes']);assert r['emergency_checkpoint']['bytes']<=policy['checkpoint_upper_bound_bytes']
  except BaseException as err:r['emergency_save_refused_or_failed']=repr(err)
 else:r['emergency_save_scope']='Only a closed four-primitive and validation boundary; prior checkpoints retained'
 save();raise
finally:
 if ctx is not None:ctx.close()

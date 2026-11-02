"""Resume the actual99700 emergency, stop normally at100000; original math unchanged."""
import argparse,io,json,os,pathlib,time,traceback
from user_vae100k_resume_protocol import ROOT,read,sha,failed_source
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['witness','train'],required=True);p.add_argument('--run',required=True);a=p.parse_args()
sp=ROOT/'configs/user-vae100k-resume-spec.json';s=read(sp)
assert os.getuid()==1013 and os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and set(os.sched_getaffinity(0))=={0,1,2,3}
assert sha(__file__)==s['entrypoint_sha256'] and s['witness_authorized'] and (a.mode=='witness' or s['train_authorized'])
v=failed_source(s);out=ROOT/'vae100k_user_handoff_20261028'/a.run;out.mkdir(exist_ok=False)
r=dict(status='STARTING_USER_VAE100K_RESUME',run=a.run,mode=a.mode,source_sha256=sha(__file__),spec_sha256=sha(sp),resume_checkpoint=s['resume_checkpoint'],output_writers_closed=False,new_optimizer_updates=0,original200k_completion_claimed=False)
def save():
 t=out/'receipt.partial';t.write_text(json.dumps(r,indent=2,allow_nan=False));t.replace(out/'receipt.json')
save();ctx=None
try:
 import torch
 from generator8278_vae_runtime import Runtime,restore_rng,batch_sha
 from generator_training_state import load_manager,save_manager,module_sha,rng_state,equal
 from checkpoint_storage_guard import check
 original=v['spec'];ctx=Runtime(original,read(ROOT/'configs/paper69-env-spec.json'),out)
 binding=dict(stage='vae',feature_dim=69,dof=29,quick_training=False,formal_training=True,paper_schedule=True,prefilter_data=False,actual_strict_input=True,paper_two_stage_route_reproduced=False,
  quality_label=v['initial_receipt']['quality_label'],numeric_sha256=v['runtime_binding_sha256'],corpus_receipt_sha256=original['corpus_receipt_sha256'])
 r['initial_resume']=load_manager(ctx.manager,'vae',ctx.clock,ctx.generators,s['resume_checkpoint']['path'],binding,s['resume_checkpoint']['sha256'])
 assert ctx.manager.max_steps==200000 and ctx.clock.attempts==ctx.clock.updates==ctx.manager.step==99700 and ctx.clock.skipped==0
 def finite(x):
  if torch.is_tensor(x):assert torch.isfinite(x).all()
  elif isinstance(x,dict):
   for z in x.values():finite(z)
  elif isinstance(x,(list,tuple)):
   for z in x:finite(z)
 finite(ctx.model.state_dict());finite(ctx.manager.optimizer.state_dict())
 if a.mode=='witness':
  records=ctx.run_batch(ctx.batch('train'));assert ctx.clock.updates==ctx.manager.step==99704 and all(x['gradients_finite'] for x in records)
  m=ctx.manager;m._to_eval_steps=4
  memory=io.BytesIO();torch.save(dict(model=ctx.model.state_dict(),optimizer=m.optimizer.state_dict(),rng=rng_state(ctx.generators),clock=ctx.clock.state(),step=m.step,pending=m._to_eval_steps),memory)
  payload=memory.getvalue();state=torch.load(io.BytesIO(payload),map_location='cpu');before=module_sha(ctx.model);expected=batch_sha(ctx.batch('train'))
  with torch.no_grad():next(ctx.model.parameters()).add_(.01)
  ctx.model.load_state_dict(state['model']);m.optimizer.load_state_dict(state['optimizer']);m.step=state['step'];m._to_eval_steps=state['pending'];ctx.clock.load(state['clock']);restore_rng(state['rng'],ctx.generators)
  assert equal(ctx.model.state_dict(),state['model']) and equal(m.optimizer.state_dict(),state['optimizer']) and equal(rng_state(ctx.generators),state['rng']) and module_sha(ctx.model)==before
  assert batch_sha(ctx.batch('train'))==expected;restore_rng(state['rng'],ctx.generators)
  ctx.run_batch(ctx.batch('val'),is_eval=True);assert ctx.clock.updates==ctx.manager.step==99704 and not m.should_eval() and module_sha(ctx.model)==before
  r.update(status='USER_VAE100K_RESUME_WITNESS_PASSED',new_optimizer_updates=4,finite_gradients=True,exact_model_adam_rng_nextbatch_restore=True,pending_validation_without_updates=True,diagnostic_state_reused_for_training=False,persistent_checkpoint_created=False)
 else:
  assert len(s['witnesses'])==2
  for w in s['witnesses']:
   from filtered15000_vae_completion import closed_job
   rp=ROOT/'vae100k_user_handoff_20261028'/w['run']/'receipt.json';wr=read(rp);proof=closed_job('user_vae100k_resume_'+w['run'],rp,w['receipt_sha256'],s['supervisor_sha256'])
   assert proof['exit_sha256']==w['exit_sha256'] and wr['status']=='USER_VAE100K_RESUME_WITNESS_PASSED' and wr['resume_checkpoint']==s['resume_checkpoint']
   assert wr['finite_gradients'] and wr['exact_model_adam_rng_nextbatch_restore'] and wr['pending_validation_without_updates'] and not wr['diagnostic_state_reused_for_training']
  dest=pathlib.Path(original['checkpoint_dir']);allowed=pathlib.Path(original['checkpoint_allowed_root']);policy=original['checkpoint_policy']
  r['storage_before']=check(dest,allowed,policy,remaining_saves=2)
  r['status']='RUNNING_USER_VAE100K_RESUME';save()
  while ctx.manager.step<100000:
   records=ctx.run_batch(ctx.batch('train'))
   while ctx.manager.should_eval():ctx.run_batch(ctx.batch('val'),is_eval=True)
   assert ctx.manager.step<=100000
   r.update(completed_primitive_attempts=ctx.clock.attempts,optimizer_updates=ctx.clock.updates,skipped_updates=ctx.clock.skipped,new_optimizer_updates=ctx.clock.updates-99700)
   if ctx.manager.step%100==0:r['storage_before']=check(dest,allowed,policy,remaining_saves=2);save()
  assert ctx.manager.step==ctx.clock.attempts==ctx.clock.updates==100000 and ctx.clock.skipped==0 and not ctx.manager.should_eval()
  ck=save_manager(ctx.manager,'vae',ctx.clock,ctx.generators,s['target_checkpoint_path'],binding,policy['reserve_bytes']+policy['log_allowance_bytes'],policy['checkpoint_upper_bound_bytes'])
  r['selected_checkpoint']=ck;r['latest_file_resume_verification']=load_manager(ctx.manager,'vae',ctx.clock,ctx.generators,ck['path'],binding,ck['sha256'])
  assert sha(s['resume_checkpoint']['path'])==s['resume_checkpoint']['sha256'];r.update(status='USER_VAE100000_RESUME_FINISHED',completed_primitive_attempts=100000,optimizer_updates=100000,skipped_updates=0,new_optimizer_updates=300)
 ctx.close();ctx=None;r['output_writers_closed']=True;save();print(json.dumps(r),flush=True)
except BaseException as error:
 r.update(status='FAILED_USER_VAE100K_RESUME',error=repr(error),traceback=traceback.format_exc());save();raise
finally:
 if ctx is not None:ctx.close()

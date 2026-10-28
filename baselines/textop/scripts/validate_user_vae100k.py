"""Restore the real100k prefix and assess the original fixed batch, zero training."""
import argparse,hashlib,json,os,pathlib,time,traceback
from user_vae100k_completion import ROOT,read,sha,stopped_vae100k
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();assert a.run=='user_vae100k_quality_v1'
sp=ROOT/'configs/user-vae100k-quality-spec.json';s=read(sp);assert s['quality_authorized'] and sha(__file__)==s['entrypoint_sha256']
assert os.getuid()==1013 and os.environ.get('CUDA_VISIBLE_DEVICES')=='1'
for n,h in s['bound_sources'].items():assert sha(ROOT/n)==h
v=stopped_vae100k(s['vae_training']);out=ROOT/'vae100k_user_handoff_20261028'/a.run;out.mkdir(exist_ok=False)
r=dict(status='STARTING_USER_VAE100K_QUALITY',run=a.run,source_sha256=sha(__file__),spec_sha256=sha(sp),
 selected_checkpoint=v['checkpoint'],user_variant='VAE100000_PREFIX_FROM_UNMODIFIED_200K_PROGRAM',paper200k_completed=False,
 training_optimizer_updates_performed=0,output_writers_closed=False)
def save():
 t=out/'receipt.partial';t.write_text(json.dumps(r,indent=2,allow_nan=False));t.replace(out/'receipt.json')
save();ctx=None
try:
 import random,numpy as np,torch
 from generator8278_vae_runtime import Runtime,restore_rng,batch_sha
 from generator_training_state import load_manager,module_sha,rng_state,equal
 original=v['spec'];ctx=Runtime(original,read(ROOT/'configs/paper69-env-spec.json'),out)
 binding=dict(stage='vae',feature_dim=69,dof=29,quick_training=False,formal_training=True,paper_schedule=True,prefilter_data=False,actual_strict_input=True,paper_two_stage_route_reproduced=False,
  quality_label=v['initial_receipt']['quality_label'],numeric_sha256=v['runtime_binding_sha256'],corpus_receipt_sha256=original['corpus_receipt_sha256'])
 r['full_state_restore']=load_manager(ctx.manager,'vae',ctx.clock,ctx.generators,v['checkpoint']['path'],binding,v['checkpoint']['sha256'])
 assert ctx.clock.attempts==ctx.manager.step==100000 and ctx.clock.updates==v['checkpoint']['actual_optimizer_updates']
 outer=rng_state(ctx.generators);seed=original['seed']+99
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);ctx.generators['val'].manual_seed(seed)
 fixed=ctx.batch('val');restore_rng(outer,ctx.generators);assert equal(rng_state(ctx.generators),outer)
 r['fixed_validation_batch_sha256']=batch_sha(fixed);assert r['fixed_validation_batch_sha256']==v['initial_receipt']['fixed_validation_batch_sha256']
 before=module_sha(ctx.model);updates=ctx.clock.updates
 r['initial_quality']=v['initial_receipt']['initial_quality']
 r['final_quality']={'teacher_forced':ctx.quality(fixed,False),'self_rollout':ctx.quality(fixed,True)}
 r['vae_quality_gate_passed']=all(r['final_quality'][k]['normalized_reconstruction_huber']<r['initial_quality'][k]['normalized_reconstruction_huber'] for k in r['final_quality'])
 assert r['vae_quality_gate_passed'],'Keep original100k; do not relax quality or resume VAE without user instruction'
 assert ctx.clock.updates==updates and ctx.clock.attempts==100000 and module_sha(ctx.model)==before
 assert sha(v['checkpoint']['path'])==v['checkpoint']['sha256']
 r.update(status='USER_VAE100K_QUALITY_VALIDATION_PASSED',completed_primitive_attempts=100000,optimizer_updates=updates,skipped_updates=ctx.clock.skipped,
  trained_model_sha256=before,original_runtime_binding_sha256=v['runtime_binding_sha256'],training_optimizer_updates_performed=0,
  corpus_receipt_sha256=original['corpus_receipt_sha256'],data_acceptance_sha256=original['data_acceptance_sha256'],strict_audit_receipt_sha256=original['strict_audit_receipt_sha256'],
  original200k_completion_claimed=False,output_writers_closed=True)
 ctx.close();ctx=None;save();print(json.dumps(r),flush=True)
except BaseException as error:
 r.update(status='FAILED_USER_VAE100K_QUALITY',error=repr(error),traceback=traceback.format_exc());save();raise
finally:
 if ctx is not None:ctx.close()

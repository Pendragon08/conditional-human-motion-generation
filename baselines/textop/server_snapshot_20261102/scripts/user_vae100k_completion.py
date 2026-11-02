"""Explicit user100k prefix provenance; never turn the old exit into200k success."""
import json,pathlib
from filtered15000_vae_completion import ROOT,sha,read,owned,closed_job

def stopped_vae100k(binding):
 if binding.get('kind')=='USER_VAE100000_RESUMED_FROM_CAPACITY_STOP':
  from user_vae100k_resume_protocol import completed_resume
  return completed_resume(binding)
 assert binding['kind']=='USER_REQUESTED_VAE100000_PREFIX' and binding['selected_step']==100000
 run=binding['control_run'];assert run=='user_vae100k_control_v1'
 cp=owned(ROOT/'vae100k_user_handoff_20261028'/run/'receipt.json')
 assert sha(cp)==binding['control_receipt_sha256'];control=read(cp)
 assert control['status']=='USER_VAE100000_STOP_ACCEPTED_ORIGINAL_EXITS_RETAINED'
 assert control['signals_sent']==1 and control['common_lock_released'] and not control['original200k_completion_claimed']
 policy_path=ROOT/'configs/vae100k-user-control-spec.json';policy=read(policy_path)
 assert policy['control_authorized'] and control['spec_sha256']==sha(policy_path)
 assert control['source_sha256']==policy['entrypoint_sha256']
 assert sha(ROOT/'scripts/watch_vae100k_handoff.py')==policy['entrypoint_sha256']
 assert sha(ROOT/'scripts/supervise_vae100k_handoff.py')==policy['supervisor_sha256']
 for n,h in policy['bound_sources'].items():assert sha(ROOT/n)==h
 closure=closed_job('vae100k_control_'+run,cp,sha(cp),policy['supervisor_sha256'])
 assert closure['exit_sha256']==binding['control_exit_sha256']
 original=policy['original_run'];sp=ROOT/'configs/filtered15000-vae-spec.json';spec=read(sp)
 assert sha(sp)==policy['original_spec_sha256'] and spec['vae']['stages']==[100000,50000,50000]
 old=ROOT/'filtered15000_20261027/vae_runs'/original/'receipt.json';final=read(old)
 assert sha(old)==control['original_final_receipt_sha256']
 assert final['status']=='FAILED_FILTERED15000_VAE' and final.get('error')=='KeyboardInterrupt()'
 for key,suffix in [('exit','_exit.json'),('process','_process_exit.json'),('outer','_outer_exit.json')]:
  p=ROOT/'receipts'/('filtered15000_vae_'+original+suffix)
  assert sha(p)==control['original_exit_sha256'][key]
 before=read(cp.parent/'producer_before_signal.json')
 assert before['status']=='RUNNING_FILTERED15000_VAE_TRAINING'
 assert before['quality_baseline_manager_step']==0
 assert before['spec_sha256']==policy['original_spec_sha256'] and before['runtime_binding_sha256']==policy['original_runtime_binding_sha256']
 ck=control['selected_checkpoint'];assert ck['manager_step']==100000 and ck['path']==policy['checkpoint_path']
 assert any(x==ck for x in before['checkpoints']) and any(x==ck for x in final['checkpoints'])
 from vae100k_control_protocol import verify_checkpoint
 verify_checkpoint(ck,spec['checkpoint_allowed_root'])
 for k in ['corpus_receipt_sha256','data_acceptance_sha256','strict_audit_receipt_sha256']:
  assert before[k]==final[k]==spec[k]==policy[k]
 return dict(spec=spec,policy=policy,control=control,checkpoint=ck,initial_receipt=before,
  runtime_binding_sha256=policy['original_runtime_binding_sha256'],control_closure=closure)

def accepted_vae100k(binding):
 v=stopped_vae100k(binding);run=binding['quality_run']
 assert run=='user_vae100k_quality_v1'
 p=owned(ROOT/'vae100k_user_handoff_20261028'/run/'receipt.json')
 assert sha(p)==binding['quality_receipt_sha256'];r=read(p)
 s=read(ROOT/'configs/user-vae100k-quality-spec.json')
 assert s['quality_authorized'] and r['spec_sha256']==sha(ROOT/'configs/user-vae100k-quality-spec.json')
 assert sha(ROOT/'scripts/supervise_user_vae100k_quality.py')==s['supervisor_sha256']
 assert r['source_sha256']==s['entrypoint_sha256'] and sha(ROOT/'scripts/validate_user_vae100k.py')==s['entrypoint_sha256']
 for n,h in s['bound_sources'].items():assert sha(ROOT/n)==h
 proof=closed_job('user_vae100k_quality_'+run,p,sha(p),s['supervisor_sha256'])
 assert proof['exit_sha256']==binding['quality_exit_sha256']
 assert r['status']=='USER_VAE100K_QUALITY_VALIDATION_PASSED' and r['output_writers_closed']
 assert r['vae_quality_gate_passed'] and r['training_optimizer_updates_performed']==0
 assert r['selected_checkpoint']==v['checkpoint'] and r['completed_primitive_attempts']==100000
 assert r['fixed_validation_batch_sha256']==v['initial_receipt']['fixed_validation_batch_sha256']
 assert r['initial_quality']==v['initial_receipt']['initial_quality']
 assert all(r['full_state_restore'][k] for k in ['model_restored_exactly','optimizer_restored_exactly','rng_and_generator_restored_exactly','clock_restored_exactly'])
 v.update(receipt=r,receipt_sha256=sha(p),quality_closure=proof)
 return v

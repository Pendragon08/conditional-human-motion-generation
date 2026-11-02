"""Explicit capacity-failure recovery; no signal and no rewriting old failures."""
import gzip,json,os,pathlib
from filtered15000_vae_completion import ROOT,sha,read,owned,closed_job
from vae100k_control_protocol import verify_checkpoint
OLD='filtered15000_vae200k_train_v1'

def failed_source(s):
 sp=ROOT/'configs/filtered15000-vae-spec.json';old=read(sp)
 assert sha(sp)==s['original_spec_sha256']=='b7eb6e95b082fd3902099ea718ec07bd42b3418c49c179f88f5438e5c5d2c6fa'
 for n,h in s['bound_sources'].items():assert sha(ROOT/n)==h,n
 rp=ROOT/'filtered15000_20261027/vae_runs'/OLD/'receipt.json';r=read(rp)
 assert sha(rp)==s['original_receipt_sha256'] and r['status']=='FAILED_FILTERED15000_VAE'
 assert r['completed_primitive_attempts']==r['optimizer_updates']==99700 and r['skipped_updates']==0
 assert r['error'].startswith("AssertionError('Insufficient checkpoint capacity:")
 assert r['emergency_checkpoint']==s['resume_checkpoint'] and s['resume_checkpoint']['manager_step']==99700
 prefix='filtered15000_vae_'+OLD
 paths={k:ROOT/'receipts'/(prefix+x) for k,x in [('exit','_exit.json'),('process','_process_exit.json'),('outer','_outer_exit.json')]}
 ex,pr,outer=[read(paths[k]) for k in ['exit','process','outer']]
 for k,p in paths.items():assert sha(p)==s['original_exit_sha256'][k]
 assert outer['actual_returncode']==1
 for e in [ex,pr]:assert e['child_exit_code']==1 and e['child_reaped'] and not e['timed_out'] and e['supervisor_error'] is None and not e.get('cleanup_error') and e['sample_errors']==[]
 assert ex['evaluation_receipt_sha256']==sha(rp) and ex['process_exit_sha256']==sha(paths['process'])
 resources=ROOT/'logs'/(prefix+'_resources.jsonl');samples=[json.loads(x) for x in resources.read_text().splitlines()]
 assert sha(resources)==ex['resources_sha256'] and len(samples)==ex['samples_written']==pr['samples_written'] and samples[-1]['phase']=='after_exit'
 for name in ['scripts/run_filtered15000_vae.py','scripts/supervise_filtered15000_vae.py','scripts/watch_vae100k_handoff.py','scripts/supervise_vae100k_handoff.py']:
  for p in pathlib.Path('/proc').iterdir():
   if not p.name.isdigit():continue
   try:
    if p.stat().st_uid==1013:assert str(ROOT/name).encode() not in (p/'cmdline').read_bytes().split(b'\0'),'Original worker still present'
   except (FileNotFoundError,PermissionError):pass
 control=read(ROOT/'vae100k_user_handoff_20261028/user_vae100k_control_v1/receipt.json')
 assert control['status']=='FAILED_VAE100K_CONTROL_RETAIN_ALL_EVIDENCE' and control['signals_sent']==0
 assert not (ROOT/'vae100k_user_handoff_20261028/user_vae100k_control_v1/signal_intent.json').exists()
 proof=verify_checkpoint(s['resume_checkpoint'],old['checkpoint_allowed_root'])
 assert r['quality_baseline_manager_step']==0 and r['runtime_binding_sha256']==s['original_runtime_binding_sha256']
 for k in ['corpus_receipt_sha256','data_acceptance_sha256','strict_audit_receipt_sha256']:assert r[k]==old[k]
 return dict(spec=old,initial_receipt=r,checkpoint=s['resume_checkpoint'],checkpoint_proof=proof,runtime_binding_sha256=s['original_runtime_binding_sha256'])

def completed_resume(binding):
 assert binding['kind']=='USER_VAE100000_RESUMED_FROM_CAPACITY_STOP' and binding['selected_step']==100000
 sp=ROOT/'configs/user-vae100k-resume-spec.json';s=read(sp);v=failed_source(s)
 assert s['train_authorized'] and binding['run']=='user_vae100k_resume_train_v1'
 rp=ROOT/'vae100k_user_handoff_20261028'/binding['run']/'receipt.json';r=read(rp)
 assert sha(rp)==binding['receipt_sha256'] and r['status']=='USER_VAE100000_RESUME_FINISHED' and r['output_writers_closed']
 assert r['source_sha256']==s['entrypoint_sha256'] and sha(ROOT/'scripts/run_user_vae100k_resume.py')==s['entrypoint_sha256'] and r['spec_sha256']==sha(sp)
 proof=closed_job('user_vae100k_resume_'+binding['run'],rp,sha(rp),s['supervisor_sha256']);assert proof['exit_sha256']==binding['exit_sha256']
 assert r['completed_primitive_attempts']==r['optimizer_updates']==100000 and r['skipped_updates']==0 and r['new_optimizer_updates']==300
 assert r['resume_checkpoint']==s['resume_checkpoint'] and all(r['initial_resume'][k] for k in ['model_restored_exactly','optimizer_restored_exactly','rng_and_generator_restored_exactly','clock_restored_exactly','pending_evaluation_fields_restored'])
 assert all(r['latest_file_resume_verification'][k] for k in ['model_restored_exactly','optimizer_restored_exactly','rng_and_generator_restored_exactly','clock_restored_exactly','pending_evaluation_fields_restored'])
 ck=r['selected_checkpoint'];assert ck['manager_step']==ck['actual_optimizer_updates']==100000
 assert ck['path']==s['target_checkpoint_path'];verify_checkpoint(ck,v['spec']['checkpoint_allowed_root'])
 v.update(checkpoint=ck,resume_receipt=r,resume_closure=proof)
 return v

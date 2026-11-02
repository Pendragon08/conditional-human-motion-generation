"""Read-only provenance gate for the new natural-ending 5000 PPO retry.

No torch, model, arrays or simulator are imported here. Actual checkpoint
contents must be independently audited later under the shared model lock.
"""
import json,os,pathlib
from step4_ppo5000_frozen_protocol_v2 import ROOT,owned,read,sha,fingerprint,verify_exit
RUN='user_step4_train_only_ppo5000_natural_train_v2'
REQUEST_SHA='344ebe00f2bd50457249350707f6315acb149b1781e97b5bb4d012734fd6f05d'
ENTRY_SHA='d238ee4c4190d10d6c60bcb556957b03d873d68556af92393918716544f11cc5'
SUP_SHA='a74a5e664a9f115437eda23f6bd61dcddbbbcaecf4c68af183235b86bf5cc11e'
ORIGINAL_RECEIPT_SHA='4d73595d9f2225c0eb361c1b6e663742307d60f4abe4ace7aa3756b448b3c661'
CONTROL_RECEIPT_SHA='8f66844391017e8b4b302d3b2f236dd6d6ee9f15bc6516b4e0371be77bdbb68d'

def accepted_natural_training(binding=None):
 parent=owned(ROOT/'step34_numeric_v1');out=owned(parent/RUN);log=owned(parent/(RUN+'_supervision'))
 prefix='step4_train_only_'+RUN;request=owned(ROOT/'receipts'/(prefix+'_request.json'));assert sha(request)==REQUEST_SHA
 s=read(request);assert s['run']==RUN and s['role']=='train' and s['policy']['training_authorized'] is True
 assert s['policy']['target_RL_iterations']==s['policy']['human_selected_target_iterations']==5000
 assert s['policy']['stage3_status']=='POSTPONED_BY_USER' and s['author_checkpoint'] is None
 assert s['executor_sha256']==ENTRY_SHA and s['supervisor_sha256']==SUP_SHA
 assert sha(owned(ROOT/'scripts/run_step4_train_only_ppo5000_natural_v2.py'))==ENTRY_SHA
 assert sha(owned(ROOT/'scripts/supervise_step4_train_only_ppo5000_natural_v2.py'))==SUP_SHA
 for name,digest in s['bound_sources'].items():assert sha(owned(ROOT/name))==digest
 paths=dict(receipt=out/'receipt.json',exit=log/'exit.json',process=log/'process_exit.json',outer=ROOT/'receipts'/(prefix+'_outer_exit.json'))
 files={k:read(p) for k,p in paths.items()};hashes={k:sha(p) for k,p in paths.items()};rec=files['receipt'];ex=files['exit']
 for key in ('exit','process'):verify_exit(files[key],True)
 assert files['outer']['actual_returncode']==0 and ex['source_sha256']==SUP_SHA
 assert ex['request_sha256']==sha(request) and ex['receipt_sha256']==hashes['receipt'] and ex['process_exit_sha256']==hashes['process']
 assert ex['resource_sha256']==sha(owned(log/'resources.jsonl'))
 last=json.loads((log/'resources.jsonl').read_text().splitlines()[-1]);assert last['phase']=='after_exit'
 assert last['owned_group_processes']==last['owned_group_gpu_memory']==[]
 assert rec['status']=='STAGE4_ORIGINAL_TRAIN_ONE_STAGE_PPO_FINISHED' and rec['output_writers_closed'] and rec['inputs_unchanged']
 assert rec['source_sha256']==ENTRY_SHA and rec['request_sha256']==REQUEST_SHA
 assert rec['training_data_mode']=='original_train_only' and rec['stage3_status']=='POSTPONED_BY_USER'
 assert rec['formal_completed_iterations']==5000 and rec['formal_optimizer_updates']==100000
 assert rec['optimizer_attempts_in_process']==rec['actual_optimizer_calls_in_process']==100000
 assert rec['old_teacher_training_updates']==rec['VAE_updates']==rec['LDM_updates']==0
 assert rec['task_signal_handlers_installed_after_sdk_start'] and rec['task_signal_handlers_verified_before_learning']
 assert rec['input_library_binding']==s['library_binding']
 names=['completed_1000.private.pth','completed_2000.private.pth','completed_3000.private.pth','completed_4000.private.pth','completed_5000.private.pth']
 assert len(rec['checkpoints'])==5 and [pathlib.Path(x['path']).name for x in rec['checkpoints']]==names
 for item,name in zip(rec['checkpoints'],names):
  p=owned(item['path']);assert p==out/name and p.stat().st_size==item['bytes']<=s['state_slot_bytes'] and sha(p)==item['sha256']
  assert isinstance(item['content_sha256'],str) and len(item['content_sha256'])==64
 for proc in pathlib.Path('/proc').iterdir():
  if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
  try:
   if proc.stat().st_uid!=1013:continue
   argv=(proc/'cmdline').read_bytes().decode().split('\0')
   assert not (str(request) in argv and any(str(ROOT/'scripts'/n) in argv for n in ('run_step4_train_only_ppo5000_natural_v2.py','supervise_step4_train_only_ppo5000_natural_v2.py')))
  except (FileNotFoundError,ProcessLookupError):pass
 original=owned(parent/'user_step4_train_only_ppo10000_train_v4/receipt.json');control=owned(parent/'user_ppo5000_control_v1/receipt.json')
 assert sha(original)==ORIGINAL_RECEIPT_SHA and sha(control)==CONTROL_RECEIPT_SHA
 cp=rec['checkpoints'][-1];fp=fingerprint(cp['path']);assert fp['bytes']==cp['bytes'] and fp['sha256']==cp['sha256']
 actual=dict(kind='NATURAL_PPO5000_FRESH_RETRY_FULLSTATE_UNAUDITED',run=RUN,request_sha256=REQUEST_SHA,
  closure_sha256=hashes,input_library_binding=s['library_binding'],checkpoint=cp,checkpoint_fingerprint=fp,
  original_failed_receipt_sha256=ORIGINAL_RECEIPT_SHA,original_control_failure_sha256=CONTROL_RECEIPT_SHA,
  actual_checkpoint_content_not_yet_cpu_audited=True,original_run_recovered=False)
 if binding is not None:assert binding==actual
 return dict(binding=actual,request=s,receipt=rec)

"""Natural-retry CPU export gate, separate from interrupted-5000 history.

No neural runtime or simulator is imported here. Natural training and its
original failure binding must independently close before this is called.
"""
import json
from step4_ppo5000_frozen_protocol_v2 import ROOT,RESERVE,LOG,owned,read,sha,fingerprint,verify_exit
from ppo5000_natural_training_acceptance_v1 import accepted_natural_training
accepted_stop=accepted_natural_training

def accepted_export(binding=None):
 """Accept an actual bounded CPU export with separate real closure proof."""
 data=owned(ROOT/'step34_numeric_v1');run='user_step4_ppo5000_natural_frozen_export_v1'
 out=owned(data/run);log=owned(data/(run+'_supervision'))
 prefix='ppo5000_natural_frozen_'+run
 sp=owned(ROOT/'receipts'/(prefix+'_request.json'));s=read(sp)
 assert s['authorized'] is True and s['run']==run and s['role']=='export'
 assert s['training_binding'] and s['export_binding'] is None
 paths=dict(receipt=out/'receipt.json',exit=log/'exit.json',process=log/'process_exit.json',
  outer=ROOT/'receipts'/(prefix+'_outer_exit.json'))
 files={k:read(v) for k,v in paths.items()};hashes={k:sha(v) for k,v in paths.items()}
 for name in ['exit','process']:verify_exit(files[name],True)
 ex=files['exit'];rec=files['receipt']
 assert files['outer']['actual_returncode']==0 and ex['source_sha256']==s['supervisor_sha256']
 assert ex['request_sha256']==sha(sp) and ex['receipt_sha256']==hashes['receipt']
 assert ex['process_exit_sha256']==hashes['process'] and ex['resource_sha256']==sha(owned(log/'resources.jsonl'))
 last=json.loads((log/'resources.jsonl').read_text().splitlines()[-1])
 assert last['phase']=='after_exit' and last['owned_group_processes']==last['owned_group_gpu_memory']==[]
 assert rec['status']=='PPO5000_NATURAL_FROZEN_CPU_EXPORTED' and rec['output_writers_closed'] is True
 assert rec['source_sha256']==s['executor_sha256'] and rec['request_sha256']==sha(sp)
 assert rec['training_binding']==s['training_binding'] and rec['fullstate_exact_5000_100000'] is True
 assert rec['independent_actor_RMS_reload_exact'] is True and rec['synthetic_probe_only'] is True
 assert rec['full_model_and_cpu_Adam_restored_exact'] is True
 assert rec['optimizer_updates_performed']==rec['new_CUDA_contexts']==0 and rec['CUDA_RNG_restored'] is False
 artifact=rec['inference'];path=owned(artifact['path']);assert path==out/'actor_RMS.private.pth'
 fp=fingerprint(path);assert fp['bytes']==artifact['bytes'] and fp['sha256']==artifact['sha256'] and 0<fp['bytes']<=32*1024**2
 actual=dict(run=run,request_sha256=sha(sp),closure_sha256=hashes,inference=artifact,
  training_binding=s['training_binding'])
 if binding is not None:assert binding==actual
 return dict(binding=actual,receipt=rec,request=s)

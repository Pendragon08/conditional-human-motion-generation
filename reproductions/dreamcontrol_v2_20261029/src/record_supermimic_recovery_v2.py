import json,hashlib,fcntl,os
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');S=R/'stage3_pick29_20261103';D=R/'stage3_supermimic29_20261103';Q=R/'stage3_skill_sequence_20261103';W=D/'parallel_pick500_20261104/numerical_repair_v2'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def atomic(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
run=D/'runs/pick500_8912_72000_parallel_v2';s=read(run/'status.json');p=read(S/'runs/pick500_8912_3000_v1/status.json')
assert s['status']=='online_training' and s['formal_run'] and s['progress']['actual_vector_steps']>600
assert s['counters']['policy_updates']>166 and p['status']=='ppo_running'
stress=read(W/'stress600_passed.json');assert stress['actual_vector_steps']==600
cp=s['latest_checkpoint'];assert cp['roundtrip'] and cp['vector_steps']==96
assert Path(cp['path']).stat().st_size==cp['bytes'] and sha(cp['path'])==cp['sha256']
assert s['reference_manifest_sha256']==p['reference_manifest_sha256']
receipt={'status':'repair_verified_formal_training_running','PPO_updates':p['policy_updates'],'PPO_pid':p['pid'],
 'SuperMimic_pid':s['pid'],'SuperMimic_run':str(run),'progress':s['progress'],'counters':s['counters'],
 'stress600':stress,'early_checkpoint_verified':cp,'reference_manifest_sha256':s['reference_manifest_sha256'],
 'scope':'observed stability beyond previous failure and recoverable checkpoint; not completed training or task success',
 'formal_status_sha256':sha(run/'status.json')}
if not (W/'recovery_verified_receipt.json').exists():(W/'recovery_verified_receipt.json').open('x').write(json.dumps(receipt,indent=2))
a=read(D/'adaptation_status.json');a.update(phase='v2_repaired_formal_training_running',formal_training_started=True,
 formal_run_directory=str(run),recovery_receipt=str(W/'recovery_verified_receipt.json'),repair_controller=str(W/'status.json'),
 preserved_v1_failure=True);atomic(D/'adaptation_status.json',a)
with (Q/'controller.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);q=read(Q/'progress.json');q.update(status='pick500_PPO_and_SuperMimic_v2_formal_running',
  supermimic_recovery_receipt=str(W/'recovery_verified_receipt.json'));atomic(Q/'progress.json',q)
print(json.dumps(receipt,indent=2))

"""CPU provenance checks against the real closed10k handoff; no Torch/SDK."""
import copy,json,pathlib
from teacher10000_pilot_protocol import load_teacher,authorize_teacher,gate,select_sources
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
s=json.loads((ROOT/'configs/teacher10000-pilot-spec.json').read_text());t=load_teacher(ROOT,s)
checks=['real_10k_handoff_authorized']
def reject(name,fn):
    try:fn()
    except (AssertionError,KeyError):checks.append(name);return
    raise AssertionError('Unexpected acceptance: '+name)
reject('disabled_before_sdk',lambda:gate('smoke',s))
v=copy.deepcopy(s);v['pilot_launch_enabled']=True
gate('smoke',v);checks.append('smoke_only_flag')
reject('sample_requires_actual_witness_activation',lambda:gate('sample',v))
for name,path,value in [
 ('reject_old5k_count',['stop','recoverable_completed'],5000),
 ('reject_spoofed40k_count',['stop','recoverable_completed'],40000),
 ('reject_wrong_optimizer_count',['stop','recoverable_optimizer_updates'],100000),
 ('reject_unreleased_lock',['stop','common_sdk_lock_released'],False),
 ('reject_wrong_final_receipt',['stop','final_training_receipt_sha256'],'0'*64),
 ('reject_wrong_backup',['stop','backup_receipt_sha256'],'0'*64),
 ('reject_unreaped_original',['process','child_reaped'],False),
 ('reject_fullstate_not_backed_up',['backup','private_full_two_location_sha_verified'],False),
 ('reject_checkpoint_public_upload',['backup','full_checkpoint_publicly_uploaded'],True),
 ('reject_wrong_latest_checkpoint',['training','latest_checkpoint','sha256'],'0'*64),
 ('reject_unverified_resume',['training','resume_exact_state_verified'],False),
 ('reject_early_backup_optimizer',['backup','audit','optimizer_step_min'],100000)]:
    c=copy.deepcopy(t);z=c
    for key in path[:-1]:z=z[key]
    z[path[-1]]=value
    reject(name,lambda:authorize_teacher(s,c))
plan=json.loads((ROOT/'configs/filter-training-plan.json').read_text());rows=[]
for n in plan['input_reference_exports']:
    p=pathlib.Path('/dev/shm/ouyanghao_textop_20261025/tracker_references')/n/'manifest.private.jsonl'
    rows.extend(x for x in map(json.loads,p.read_text().splitlines()) if not x['mirrored'])
assert len(select_sources(rows,'smoke',s))==8;checks.append('exact_existing8_smoke_identities')
assert len(select_sources(rows,'sample',s))==64;checks.append('exact_existing64_sample_identities')
print(json.dumps({'status':'TEACHER10000_PROVENANCE_CPU_CHECKS_PASSED','checks':checks,'count':len(checks),'sdk_runs':0,'models_loaded':0,'training_signals':0,'actual_physics_validation':False}))

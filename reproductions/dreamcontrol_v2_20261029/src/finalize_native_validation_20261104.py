import fcntl,hashlib,json,pathlib,shutil
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');D=R/'stage3_supermimic29_20261103';N=D/'native_universal_20261104';Q=R/'stage3_skill_sequence_20261103'
def read(p):return json.loads(p.read_text())
def atomic(p,j):
 t=p.with_suffix('.complete.tmp');t.write_text(json.dumps(j,indent=2));t.replace(p)
evidence={}
for name in ['smoke_author_v1','capacity_8912_v1','benchmark_8912_v1','evaluation_entry500_v1','resume_8912_v1']:
 p=N/(name+'_guard.json');j=read(p);assert j['status']=='passed';evidence[name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
resume=read(N/'resume_8912_v1/status.json');assert resume['optimizer_updates']=={'opt_wm':120,'opt_pi':11,'opt_v':220} and resume['real_environment_transitions']==1408096
receipt={'status':'native_universal_trainer_validation_complete_formal_launch_pending','formal_training_started':False,
 'reference_count':500,'body_dof':29,'future_reference_actor':True,'evidence':evidence,'resume_checkpoint':resume['checkpoint'],
 'resume_semantics':'model and optimizer restored; replay rebuilt and64additional actual vectorsteps counted; RNG not restored',
 'task_interaction_certified':False,'free_bytes':shutil.disk_usage('/data').free,'next':'formal compute/checkpoint budget and protected native training launch; PPO continues'}
(N/'validation_complete_receipt.json').open('x').write(json.dumps(receipt,indent=2));atomic(N/'status.json',receipt)
p=D/'adaptation_status.json';j=read(p);j.update(phase=receipt['status'],formal_training_started=False,native_validation_receipt=str(N/'validation_complete_receipt.json'));atomic(p,j)
atomic(D/'active_native_universal_run.json',{'working_directory':str(N),'status_path':str(N/'status.json'),'validation_receipt':str(N/'validation_complete_receipt.json'),'formal_training_started':False})
with (Q/'controller.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=Q/'progress.json';j=read(p);j['SuperMimic_policy_mode']['native_validation_receipt']=str(N/'validation_complete_receipt.json');j['status']='Pick_PPO_running_native_universal_validation_complete_formal_launch_pending';atomic(p,j)
print(json.dumps(receipt,indent=2))

import csv,hashlib,json,pathlib,statistics,fcntl,os,shutil
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');D=R/'stage3_supermimic29_20261103';N=D/'native_universal_20261104';Q=R/'stage3_skill_sequence_20261103'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p,j):
 t=p.with_suffix('.record.tmp');t.write_text(json.dumps(j,indent=2));t.replace(p)
evidences={}
for name in ['smoke_author_v1','capacity_8912_v1','benchmark_8912_v1']:
 p=N/(name+'_guard.json');j=read(p);assert j['status']=='passed';evidences[name]={'path':str(p),'sha256':sha(p),'result':j['worker_result']}
metrics=list(csv.DictReader((N/'benchmark_8912_v1/logs/train_log.csv').open()))
ips=[float(r['it_per_s']) for r in metrics if int(r['iteration'])>=3];median=statistics.median(ips)
proof={'status':'native_main_trainer_compilation_batching_and8912_capacity_verified','formal_training_started':False,
 'native_main_sha256':sha(N/'source/supermimic/train.py'),'source_mode':'one universal reference-conditioned native body29 tracker',
 'evidence':evidences,'warm_benchmark_iterations_per_second_median':median,'warm_sample_count':len(ips),
 'sample_scope':'10iteration smoke; not long-run ETA or task performance',
 'remaining':['native read-only evaluator entry and checkpoint resume witness','formal resource/checkpoint budget and protective launch','hands/object integration and interaction success evaluation'],
 'PPO_uninterrupted':read(R/'stage3_pick29_20261103/runs/pick500_8912_3000_v1/status.json')['pid']==1436632,
 'free_bytes':shutil.disk_usage('/data').free}
(N/'native_compiled_benchmark_receipt.json').open('x').write(json.dumps(proof,indent=2))
p=D/'adaptation_status.json';j=read(p);j.update(phase='native_universal_compiled8912_verified',formal_training_started=False,native_benchmark_receipt=str(N/'native_compiled_benchmark_receipt.json'));atomic(p,j)
with (Q/'controller.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 p=Q/'progress.json';j=read(p);j['SuperMimic_policy_mode']['native_verification_receipt']=str(N/'native_compiled_benchmark_receipt.json');j['status']='Pick_PPO_running_native_universal_compiled8912_verified';atomic(p,j)
print(json.dumps({'status':proof['status'],'warm_median_it_per_second':median,'free_bytes':proof['free_bytes']}))
for name in ['evaluation_entry500_v1_guard.json','evaluation_entry500_v1/status.json']:
 p=N/name
 if p.exists(): print(name,p.read_text()[-4500:])

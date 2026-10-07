"""Unique GPU1 queue. Never mutates or signals existing training/viewer jobs."""
import fcntl,hashlib,json,os,signal,subprocess,time,traceback,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'configs/feedback_paused_isaac_20261007.json'
SOURCES=ROOT/'configs/feedback_paused_sources_20261007.json'
OUT=ROOT/'artifacts/feedback_paused_isaac_20261007'
MON=ROOT/'artifacts/monitoring/r3_feedback_paused_20261007'
WRAP=ROOT/'scripts/r3_feedback_paused_stage_20261007.sh'
OWN_STATE=False
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def free():
 return int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).strip())
def state(stage,seed=None,arm=None,status='RUNNING',**extra):
 atomic(MON.with_suffix('.state'),{'status':status,'stage':stage,'seed':seed,'arm':arm,'gpu':1,'updated_unix':time.time(),**extra})
def resources(conf):
 assert free()>=conf['gpu_min_free_mib'],'GPU1 headroom below stage-start reserve'
 import shutil
 assert shutil.disk_usage(ROOT).free/1024**3>=conf['disk_min_free_gib'],'Disk reserve'
def spawn(stage,seed,output,arm=''):
 log=MON.parent/f'{MON.name}.{stage}.{seed}.{arm or "none"}.log'
 assert not log.exists(),'Preserve existing stage log'
 f=log.open('w');p=subprocess.Popen(['bash',str(WRAP),stage,str(seed),str(output),arm],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 f.close();return p,log
def stop_owned(p):
 if p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=35)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
def run_stage(stage,seed,output,arm,conf,physics):
 resources(conf);state(stage,seed,arm)
 p,log=spawn(stage,seed,output,arm)
 try:
  while p.poll() is None:
   if physics.poll() is not None:raise RuntimeError('Paused physics process exited during '+stage)
   if free()<conf['gpu_reserve_mib']:raise RuntimeError('GPU1 reserve exhausted; stop only this new test')
   time.sleep(10)
  assert p.returncode==0,f'{stage} failed exit={p.returncode}; preserve {log}'
 finally:
  if p.poll() is None:stop_owned(p)
def analyze(summaries,conf):
 import numpy as np
 rows=[{**row,'seed':s['seed']} for s in summaries for row in s['rows']]
 estimates={}
 for arm in conf['arms']:
  selected=[r for r in rows if r['arm']==arm];push=[r for r in selected if r['domain']=='push'];clean=[r for r in selected if r['domain']=='clean']
  estimates[arm]={'attempted':len(selected),'retarget_missing':sum(r['retarget_missing'] for r in selected),'push_original_tracking_auc_rad_s':float(np.mean([r['post_branch_original_joint_rmse_auc_rad_s'] for r in push])),'clean_original_joint_rmse_rad':float(np.mean([r['whole_episode_original_joint_rmse_rad'] for r in clean])),'goal_success':float(np.mean([r['virtual_goal_success'] for r in selected])),'first_fall_rate':float(np.mean([r['first_fall_step'] is not None for r in selected])),'failure_including_missing':float(np.mean([r['failure'] for r in selected])),'switch_delta_p95_rad':float(np.percentile([r['switch_delta_p95_rad'] for r in selected],95)),'prefix_joint_rmse_rad':float(np.mean([r['prefix_joint_rmse_mean_rad'] for r in selected]))}
 contrasts={};rng=np.random.default_rng(20261007)
 for other in ['fixed','planned','alignment','shuffled']:
  a=[];b=[]
  for rid in conf['references']:
   a.append(np.mean([r['post_branch_original_joint_rmse_auc_rad_s'] for r in rows if r['reference']==rid and r['domain']=='push' and r['arm']=='actual']))
   b.append(np.mean([r['post_branch_original_joint_rmse_auc_rad_s'] for r in rows if r['reference']==rid and r['domain']=='push' and r['arm']==other]))
  a=np.array(a);b=np.array(b);index=rng.integers(0,len(a),(10000,len(a)))
  effect=1-a.mean()/max(b.mean(),1e-12);boot=1-a[index].mean(1)/np.maximum(b[index].mean(1),1e-12)
  contrasts[other]={'actual_relative_AUC_improvement':float(effect),'reference_cluster_bootstrap_95pct_CI':np.quantile(boot,[.025,.975]).tolist(),'reference_clusters':len(a)}
 target=estimates['actual'];g=conf['gates']
 acceptance=target['retarget_missing']==0
 improvement=all(contrasts[x]['actual_relative_AUC_improvement']>=g['relative_auc_improvement'] and contrasts[x]['reference_cluster_bootstrap_95pct_CI'][0]>0 for x in ['planned','alignment']) and contrasts['shuffled']['reference_cluster_bootstrap_95pct_CI'][0]>0
 safety=all(target['goal_success']>=estimates[x]['goal_success']-g['success_loss_pp']/100 and target['first_fall_rate']<=estimates[x]['first_fall_rate'] for x in ['planned','alignment'])
 clean=target['clean_original_joint_rmse_rad']<=estimates['fixed']['clean_original_joint_rmse_rad']*(1+g['clean_rmse_worsening_fraction'])
 spike=target['switch_delta_p95_rad']<=estimates['planned']['switch_delta_p95_rad']*(1+g['switch_spike_worsening_fraction'])
 return {'status':'COMPLETE','scope':conf['scope'],'seeds':conf['seeds'],'references':conf['references'],'episodes':len(rows),'attempted_denominator_retained':True,'summary_hashes':[sha(OUT/f'seed{s["seed"]}'/'summary.json') for s in summaries],'protocol_sha256':sha(PROTOCOL),'estimates':estimates,'contrasts':contrasts,'offline_component_gates':{'retarget':acceptance,'AUC_effect':improvement,'goal_fall_safety':safety,'clean_noninferiority':clean,'switch_spike':spike},'offline_component_gate_passed':all([acceptance,improvement,safety,clean,spike]),'innovation_validated':False,'causal_design':'Randomized physical slot assignment by reference/condition/seed; no exact cross-arm PhysX clone claim','online_realtime_gate_evaluated':False,'full_humanml_prefix':False,'physical_entity_contact_evaluated':False,'rows':rows}
def main():
 global OWN_STATE
 os.nice(10);MON.parent.mkdir(parents=True,exist_ok=True)
 lock=MON.with_suffix('.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not MON.with_suffix('.state').exists(),'Unique queue already dispatched; inspect rather than retry'
 conf=json.loads(PROTOCOL.read_text());sources=json.loads(SOURCES.read_text())
 for rel,value in sources.items():assert sha(ROOT/rel)==value,f'Source changed: {rel}'
 witness=json.loads((ROOT/'results/feedback_paused_isaac_20261007/independent_witness.json').read_text())
 assert witness['status']=='PASS' and witness['protocol_sha256']==sha(PROTOCOL)
 assert witness['isaac_source_sha256']==sha(ROOT/'scripts/r3_feedback_paused_isaac_20261007.py')
 assert not OUT.exists();OUT.mkdir(parents=True)
 OWN_STATE=True
 atomic(OUT/'launch.json',{'time_unix':time.time(),'sources':sources,'protocol':conf,'witness_sha256':sha(ROOT/'results/feedback_paused_isaac_20261007/independent_witness.json'),'GPU1_free_MiB':free(),'existing_training_untouched':True})
 summaries=[]
 for seed in conf['seeds']:
  resources(conf);output=OUT/f'seed{seed}';state('capture',seed)
  physics,log=spawn('physics',seed,output)
  try:
   started=time.time()
   while not (output/'branch_ready.json').exists():
    if physics.poll() is not None:raise RuntimeError(f'Capture failed exit={physics.returncode}; {log}')
    if time.time()-started>600:raise TimeoutError('Capture did not reach branch in 600s')
    time.sleep(2)
   run_stage('prepare',seed,output,'',conf,physics)
   for arm in conf['generation_arms']:run_stage('generate',seed,output,arm,conf,physics)
   manifests={}
   for arm in conf['generation_arms']:
    run_stage('retarget',seed,output,arm,conf,physics)
    folder=ROOT/f'third_party/DreamControl/TrajGen/sample_feedback_paused_20261007_seed{seed}_{arm}/Kick_sim2'
    manifests[arm]={'folder':str(folder),'files':{f.name:sha(f) for f in folder.glob('*_n.pkl')}}
   inputs=json.loads((output/'generation_inputs.json').read_text())
   atomic(output/'candidates_ready.json',{'seed':seed,'branch_ready_sha256':sha(output/'branch_ready.json'),'pairs':inputs['pairs'],'arms':manifests})
   state('continuation',seed)
   while physics.poll() is None:
    if free()<conf['gpu_reserve_mib']:raise RuntimeError('Continuation GPU reserve exhausted')
    time.sleep(2)
   assert physics.returncode==0,f'Continuation failed exit={physics.returncode}; {log}'
   summary=json.loads((output/'summary.json').read_text())
   assert summary['status']=='COMPLETE' and summary['steps']==500 and summary['same_episode_branch_without_state_reset']
   summaries.append(summary)
   atomic(OUT/'partial_summary.json',{'status':'PARTIAL','seeds_completed':[s['seed'] for s in summaries],'innovation_validated':False})
  except BaseException as ex:
   if output.exists():atomic(output/'abort.json',{'reason':str(ex),'preserve_logs':True})
   stop_owned(physics);raise
 interpreter='/data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python'
 result=json.loads(subprocess.check_output([interpreter,str(Path(__file__).resolve()),'--analyze'],text=True,timeout=120))
 atomic(OUT/'summary.json',result)
 state('finished',status='COMPLETE',summary=str(OUT/'summary.json'),innovation_validated=False)
 print('FEEDBACK_PAUSED_QUEUE_COMPLETE',flush=True)
if __name__=='__main__':
 try:
  if sys.argv[1:]==['--analyze']:
   conf=json.loads(PROTOCOL.read_text());summaries=[json.loads((OUT/f'seed{seed}'/'summary.json').read_text()) for seed in conf['seeds']]
   print(json.dumps(analyze(summaries,conf)))
  else:main()
 except BaseException as ex:
  traceback.print_exc()
  if OWN_STATE:state('failed',status='FAILED',reason=str(ex),preserve_logs=True)
  raise

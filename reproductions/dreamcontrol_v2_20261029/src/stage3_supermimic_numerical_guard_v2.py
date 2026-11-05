"""One isolated numerical diagnostic, then one conditional new formal version.

Never restart the failedv1 or edit running PPO/guard sources. Per-job caps and
combined disk/checkpoint reserve still apply. Finite-gradient failures stop.
"""
import ast,hashlib,json,os,subprocess,sys,time,shutil,traceback,math
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');S=R/'stage3_pick29_20261103';D=R/'stage3_supermimic29_20261103';O=S/'reference_scale500_20261104'
W=D/'parallel_pick500_20261104/numerical_repair_v2';RUN=D/'runs/pick500_8912_72000_parallel_v2';MAN=O/'final_library_v1/reference_bank.json'
sys.path.insert(0,str(D/'src'))
from stage3_pick_parallel_controller import resources,stop_group,alive
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(j):
 t=W/'status.tmp';t.write_text(json.dumps(j,indent=2,allow_nan=False));t.replace(W/'status.json')
def main():
 W.mkdir(exist_ok=False);state={'status':'CPU_numerical_verification','pid':os.getpid(),'formal_started':False,'PPO_not_interrupted':True};save(state)
 paths=[D/'src/stage3_supermimic_pick500_train_v2.py',D/'src/stage3_supermimic_pick_core_v2.py']
 for p in paths:ast.parse(p.read_text())
 (W/'source_binding.json').open('x').write(json.dumps({str(p):sha(p) for p in paths},indent=2))
 cpu="""import sys,torch,json
sys.path[:0]=['SOURCES','PICKSOURCE']
from stage3_supermimic_pick_core_v2 import stable_clip
torch.manual_seed(19)
a=torch.nn.Parameter(torch.zeros(200));b=torch.nn.Parameter(torch.zeros(200));g=torch.randn(200);a.grad=g.clone();b.grad=g.clone()
torch.nn.utils.clip_grad_norm_([a],1.);stable_clip([b],1.)
assert torch.allclose(a.grad,b.grad,atol=1e-7,rtol=1e-6)
c=torch.nn.Parameter(torch.zeros(200));c.grad=torch.full((200,),1e30);old=float(c.grad.norm());n=stable_clip([c],1.)
assert torch.isfinite(n) and torch.isfinite(c.grad).all() and abs(float(c.grad.norm())-1)<1e-6
c.grad[0]=float('nan')
try:stable_clip([c],1.)
except RuntimeError as e:assert 'nonfinite gradient components' in str(e)
else:raise AssertionError('NaN gradient was accepted')
print(json.dumps({'passed':True,'ordinary_FP32_clip_equivalent':True,'finite_huge_gradient_FP32_norm_overflows':old==float('inf'),'NaN_rejected':True}))
""".replace('SOURCES',str(D/'src')).replace('PICKSOURCE',str(S/'src'))
 P='/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python'
 out=subprocess.check_output([P,'-c',cpu],env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'},universal_newlines=True)
 (W/'CPU_gradient_verification.json').open('x').write(out)
 spec=read(D/'runtime_bridge_spec.json');env={**os.environ,**spec['env']};child=None;log=None
 def launch(name,output,extra):
  nonlocal child,log
  assert not output.exists()
  command=[spec['python'],'-u',str(paths[0]),'--headless','--device','cuda:0','--reference-manifest',str(MAN),'--output-dir',str(output),'--num-envs','8912']+extra
  log=(W/(name+'.log')).open('x');child=subprocess.Popen(command,cwd=spec['cwd'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  (W/(name+'_launch.json')).open('x').write(json.dumps({'pid':child.pid,'command':command,'report':str(output/'status.json')},indent=2))
  state.update(status=name,child_pid=child.pid,child_output=str(output));save(state)
 def wait(output,target_steps,timeout):
  started=time.monotonic();done=None
  while True:
   own=resources(child.pid);ppo=resources(1436626);free=shutil.disk_usage('/data').free
   state.update(own_resources=own,PPO_resources=ppo,free_bytes=free);save(state)
   assert own['gpu_bytes']<=12*1024**3 and own['rss_bytes']<=32*1024**3
   assert own['gpu_bytes']+ppo['gpu_bytes']<=24*1024**3 and own['rss_bytes']+ppo['rss_bytes']<=64*1024**3
   assert free>=8*1024**3+356*1024**2
   if (output/'status.json').exists():
    s=read(output/'status.json');state.update(last_phase_status=s['status'],progress=s.get('progress'),counters=s.get('counters'));save(state)
    if s['status']=='failed':raise RuntimeError(s.get('error'))
    if s['status']=='interaction_budget_completed':
     assert s['progress']['actual_vector_steps']==target_steps and s['latest_checkpoint']['roundtrip'] and s['reference_count']==500
     if done is None:done=time.monotonic()
     if child.poll() is not None or time.monotonic()-done>40:
      stop_group(child.pid);log.close();return s
   if child.poll() is not None:raise RuntimeError('phase exited before a complete receipt')
   assert time.monotonic()-started<timeout
   time.sleep(5)
 try:
  diagnostic=W/'stress600_v2';launch('stress600',diagnostic,['--diagnostic','--macro-rounds','25','--model-warmup-updates','500'])
  s=wait(diagnostic,600,2400);child=None
  assert s['counters']['policy_updates']>166
  (W/'stress600_passed.json').open('x').write(json.dumps({'report':str(diagnostic/'status.json'),'sha256':sha(diagnostic/'status.json'),
   'actual_vector_steps':600,'counters':s['counters'],'same_seed_full8912_configuration':True,
   'not_a_long_run_success_claim':True,'original_failure_last_confirmed_step':480},indent=2))
  full=s['latest_checkpoint']['bytes'];ppo_dir=S/'runs/pick500_8912_3000_v1/ppo';cp=max(p.stat().st_size for p in ppo_dir.glob('*.pt'))
  ppo_budget=max(0,63-len(list(ppo_dir.glob('*.pt'))))*math.ceil(cp*1.25/1024**2)*1024**2
  required=max(11*1024**3,8*1024**3+356*1024**2+math.ceil(full*1.15)*4+60*math.ceil(full*.006/1024**2)*1024**2+ppo_budget+64*1024**2)
  state.update(status='stress_passed_waiting_formal_shared_disk',required_start_free_bytes=required);save(state)
  while shutil.disk_usage('/data').free<required:time.sleep(30)
  launch('formal_v2',RUN,['--formal','--macro-rounds','3000','--model-warmup-updates','500'])
  active={'run_dir':str(RUN),'controller_pid':os.getpid(),'controller_status':str(W/'status.json'),'reference_manifest':str(MAN),
   'reference_manifest_sha256':sha(MAN),'device':'cuda:0','num_envs':8912,'real_vector_steps':72000,'real_transitions':641664000,
   'previous_failed_run_preserved':str(D/'runs/pick500_8912_72000_parallel_v1'),
   'not_resume':'v1 had no persisted optimization checkpoint; this is an explicit new run, previous compute separately recorded',
   'early_full_checkpoint_at_vector_step':96,'full_checkpoint_rounds':[1500,3000]}
  tmp=D/'active_pick500_run.tmp';tmp.write_text(json.dumps(active,indent=2));tmp.replace(D/'active_pick500_run.json')
  state['formal_started']=True;save(state);s=wait(RUN,72000,72*3600);child=None
  state.update(status='formal_budget_completed_evaluation_pending',final_checkpoint=s['latest_checkpoint']);save(state)
 except BaseException as error:
  state.update(status='failed_preserved',error=repr(error),traceback=traceback.format_exc());save(state)
  if child:stop_group(child.pid)
  raise
if __name__=='__main__':main()

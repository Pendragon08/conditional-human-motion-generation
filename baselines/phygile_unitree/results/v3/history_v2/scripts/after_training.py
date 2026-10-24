"""One-shot dependent evaluation/export pipeline; never stops another process."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess,time
p=argparse.ArgumentParser();p.add_argument('--training-run',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];runtime=root.parent;parent=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017')
a.output.mkdir(parents=True,exist_ok=False)
receipt={'status':'WAITING_FOR_TRAINING','training_run':a.training_run,'source':str(root),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'evaluation_results':[]}
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def save():
 q=a.output/'pipeline.partial';q.write_text(json.dumps(receipt,indent=2)+'\n');q.replace(a.output/'pipeline.json')
save();deadline=time.monotonic()+6*3600
try:
 while True:
  q=runtime/'runs'/a.training_run/'receipt.json'
  if q.exists():
   training=json.loads(q.read_text())
   if training['status']=='FAILED':raise RuntimeError('Training failed; no evaluations launched')
   if training['status']=='BOUNDED_TRAINING_COMPLETED':break
  if time.monotonic()>deadline:raise TimeoutError('Training completion was not observed in six hours')
  time.sleep(30)
 assert training['additional_ppo_iterations']==500 and training['actual_adam_calls']==10000
 assert training['executed_sources_unchanged'] and training['parent_unchanged']
 checkpoint=Path(training['final_checkpoint']);assert sha(checkpoint)==training['final_sha256']
 receipt.update(status='EVALUATING',training_receipt=training);save()
 env=os.environ.copy();env.pop('DISPLAY',None)
 env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',WANDB_MODE='disabled')
 for clip,motion in enumerate(sorted((parent/'data/test').glob('*.npz'))):
  out=a.output/('isaac_clip'+str(clip))
  command=['taskset','-c','16-19','bash',str(root/'scripts/run_isaac.sh'),str(root/'scripts/lab_run.py'),'--mode','evaluate','--checkpoint',str(checkpoint),'--output',str(out),'--num-envs','1','--motion-file',str(motion),'--headless','--device','cuda:1','--kit_args=--/renderer/multiGpu/enabled=false --/renderer/activeGpu=1 --/plugins/carb.tasking.plugin/threadCount=4 --/plugins/omni.tbb.globalcontrol/maxThreadCount=4']
  with (a.output/('isaac_clip'+str(clip)+'_stdout.txt')).open('x') as so,(a.output/('isaac_clip'+str(clip)+'_stderr.txt')).open('x') as se:
   result=subprocess.run(command,env=env,stdout=so,stderr=se,timeout=600)
  e=json.loads((out/'receipt.json').read_text()) if (out/'receipt.json').exists() else None
  receipt['evaluation_results'].append({'command':command,'exit_code':result.returncode,'receipt':e});save()
  if result.returncode or e['status']!='FIRST_EPISODE_EVALUATED':raise RuntimeError('Evaluation failed; retained all outputs')
 export=a.output/'active_policy'
 command=['taskset','-c','16-19','bash',str(root/'scripts/run_isaac.sh'),str(root/'scripts/export_active_policy.py'),'--checkpoint',str(checkpoint),'--contract',str(a.output/'isaac_clip0/contract.json'),'--encoder-source',str(parent/'upstream/source/whole_body_tracking/whole_body_tracking/utils/encoder_actor_critic.py'),'--output',str(export)]
 with (a.output/'export_stdout.txt').open('x') as so,(a.output/'export_stderr.txt').open('x') as se:
  result=subprocess.run(command,env=env,stdout=so,stderr=se,timeout=600)
 receipt['export']={'command':command,'exit_code':result.returncode};save()
 if result.returncode:raise RuntimeError('Active inference export failed')
 proof=json.loads((export/'artifact_proof.json').read_text());assert proof['status']=='BITWISE_ACTIVE_INFERENCE_PARITY'
 receipt.update(status='TRAINING_AND_ISAAC_EVALUATION_EXPORT_COMPLETED',export_proof=proof,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
except BaseException as error:
 receipt.update(status='FAILED',error=repr(error));save();raise

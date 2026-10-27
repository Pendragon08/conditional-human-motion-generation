"""Immutable bounded invocation for a fresh document-following witness."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--name',default='document_run');a=p.parse_args()
if a.name not in ('document_run','development_run'):raise ValueError('Explicit bounded run name required')
OUT=ROOT/'results/completion_v1'/a.name
OUT.mkdir(parents=True,exist_ok=False)
parent=ROOT/'logs/rsl_rl/phygile_moe_pilot/2026-10-17_11-25-01_extended2000_seed42/model_1999.pt'
plan=json.loads((ROOT/'configs/completion_plan.json').read_text())
assert hashlib.sha256(parent.read_bytes()).hexdigest()==plan['parent_checkpoint_sha256']
state={'status':'RUNNING','started_unix':time.time(),'plan_sha256':hashlib.sha256((ROOT/'configs/completion_plan.json').read_bytes()).hexdigest(),'stages':[]}
def stage(name,command,timeout=600):
 state['stages'].append({'name':name,'command':command,'status':'RUNNING'})
 (OUT/'pipeline.json').write_text(json.dumps(state,indent=2)+'\n')
 code=subprocess.call([sys.executable,'scripts/run_prefix_recorded.py','--name',
      f'completion_v1/{a.name}/{name}','--timeout',str(timeout),'--',*command],cwd=ROOT)
 state['stages'][-1].update(exit_code=code,status='COMPLETED' if code==0 else 'FAILED')
 state['status']='RUNNING' if code==0 else 'FAILED'
 (OUT/'pipeline.json').write_text(json.dumps(state,indent=2)+'\n')
 if code:raise SystemExit(code)
 # Some Isaac launch wrappers return zero after a signal. Require primary
 # artifacts and completion status, never the transport exit code alone.
 required={
  'prepare':ROOT/'cache/closed_loop_upstream/COMPLETION_HOOK.json',
  'regression':OUT/'regression/verification.json',
  'codec':OUT/'codec/output/verification.json',
  'curriculum':OUT/'curriculum/output/evaluation.json',
  'expert_mask':OUT/'expert_mask/output/evaluation.json',
  'training_hook':OUT/'training_hook/output/verification.json',
  'closed_loop':OUT/'closed_loop/output/verification.json'}[name]
 try:
  payload=json.loads(required.read_text())
  if name!='prepare' and payload.get('status') not in ('PASS','COMPLETED','COMPLETED_ENGINEERING_VALIDATION'):
   raise ValueError('Missing primary completion status')
 except Exception:
  state['status']='FAILED';state['stages'][-1]['status']='FAILED_MISSING_PRIMARY_EVIDENCE'
  (OUT/'pipeline.json').write_text(json.dumps(state,indent=2)+'\n')
  raise
stage('prepare',['python3','scripts/prepare_completion_upstream.py'],120)
stage('regression',['bash','scripts/run_isaac.sh','reimplementation/test_completion.py','--output',str(OUT/'regression/verification.json')],120)
stage('codec',['bash','scripts/run_isaac.sh','scripts/validate_codec262.py','--output',str(OUT/'codec/output')],120)
service=plan['curriculum_service']
request={'checkpoint':str(parent),'checkpoint_sha256':plan['parent_checkpoint_sha256'],
 'active_experts':1,
 'files':[str(ROOT/f) for f in service['files']],'attempts_per_file':service['attempts_per_file'],
 'seed':service['seed'],'device':'cuda:0','training_config':str(ROOT/'configs/moe_pilot.yaml')}
(OUT/'curriculum_request.json').write_text(json.dumps(request,indent=2)+'\n')
stage('curriculum',['python3','scripts/curriculum_evaluate.py','--request',str(OUT/'curriculum_request.json'),
 '--output',str(OUT/'curriculum/output')])
mask_request={**request,'active_experts':2,'attempts_per_file':2,'files':request['files'][:1],
 'purpose':'mask compatibility fixture only; expert 1 is untrained, no tracking performance claim'}
(OUT/'mask_request.json').write_text(json.dumps(mask_request,indent=2)+'\n')
stage('expert_mask',['python3','scripts/curriculum_evaluate.py','--request',str(OUT/'mask_request.json'),
 '--output',str(OUT/'expert_mask/output')])
stage('training_hook',['bash','scripts/run_isaac.sh','scripts/validate_curriculum_training.py',
 '--output',str(OUT/'training_hook/output')])
stage('closed_loop',['bash','scripts/run_isaac.sh','scripts/run_generated_closed_loop.py',
 '--checkpoint',str(parent),'--motion_file','data/train/sample_000_actor0.npz',
 '--normalization',str(OUT/'codec/output/normalization.npz'),'--output',str(OUT/'closed_loop/output'),
 '--headless','--device','cuda:0','--seed','2026'])
state.update(status='COMPLETED',finished_unix=time.time())
(OUT/'pipeline.json').write_text(json.dumps(state,indent=2)+'\n')
print('PHYGILE_COMPLETION_VALIDATION_COMPLETE',flush=True)

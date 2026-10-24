"""Wait for the recorded pilot, then evaluate predeclared clips/checkpoints."""
from pathlib import Path
import json, subprocess, time, sys, datetime

root=Path(__file__).resolve().parents[1]
receipt=root/'results/pilot_train20_nostate_seed42/receipt.json'
deadline=time.monotonic()+7400
while True:
 status=json.loads(receipt.read_text())
 if status['status']!='RUNNING':break
 if time.monotonic()>deadline:raise TimeoutError('Training receipt still running after bounded wait')
 time.sleep(20)
assert status['status']=='COMPLETED' and status['exit_code']==0,status['status']
runs=sorted((root/'logs/rsl_rl/phygile_moe_pilot').glob('*train20_nostate_seed42'))
assert len(runs)==1,runs
run=runs[0]
suite={'status':'RUNNING','selection':'predeclared train clips 0,8,10,14 plus all four source-group-disjoint test clips; checkpoints after 1 and 500 updates','entries':[]}
dest=root/'results/evaluation_suite.json'
for iteration in [0,499]:
 cp=run/f'model_{iteration}.pt';assert cp.is_file(),cp
 for split,indices in [('train',[0,8,10,14]),('test',[0,1,2,3])]:
  for i in indices:
   name=f'iter{iteration:03d}_{split}_{i:03d}'
   args=['bash','scripts/run_isaac.sh','scripts/evaluate_pilot.py','--checkpoint',str(cp),'--motion_file',f'data/{split}/sample_{i:03d}_actor0.npz','--output',f'results/evaluation/{name}','--headless','--device','cuda:0']
   command=['python3','scripts/run_recorded.py','--name',f'evaluation_logs/{name}','--timeout','360','--']+args
   proc=subprocess.run(command,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
   entry={'name':name,'checkpoint_iteration':iteration,'optimizer_updates':iteration+1,'split':split,'clip_index':i,'command':args,'exit_code':proc.returncode}
   suite['entries'].append(entry);dest.write_text(json.dumps(suite,indent=2)+'\n')
   print(name,proc.returncode,flush=True)
   if proc.returncode:
    suite['status']='FAILED';dest.write_text(json.dumps(suite,indent=2)+'\n');raise SystemExit(proc.stdout[-3000:])
suite['status']='COMPLETED';suite['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
dest.write_text(json.dumps(suite,indent=2)+'\n')
print('PHY_GILE_EVALUATION_SUITE_COMPLETE',flush=True)

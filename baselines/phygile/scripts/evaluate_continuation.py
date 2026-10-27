"""Evaluate the fixed 2000-update checkpoint; preserve all first-episode data."""
from pathlib import Path
import datetime, hashlib, json, subprocess, sys
root=Path(__file__).resolve().parents[1]
receipt=json.loads((root/'results/continuation_2000/train/receipt.json').read_text())
assert receipt['status']=='COMPLETED' and receipt['exit_code']==0
runs=list((root/'logs/rsl_rl/phygile_moe_pilot').glob('*extended2000_seed42'))
assert len(runs)==1,runs
run=runs[0];checkpoint=run/'model_1999.pt';assert checkpoint.is_file()
resume=json.loads((run/'resume_provenance.json').read_text())
assert resume['parent_optimizer_updates']==500 and resume['additional_updates_requested']==1500
plan=json.loads((root/'configs/continuation_2000_plan.json').read_text())
out=root/'results/continuation_2000/evaluation_suite.json'
assert not out.exists(), 'Refusing to overwrite an evaluation suite'
suite={'status':'RUNNING','fixed_total_updates':2000,'selection_rule':plan['selection_rule'],
       'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'entries':[]}
for split in ['train','test']:
    for index in plan['evaluation'][split+'_clips']:
        name=f'iter1999_{split}_{index:03d}'
        args=['bash','scripts/run_isaac.sh','scripts/evaluate_pilot.py','--checkpoint',str(checkpoint),
              '--motion_file',f'data/{split}/sample_{index:03d}_actor0.npz',
              '--output',f'results/continuation_2000/evaluation/{name}',
              '--num_envs','32','--seed','2026','--headless','--device','cuda:0']
        command=[sys.executable,'scripts/run_recorded.py','--name',f'continuation_2000/evaluation_logs/{name}',
                 '--timeout','360','--']+args
        result=subprocess.run(command,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        suite['entries'].append({'name':name,'checkpoint_iteration':1999,'optimizer_updates':2000,
                                 'split':split,'clip_index':index,'command':args,'exit_code':result.returncode})
        out.write_text(json.dumps(suite,indent=2)+'\n')
        print(name,result.returncode,flush=True)
        if result.returncode:
            suite['status']='FAILED';out.write_text(json.dumps(suite,indent=2)+'\n')
            raise RuntimeError(result.stdout[-3000:])
suite['status']='COMPLETED';suite['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
out.write_text(json.dumps(suite,indent=2)+'\n')

"""Bounded argv-based witness dispatch; no nested Python/shell source strings."""
from pathlib import Path
import json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
MON=ROOT/'artifacts/monitoring/r3_feedback_paused_witness_20261007'
OUT=ROOT/'artifacts/feedback_paused_witness_20261007'
SESSION='r3_feedback_paused_witness_20261007'
if sys.argv[1:]==['--child']:
 rc=255;MON.parent.mkdir(parents=True,exist_ok=True)
 try:
  with MON.with_suffix('.log').open('x') as f:
   rc=subprocess.call(['bash',str(ROOT/'scripts/r3_feedback_paused_stage_20261007.sh'),'witness','42',str(OUT)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 finally:MON.with_suffix('.exit.json').write_text(json.dumps({'exit_code':rc,'updated_unix':time.time()})+'\n')
 sys.exit(rc)
else:
 assert not OUT.exists() and not MON.with_suffix('.log').exists() and not MON.with_suffix('.exit.json').exists(),'Do not overwrite or blindly retry witness'
 p=subprocess.run(['tmux','has-session','-t',SESSION],capture_output=True)
 assert p.returncode!=0,'Witness session already active'
 free=int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=14000
 command='/usr/bin/python3 '+str(Path(__file__).resolve())+' --child'
 subprocess.run(['tmux','new-session','-d','-s',SESSION,command],check=True)
 print(json.dumps({'dispatched':True,'session':SESSION,'command':command,'free_GPU1_MiB':free}))

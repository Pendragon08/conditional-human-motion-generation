"""Sync only new task code; diagnostics remain separate from full training."""
import pathlib,subprocess,sys
from session import REPO,remote
BASE=REPO/'baselines/textop';ROOT='/data/code/ouyanghao/oyh/textop-reproduction-20261024'
files=['scripts/export_tracker_references.py','scripts/tracker_ppo_resume_witness.py',
       'scripts/supervise_tracker_job.py','configs/tracker-ppo-diagnostic.json']
for rel in files:
    compile((BASE/rel).read_text(encoding='utf-8'),rel,'exec') if rel.endswith('.py') else None
    subprocess.run(['scp','-q','-o','BatchMode=yes','-i','C:/Users/Pende/.ssh/id_ed25519',str(BASE/rel),'ouyanghao@10.147.17.32:'+ROOT+'/'+rel],check=True)
print(remote("""import subprocess,pathlib,json
r=pathlib.Path('"""+ROOT+"""');log=r/'logs/tracker_supervisor_export_author_v1.log'
with log.open('xb') as f:
 p=subprocess.Popen(['python3',str(r/'scripts/supervise_tracker_job.py'),'--job','export_witness','--run','export_author_v1'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'supervisor_pid':p.pid,'job':'export_witness','scope':'two complete reference motions plus mirrors, no dynamics or training'}))
""",timeout=30))

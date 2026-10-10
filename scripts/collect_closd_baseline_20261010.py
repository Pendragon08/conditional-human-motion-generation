"""Bounded, one-job result transfer. No training launch, scheduling or messages."""
import datetime
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/closd_baseline_20261010'
REMOTE='/data/code/ouyanghao/oyh/conditional-human-motion-generation/results/closd_baseline_20261010'
HOST='ouyanghao@10.147.17.32'
OPTS=['-i',str(Path.home()/'.ssh/id_ed25519'),'-o','BatchMode=yes','-o','ConnectTimeout=15',
      '-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2']
OUT.mkdir(parents=True,exist_ok=True)
def cmd(argv):
    return subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                          text=True,encoding='utf-8',errors='replace',timeout=90)
def fetch(rel):
    dest=OUT/rel
    dest.parent.mkdir(parents=True,exist_ok=True)
    partial=dest.with_name(dest.name+'.download')
    result=cmd(['scp',*OPTS,HOST+':'+REMOTE+'/'+rel,str(partial)])
    if result.returncode==0 and partial.exists() and partial.stat().st_size>0:
        partial.replace(dest)
        return True
    return False
deadline=time.monotonic()+8*3600
while time.monotonic()<deadline:
    try:
        status=cmd(['ssh',*OPTS,HOST,'cat '+REMOTE+'/pipeline.state; cat '+REMOTE+'/finalize.state 2>/dev/null'])
        remote_state=status.stdout.strip()
        for rel in ['pipeline.state','baseline_report.html','summary.json','provenance.sha256','released_evaluation/eval.log']:
            fetch(rel)
        # A receipt exists only after the official collector completed normally.
        receipt_status=cmd(['ssh',*OPTS,HOST,'test -s '+REMOTE+'/fresh_generation/receipt.json'])
        if receipt_status.returncode==0:
            for rel in ['fresh_generation/receipt.json','fresh_generation/config.yaml','fresh_generation/eval.log']:
                fetch(rel)
        completed='COMPLETE_REPORT_AND_PREVIEWS' in remote_state
        failed='FAILED' in remote_state
        if completed:
            result=cmd(['scp',*OPTS,'-r',HOST+':'+REMOTE+'/previews',str(OUT)])
            if result.returncode!=0:
                raise RuntimeError('Preview transfer failed; will retry without restarting any experiment.')
        local={'updated_at':datetime.datetime.now().astimezone().isoformat(),
               'remote_state':remote_state,'collector':'single experiment, bounded 8h; no auto-relaunch',
               'complete':completed,'failed':failed,'raw_licensed_motions_downloaded':False}
        (OUT/'local_sync.json').write_text(json.dumps(local,indent=2),encoding='utf-8')
        print(json.dumps(local),flush=True)
        if completed or failed:
            break
    except Exception as error:
        print(datetime.datetime.now().isoformat(),repr(error),flush=True)
    time.sleep(60)
else:
    (OUT/'local_sync.json').write_text(json.dumps({'complete':False,'status':'transfer deadline; server process untouched'}),encoding='utf-8')

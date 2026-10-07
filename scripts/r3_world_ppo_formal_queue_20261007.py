"""Only the preregistered five arms; never launch explanation arms by stealth."""
from pathlib import Path
import json,subprocess,time,hashlib
ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'artifacts/monitoring/r3_world_ppo_formal_20261007.state'
BASE=ROOT/'artifacts/world_residual_ppo_20261007/formal'
def atomic(data):
 t=STATE.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(STATE)
def main():
 assert not STATE.exists() and not BASE.exists()
 BASE.mkdir(parents=True);completed=[]
 for seed in [42,43,44]:
  for arm in ['P0','P-cap','P-history','W-correct','W-shuffled-original']:
   for stage in ['train','eval']:
    atomic({'status':'RUNNING','stage':stage,'arm':arm,'seed':seed,'completed':completed,'updated_unix':time.time()})
    rc=subprocess.run(['bash',str(ROOT/'scripts/r3_world_ppo_formal_stage_v3_20261007.sh'),stage,arm,str(seed)]).returncode
    if rc:
     atomic({'status':'FAILED','stage':stage,'arm':arm,'seed':seed,'wrapper_exit':rc,'completed':completed,'preserve':True});return
    out=BASE/f'{stage}_{arm}_seed{seed}_v1'
    summary=json.loads((out/'summary.json').read_text());assert summary['status']=='COMPLETE'
    if stage=='train':assert summary['formal_PPO_updates']==500 and summary['actual_transitions']==98304000
    else:assert summary['PPO_updates']==0 and summary['actual_evaluation_transitions']==1024000
    completed.append({'stage':stage,'arm':arm,'seed':seed,'output':str(out),
     'summary_sha256':hashlib.sha256((out/'summary.json').read_bytes()).hexdigest()})
 rc=subprocess.run([str(ROOT/'third_party/isaacsim-4.5.0/python.sh'),str(ROOT/'scripts/r3_world_ppo_gate_20261007.py'),
  '--root',str(BASE),'--output',str(BASE/'control_gate.json')]).returncode
 if rc:atomic({'status':'FAILED','stage':'gate_integrity','wrapper_exit':rc,'completed':completed});return
 gate=json.loads((BASE/'control_gate.json').read_text())
 atomic({'status':'COMPLETE','completed':completed,'control_gate':gate,'explanation_queue_started':False,
  'sim_to_sim_started':False,'all_source_weights_and_physics_outputs_preserved':True})
 print('WORLD_PPO_FORMAL_QUEUE_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as error:
  atomic({'status':'FAILED','stage':'queue_integrity','exception':repr(error),'preserve':True})
  raise

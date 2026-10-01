"""Archive completed official-checkpoint evaluations; never launches simulation."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
MON = ROOT / 'artifacts/monitoring'
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1<<20), b''): digest.update(chunk)
    return digest.hexdigest()

records=[]
for model in ('CLoSD_multitask_finetune','CLoSD_no_finetune'):
    prefix=f'r3_closd_{model}_eval4096_20261001'
    state=(MON/(prefix+'.state')).read_text()
    assert state.startswith('COMPLETE ') and 'exit=0' in state, state
    log=MON/(prefix+'.log'); text=log.read_text(errors='replace')
    assert 'Collected enough data (1000 per task) success rates:' in text
    assert 'Eval plot ended' in text and 'CLOSD_NATIVE_CLEANUP_COMPLETE' in text
    rates={name:float(value) for name,value in re.findall(r'^STATES\.(\w+): ([\d.]+)$', text, re.M)}
    assert set(rates)=={'REACH','STRIKE_KICK','STRIKE_PUNCH','SIT','GET_UP'}, rates
    modelpath=ROOT/'third_party/CLoSD/output/CLoSD'/model/'Humanoid.pth'
    record={'status':'official_released_checkpoint_evaluation_complete',
        'checkpoint_source':'official_released_weights_not_our_retraining','model':model,
        'checkpoint_sha256':sha(modelpath),'log_sha256':sha(log), 'exit_code':0,
        'num_envs':4096,'episode_length':500,'cfg':7.5,'seed':0,
        'minimum_resets_per_state':1000,'success_rate_rounded_2dp':rates,
        'strike_kick_punch_pooled_upstream':True,'native_cleanup_completed':True,
        'formal3072_retraining_complete':False,'evaluation_video_verified':False,
        'video_note':'No evaluation video was recorded; live-scene telemetry is not a saved result video.'}
    records.append(record)
    target=ROOT/'results'/f'r3_closd_{model}_official_eval4096_20261001.json'
    target.write_text(json.dumps(record,indent=2)+'\n')
(ROOT/'results/r3_closd_official_eval4096_20261001_summary.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records))

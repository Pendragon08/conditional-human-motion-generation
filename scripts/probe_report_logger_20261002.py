from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import json
R=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation')
O=Path('/data/code/ouyanghao/oyh/humanoid-combat-motion')
runs=[('Kick',O/'third_party/DreamControl/Training/logs/rsl_rl/g1/2026-09-27_03-19-31-Kick-a'),
('Punch_initial',R/'third_party/DreamControl/Training/logs/rsl_rl/g1/2026-09-27_07-24-06-Punch-a'),
('Punch_resume',R/'third_party/DreamControl/Training/logs/rsl_rl/g1/2026-09-29_12-37-46-Punch-a'),
('Jump',R/'third_party/DreamControl/Training/logs/rsl_rl/g1/2026-10-01_14-01-43-Jump-a'),
('Sit',R/'third_party/DreamControl/Training/logs/rsl_rl/g1/2026-10-01_16-54-27-Sit-a'),
('Pick',R/'third_party/DreamControl/Training/logs/rsl_rl/g1/2026-10-01_19-59-36-Pick-a'),
('Bimanual-Pick',R/'third_party/DreamControl/Training/logs/rsl_rl/g1/2026-10-02_00-21-24-Bimanual-Pick-a')]
for name,p in runs:
    ea=EventAccumulator(str(p),size_guidance={'scalars':0});ea.Reload()
    rows=[]
    for t in ea.Tags()['scalars']:
        a=ea.Scalars(t)
        rows.append({'tag':t,'n':len(a),'first':[a[0].step,a[0].value],'last':[a[-1].step,a[-1].value]})
    print(json.dumps({'task':name,'events':[{'file':q.name,'bytes':q.stat().st_size} for q in p.glob('events*')],'scalars':rows}),flush=True)

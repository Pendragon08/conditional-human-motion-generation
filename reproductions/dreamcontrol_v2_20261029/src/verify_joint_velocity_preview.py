"""Independent scalar check of a nominal velocity preflight result."""
import hashlib
import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

cfg=json.loads(Path(sys.argv[1]).read_text())
observed=json.loads(Path(sys.argv[2]).read_text())
states=json.loads(Path(cfg['state_manifest']).read_text())['records']
urdf=Path(cfg['state_urdf'])
assert hashlib.sha256(urdf.read_bytes()).hexdigest()==cfg['state_urdf_sha256']
limits={j.get('name'):float(j.find('limit').get('velocity'))
        for j in ET.parse(urdf).getroot().findall('joint') if j.get('type')=='revolute'}
reported={r['sample_id']:r['declared_velocity_diagnostic'] for r in observed['state_audit']['cases']}
assert len(reported)==len(states)
out=[]
for s in states:
    p=Path(cfg['state_root'])/s['state_path']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==s['state_sha256']
    with np.load(p,allow_pickle=False) as f:
        q=f['joint_pos'].astype(float); times=f['sample_times'].astype(float)
        names=[str(x) for x in f['joint_names']]
    maxratio=-1.; count=0; frames=0; worst=None
    for i in range(len(times)-1):
        flagged=False
        for j,name in enumerate(names):
            ratio=abs(float(q[i+1,j])-float(q[i,j]))/float(times[i+1]-times[i])/limits[name]
            if ratio>maxratio:maxratio=ratio;worst=(i,name)
            if ratio>1:count+=1;flagged=True
        frames+=flagged
    result=reported[s['sample_id']]
    assert abs(maxratio-result['maximum_ratio'])<1e-12
    assert count==result['joint_interval_pairs_above_limit']
    assert frames==result['intervals_with_any_joint_above_limit']
    assert worst==(result['worst_interval'],result['worst_joint'])
    out.append({'sample_id':s['sample_id'],'max_ratio':maxratio,'above_limit_pairs':count})
print(json.dumps({'cases':out,'training_ready':False,
                  'scope':'same nominal times, robot states and URDF; scalar recomputation only'},indent=2))

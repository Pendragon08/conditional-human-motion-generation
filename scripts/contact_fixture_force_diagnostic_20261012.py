"""Locate net-force export disagreements without changing acceptance criteria."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
result={}
for name in ['contact_force_fixture_20261011','contact_force_fixture_v2_20261011']:
    root=ROOT/'results'/name
    arrays=[]
    for label in ['repeat_a','repeat_b']:
        with np.load(root/label/'fixture_arrays.npz',allow_pickle=False) as z:
            arrays.append({k:z[k].copy() for k in z.files})
    per=[]
    for i in range(4):
        delta=np.abs(arrays[0]['net_contact_force'][:,i]-arrays[1]['net_contact_force'][:,i])
        index=int(np.unravel_index(np.argmax(delta),delta.shape)[0])
        per.append(dict(fixture=i,maximum_difference_N=float(delta.max()),max_frame=index,
            net_at_max=[a['net_contact_force'][index,i].tolist() for a in arrays],
            sensor_at_max=[a['constraint_force_sensor'][index,i].tolist() for a in arrays],
            state_at_max=[a['state'][index,i].tolist() for a in arrays],
            net_sensor_error_max_N=[float(np.max(np.abs(a['net_contact_force'][:,i]-a['constraint_force_sensor'][:,i,:3]))) for a in arrays],
            net_sensor_error_p95_N=[float(np.quantile(np.linalg.norm(a['net_contact_force'][:,i]-a['constraint_force_sensor'][:,i,:3],axis=1),.95)) for a in arrays]))
    result[name]=per
out=ROOT/'results/contact_force_fixture_v2_20261011/force_disagreement_diagnostic.json'
out.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

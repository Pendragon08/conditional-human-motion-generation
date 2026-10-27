from pathlib import Path
import subprocess,json,numpy as np
r=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017')
out=r/'results/completion_v1/replay_determinism_recovery'
old=json.loads((r/'results/completion_v1/closure_final/output/simulator/invocation_001/request.json').read_text())
for i in range(2):
 d=out/str(i);d.mkdir();req=dict(old,output=str(d));p=d/'request.json';p.write_text(json.dumps(req))
 with (d/'worker.log').open('w') as f:
  rc=subprocess.run(['bash','scripts/run_isaac.sh','scripts/replay_physics_worker.py','--request',str(p),'--headless','--device','cuda:0'],cwd=r,stdout=f,stderr=subprocess.STDOUT).returncode
 if rc or not (d/'worker_result.json').exists():raise RuntimeError((i,rc))
with np.load(out/'0/raw.npz') as a,np.load(out/'1/raw.npz') as b:
 diff={k:float(np.max(np.abs(a[k].astype(float)-b[k].astype(float)))) for k in a.files}
with np.load(out/'0/initial_physical_properties.npz') as a,np.load(out/'1/initial_physical_properties.npz') as b:
 physical={k:float(np.max(np.abs(a[k]-b[k]))) for k in a.files}
res={'state_deltas':diff,'physical_property_deltas':physical}
(out/'comparison.json').write_text(json.dumps(res,indent=2));print(json.dumps(res),flush=True)
assert max(diff.values())<5e-4

from pathlib import Path
import json
import numpy as np
p=Path(__file__).resolve().parent/'raw_result'
r=json.loads((p/'output/verification.json').read_text())
print(json.dumps({k:v for k,v in r.items() if k not in ('attempts','proposals','source_sha256','raw_files','simulator_invocations')},indent=2))
for n in ['output/simulator/invocation_001/raw.npz','output/simulator/invocation_001/continuation.npz','output/simulator/invocation_001/initial_physical_properties.npz','output/nominal_standing_target.npz']:
 with np.load(p/n,allow_pickle=False) as d:print(n,{k:(str(d[k].dtype),d[k].shape) for k in d.files})
for n in sorted((p/'output/proposals').glob('*.npz')):
 with np.load(n,allow_pickle=False) as d:print(n.name,{k:(str(d[k].dtype),d[k].shape) for k in d.files})
print('WORKER_RESULTS', [json.loads(q.read_text()) for q in sorted((p/'output/simulator').glob('*/worker_result.json'))])

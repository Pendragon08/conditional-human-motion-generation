import pathlib,sys,json,hashlib
import numpy as np
W=pathlib.Path(__file__).resolve().parents[2];B=W/'work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop'
sys.path.insert(0,str(B/'scripts'));from direct_strict_trace import ExecutionTrace,FIELDS
from strict_tracking_coverage import StrictCoverage
out=B/'results/teacher15000_direct_strict_20261027/compact_trace_cpu_v1';out.mkdir(exist_ok=False)
def state(frame):
 d={k:np.zeros(shape,dtype='<f4') for k,shape in FIELDS};d['root_quat_wxyz'][0]=1;d['reference_frame'][0]=frame;return d
records=[]
for name,fail in [('complete',False),('failure',True)]:
 t=ExecutionTrace(out/name,hashlib.sha256(name.encode()).hexdigest(),4,['j'+str(i) for i in range(29)],runtime_kind='synthetic_cpu_fixture',budget={'used':0,'maximum':4096})
 t.append(state(0));c=StrictCoverage(4)
 for i in range(3):
  t.append(state(i));c.observe(terminated=fail,motion_end=(i==2))
  if c.end_reason!='running':break
 result=t.finish(c.record());arr=np.fromfile(t.path,dtype='<f4').reshape(-1,36)
 assert len(arr)==c.steps+1 and np.isfinite(arr).all() and np.array_equal(arr[0,3:7],np.array([1,0,0,0]))
 assert result['runtime_kind']=='synthetic_cpu_fixture';records.append(result)
t=ExecutionTrace(out/'invalid',hashlib.sha256(b'invalid').hexdigest(),4,['j'+str(i) for i in range(29)],runtime_kind='synthetic_cpu_fixture',budget={'used':0,'maximum':144})
bad=state(0);bad['joint_pos'][0]=np.nan
try:t.append(bad)
except AssertionError:pass
else:raise AssertionError('NaN accepted')
assert t.samples==0 and t.path.stat().st_size==0;t.append(state(0))
try:t.append(state(0))
except AssertionError:pass
else:raise AssertionError('Budget overrun accepted')
t.close_incomplete();assert json.loads((t.folder/'trace.private.json').read_text())['status']=='INCOMPLETE'
proof=dict(status='NEW_COMPACT_TRACE_CPU_FIXTURE_PASSED',physical_execution=False,complete_and_failure_checked=True,nan_rejected=True,budget_enforced=True,incomplete_preserved=True,trace_source_sha256=hashlib.sha256((B/'scripts/direct_strict_trace.py').read_bytes()).hexdigest())
(out/'proof.json').write_text(json.dumps(proof,indent=2),encoding='utf-8');print(json.dumps(proof))

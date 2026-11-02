# Independent terminal metrics from retained5k recordings

This fills a missing **analysis** comparison before10k evaluation. It reuses the
closed real `quick_pilot_sample64_v1` recordings, not simulation or new tracking.
Do not alter its original receipt's `final_reference_endpoint_comparison_performed=false`:
that remains accurate for the original execution. New post-hoc evidence is separate.

Warm runtime: existing server `ROOT/env/bin/python` with NumPy, no Torch import,
CUDA hidden, CPU0–3, one BLAS thread, no dependencies/shared files changed.
ROOT is `/data/code/ouyanghao/oyh/textop-reproduction-20261024`.
Current trainer and teacher10000 coordinator remain active and byte-frozen.

Spec `configs/recorded-endpoint-spec.json` SHA
`6db8a9171e8ed95cba7db0a31a3ad73cfe0be32f8fca0eff51d84596796d400b`
binds original receipt/exit/selection/source code. New auditor verifies original
strict decision journal,64source identities, actual process/resource receipts,
all trace binary/NPZ digests and array equality, exact joint names and full-frame
reference hashes before metrics. All42 natural completions compare recorded
state at elapsed(T−1)/50 to independent referenceT−1. Controller originally
consumedT−2. The22 earlier failures have no terminal metric and are not extrapolated.
No extra control call, dynamics, optimizer, checkpoints or dataset acceptance.

Metrics use reference world coordinates after scene-origin subtraction, without
per-frame yaw/root alignment or initial-offset subtraction. Initial root offsets
were all exactly0 in the retained recordings. New mean endpoint root norm~.1608m
andjointRMSE~.1751rad are own diagnostics; not the old aligned current-target
metrics, paper benchmark scores, semantic acceptance or new rejection thresholds.
Every failure remains. Natural completion alone is not trajectory quality.

## Fresh document invocation (read-only recomputation)

Localcwd `C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成`.
LocalPython `C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`.
Set PYTHONIOENCODING=utf-8 and run this snippet verbatim once:

```python
import pathlib,sys,json
b=pathlib.Path('work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop')
sys.path.insert(0,str(b/'provider_tools'))
from session import remote
code="""import pathlib,subprocess,os,json,hashlib
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
sp=r/'configs/recorded-endpoint-spec.json'
assert hashlib.sha256(sp.read_bytes()).hexdigest()=='6db8a9171e8ed95cba7db0a31a3ad73cfe0be32f8fca0eff51d84596796d400b'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
v=subprocess.run([str(r/'env/bin/python'),str(r/'scripts/audit_recorded_endpoints.py'),'--spec',str(sp),'--run','baseline5k_author_v1','--verify-output'],capture_output=True,text=True,env=env,timeout=180)
assert v.returncode==0,(v.stdout,v.stderr)
j=json.loads(v.stdout);assert j['status']=='EXISTING_ENDPOINT_AUDIT_INDEPENDENTLY_RECOMPUTED' and j['matched_trials']==64
print(json.dumps({'actual_exit':v.returncode,'summary':j,'stderr':v.stderr,'new_sdk_runs':0}))
"""
print(remote(code,timeout=210))
```

`--verify-output` recomputes from retained inputs and compares exact saved private
bytes/summary; it writes no server output. Read/review code and document definitions
for frame indexing, joint order, failed-trial treatment, signal-free ownership,
and absence of hidden runtime edits. Do not rerun original pilots or7CPU fixtures.
Save only small proof/REPORT/SUMMARY in fresh local
`work/textop_reproduction_20261024/recorded_endpoint_document_v1`.
No raw arrays or per-source identities may be copied to public repository.

Authoractualexit0 and7local/7server synthetic unit checks are retained in
`results/recorded_endpoints_20261027`. Private metric table remains server plus
D private backup. No10k evaluator has been launched. Its later paired comparison
must retain the64 identical sources, original strict outcomes, final endpoints
and failures as distinct observations rather than replacing the nominal criterion.

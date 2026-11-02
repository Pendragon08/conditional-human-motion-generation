# Paired 5k/10k comparison of the fixed 64 motions

This is a CPU analysis of two closed real recordings, not another SDK run.
The 10k checkpoint is `6f949e4edc50d9a763a3634d716162cb5ddecb079bd1142d3720f907f5efab45`.
Original 5k outcomes and endpoint audit remain unchanged. The 10k replay uses
the same 64 identities, reference files, lengths, seed, ground and thresholds.

Warm runtime: server `ROOT/env/bin/python`, NumPy, no Torch import; CUDA hidden,
one BLAS thread, CPU0-3. ROOT is
`/data/code/ouyanghao/oyh/textop-reproduction-20261024`.
No installation, model loading, training, signals or shared-file changes.

The comparison binds actual checkpoint/handoff, source/spec SHA, private
selection, original references, process/resource/outer exits, trial journal,
binary/NPZ array equality and closed writers before calculating outcomes.
`paired64_v1` is the unique author output. It must already exist before the
independent command below. Do not create a second output or replay the SDK.

Terminal errors use the last recorded state at `(T-1)/50` against independent
reference `T-1`, in world coordinates after scene-origin subtraction; no
per-frame root/yaw alignment. The controller consumed reference `T-2` last.
Early failures have no terminal extrapolation. Paired endpoint changes use only
motions naturally completed by both checkpoints. All-new-completion statistics
have a separate denominator. These are custom diagnostics, not paper scores,
semantic acceptance, full-library filtering or new rejection thresholds.

## Fresh document invocation

Local cwd: `C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成`.
Use existing Python
`C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`
with `PYTHONIOENCODING=utf-8`. Execute this snippet verbatim once:

```python
import pathlib,sys,json
b=pathlib.Path('work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop')
sys.path.insert(0,str(b/'provider_tools'))
from session import remote
code="""import pathlib,subprocess,os,json,hashlib
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
sp=r/'configs/teacher10000-comparison-spec.json';src=r/'scripts/compare_teacher10000_pilot.py'
assert hashlib.sha256(sp.read_bytes()).hexdigest()=='10bf2854df741534a1c31b63fb54f33068b6e14dff00bdbd513b20bba694024d'
assert hashlib.sha256(src.read_bytes()).hexdigest()=='a7123517d1e343df2be1669a8faa9ba52bbf6b67214eeaeabeaafd7a54b72154'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
v=subprocess.run([str(r/'env/bin/python'),str(src),'--run','paired64_v1','--verify-output'],capture_output=True,text=True,env=env,timeout=180)
assert v.returncode==0,(v.stdout,v.stderr)
j=json.loads(v.stdout);assert j['status']=='PAIRED64_INDEPENDENTLY_RECOMPUTED' and j['matched_sources']==64
print(json.dumps({'actual_exit':v.returncode,'summary':j,'stderr':v.stderr,'new_sdk_runs':0}))
"""
print(remote(code,timeout=210))
```

Review source for strict same-source pairing, failed-trial handling, denominators,
frame indexing and absence of changes to original runs. Verify against the
author aggregate summary; do not publish private source identities or arrays.
Write small REPORT.md and SUMMARY.json under a fresh local document directory.
Keep original errors if the verifier or invocation fails; do not hide a failure
by rerunning the physical evaluation. This analysis cannot decide that an early
teacher failure means the motion is intrinsically infeasible.

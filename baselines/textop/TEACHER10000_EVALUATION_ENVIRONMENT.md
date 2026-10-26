# User-requested10k evaluation checkpoint

The user explicitly requests assessing filtering at10000 rather than waiting for
40000, improving semantics, and notification at actual paper-level completion.
The original40000 production spec/trainer are frozen. This separate coordinator
waits for the actual10000 snapshot and server+D private full-state/GitHub neural
inference backup, then pauses only the original SDK process using its pidfd and
original start ticks. No data/checkpoint is deleted. All original exits remain;
the unchanged40000 supervisor may return nonzero after authorized interruption.
This is not a numerical failure or40000 completion and must not trigger auto-restart.

Warm environment: SERVER_NOTES.md / FULL_TEACHER_RESUME_ENVIRONMENT.md.
Server/root `/data/code/ouyanghao/oyh/textop-reproduction-20261024`,
SSH `ouyanghao@10.147.17.32`, UID1013. SystemPython3.8 stdlib control only.
CPU serialization reuses unchanged `inspect_formal_filter_checkpoint_v2.py`
under existing `env/bin/python` with CUDA hidden, one numeric thread; CPU0–3.
No new environment/package, SDK or neural validation is needed for the control.
GPU remains shared and used by the existing8192-env trainer. Do not apply the
skill's empty-GPU default against the user's explicit shared authorization.

New backup adapter binds the resumed run and original latest training receipt,
model/source/spec/reference digests and actual exits for final40000. It accepts
only10000,15000,...40000. Never repurpose the old hardcoded5k backup tool.
Positive10k file load/export/transfer remains untested until the real snapshot.
Current32 synthetic metadata checks and live pending preflight do not substitute
for that future audit. Missing/newer/wrong/unstable snapshots cannot be accepted.
Public inference excludes Adam and per-motion statistics; full states remain private.

## Fresh independent document invocation

Local working directory is `C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成`.
Use bundledPython:
`C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`.
Set PYTHONIOENCODING=utf-8. Run this Python snippet verbatim with that interpreter:

```python
import sys,pathlib,json
b=pathlib.Path('work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop')
sys.path.insert(0,str(b/'provider_tools'))
from session import remote
code="""import pathlib,subprocess,sys,json,hashlib
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
s=json.loads((r/'configs/teacher10000-evaluation-spec.json').read_text())
assert s['handoff_enabled'] is False
for n,h in s['bound_control_sources'].items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h
v=subprocess.run([sys.executable,str(r/'scripts/teacher_evaluation_control.py'),'--action','observe'],capture_output=True,text=True,timeout=45)
assert v.returncode==0,(v.stdout,v.stderr)
j=json.loads(v.stdout);assert j['status']=='WAITING_VERIFIED_10000_CHECKPOINT'
assert j['observed_completed']<10000 and j['signals_sent']==[] and j['sdk_runs']==0 and not j['target_checkpoint_ready']
assert not (r/'receipts/teacher10000_stop_requested.json').exists()
print(json.dumps({'actual_exit':v.returncode,'observation':j,'stderr':v.stderr,'no_training_signal':True}))
"""
print(remote(code,timeout=65))
```

Expected: actualexit0, WAITING_VERIFIED_10000_CHECKPOINT, identity of the resumed
trainer, no signal/SDK/export/checkpoint load. Inspect candidate control/protocol,
backup adapter and coordinator for backup-before-stop, latest-only provenance,
pidfd-original identity, pending-no-resignal, actual exits/common-lock checks and
preservation of40000 baseline. Report divergences, do not fix or enable anything.
Do not repeat the32 fixtures, old pidfd fixtures, load5k or launch a new SDK.
Write REPORT.md/SUMMARY.json and small evidence under new local task folder
`work/textop_reproduction_20261024/teacher10000_document_v1`; no raw datasets.

After clean independent pass parent enables only handoff_enabled in this new
spec, records old/new exact bytes and starts one hidden local coordinator:
`python provider_tools/teacher10000_coordinator.py --run coordinator_v1`.
Check actual local processes before launch. Never restart an existing directory.
On network/backup failure preserve stage, inspect child/receipt, no blind retry.
Pending signals prohibit another signal; finish only after original real exits.

## After handoff

First freeze separate10k evaluation schema/spec and validate its actual executor
with author/fresh runs. Do not spoof5k or40k counts in previous gates. Compare
the exact64 original references/thresholds/durations against the retained5k
42success/22failure baseline, including full endpoint error checks. Then expand
fixed coverage without selecting on outcomes. Keep all failures/unattempted and
source isolation. Decide further teacher training from evidence; 40000 remains
the paper reference budget, not a prerequisite for this diagnostic evaluation.
Evaluation launcher remains disabled/unimplemented here; no claim of physical
filter completion or semantic improvement is implied by installing the control.

Semantic work uses read-only coverage evidence plus controlled same-history,
same-noise different-text/unconditional comparisons and held-out motion/text
families after SDK lock availability. Keep CFG5/paper baseline and all negative
results; any sampler, dataset or loss variant must be explicit and tested.
Paper complete notification requires complete required data screening, full
generator/requiredWBC/augmentation and online execution/video/quality evidence.

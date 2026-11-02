# Bounded joint online prototype, declared before dispatch

## Current candidate and retained failures

The original GPU-physics online_author_v1 actually closed257steps, hot switch consumed,
no physical failure/underrun, but13.26s wall time and257 missed20ms deadlines. v2
performed16warmupcontrols then failed resetting an inference tensor outside inference
mode. v3 fixed only that context and added host-side guarded indices/no duplicate
observation compute: closed257steps,10.28s, still257misses. v4 measured component
times without changing physics/math. v5 TorchCPU threads4→1 barely changed timing.
All originals and exact source/spec versions are retained, never rewritten as passes.

v6 is an explicitly different **single-robot CPU PhysX / GPU1 neural** execution
profile. Same URDF/material/selfcollision/5msdt/4substeps/teacher/strict thresholds;
full-scale8192-env training remains GPU and unchanged. It closed257steps in5.83s,
0physicalfailures/underruns, but256misses. This is not bit-exact GPU physics and must
be identified as a runtime/backend difference in any claim.

v7 skipped training-only reward evaluation during timed inference and reused equal
557D policy/critic inputs after an actual torch.equal assertion. Zero returned
rewards were marked API placeholders, never measured scores. Model/thresholds and
training reward configuration unchanged. It failed on a pinned-API mismatch:
IsaacLab2.1 compute_group does not accept update_history. v8 corrects only that call
and records/captures partial physics steps on exceptions. v7 has16warmupcontrols
plus one incomplete timed environment step; its failed receipt was not amended.

v8 closed257 controls but202 deadline misses. v9 async private trace writer and passive
thread waiting closed257 with208 misses. v10 partitioned the SAME four cores:
generator CPU0/Torch1, SDK CPU1-3/Torch1, and reduced misses to8/max10.449ms.
v11 moves verified response SHA/NPZ reading to a bounded background CPU0 thread;
main thread still checks immutable prefix and installs only real produced frames.
The trace writer also uses CPU0, FIFO16, errors/overflow fail, drains before closing.
It additionally warms exact neural/finite-check kernels without physics/reference
advance and sets the clock only after journal/startup receipt preparation.
No thresholds, physics, network, weights, control period or deadline tolerance changed.

Current author online_author_v11 has closed257 controls,0 physical failure/underrun,
0 missed20ms deadlines and control-loop5.134738s for nominal5.14s. All child/outer exits0 were verified. Fresh online_document_v1 also actually
closed257 with0miss/underrun/physicalfailure, control-loop5.131990s and hot-command
latency0.496600s; independent63+430+41 checks passed. Both are finished: never rerun.
The documented fresh invocation below has completed exactly once; it is now a historical reproduction record, not a command to repeat.
There are16explicit cold-start warmupcontrols/64substeps followed by reset, counted
separately from257timed controls/1028substeps. No warmup samples are timing successes.
Runtime4sources/spec are frozen while each worker is alive. Source changes only after
real child/outer exits and lock release, with original byte snapshot preserved.

This profile is separate from immutable offline generation/SDK/video profiles.
Warm generator Python3.8/Torch1.13 and tracker Python3.10/Torch2.5 use the original
trained VAE2000/LDM3000 and teacher5000. No installation, shared dependency changes,
optimization, GT future, posterior conditioning or execution-pose conditioning.
Run-experiment and SERVER_NOTES.md govern warm reuse. CPU URDF-FK was checked against
all258 existing real SDK-exported poses by author and a fresh CPU document witness.

One exclusive supervisor holds full_library_capacity.lock for one deliberate
two-worker GPU1 experiment. Generator Torch cap0.15 + SDK cap0.50 =0.65. Both
combined CPU0–3: generator and SDK I/O threads CPU0, SDK main CPU1–3; only ONE SDK. The SDK's existing GPU0 auxiliary context is retained and
sampled. User authorizes shared GPUs: require actual GPU1 free20GiB, host RAM20GiB,
personal persistent data free3GiB; do not require an empty GPU. Original paused57D
and every other old job remain untouched. Parent preflights worker identities,
common lock and immutable source/weight/URDF/profile bytes before dispatch.

Protocol: generator starts with two neutral reference frames, publishes four real
8-frame primitives. SDK initializes from that real34-frame prefix. Remaining
slots are NaN and every reference-property access asserts actual available prefix;
the provisional last-frame velocity is excluded. Paper teacher reads current and
next4frames; we also reserve the following actual pose for centered derivatives.
No unproduced placeholders, implicit last-frame filling or ignored underruns.
Generated own-history2/future8, 5step CFG5, original trained weights/math remain.
Exact tracker URDF analytical FK runs on CPU; no ghost/collision asset is spawned.

SDK requests one new primitive when lead drops below24frames. Generator runs
asynchronously, with atomically published request/response/NPZ and SHA. At actual
control step90 the controller submits a new self-written text command, "wave your
right hand", replacing "walk forward" for subsequent requests. CLIP encodes the
second text after reception. This is a scripted live text hot-switch test, not a
claim of an already delivered human-facing interactive UI. Prebuffer delay is
included in command-to-consumption latency. Existing bytes cannot be rewritten.

Bounded257actual control steps/1028controlled substeps, 50Hz wall-clock schedule.
Record each scheduled/actual start, physics finish, whole-tick finish, deadline,
buffer horizon, each submission/generator receipt/publication/SDK receipt/target
consumption. Deadline = scheduled start+20ms, **zero tolerance**. Any overrun is
reported; no deadline threshold relaxation or realtime acceptance just because
the mean is below20ms. Hard process bound900s, SDK600s. Preserve all failures.
Physics strict thresholds/solidfloor/actuators unchanged. Training reward configuration unchanged; timed inference omits reward evaluation and returned zeros are explicit API placeholders. Original simulator
reset and actuator behavior preserved. End is bounded-duration completion, never
reported as natural end of a whole known source motion. Pre-autoreset actual state
is recorded, and final generated-reference endpoint remains independently unchecked.

Author once, from /data/code/ouyanghao/oyh/textop-reproduction-20261024:
```text
python3 scripts/supervise_online_quick.py --run online_author_v11
```
Use durable subprocess wrapper capturing actual returncode, not SSH lifetime.
If failure occurs, preserve exact stage/source/spec/log before any new profile/run.
After author closes and genuine evidence is reviewed, fresh agent reads this doc,
local SERVER_NOTES and run-experiment skill, checks absent run/real resource/SHA, and
executes exactly once:
```text
python3 scripts/supervise_online_quick.py --run online_document_v1
```
For durable execution from Windows use this existing exact wrapper command once:
```powershell
& 'C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' 'work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop/provider_tools/launch_online_quick.py' --run online_document_v1
```
It dispatches the preceding remote command once and records actual outer returncode.
Current frozen spec SHA818d9e0dd951eaa5b2bb3f3e5d7858f2322cf7af740348f1d59190554fca4430.
Before launch verify author child/outer exits, old paused jobs, common lock, resources,
all bound actual server byte hashes. Local FK spec previously had equivalent CRLF;
use actual server LF hash rather than reserializing or changing the bound file.
Capture independent actual exit and private evidence, source before/after. No fixes,
retries, packages, unrequested GPU jobs, changed bounds or edits to old profiles.
Document any divergence. Verify both workers reaped, zero optimizer calls, real
post-start command and generation, own-history, finite FK/buffers, all consumed
lookahead produced before use, actual control feedback and deadline counts. Compare
same-seed outcomes honestly, not require identical thread scheduling. All private
arrays/logs/source mappings stay server+D; publish aggregate REPORT/SUMMARY only.

Even perfect timing/physical completion does not accept semantics, LDM sampling
quality, final deployable WBC, full8278 filtering, or full-paper reproduction. Fixed
LDM quality .310→.608 worsened and remains unresolved. Full training stays gated
on quality review, full available-data physics/corpus and persistent capacity.

## CURRENT2026-10-26 17:44 — actual short video passed; long CPU budget repaired

Actual5k handoff and actualauthor/freshsmoke/64pilot have completed. Rendering
is nowenabled, unchangedMesa/GMRdependencies/source/signature. Existing actual
shortmovie actual_smoke_author_trial0_v1 completed52frames50fps/realexit0 and was
visually inspected; it is the first actualrecorded-physics video, not synthetic.

Long actual_sample64_longest_completed_v1 hit its900s CPU-only supervisor bound:
childexit-15/timed_outtrue,62resource samples; partialMP4 retained server+D,
NOTa valid video. Whole64physical results remain42complete/22failed, all closed
physics records and originalassets remainintact. Its serialbatch stopped before
longest_failed_v1 was launched. Do not rerunv1 or present itspartialmedia.
A budget helper first failed beforemutation with missingimportsys; error saved.
Corrected helper confirmed900 unchanged, preservedfailure and changed ONLY
actual_timeout_seconds900→2400. Numerical render_signature remains
 dbb4301b20d56abf03619cad25854a8d53769eea980950fb39ecc0de0a9cabc5.
Allrender maths/width640/height480/fps50/allframes/nointerpolation/CPU0-3/Mesa/
CUDAempty/max2001frames/source hashes and original16/control remainunchanged.
This repairs a supervision allowance, not physics/generator quality. Existing
52frame render took36.76s, so1613frames requires roughly19minutes plusoverhead.

Fresh agent-follows-doc actualinvocation (CPUONLY; no GPUempty-card criterion)
may start only after checkingv1 reaped/no activevideo worker, realclosedpilot,
currentexactsource/spec/profile and persistentdisk/RAM. From serverproject
/data/code/ouyanghao/oyh/textop-reproduction-20261024, execute these commands
ONCE, SERIALLY, preserving realfinalexit/receipt/mediaSHA:

```text
python3 scripts/supervise_execution_video.py --mode actual --run actual_sample64_longest_completed_v2 --pilot-run quick_pilot_sample64_v1 --trial 31
python3 scripts/supervise_execution_video.py --mode actual --run actual_sample64_longest_failed_v2 --pilot-run quick_pilot_sample64_v1 --trial 44
```

Firstinput1613frames32.26s nominalcompletion, SFU. Second440frames8.8s failure,
BMLmovi. Selection rule: longest actualexecuted duration withinoutcome group,
earliestfinishindex breaks ties. Neither is a newrobottrial;0newSDK/physics/
optimizer. No package/sharedfile/source/config change is permitted by witness.
Use fresh local work/textop_reproduction_20261024/execution_video_actual_document_v3.
Check repeated runfolders/receipts and activeworkers before launch. If anyreal
assertion/bound/resource failure occurs, preserve rawoutputs and stop, no blind
retry/renaming/criterion relaxation. Waitactualexit; STARTING/partialMP4 is not
success. Keep originalfailure, root/source maps and poses private.

Afteractualcomplete, parentcollectors collect_quick_recorded_video.py --run
<existing_v2_run> verify proof/media and copyDprivate. Parentmust inspectframes.
SFU terms permit noncommercial research with attribution; parenthas read the
officiallicense and will use publish_quick_sfu_video.py --run
actual_sample64_longest_completed_v2 only AFTER mediaSHA/realexit/visualcheck,
code/reportpublication. No rawmotions/indices/models published. BMLmovi failure
video staysprivate due its more restrictive distribution wording; parentreport
links localvideo and publishes aggregatefailures. Do NOT runoldCPUrenderer
fixtures or syntheticstagecommands again. Fullgenerator/closedloop stillpending.

# CPU software video renderer — no SDK or new control simulation

This profile reuses the existing GMR Python and35 read-only Unitree visual mesh
assets. The declarative `configs/execution-video-spec.json` was written before
rendering. No packages, shared model, training files or live controller changed.
MuJoCo3.3.7 computes visual FK only; `mj_step` is prohibited. This is not a new
MuJoCo tracking experiment or a sim-to-sim control result.

GLVND is forced to its Mesa vendor JSON, CUDA visibility is empty, and llvmpipe
uses CPU0–3 only. First rendering checks actual GL vendor/renderer strings and
rejects any loaded libnvidia library. No Isaac SDK is imported; the current
teacher keeps its common SDK lock and continues to5000. The video worker owns
only its newly created process group and outputs. Its resource/exit records are
separate from all trainer records.

## Completed author execution; do not repeat

The first author attempt stopped BEFORE spawning a video worker: a string path
was passed to a Path-only checksum lambda in the new supervisor. The raw error,
old source bytes and exit1 are retained. Converting the helper argument with
pathlib.Path fixed only this new supervisor. Source-bound spec updated before
the next invocation; no shared or teacher file changed.

`renderer_author_v2` actually exited0 after24 frames were rendered, encoded and
decoded at50fps. GL was Mesa/X.org / llvmpipe(LLVM12.0.0), no libnvidia loaded;
original16 and live controller hashes matched. Frames are SYNTHETIC and carry
an explicit watermark; there is no actual experiment/policy video yet.
The old MP4 is462654B/SHA2a69103f2c0c201f7d49a25ad0e808d124a16a54371d7a50a732d62a0e8d92a2.

Fresh document_v1 rendered successfully but visual inspection found coplanar
floor z-fighting. That media is retained as a diagnostic, not an accepted visual
profile. An intermediate author_v3 correction inspected only the first worldbody
section; its parsed count1 was misleading and its MP4 remained byte-identical.
Read-only inspection then confirmed the asset has TWO worldbody sections, merged
by MuJoCo, with its existing floor in the second. Latest code handles all sections,
reuses the existing plane and asserts exactly one plane in the COMPILED model.
Old code/receipts and the inaccurate intermediate metadata remain with this
explicit correction. The actual-mode title now explicitly says RECORDED ISAACLAB.
Only the corrected author_v4 and fresh document_v2 may witness the latest profile.

## Fresh independent actual software-rendering witness

From the local task workspace run this once, verbatim, with the new names below:

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' 'work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop/provider_tools/witness_execution_video.py' --run renderer_document_v2 --output 'work/textop_reproduction_20261024/execution_video_document_v2'
```

It sends exactly one new CPU-only renderer_test to the original server, bounded
to120s by the supervisor. It copies the actual decoded24-frame MP4 and first
frame to the new local folder and verifies their SHA. Inspect the saved first
frame and report visual or document divergence. Do not run SDK, change flags,
rerun existing witnesses, touch original training, or present this fixture as a
paper result. A failed stage retains its receipt and media; inspect rather than
blindly relaunching. The expected status is
`SOFTWARE_RENDERER_SYNTHETIC_WITNESS_PASSED`, actual supervisor exit0,
24 encoded/decoded frames,50fps, llvmpipe, no NVIDIA libraries, exactly ONE compiled
world plane (existing plane reused across two XML sections), zero SDK/physics/
optimizer/training-signal activity. Metadata checks alone are insufficient.

## Future actual recorded-pose videos

`actual_rendering_enabled` is currentlyfalse. After both real software witnesses
are bound, actual mode also requires the original5000 checkpoint/backup/controlled
exit and an actually completed quick-pilot receipt, supervisor/resource hashes,
private execution index and matching finished physical-trial NPZ/binary hashes.
Reference-motion files and synthetic fixtures cannot be used as actual inputs.
The same actual trace must say isaaclab_physics, pre-autoreset capture, and
contain initial plus all executed control-step states.

The GMR visual MJCF is a distinct asset from the tracker URDF. Rendering maps
all29 joints BY NAME and uses the recorded root xyz/wxyz; it does not rerun a
policy, integrate dynamics, modify poses or prove identical mesh geometry.
Video text explicitly distinguishes recorded IsaacLab tracking from MuJoCo
visualization. Current pilots track reference motions; the motion generator is
not yet evaluated. Failed trials remain labeled failures. T−2 current-target
timing and unchecked final endpoint remain limitations; no paper score claim.

Future actual commands are server-relative, NOT part of this witness:

```text
python3 scripts/supervise_execution_video.py --mode actual --run pilot_smoke_video0_v1 --pilot-run quick_pilot_smoke_author_v1 --trial 0
```

Trial numbering follows the finished-trial index, not source order. Actual
outputs go to recorded_videos_private first; raw poses, annotations and per-source
indexes stay private. Retain failures and verify media before any permitted
research-video publication. Max2001frames/128MiB,50fps, no dropped/interpolated
frames; actual output timeout900s. Do not treat synthetic validation media as a
new result or the existence of a renderer as an executed closed loop.

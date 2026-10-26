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

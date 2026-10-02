# Frozen completed tasks: Isaac Lab to MuJoCo, 2026-10-02

User authorized the four newly completed tasks Jump, Sit, Pick, Bimanual-Pick.
Kick/Punch transfers are already complete and excluded. No learning, extra IK,
reference change, reward change, training interruption or package installation.

Warm reuse `configs/mujoco_sim2sim_env.json`, canonical hash91455383, ledger
`.aris/compute/ssh-mujoco.md`. MuJoCo3.2.3 CPU dynamics, Torch1.13.1 CUDA
inference on GPU0, NVIDIA EGL GPU0 render. Isaac single-environment exports use
the existing Isaac Sim4.5/Isaac Lab runtime on GPU0; the running8192-env PPO is
on GPU1. Exports use official USD I/O, with no camera. A failed stage blocks
this separate queue; it must not modify the official training queue.

Exact read-only input check from Windows PowerShell:

```powershell
ssh -b 10.147.17.250 -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 "python3 /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/r3_sim2sim_completed_queue.py --stage check"
```

Expected exit0 and `COMPLETED_SIM2SIM_PREFLIGHT_PASS`, checking four exact
final1999 checkpoint hashes against archived results. Not a task result.

First export stage, run once on the server:

```bash
python3 scripts/r3_sim2sim_completed_queue.py --stage export --attempt 2
```

This records original reference tensors,30 actual observation/action snapshots,
default gains, mass, hand binary open/close commands and primitive task scene
dimensions. Per-task logs/states are in artifacts/monitoring with prefix
r3_sim2sim_completed_20261002. Completed stages skip; existing failed attempt
logs block and are never overwritten automatically. Initial Jump export failed
before any trial because this installed ObservationManager lacks get_term_cfg.
That diagnostic log/state remain preserved. One audited API correction uses its
actual observation config list and1D binary hand commands; attempt2 exported
all four normally. This is not a MuJoCo policy failure or a scored trial.

Exact numerical/GPU/physics witness after exports, from Windows PowerShell:

```powershell
ssh -b 10.147.17.250 -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 "python3 /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/r3_sim2sim_completed_queue.py --stage witness"
```

Require exit0; individual task witness logs contain `COMPLETED_INTERFACE_PARITY_PASS`
and `COMPLETED_SIM2SIM_WITNESS` (one each per task). The queue console prints
the input-check sentinel. Inspect all logs/states and output JSON, not just
the queue exit code. Compare observation/action and injected MuJoCo
state on30 real Isaac snapshots, max error below0.0002. Check actor/normalizer
tensor equality to original final checkpoint, seeded CUDA inference, moving
finite floating-base dynamics and nonempty NVIDIA EGL frame. The fresh
doc-follow agent runs the exact command without improvising or fixing. An
additional audit compares all70 original references at16 times against the
unchanged official MotionLibBase numerical functions, including negative time
and future offsets; require maximum error below0.0002 and zero phase mismatches.

After passing, run once `python3 scripts/r3_sim2sim_completed_queue.py --stage eval`.
For every original reference, one start-at-zero trial up to500 control steps
(50Hz), no autoreset. Falling stops and stays in the denominator. Record all
trials, the first two reference-index videos without selecting favorable runs,
model/reference/adapter hashes and actual task collisions. Jump/Sit/Bimanual
have one reference each; Pick has67. No reference motion playback.

Metrics: report the original task criterion as a **proxy**, survival and tracking.
Jump/Sit: torso/pelvis within0.05m only in the original event window. Pick:
object center above0.95m during closed phase; Bimanual above0.7m. Absolute
height may already hold at reset, so it does not by itself prove lifting.
Separate additional checks: Jump both feet airborne at least2 control frames
after initial ground contact (ignore spawn-height drop)
and subsequent stable landing at least1s; Sit seat/pelvis-or-torso contact held
at least1s; Pick right-hand object contact plus relative lift at least0.05m
held1s; Bimanual both hands contact plus same lift/hold. These are explicitly
added checks, not paper metrics. Record their negative outcomes too.

Task cuboids use training geometry/mass/reset positions. No welded or scripted
object motion: free bodies respond only to gravity/contact. Static colliders
and object contacts are active, robot self-collision disabled as in training.
Object mass is the seed0 actual startup mass, fixed across trials. Photorealistic
play assets, USD/MJCF collision/inertia parity, implicit/explicit drive parity,
and sampling distribution remain limitations. These trial rates cannot be
directly subtracted from official Isaac1000-env autoreset rates.

Verify exit0, positive FFprobe duration, hashes and actual image content before
archiving new results/video in local results/reports/media and the new private
GitHub. Original tensors, licensed upstream assets and trained weights stay
on the server.

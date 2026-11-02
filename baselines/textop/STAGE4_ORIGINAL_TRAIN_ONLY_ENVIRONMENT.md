# CURRENT — 2026-10-29: human postponed Stage3 and requested direct Stage4

Stage3 is **POSTPONED_BY_USER**. Do not launch augmentation witnesses or the
5368-clip production job. The older mixed-data factory and disabled spec remain
historical preparation. Never fabricate its mixed-library receipt to bypass it.

The new `step4-original-train-only-ppo-spec.json` explicitly selects the5361
accepted original, unmirrored TRAIN families and5551 slices below40s. Their
2,835,665 T-1 intervals cover the original TRAIN data once; adjacent slices
share only the boundary pose. The1774 original VAL families and2 quarantined
duplicate TRAIN families are excluded. TMR's268 internal-check families are
still original TRAIN and may be used by the motion tracker, without using VAL.
Generated training clips=0. This is a MoCap-only ablation, with no claim that it
reproduces the paper's augmented-data result.

VAE/LDM remain at100000, TMR is finished at100epochs/101500 updates, and the old
557D privileged teacher remains frozen. The new policy is independently
initialized actor431D/critic557D, future5 reference frames,29 actions at50Hz,
MLP[2048,1024,512]ELU, Isaac Lab one-stage PPO. No DAgger/distillation.
The150000 iteration setting remains a paper-derived candidate; no new PPO
throughput or completion time has been measured. The new PPO flags stay false
until its own real environment/recovery witnesses are accepted.

## Warm environment and immutable inputs

Use the existing `scripts/run_tracker.sh` and `configs/tracker-env-spec.json`:
Isaac Sim4.5, Isaac Lab2.1, Torch2.5.1+cu118, RSL-RL2.3.1. No install/shared
source edit. URDF sha256 dbc8471004e9b0b85c27a4ebc1f0d90a727c1cf7ca9ef507b805596f2004d64b.
Exact equivalence to the author's unreleased asset is unestablished.

Server ROOT=/data/code/ouyanghao/oyh/textop-reproduction-20261024. New outputs
ROOT/step34_numeric_v1. UID1013/no symlinks/ext4/write+fsync accepted; previous
28.93GB was a measurement, not a reservation. Each job rechecks2GiB reserve+
128MiB logs+all remaining outputs. GPU1 shared/.65, CPU0-3 total4, headless,
16GiB available RAM, GPUfree>=20GiB. All model/SDK jobs use the existing
receipts/full_library_capacity.lock, even CPU model verification. Immutable
TMR private byte copying may continue independently. No more ACS actions.

Input metadata is reused from the already accepted immutable scope (SHA
ac49c12afeedc8703cf2c3b8c8e19df4c25fc691c2220df9d927d32d3552cbb4),
**only its original_RL_clips field**. No augmentation is generated or required.
The actual TRAIN receipt SHA is
f355de1c947487d750cbe3d59c51d78c8853a36315c659a3b08b4876df7273d2.
`step4_train_only_protocol.scope()` joins every clip to the original TRAIN
source SHA and original named29-joint metadata, rechecks complete source
interval coverage, and prohibits VAL/generated/mirrored inputs.

## Actual reference author and fresh agent-follows-document pass

Author invocation from W=C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成,
using bundledPython -B, PYTHONIOENCODING=utf-8:

    python -B BASE/provider_tools/step4_train_only.py refs_launch --role author --evidence s4_ref_a_launch1
    python -B BASE/provider_tools/step4_train_only.py refs_collect --role author --evidence s4_ref_a_collect1

BASE is W/work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop;
python is C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe.
Read this full document and the actual source before using these invocations.
If any intent/run already exists, inspect the original process and outcome;
never dispatch the same role again. Live collection is metadata-only.

The author loads the actual first original TRAIN slice, hashes its complete
source file, writes the poses to128 Isaac articulation instances, runs
SimulationContext.forward (no dynamics), maps29 joints by name and reads30
bodies. It computes the unchanged original .02s numpy gradients and world
angular-velocity rule. No joint/contact clamp, policy/model, physics steps or
optimization. The bound full request/source SHA precedes launch. Output upper
bound is2152B/frame+8192B/clip+64MiB asset allowance; float64 angular velocity
is included. All finite/quaternion/source-SHA checks must pass. Native close
may use the documented bounded20s fallback; do not call that native success.

Actual accepted author closure requires STAGE4_ORIGINAL_TRAIN_ISAAC_REFERENCES_FINISHED,
process/supervisor/outer0, child reaped, after_exit owned processes empty,
writers closed, source signatures unchanged, model lock free, and the terminal
collector's complete NPZ byte SHA/CRC, each array SHA/shape/dtype and exact
original TRAIN source join. A dispatcher, source parse or live sample is not
an accepted numerical result.

Only after the author has actually passed, a **fresh agent** follows this same
document verbatim and executes:

    python -B BASE/provider_tools/step4_train_only.py refs_launch --role document --evidence s4_ref_d_launch1
    python -B BASE/provider_tools/step4_train_only.py refs_collect --role document --evidence s4_ref_d_collect1

It independently compares the real author/document array hashes/shapes/dtypes
(numeric_signature), real closures, resources and source bindings. It must
report actual findings, preserve failures and write an honest SUMMARY under
W/work/textop_reproduction_20261024/step4_train_only_reference_document_v1.
Status STAGE4_TRAIN_ONLY_REFERENCE_FRESH_DOCUMENT_VERIFIED is permitted only
with actual_exit_code0 and no unresolved findings. Copy that real summary to
BASE/results/step4_train_only_20261029/fresh_document_summary.json. Do not fill
findings[] until actually reviewed. Do not reuse older FK/teacher witnesses.

After real fresh-document acceptance, parent may launch `refs_launch --role
library --evidence s4_ref_library_launch1`, then use new unique refs_collect
evidence. The library covers all5551 original TRAIN slices, no generated or
VAL motion. Its independently verified terminal manifest becomes the explicit
TRAIN-only RL input; FK alone is not physical acceptance or PPO readiness.

## Remaining real PPO work

`step4_train_only_isaaclab_config.py` provides the separate TRAIN-only factory,
without weakening the historical Stage3 mixed-data contract. New live SDK
observation dimensions, complete source library consumption, finite gradients,
fresh model/Adam/normalizer/CPU-CUDA RNG/ADS state recovery and recorded-rollout
replay still require real author/fresh acceptance before formal training.
Measure fullstate size and actual8192-environment PPO throughput to accept
state slots and estimate duration. No old teacher weight transfer or invented
performance. Simulator cache replay, if unsupported, must be explicitly
documented as episode-boundary restart, never claimed bit-exact dynamics.

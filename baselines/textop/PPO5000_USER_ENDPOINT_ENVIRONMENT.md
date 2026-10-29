# User-selected PPO5000 endpoint — actual training remains active

The user changed the new PPO budget from10000 to5000 completed iterations.
Keep the existing original-TRAIN-only run and its frozen source/spec/Adam/RNG.
Stage3 remains postponed. Existing training has no regular save at5000: its
normal immutable target is10000 and it retains last complete learning state in
CPU RAM. This controller is a separate CPU process, not an edited training job.

It observes the trainer's fsynced receipt every0.1s. Only after it records
exactly5000 completed iterations/100000 finite Adam calls, matching original
request/source, fresh-policy scope, original TRAIN library, PID creation identity
and no preexisting emergency state, it writes one durable signal intent and
uses a pidfd to send a single SIGINT to the bound original Python worker. It
never signals the supervisor, a process group, or any other project. The normal
exception handler saves the latest fully completed learning state to the new
private emergency file. Original interrupted exit codes and any partial
post-target in-memory optimizer calls remain factual evidence.

This is **not** a preexisting producer checkpoint at5000. There is a short
race between receipt, signal delivery and the next iteration; neither the
observer nor a reported iteration number can prove that the resulting private
file contains the exact target. After all original workers really exit and the
common model lock is free, a separate new CPU audit must deserialize the saved
state and verify exactly5000/100000, all finite model/Adam/RMS/ADS/clock/RNG,
source binding, full SHA and original exit facts. If the actual saved state is
not exactly5000, report it honestly and do not present it as the user's target.
There is no warrant to rearm or resend a signal when intent or outcome is
unknown. Continue the already authorized CPU/fresh export and new WBC heldout/
online effects only from a genuinely accepted state.

W=C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成;
BASE=W/work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop;
ROOT=/data/code/ouyanghao/oyh/textop-reproduction-20261024;
T=W/work/textop_reproduction_20261024.
Every Windows command explicitly sets PYTHONIOENCODING=utf-8, uses bundled
Python -B, and sets the working directory to W. Unique short ASCII evidence.

Actual disabled spec and server sources have been created. The author fixture
is real exit0/closed:14 invalid metadata cases,2 wrong identities, duplicate
intent refused; exactly one SIGINT sent to its own throwaway child,0 production
signals,0 model/checkpoint/GPU/optimizer work. Author receipt SHA
175542780428a01254e34d568359ab9af51504e07a6c9071e806ba4a5835b10d.
This fixture does not prove a production save or future physics continuation.

The independent fresh document pass must review this guide and these precise
sources: `scripts/ppo5000_endpoint_protocol.py`,
`scripts/watch_ppo5000_endpoint.py`, `scripts/supervise_ppo5000_endpoint.py`,
`scripts/test_ppo5000_endpoint.py`, and `provider_tools/ppo5000_endpoint.py`.
It must run exactly one new isolated document fixture:

    python -B BASE/provider_tools/ppo5000_endpoint.py fixture --role document --evidence p5000_fd1

Read its real result at
`BASE/results/step4_train_only_20261029/ppo5000_endpoint/p5000_fd1/result.json`.
Review actual code for unsafe signaling, identity errors, target overshoot,
model-lock overlap, source mutation or falsely accepted partial checkpoints.
Do not start or signal production as part of this independent fixture.
Write an honest SUMMARY at `T/ppo5000_endpoint_document_v1/SUMMARY.json` with
status `PPO5000_FRESH_DOCUMENT_VERIFIED` only for exit0 and no unresolved
findings; otherwise use a failure status and report the findings. Required
fields: `actual_exit_code`, `findings`, `watcher_sha256`, `protocol_sha256`,
`fixture_receipt_sha256`, `actual_production_signals=0` and
`actual_fixture_signals=1`, all from actual result/source bytes.

After genuine author and fresh acceptance, the parent alone enables and
launches the one production observer with unique evidence:

    python -B BASE/provider_tools/ppo5000_endpoint.py enable --document-summary T/ppo5000_endpoint_document_v1/SUMMARY.json --evidence p5000_enable1
    python -B BASE/provider_tools/ppo5000_endpoint.py launch --evidence p5000_launch1
    python -B BASE/provider_tools/ppo5000_endpoint.py collect --evidence p5000_live1

The controller is not armed merely by enable or dispatch. Confirm its worker
identity, WAITING status and0 signals by collector. Unknown client or prior
intent requires reconciliation, never blind retry. Once armed keep this sole
controller; do not rearm, signal manually, restart the old training or edit
its active source/spec/budget. The observer uses a separate metadata lock,
CPU0-3,1GiB available RAM,2GiB+128MiB DATA floor and15s resource supervision.
It does not take the training model lock or load a neural model/SDK/GPU.

After signal, its own result `PPO5000_SIGNAL_AND_SAVED_STATE_AWAITING_CPU_AUDIT`
proves a file and original exits only; it is not an accepted5000 checkpoint.
Original training process/supervisor/outer exits remain nonzero and must not
be rewritten as success. Formal10000 checkpoint and old10000 export protocol
must not be used for this target. Source/resume/CPU/export acceptance must be
explicitly adapted to the real5000 emergency provenance, with complete state
and exact step validation. No new RL iteration or generator/old-teacher update
is authorized after the selected endpoint.

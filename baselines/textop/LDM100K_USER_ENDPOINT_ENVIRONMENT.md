<!--ldm100k-prefix-preparation-start-->
CURRENT新增：真正100k专用CPU完整性/推理导出源码与本地启用/派发/轻量收集桥已实现；服务器已部署禁用，export_authorized=false/ldm_training=null。26个独立元数据破坏负例实际CPU0/reaped/streams_final通过，0生产checkpoint读取/模型/GPU/信号/真实导出；不是实际验收。读[LDM100K_PREFIX_EXPORT_ENVIRONMENT.md](LDM100K_PREFIX_EXPORT_ENVIRONMENT.md)和prefix_export_preparation_v2.json。不要重跑准备或提前pending激活。原训练11800尝试/11800Adam/0skip（2026-10-28T14:10:41.709363+08:00），原监督与worker身份未变；唯一100k控制另采样11900，仍等待真实保存和精确恢复，0生产信号。源/spec/共同锁均保持。私有100k字节备份现有worker/client创建身份实核仍在传，partial不是已完成备份。
<!--ldm100k-prefix-preparation-end-->

# CURRENT — user requests LDM100000 endpoint, original300k math frozen

Latest human: “LDM也训练100000步吧，不需要300000步。” This overrides all older
300k completion requirements. VAE100000 is already complete. Continue the same
running LDM weights/Adam/RNG to the real100000 checkpoint, then stop the original
program and proceed to actual integrity/generation/semantic/physical evaluation.
No restart, course compression, optimizer change, or new user confirmation.
The original300k run name/schedule/source remain historical provenance; a100k
prefix is not a completed300k schedule or exact paper reproduction.

Original run user_vae100k_ldm300k_train_v1. Original specSHA
654332646f2aa4bb69fea4cc779edf88138735a0ca9f5e5906e783948868c770;
entry eb613778716602f83051e54a2c40d1fc8b6a2820aaf5bd8416a75b18f47b3b64;
numeric/layout binding323349d38cef74e07215ab81c8919879512a2391552c9187151eb6a8a521b7ff.
At13:26 actual3900attempts/3900Adam/0skip, same worker304272/start1436142105
and supervisor304265/start1436142089. PIDs are clues, verify argv/start identity.
Original GPU1shared/.65/CPU0-3 and common GPU lock stay unchanged. No overlapping
GPU/SDK/model jobs. Existing private byte worker is independent CPU transport.

New scripts/ldm100k_control_protocol.py,watch_ldm100k_handoff.py,
supervise_ldm100k_handoff.py and provider_tools/test_ldm100k_control_cpu.py were
deployed and parsed by real server Python3.8. New disabled control specSHA
1f5d9d2895c75905a2846d4cdbe55915f1c502aa18e07f45d17a55fb48c282a7.
Author CPU fixture has actually exited0/reaped/final streams,10negative metadata
cases plus wrong identity/unknown or duplicate intent rejection; exactly one
fixture-owned CPU sleep received SIGINT and exited-2. Production signals0,
production checkpoint reads0,newGPU0. It also proves zero history/semantic
quality counters do not authorize training past the user's endpoint.

The new observer is NOT yet armed until the actual fresh document pass below.
Old VAE observers are dead and must never be rearmed or signaled again.

## Fresh CPU document invocation

Use local bundled Python, PYTHONIOENCODING=utf-8 and BASE scripts. No package
installation or source/spec edits. Run exactly once:

```
provider_tools/run_ldm100k_control_fixture.py --run document_v1
```

The helper dispatches the documented stdlib fixture only, not an observer or
production signal. It records actual remote child PID/exit/reaping/streams,
checks original and new source/spec hashes before/after, original LDM unchanged.
Local result is results/ldm100k_user_handoff_20261028/
control_fixture_document_v1.json; raw client outcome fix_document_v1_client.json.
Remote SUMMARY lives at ROOT/ldm100k_user_handoff_20261028/
control_fixture_document_v1/SUMMARY.json. Inspect actual hashes and client
exit0/reaped/final streams, not just printed capability or source parsing.
No GPU lock is taken by these CPU fixtures; all signaled PIDs must be created
inside the fixture itself. Unknown intent/client means inspect, never rerun.

Return a genuine local SUMMARY.json with fields:

```
status: LDM100K_CONTROL_FRESH_DOCUMENT_VERIFIED
actual_exit_code: 0
findings: []
actual_fixture_child_reaped: true
actual_fixture_child_streams_final: true
fresh_document_fixture: actual result.fixture
client: actual recorded client outcome
current_source_spec_sha256: actual result.current_source_spec_sha256
remote_fixture_summary_sha256: actual result.remote_fixture_summary_sha256
```

Only after the real author and fresh CPU successes, parent invokes:

```
provider_tools/arm_ldm100k_control.py --document-summary ACTUAL_ABSOLUTE_SUMMARY
provider_tools/collect_ldm100k_control.py --kind ldm_control --run user_ldm100k_control_v1 --evidence lc0
```

The arm operation is unique, archives the disabled spec, binds two real CPU
proofs and verifies the original live pidfd identity. It starts a durable CPU
observer with separate ldm100k_user_control.lock, not the occupied GPU lock.
Do not arm again or send manual extra signals. The observer is bounded7days,
CPU0-3,1GiB RAM/2GiBdata floor,15s resource logs. Its supervisor owns only its
own observer group and never SIGKILLs the original producer.

Every second the observer checks the immutable original source/spec binding,
current process identity and receipt. Only a named real100000 full checkpoint
after atomic fsync/rename, completed native validation, producer exact reload
of model/Adam/EMA/RNG/clock/pending fields and restored history counters may be
selected. It independently checks full bytes/SHA/owner/no symlink/stable inode,
then fsyncs one exclusive signal intent and sends one pidfd-bound SIGINT. An
unknown intent is not permission to resend. Semantic quality is evaluated AFTER
the stop and must not keep the trainer running beyond100000.

Original KeyboardInterrupt/nonzero process/supervisor/outer exits are retained.
Post-target in-memory work can be interrupted and is discarded; never repair
or relabel it as an accepted checkpoint. The selected immutable100k state is
the only generator source. Require no original live workers, all actual exit
records, closed observer process/supervisor/outer0 and released GPU lock before
any new model/SDK/GPU evaluation. Positive control status:
USER_LDM100000_STOP_ACCEPTED_ORIGINAL_EXITS_RETAINED. Its target history counters
are recorded, not fabricated; zero/failed counters remain real quality findings.

Continue with actual100k fullstate finite model/Adam/clock/history/aux-counter
audit, inference reload and heldout multi-seed text/unconditional/TMR metrics,
physical endpoints and films. These post100k steps are not yet accepted and
must not use the old300k-completion gate or claim complete paper effects. All
sources, failed trials and checkpoints remain private/retained. No training
past the new100k endpoint, no teacher/prefilter8100/quick restarts. If the old
program exits early due to capacity/errors, inspect real emergency/fullstate
and actual exits; do not signal/rearm blindly or count unexecuted steps.

Keep source/private transport and old57D/PhyGile/ReactiveBFM protections in
USER_VAE100K_LDM_ENVIRONMENT.md. VAE100k D copy p100k_v1 is still independent;
pending is not verified,2GiB floors remain, no partial deletion or raw public
weights/text/trajectory/SDK upload. Health/pending stays quiet; only material
result/failure/input/completion notifies. Existing textop monitor only.

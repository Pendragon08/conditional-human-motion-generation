# PPO5000: disabled CPU follow-up, actual training still active

The human-selected target is5000 PPO iterations/100000 Adam calls. The single
original-TRAIN PPO job and its single bound-PID stop controller are active;
Stage3 remains postponed. Neither active source/request/budget was edited.
The original producer still names10000 internally; its name/spec are provenance,
not permission to ignore the5000 endpoint. The controller will request the
producer's last-good emergency save by one SIGINT after observing the5000
receipt. Signal delivery can race the next iteration. Only the real saved-state
CPU audit can prove its exact5000/100000 contents.

This guide supersedes the fresh-agent launch sequence in the historical
`PPO5000_USER_ENDPOINT_ENVIRONMENT.md`. The two isolated CPU boundary fixtures
genuinely passed, but two fresh-AI-agent startup attempts failed before any
tools because the account did not support the requested model. The parent
review and the limitations are explicitly recorded in
`PPO5000_USER_ENDPOINT_REVIEW_FALLBACK.md`; no agent review is claimed.

## New disabled successor — not yet an evaluated model

`provider_tools/ppo5000_frozen_cpu_v2.py` and these four new distinct sources
are deployed on the user's server with the **disabled** template
`configs/step4-ppo5000-frozen-cpu-v3-spec.json`, SHA
904d8932e978c3f05ef3a0ffacac391a7ce1de493a30f23c2cd063b114105e4d:

- `scripts/step4_ppo5000_frozen_protocol_v2.py`, SHA
  8de84fe222c8045aae460bcda10d8f92e25afc9e8b0c1351b0351004b39f2e40;
- `scripts/step4_ppo5000_cpu_core_v2.py`, SHA
  6d9e10e0fb0a7c02aeb7c1ba7d7d0f9004fc6dd12bcc3870303f92f4b06c9a3e;
- `scripts/run_ppo5000_frozen_cpu_v2.py`, SHA
  135c64943bd55bf3b87625ce2cf2da887282fe0be6b560e8f9c189e939a49d09;
- `scripts/supervise_ppo5000_frozen_cpu_v2.py`, SHA
  11fe87478a5615c8374932c77afc052a023166804b26ea97b6c322949eb8f962.

No CPU checkpoint, actor, Adam or SDK has been loaded by these preparations.
Two synthetic positive and25 negative provenance examples, two positive and
nine negative exit examples, and an additional deployed v2 live-predecessor/
unsupervised-entry refusal with three incorrect native-source hashes passed.
These are metadata/code gates, not real neural-state witnesses. On server the
actual native PPO source hash `ec0286fdbdd30360b21f2bea314f676058793f9241649d30545d656a961a22a2`
shows `torch.optim.Adam`. Actor and runner source SHA match actual training
receipts. Normalizer SHA was checked while training, not directly recorded at
its initialization; do not assert more provenance than observed.

An earlier disabled source version is retained. An unexecuted local draft
assigned audit fields before constructing the audit dictionary; that defect
was corrected before the v2 sources were deployed, with old bytes archived in
`sources/ppo5000_cpu_v3_before_preflight_repair`. The v2 collector's worker
names were also corrected before deployment and its old bridge bytes retained.
None of these edits changed the active trainer or controller. The sources are
not a completed, independently verified model or an effects result.

## After the actual controller and original trainer finish

First observe the existing controller and original processes, their actual
exit/intent and the common lock. Do not rearm or send another signal. The
controller's `PPO5000_SIGNAL_AND_SAVED_STATE_AWAITING_CPU_AUDIT` is only a
file/provenance observation; its original child/supervisor/outer nonzero exits
must remain intact. The saved state must not be called5000 until its payload has
been read by the accepted CPU job. A different count is a real failure, not a
reason to relabel, skip the check, or select old teacher weights.

Only when original training and the controller have closed, all their workers
have exited, the original common lock is free, and new capacity is accepted:

    python -B BASE/provider_tools/ppo5000_frozen_cpu_v2.py activate --stage export --evidence p5_cpu_act1
    python -B BASE/provider_tools/ppo5000_frozen_cpu_v2.py launch --stage export --evidence p5_cpu_launch1
    python -B BASE/provider_tools/ppo5000_frozen_cpu_v2.py collect --stage export --evidence p5_cpu_collect1

The immutable activated job request binds actual signal intent/result, original
nonzero exits, controller0 closure, final state bytes/SHA/content, all source
hashes and original TRAIN library. A new CPU supervisor acquires the same model
lock. The entry refuses to construct a model outside its owned supervisor. It
must audit exact5000/100000, CPU-finite full state and all Adam counters,
reconstruct the native431/557 model and `torch.optim.Adam` and exactly reload
their saved state without taking a step, inspect CPU/CUDA RNG bytes without
restoring CUDA, and export the actor plus its learned observation RMS. The
policy payload emits raw29D deterministic actions before actuator scaling.
No new clamp, generator, teacher, optimization or physics simulation is added.

Only after actual export receipt, child/supervisor/outer0, no workers, common
lock free and terminal artifact binding are verified, repeat the same bridge
with `--stage verify` and fresh unique evidence. The second process reloads
the full state and previously saved actor/RMS, comparing parameters and
synthetic deterministic forwards exactly. `FRESH_CPU_VERIFIED` means a new
OS process; it does not imply a new AI-agent review. Both real jobs are still
pending. Their synthetic forward results establish serialization integrity,
not heldout tracking, online20ms latency or complete paper reproduction.

Each job needs actual DATA reserve2GiB+128MiB logs+remaining32MiB export,
16GiB available host RAM, CPU0-3, no visible CUDA and900s bounded supervision
with15s resource samples. These are checks, not reservations. Keep all partials,
failed receipts and original nonzero exits. Unknown dispatch/exit must be
reconciled from original intent/process identity; never blind-retry or signal.
After a real independent CPU verify, continue genuine new-policy heldout/online
tracking and a separate private fullstate backup. Recheck C capacity before any
backup; D has been below its2GiB floor. No old10000/TMR/VAE acceptance tool,
old557 teacher physical result or old videos can substitute for this work.

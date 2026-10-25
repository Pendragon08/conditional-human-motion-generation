# Contact-consistent physical filtering v3 — inactive candidate

2026-10-26: formal privileged-policy training continues separately. This new
exporter/supervisor/finalizer is **disabled and has not been executed in the SDK**.
It does not establish physical acceptance, a trained generator, or evaluator readiness.
Do not modify the 16 byte-bound files used by the active formal training run.

## Reason and exact boundary

Formal policy training now uses the validated static solid-floor contact profile.
The historical strict replay executor uses the old Plane and the historical v2
validator did not bind contact configuration or the evaluator's actual exit.
Keep that original code and evidence unchanged. New candidate files are:

- scripts/evaluate_strict_filter_contact.py: same strict thresholds and complete
  frame0 through T-1 coverage, with the same validated_contact_profile as training;
  no optimizer updates; writes completed trials incrementally to private JSONL.
- scripts/supervise_strict_contact_replay.py: separate, disabled formal entry,
  requires a genuinely finished40k policy, same common SDK lock and resource
  preflight. Persists actual exit and final receipt SHA before acceptance.
- scripts/finalize_contact_filter.py: separate CPU-only post-exit step. Writes
  private evidence/decisions plus public aggregate counts; corpus acceptance and
  generator training remain false until the later final-corpus checks pass.
- scripts/apply_tracking_filter_v3.py: all v2 checkpoint/training/attempt/EMA/
  whole-motion checks plus matching contact profile, source/spec/checkpoint hashes,
  closed-output/evaluator-exit checks and a lossless trial-journal comparison.

The exact static floor is4000x4000x0.2m, topz0, friction1/restitution0/multiply,
self-collision preserved. Author collision geometry remains unpublished. v3 keeps
pelvis0.25m/end-effector0.5m thresholds; it does not delete difficult motions or
clip rewards. One deterministic strict replay per source is an explicit local
choice where paper repetition details are unavailable.

## What was actually checked

Local CPU author_v3 and server Python3.8.10 each passed23 synthetic protocol cases.
The server process used CPU0–3 and hidden CUDA, invoked no SDK and no optimizer.
The positive fixture contains explicit synthetic checkpoint/source markers; it is
not a real policy or accepted motion. Negative cases include running training,
wrong contact profile/self-collision/friction, failed/timeout/unclosed evaluator,
wrong source/checkpoint/reference, disabled or unvalidated spec, missing/duplicate
journal rows, partial trajectories and never-attempted sources. Fixtures remain.

An additional check rejected the **real running100-round/2000-update receipt**.
The disabled supervisor exited before dispatching SDK work or creating an output
run. All16 active runtime files had identical frozen hashes before and after
staging the six new files. No active configuration was modified.

Receipts: results/filter_contact_protocol_20261026/local_cpu_v3.json and
server_cpu_v1.json. Earlier20/22-case receipts are retained. The final local
collection step initially used the wrong parent directory after server checks had
already passed; its assertion failure and local-only recovery are retained in
collection_path_failure_and_recovery.json. Server checks were not rerun.

Git autocrlf changes checkout bytes. candidate_runtime_bytes.json preserves exact
six-file deployed bytes/SHA; this is separate from the immutable formal-training
bundle. Restore only into a new owned folder, never over active runtime files.

## Remaining work before any actual formal evaluation

1. Let the current40000-round policy finish; validate and privately back up its
   genuine final checkpoint, statistics, receipt and actual supervisor exit.
2. Prepare a distinct bounded SDK-validation path for this new exporter. Its
   present CLI only accepts formal mode; there is deliberately no shortcut to
   turn the active incomplete policy into accepted evidence. The real-validation
   flag cannot be set merely because these CPU tests pass. Retain the legacy
   witness for history; do not reuse its old Plane as the new evaluator.
3. Add/review continuous resource sampling and failure handling for the new
   supervisor, then perform author and fresh independent actual SDK validation
   when no other SDK task holds the common lock. CPU logic checks are not this
   validation. Ensure all completed failed trials survive errors/timeouts.
4. Bind the actual validation receipts and exact source bytes, latest storage/RAM/
   GPU budget and completed policy before enabling the new spec. The current
   formal_evaluation_enabled and real_sdk_validation_completed stay false.
5. Run full real relaxed-statistics plus strict whole-motion coverage under the
   same contact profile. After actual exit0 and closed writers, use the separate
   finalizer. Then independently accept the source-isolated69D corpus, train-only
   normalization, full realCLIP cache and resume state. Only then automatically
   start the already authorized VAE200k→LDM300k and final control/closed-loop work.

Hashes and internally consistent logs cannot independently authenticate their
own truth. Trusted actual execution and independent observation remain required.
All arrays, per-source/text mappings, trial journals and decisions stay private.

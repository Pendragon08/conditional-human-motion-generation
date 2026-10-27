# Experiment Audit Follow-up — Commit 6850b6e

**Date:** 2026-10-22  
**Auditor:** Codex GPT-5.6-Sol ultra, same-family reviewer, read-only, provisional  
**Follow-up commit:** `6850b6e404565e4716588943b03fa07f4141e085`  
**Initial frozen verdict:** WARN at `104393d48be438e923a30f3e42c466b861fb8639`  
**Follow-up verdict:** WARN — meaningful partial remediation, residual safety gaps  

## Verdict relationship

This is a separate bounded review of the requested backup and recovery changes. It does not rewrite the initial A–F findings. The scientific statuses remain **A PASS, B PASS, C WARN, D PASS, E WARN, F PASS**. Commit `6850b6e` closes the initial cached-evaluation-file gap and adds a useful manual recovery-plan guard, but it does not complete training, exercise a real checkpoint, or resolve the remaining checkpoint/status/cleanup risks. Operational integrity therefore remains **WARN**.

## What the commit demonstrably fixes

### 1. Existing downloaded evaluation artifacts are revalidated: PASS

`Worker.download()` now obtains the current remote byte count and SHA-256 whenever the caller does not already supply an expected digest, compares any existing local file, validates a newly copied partial, and only then renames it (`baselines/reactivebfm/scripts/backup_full_run.py:107-123`). This covers evaluation receipts, feedback JSON, `args.json`, and evaluation archives because those call sites use `download()` (`:161-189,262`). The initial behavior that accepted any already-present file when `expected=None` is gone.

Evaluation receipts also receive a sidecar seal on first capture and must pass it before reuse (`backup_full_run.py:45-54,140-170`). A successful physical receipt is then compared again with current remote bytes (`:168-170`). Feedback and archive downloads are revalidated on every pass (`:171-189`). The two added unit tests exercise a mismatched existing download and a changed sealed receipt (`baselines/reactivebfm/scripts/test_backup_full_run.py:97-109`). The preserved test output records eight passing tests (`baselines/reactivebfm/results/full_training_20261021/backup_worker_unit_v4_stderr.txt:1-5`).

This resolves initial operational finding 2 for ordinary, non-optimized execution. The locally generated `EVALUATION_LAUNCH_FAILED` receipt remains intentionally frozen and is not upgraded if a remote receipt appears later; this is conservative and does not invent a pass.

### 2. Automatic selection of an unmanifested highest filename is mitigated for documented recovery: PASS WITH CONDITIONS

`prepare_verified_resume.py` scans only `checkpoint_ready_*.json`, requires model and optimizer names for the same step, checks containment/no symlink, streams each file's size and SHA-256, checks the configuration hash, and chooses the highest verified update (`baselines/reactivebfm/scripts/prepare_verified_resume.py:23-55`). It refuses an existing recovery output and emits a plan only; it does not launch training (`:45-75`). The generated command uses a fresh output directory and an explicit model path (`:55-66`). Because the upstream scanner sees the fresh output's empty checkpoint directory, it cannot override that explicit resume path with an unmanifested higher file from the failed attempt.

The five tests cover an unmanifested higher filename, a corrupt latest pair, no valid pair/existing destination, plan-only behavior, and a changed configuration (`baselines/reactivebfm/scripts/test_verified_resume.py:28-50`). The preserved output records five passing tests (`baselines/reactivebfm/results/full_training_20261021/verified_resume_unit_v2_stderr.txt:1-5`). The operations guide accurately says the helper is plan-only, requires a fresh attempt/release namespace, cannot reproduce RNG/data-iterator state bitwise, and has not been tested on a real full-run checkpoint (`baselines/reactivebfm/docs/full_run_operations.md:17-32`).

This is a valid mitigation for initial finding 1 when the helper is executed normally and the pair is revalidated immediately before launch. It does not make the original checkpoint write itself atomic.

### 3. Deployment/metadata evidence is internally consistent: PASS AS LOCAL RECEIPT EVIDENCE

The metadata receipt records an actual SSH metadata download/cached-file verification, not a training or physics result (`baselines/reactivebfm/results/full_training_20261021/metadata_download_v1/receipt.json:2-16`). Its claimed hashes match the committed bytes:

- `args.json`: 2,536 bytes, SHA-256 `c7f934811ef3e2fe166b0e53b27099ec7aef577058a5ed4066047fc91a4e491c` (`metadata_download_v1/args.json:1-87`; receipt `:5-6`).
- `train_full.py`: `ce21ec77…d57de`; `run_closed_loop.py`: `a05ba789…3c16`; `prepare_verified_resume.py`: `5985d062…9dcb`; config: `89428ed9…b53a` (receipt `:8-12`).
- Revised worker: `6cdac41b19a23790cfc50eaf6596844332bbabbee19e01208a13e68269ee39c1` (receipt `:14`).
- The previous closed-loop receipt hash remains `282459de…e1ac` (receipt `:7`).

The committed launch record names PID 53588 and distinct v3 logs (`baselines/reactivebfm/results/full_training_20261021/backup_worker_launch_v3.json:1-6`). A separate read-only local observation found that PID running the expected worker/config command; `D:/具身智能/ReactiveBFM复现_20261021/full_training_20261022/worker_v3_stdout.txt:1-3` records the exact revised worker hash and successive snapshots at 590 and 660 updates. At 660 updates both ranks were still `TRAINING`, the process was present, and no checkpoint manifest existed (`current_training_status.json:1-175`). This demonstrates deployment/liveness at that observation time, not future continuity.

## Report and documentation claims

The new supplement accurately distinguishes protocol hardening from scientific validation. It says the audit is provisional WARN, states that the trainer's underlying writes remain non-atomic, notes that the recovery helper only produces a plan, says failures do not auto-restart training, and explicitly says there has been no real recovery rehearsal (`baselines/reactivebfm/reports/ReactiveBFM完整训练与闭环进展报告_20261022.md:97-110`). The operations guide repeats these limits (`full_run_operations.md:9,17-32,34-36`).

One minor inconsistency remains: the older report text still says “six backup unit tests” (`report:75`), while the supplement and primary output correctly say eight (`report:101`; `backup_worker_unit_v4_stderr.txt:1-5`). It should say that the original six were extended to eight, or update the count.

## Residual gaps

### 1. Recovery validation still depends on `assert`: material WARN

Every critical pair check in `select_verified()`—step/update types, manifest filename, two-file cardinality, exact names, path containment, symlink rejection, hash format, size, and SHA-256—is implemented with `assert` (`prepare_verified_resume.py:23-40`). Python removes those checks under `-O`/`PYTHONOPTIMIZE`.

I confirmed this with a bounded CPU test in the allowed audit-output area: under `python -O`, a manifest containing one nonexistent model path, no optimizer row, a false size, and a zero digest still produced `status=VERIFIED_RECOVERY_PLAN_ONLY`; the observed fields were `selected_files=1`, `checkpoint_exists=false`, and `optimizer_row_present=false`. No training was launched. This contradicts the helper's unconditional “only verified complete pairs” safety claim if optimized Python is used.

Replace these assertions with explicit validation branches that raise `ValueError`/`RuntimeError`, and add a subprocess test that invokes the helper with `python -O` and proves the malformed pair is rejected. The same correction remains necessary for deletion-critical assertions in `backup_full_run.py:226-235,251-257,287-292`.

### 2. Plan generation and launch are separate, with no launch-time revalidation: material WARN

The plan embeds the selected manifest and hashes, but its command directly references the original model path (`prepare_verified_resume.py:55-66`). There is no executor that re-hashes the model/optimizer/manifest immediately before spawning `torchrun`, and the verified pair is not copied or atomically linked into a sealed recovery staging directory. A file can change after plan creation and before manual execution. The generated command then bypasses the helper entirely.

Add a plan executor that verifies the plan file, manifest hash, configuration hash, model and optimizer hashes immediately before launch, refuses an existing destination, records the exact launch evidence, and then launches once. Keeping plan generation non-launching is sound; the execution step needs the same fail-closed checks.

### 3. Original checkpoint publication remains non-atomic: WARN, mitigated but not fixed

The trainer still writes model and optimizer directly to final names before publishing its atomic ready manifest. Commit `6850b6e` does not change `train_full.py` or the upstream saver. The helper can reject incomplete pairs after a crash, which materially lowers resume risk, but partial final-named files still exist and can be selected if an operator bypasses the documented helper. The report correctly discloses this (`report:105`; operations guide `:19-21`).

### 4. Worker completion/liveness and cleanup gaps remain: WARN

The follow-up does not change these initial findings:

- Status parse errors are silently omitted, and completion does not require the exact fresh rank set (`backup_full_run.py:78-90,298-301`).
- Process liveness is a generic command substring match, with no run/output/PID binding or stale-progress detector (`:88-90`).
- Explicit failure or process absence stops the worker without verified automatic recovery; empty/corrupt statuses plus an absent process can poll indefinitely (`:302-312`).
- Release asset discovery is not paginated (`:92-94`).
- Critical deletion and manifest-shape guards remain `assert` based (`:226-235,251-257,287-292`).

### 5. Tests remain protocol simulations, and no real checkpoint path has run: WARN

The eight backup tests and five recovery tests use dummy bytes. The operations guide and report disclose that limitation (`full_run_operations.md:30`; report `:101-105`). At the follow-up observation the full run had 660 updates and still no first-1,000 checkpoint. Thus no real checkpoint has exercised model/optimizer publication, GitHub upload, diagnostic fan-out, retention cleanup, verified recovery planning, deserialization, or resumed optimization.

The metadata download receipt proves the revised generic download/hash path against `args.json`; it is useful but is not a substitute for the first real checkpoint acceptance test.

### 6. Small reproducibility/documentation limits remain

- The helper hashes `args.json` but has no trusted expected args hash and the generated training command does not consume that file (`prepare_verified_resume.py:49-59`). Treat `args_sha256` as recorded metadata, not an integrity comparison.
- The helper validates bytes but does not perform a bounded deserialization/load check for model and optimizer. A ready manifest strongly indicates `torch.save` returned, but a pre-launch load validation would detect a semantically unreadable pair before allocating a new recovery attempt.
- `full_run_operations.md:13` instructs a hidden background window but does not itself record the executable flags/environment. Preserve the actual command, source hash, configuration hash, and process identity on every restart, as the v3 evidence did.
- Present-tense progress and process statements remain point-in-time claims.

## Follow-up action items

1. Replace all recovery and deletion safety `assert` statements with explicit exceptions; add optimized-Python rejection tests.
2. Add a launch-time verifier/executor or sealed staging copy so the pair cannot change between plan creation and `torchrun` start.
3. At the first 1,000-update checkpoint, run a bounded acceptance sequence: validate the ready manifest and both hashes, load model/optimizer without training, download/upload with verified hashes, confirm immutable evidence, exercise all nine diagnostic attempts, and record retention decisions. Do not delete the only verified server pair during this acceptance.
4. Require exact rank membership, freshness, configuration/source agreement, and run-specific process identity before completion or failure decisions; add a stale-progress state and publish abrupt-exit evidence.
5. Replace single-page release asset discovery with pagination.
6. Update report line 75 to distinguish the original six tests from the current eight.
7. Keep the recovery helper's stated ceiling: verified plan only, no exact RNG/data replay, no demonstrated real recovery until that test exists.

## Claim impact after 6850b6e

| Claim | Follow-up impact |
|---|---|
| Existing evaluation downloads are checked against current remote bytes | **Supported** for normal execution by code, two added tests, and an actual metadata download receipt |
| Cached evaluation receipts cannot silently change | **Supported** for accidental/stale changes under normal execution; locally generated failure seals are not cryptographic authentication against a malicious writer |
| The documented helper ignores unmanifested higher files and corrupt pairs | **Supported with conditions** under ordinary Python and at plan-generation time |
| The helper is safe under all Python modes | **Unsupported**; bounded `python -O` verification bypassed all assert-based pair checks |
| A verified plan guarantees the same bytes will be launched later | **Unsupported** without launch-time revalidation or sealed staging |
| Real full-run recovery has been demonstrated | **Unsupported and explicitly disclaimed** |
| Full checkpoint publication/cleanup is operationally proven | **Unsupported; no checkpoint manifest existed at 660 updates** |
| Initial scientific progress/diagnostic claims changed | **No change**; the initial A–F verdict remains in force |

## Follow-up conclusion

Commit `6850b6e` is a real improvement. It closes the cached-file trust hole and gives operators a correct recovery selection path under normal interpreter settings. The remaining `assert` dependence and the gap between plan-time verification and launch prevent upgrading operational integrity to PASS. The overall experiment verdict stays **WARN**, while the specific cached-download finding is resolved and the resume-filename finding is reduced from an unguarded risk to a documented, conditionally safe manual procedure.

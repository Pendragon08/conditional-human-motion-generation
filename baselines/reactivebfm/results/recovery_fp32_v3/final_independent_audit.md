# ReactiveBFM FP32 v3 Final Independent Audit

**Date:** 2026-10-22  
**Auditor:** fresh Codex reviewer, read-only  
**Review class:** same-family, provisional  
**Audited commit:** `bfbd1d18793a21fae8e46b911462af2663325ac4`  
**Overall verdict:** **WARN**  
**Integrity status:** **warn**

The v3 recovery training core is operational for the bounded evidence inspected. The independently executed two-rank witness completed eight resumed updates; the saved update-10,041 checkpoint has matching model/optimizer hashes, finite model, EMA, and optimizer tensors, and all 365 AdamW state counters equal 10,041; and the latest captured health rows show both ranks at 10,348 with zero skips and equal step-call/validated-update counters. I found no critical defect in the final v3 training state machine or final backup code.

This is not a full-schedule or root-cause PASS. The latest validated live point, update 10,348, has not reached the original run's first recorded sharp rise at update 10,501, much less its divergence/stall region around updates 11,401–11,503 or the 1,000,000-update target. The large checkpoint upload, GitHub verification, and newly hash-bound nine-trial evaluation remain pending. No paper-performance claim is supported.

## Proven findings

### 1. Original counter drift was a control-flow defect

The upstream FP32 optimizer path returns `False` when pre- or post-clipping norms are non-finite (`upstream/reactivebfm/model/motion_planner/objectives/diffusion/fp16_util.py:220-235`). The scheduled-forcing loop ignored that return value, updated EMA after the call, advanced LR, and incremented the loop step (`upstream/reactivebfm/train/scheduled_forcing_loop.py:607-613,668`). A primitive-loss rejection also advanced LR and the loop step without an optimizer update (`scheduled_forcing_loop.py:576-590,607-613,668`). Meanwhile the experiment wrapper's optimizer counter increments only through the optimizer's post-step hook (`scripts/train_full.py:86-92`).

That combination directly explains why loop steps could advance while true optimizer updates stalled. The raw failure record shows optimizer updates 10,501 at official step 10,500, then only 11,495 at official step 11,500, and a fixed 11,503 updates across official steps 11,800–12,000 (`results/full_training_20261021/monitor_20261022_145249/manual_anomalies_20261022_1451.json:41-56,122-182`). It also records 566 non-finite warnings (`manual_anomalies_20261022_1451.json:185`).

The initiating cause of the gradient explosion is **not proven**. Gradients rose from 6.49 at update 10,101 to 188.52 at 10,401, 8,751.63 at 10,501, 783,660.69 at 10,601, and 2.44e13 at 11,401 while post-clipping norms stayed near one (`manual_anomalies_20261022_1451.json:5-56,122-128`). FP32 computation and FP64 norm/clipping reduce risk and make failure observable; they do not establish which data/model/dynamics condition initiated that growth.

### 2. V3 update, DDP, EMA, and failure ordering is correct for the audited path

`numerical_guard.py` computes parameter and gradient norms in FP64 and clips with one scalar coefficient (`scripts/numerical_guard.py:26-47`). It makes pre-step and post-step decisions collective across ranks (`numerical_guard.py:18-23,103-145`). After `optimizer.step()`, it checks every trainable parameter and every tensor in every optimizer state; a bad state aborts before the validated-update counter, EMA, LR advance, or save (`numerical_guard.py:135-145,211-229`). EMA is checked before and after its update and runs only after a validated optimizer step (`numerical_guard.py:211-218`). The watchdog records the attempt before LR annealing, and the early checkpoint occurs only after a validated step and EMA (`numerical_guard.py:220-229`).

The guard separately records optimizer-step calls and validated optimizer updates (`numerical_guard.py:68-72,135-143,147-180`). This fixes the prior ambiguity between loop attempts and effective updates. Its DDP collective decision means a non-finite rank cannot silently diverge from the other rank.

The remaining counter limitation is a proof gap, not an observed defect: `optimizer_steps()` scans AdamW's per-parameter counters at initialization (`numerical_guard.py:50-54,68`), while later live health rows increment guard counters after each validated step rather than re-scanning all AdamW counters on every update (`numerical_guard.py:135-143`). The saved update-10,041 CPU inspection closes this gap at that checkpoint: 365 optimizer states have the single step value 10,041 and all optimizer tensors are finite (`checkpoint_cpu_v3.json:31-39`). Live update 10,348 is supported by call/validation counters, not a second full Adam-state deserialization.

### 3. Resume semantics preserve model, EMA, and AdamW as claimed

The upstream loader restores model and `model_avg`, synchronizes both, derives the resume step from the filename, and loads the matching optimizer file (`scheduled_forcing_loop.py:722-783`). Its resume convention increments the loop-local step before adding the filename step (`scheduled_forcing_loop.py:725-730`); therefore a checkpoint named at official step `s` can legitimately contain `s+1` optimizer updates. `train_repaired.py` rejects any manifest that does not use exactly that convention, rejects count mismatch, requires the exact model/optimizer pair, re-hashes both files, and verifies the requested resume path (`scripts/train_repaired.py:24-48`).

The document witness recorded both model and EMA loading and optimizer loading (`document_witness_v3/raw_outputs/supervisor_stdout.txt:46-48`). Its rank-0 and rank-1 health records each ended at update 10,029 after eight attempts, with zero skips, `optimizer_step_calls == validated_optimizer_updates == optimizer_updates`, and step/EMA validation true (`document_witness_v3/raw_outputs/health_rank0.json:5-16`; `health_rank1.json:5-16`). The invocation exited zero (`document_witness_v3/invocation_result.json:36-41`).

The production checkpoint manifest identifies official step 10,040/update 10,041 and exact pair hashes (`trace_validation_v3/trace_receipt.json:3-30`). Independent CPU deserialization found 388 finite model tensors, 388 finite EMA tensors, 365 finite optimizer states, and the sole AdamW step value 10,041 (`checkpoint_cpu_v3.json:19-39`). The latest bounded snapshot records both ranks at update 10,348, 327 resumed updates, zero skips, finite norms, and validated step/EMA state (`current_v3_snapshot.json:237-286`), with the training processes still running (`current_v3_snapshot.json:289-295`).

Exact RNG and data-iterator replay is explicitly false (`configs/recovery_fp32_v3.json:36`). This is a valid state continuation, not an exact trajectory replay.

### 4. Source and output isolation are adequate

The launcher checks six recovery sources plus the complete 117-file upstream Python tree, CPU-inspects the resume checkpoint, and creates an absent output directory (`scripts/launch_recovery.py:19-54`). The current recovery config pins the initial optimizer count to 10,021 and names the fresh `training_fp32_v3` output (`configs/recovery_fp32_v3.json:24-45`). I independently re-hashed all six local files and matched the archived source pins and upstream-tree digest `cc0d1e8b...`; the actual archived status records identify ranks 0 and 1 with world size two.

One terminal metadata limitation remains: each guard health row hardcodes `full_training_completed: false` (`scripts/numerical_guard.py:155-170`), and the launcher's terminal rewrite copies process status and exit code but does not update that field (`scripts/launch_recovery.py:61-71`). A future completed run must therefore use the rank status files plus `supervisor_exit.json` for completion, rather than the health-file completion flag. This does not affect optimizer/EMA behavior.

## Backup and evidence preservation

The original backup implementation remains a historical **FAIL**. The worker identified by source SHA `104295a1ee84523ef2b4c15b7955aa89fa4032f33252ba576184650835441594` could evaluate before the checkpoint transfer and remote digest validation. Its nine evaluation receipts are preserved under `full_training_v3` as unbound/provisional; `backup_rebind_v3.json:3-7` records the correction and fresh `full_training_v3_verified` destination. These records must never be promoted into verified results.

The final backup source SHA is `e1a13bfdd8fd64647afdf61a9219b0f1bebee71637d7454e30175676a34ce4ee`. It now downloads and verifies both manifest members before trace capture or evaluation (`scripts/backup_full_run.py:368-402`). Each evaluation uses a digest-bearing label, skips an orphan local receipt or pre-existing remote destination into a fresh retry directory, checks the remote model bytes immediately before and after the trial, and binds the full model identity to the exact receipt digest (`backup_full_run.py:144-226`). The regression test simulates a model change during a trial, restoration, and retry, then proves the orphan remains unbound and all nine retries use fresh destinations (`scripts/test_backup_full_run.py:143-169`).

Trace capture now validates configuration identity, required source pins, CPU checkpoint inspection, rank/world-size status metadata, and every selected health row's embedded rank before creating an archive (`backup_full_run.py:269-307`). It then embeds the exact checkpoint manifest and per-member digests, reopens the frozen archive, revalidates mandatory metadata, embedded ranks, endpoint counters, step validation, and EMA validation (`backup_full_run.py:308-336`). Tests reject copied rank-0 data under rank 1 and empty provenance (`test_backup_full_run.py:115-135`).

A transient test failure caused by a global placeholder replacement was preserved in `backup_worker_provenance_test_failure.json:2-5`. The unique `__MANIFEST_LITERAL__` placeholder fixed it (`backup_full_run.py:276-336`). The final receipt binds unchanged code/test hashes before and after 16 passing tests (`backup_worker_final_tests.json:2-18`). The actual frozen u10,041 archive passed the strengthened schema and embedded-rank validation (`trace_schema_validation_v3.json:2-5`), and worker PID 41616 started with the final source hash (`full_training_v3_verified/worker_stdout_provenance_v3.txt:1-2`).

I independently opened `training_trace_000010040.tar.gz`, recomputed its SHA-256 `2cdf5455d7cc44b9a22692621233bb5880447dc8b1f666ffab5fcbcdfd221b9d`, checked every listed member size/hash, and confirmed there are exactly eight proof-listed members plus `trace_integrity.json`. Rank 0 and rank 1 each contain 20 consecutive rows from updates 10,022 through 10,041, with embedded ranks 0 and 1 respectively, zero skips, validated steps/EMA, and equal call/validated/update counts. This validates the existing checkpoint-scoped trace despite the earlier generic verifier gap.

The completed large backup is still pending: the latest worker state has no processed checkpoint receipt yet, so model/optimizer transfer, GitHub server-side digest verification, evidence package, and nine newly bound simulation receipts are not certified by this audit. The backup worker may continue; this report does not claim completion.

## Experiment-integrity checks

| Check | Status | Finding |
|---|---|---|
| A. Ground-truth provenance | PASS | No dataset-ground-truth performance claim is made. The closed-loop runs are explicitly diagnostic proxies; the generated report states that the height threshold is not the paper success rate (`backup_full_run.py:248-260`). |
| B. Score normalization | PASS | No metric is divided by a statistic of the model's own output. The reported root-height threshold remains a raw simulation proxy (`backup_full_run.py:248-260`). |
| C. Result-file existence | WARN | Witness, checkpoint, trace, schema-validation, and source-bound test records exist and match. Full verified upload/evaluation receipts are pending; original unbound receipts remain provisional. |
| D. Dead-code/runtime path | PASS | The numerical guard executed on two ranks, the actual trace passed the strengthened validator, and the final backup tests passed. The final full backup/evaluation pipeline has not completed end to end. |
| E. Scope | WARN | Evidence covers 327 resumed updates at the latest snapshot and one saved checkpoint. It has not crossed the original anomaly region or completed 1,000,000 updates. |
| F. Evaluation type | `simulation_only` plus engineering validation | MuJoCo diagnostics are simulation-only proxies. Witness, counter, finiteness, checkpoint, and provenance checks are engineering validation, not `real_gt` benchmark evaluation. |

## Claim impact

- **Supported:** “The v3 code resumed the verified model/EMA/AdamW state and executed bounded two-rank FP32 updates with collective numerical guards.”
- **Supported:** “The update-10,041 saved checkpoint pair matches its manifest and contains finite model, EMA, and optimizer state with all 365 AdamW counters at 10,041.”
- **Supported with qualifier:** “The recovery is operational.” Qualify this as bounded through the inspected witness/checkpoint/live snapshot and pending full backup completion.
- **Unsupported:** “The gradient-explosion root cause has been identified or eliminated.”
- **Unsupported:** “The run has passed the prior failure region, completed the schedule, exactly replayed RNG/data order, reproduced paper metrics, or completed the verified large backup/evaluation.”

## Required evidence before broader claims

1. For a checkpoint-backup completion claim, require the worker's processed receipt, complete model and optimizer bytes, GitHub server-side digest matches, the frozen evidence package, and all nine new checkpoint-identity bindings.
2. For a stability claim, require validated health and saved-checkpoint evidence beyond the prior update-11,503 stall region; for a completion claim, require both terminal rank status files and `supervisor_exit.json`.
3. Treat the live guard counters as operational evidence and use periodic saved-checkpoint AdamW counter inspection when a stronger counter proof is needed.

No training process or artifact was modified by this audit. Only this audit report and its JSON companion were written.

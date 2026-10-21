# Experiment Audit Report — Initial Frozen Review

**Date:** 2026-10-22  
**Auditor:** Codex GPT-5.6-Sol ultra, fresh same-family reviewer, read-only, provisional  
**Project:** ReactiveBFM full-training and closed-loop reproduction  
**Audited commit:** `104393d48be438e923a30f3e42c466b861fb8639`  
**Acceptance status:** provisional (`review_independence: same-family`)  

## Overall Verdict: WARN

## Integrity Status: warn

The existing progress claims are numerically consistent with the frozen primary artifacts. I found no unlabelled prediction-derived ground truth, no prediction-based score denominator, and no fabricated completed run or paper benchmark. The warning is driven by two different limits: the scientific evidence is still a small, explicitly labelled progress/diagnostic set, and the audited checkpoint/backup control plane has material crash-recovery and state-validation gaps that have not yet been exercised by a real full-run checkpoint.

This report is frozen to commit `104393d48be438e923a30f3e42c466b861fb8639`. Working-tree revisions made after that commit are outside this initial verdict and require a separate follow-up review.

## Audit boundary and method

- I read every requested training, evaluation, feedback, backup, test, runtime, configuration, documentation, tracker, and upstream deployment file line by line. I also followed the metric call chain into `losses.py`, `flow_matching_smooth.py`, and the composite dataset implementation.
- I inspected the JSON/text evidence under `baselines/reactivebfm/results/full_training_20261021/`, the acquisition/data audits, the compute ledger, and the external `closed_loop_simclock_v1.tar.gz` directly.
- Bounded CPU checks parsed all 59 JSON files and both JSONL files in the requested results tree, validated the tar/gzip structure and member hashes, checked both NPZ ZIP CRCs, and recomputed the recorded feedback invariants from the archived arrays.
- I did not contact GitHub, the training server, or any other external service. Remote publication claims are therefore checked against preserved local receipts, not against current remote state.
- A read-only local observation at 2026-10-22 00:52:50+08 found the backup worker recording both ranks at 530 updates, with no checkpoint manifest yet (`D:/具身智能/ReactiveBFM复现_20261021/full_training_20261022/worker_v2_stdout.txt:6`). The report's 270-update claim remains correct for its timestamped frozen snapshot.

## A. Ground Truth Provenance: PASS

The training target comes from released motion records, not from the model's own predictions. The scheduled-forcing loop slices `gt_pred` from the loaded `motion_full` tensor and passes it to the flow objective (`baselines/reactivebfm/upstream/reactivebfm/train/scheduled_forcing_loop.py:463-496,556-560`). The objective builds its velocity target as `noise - x_start` and compares reconstructed motion with `x_start` (`baselines/reactivebfm/upstream/reactivebfm/model/motion_planner/objectives/flow/flow_matching_smooth.py:145-171,233-240`). The acquisition receipt identifies the source as the official ReactiveBFM release and records 24/24 files and 870,700,765 bytes (`baselines/reactivebfm/results/acquisition_v1/acquisition.json:3-6`). The numeric audit reports zero key and exact-motion-hash overlap across train/validation/test splits for all three datasets (`baselines/reactivebfm/results/data_audit_v1.json:14-23,137-146,193-202`).

The non-dataset references are labelled correctly:

- The tracker hold uses a scripted static A-pose and labels itself a diagnostic rather than a generative or paper metric (`baselines/reactivebfm/scripts/check_tracker_hold.py:28-50`; `baselines/reactivebfm/results/full_training_20261021/tracker_hold_v1_stdout.txt:1`).
- Feedback verification reconstructs prefixes from recorded MuJoCo states and explicitly declares `evaluation_type: simulation_only` and no motion-quality or paper-performance claim (`baselines/reactivebfm/scripts/verify_feedback.py:19-65`; `baselines/reactivebfm/results/full_training_20261021/closed_loop_simclock_v1/feedback_verification.json:2-10`).
- Recomputation parity compares two executions of the same model/batch and explicitly limits itself to implementation parity (`baselines/reactivebfm/scripts/check_recomputation.py:32-53`).
- The report says the old 101-update model, height threshold, and static reference are not ground truth or paper performance (`baselines/reactivebfm/reports/ReactiveBFM完整训练与闭环进展报告_20261022.md:53-58`).

Official paper benchmarks and goal-reaching data remain unavailable/TODO (`baselines/reactivebfm/results/EXPERIMENT_TRACKER.json:15,19`; report `:93`). That absence is disclosed and does not convert simulator state or model output into ground truth.

## B. Score Normalization: PASS

The audited loss denominator depends on the count of valid target entries in the mask, not on prediction statistics (`baselines/reactivebfm/upstream/reactivebfm/utils/training/losses.py:17-31`). The reported simulator values are raw counts, seconds, metres, RMSE values, and componentwise absolute errors. No evaluated score divides by a prediction-derived maximum, norm, or variance.

The suspiciously small values are implementation checks with direct explanations rather than performance scores. The recomputation witness compares 365 nonzero gradient tensors and reports maximum absolute error 0 for two deterministic FP32 executions (`baselines/reactivebfm/results/full_training_20261021/document_witness_v1/witness.json:243-258`). The feedback verifier reports linear error `5.960466498144967e-08`, quaternion-component error `4.042602706477716e-07`, and anchor-time error `1.021405182655144e-14` while labelling the check as wiring consistency (`feedback_verification.json:3-10`). Independent recomputation from the archived arrays reproduced those invariants within the verifier thresholds.

The three released Mean/Std pairs are input normalizers, not score denominators. Each composite component loads its own `Mean.npy` and `Std.npy` (`baselines/reactivebfm/upstream/reactivebfm/data/datasets/dataset.py:37-48,133-160`; `motion_text.py:376-391`). The data audit shows that the released Std arrays differ from a simple recomputation on the current training splits by as much as 1.0 (`data_audit_v1.json:28-31,151-154,207-210`). This is a comparability limitation handled under scope, not denominator fraud.

The upstream online generator clips generated reference root height to `[0.60, 0.85]` (`baselines/reactivebfm/upstream/deploy/run_online_generation.py:61-64,474-478`), but the reported fall proxy uses actual simulator root height, and the old model fails it. The clip therefore does not explain away or inflate the reported outcome.

## C. Result File Existence and Claim Matching: WARN

All material numerical claims in the timestamped report match primary result keys:

- The frozen snapshot has two ranks at `optimizer_updates=270`, `official_step=269`, `full_training_completed=false`, no checkpoint manifests, global batch 1,024, 107,045,924 trainable parameters, and component lengths 1,403/6,268/687 (`baselines/reactivebfm/results/full_training_20261021/live_snapshot_20261022_0036/server_snapshot.json:1`). This is 270 shared DDP optimizer updates, not 540. The target is 1,000,000, so the snapshot is 0.027% complete.
- The recorded 1,155.66 seconds yields 4.2802 seconds/update and about 49.53 days remaining under a constant-rate extrapolation, matching the rounded report values (`report:89`). The roughly 10.69 GiB peak is per rank; the wording at report line 89 should say so explicitly.
- The acquisition and server manifests contain the reported 24 matching files and 870,700,765 bytes (`acquisition.json:3-6`; `baselines/reactivebfm/results/full_training_20261021/server_data_verified.json:2-123`). The filtered component counts total 8,358 and match the runtime receipt (`server_snapshot.json:1`; report `:18-25`).
- The four-update witness records both ranks completing four updates with global batch 1,024, 365 nonzero gradients, and zero maximum parity error (`document_witness_v1/witness.json:118-127,185-194,243-258`).
- Static hold records 400 controls, eight simulated seconds, and minimum root height 0.788038 m (`tracker_hold_v1_stdout.txt:1`).
- The old-checkpoint loop records 100 policy calls, 400 MuJoCo integrations, two simulated seconds, 12 planner calls, 12.246044 seconds wall time, minimum root height 0.082927 m, and `upright=false` (`closed_loop_simclock_v1/receipt.json:2-15,201-209`). Feedback records 100 states and the exact small consistency errors cited above (`feedback_verification.json:2-10`).
- Six backup tests are recorded as passing (`baselines/reactivebfm/results/full_training_20261021/backup_worker_unit_v3_stderr.txt:1-5`). Their source explicitly says they use dummy bytes and are not scientific results (`baselines/reactivebfm/scripts/test_backup_full_run.py:1-12`).
- The tracker is consistent with the snapshot: `full_training=IN_PROGRESS`, `paper_benchmarks=TODO`, and full checkpoint backup waiting for the first 1,000-update checkpoint (`baselines/reactivebfm/results/EXPERIMENT_TRACKER.json:13-19`).

The status is WARN rather than PASS for evidence-bound claims around the numbers:

1. No full-training checkpoint existed at the frozen snapshot or at the later 530-update observation. The planned milestone evaluations are future work, not results. External release availability was not independently queried under the review constraint.
2. The compute ledger is stale: it says the full-runtime witness is awaiting independent invocation (`.aris/compute/reactivebfm-20261021.md:18`), while the later primary witness records one successful invocation and 24 checks passed (`document_witness_v1/witness.md:3-20`). The witness is the later authoritative artifact, but the ledger should be reconciled.
3. “Dual RTX A6000” and uninterrupted continuity of unrelated jobs are not fully established by the frozen snapshot. The witness itself says unrelated-job PID continuity was not independently audited (`document_witness_v1/witness.json:364-372`).
4. README's present-tense “is running” is time-sensitive (`baselines/reactivebfm/README.md:3`). It was supported by the frozen snapshot and by the later local worker observation, but should be attached to an observation time.

## D. Dead Code Detection: PASS

The scientific metrics in the current claims are on executed paths and have recorded outputs.

- `masked_motion_metrics` is called by the flow objective (`flow_matching_smooth.py:233-240`). The scheduled loop accumulates every returned term, reduces it across ranks, and sends it to the reporting platform (`scheduled_forcing_loop.py:596-658`). The wrapper's platform writes those values to `metrics.jsonl` (`baselines/reactivebfm/scripts/train_full.py:69-76`).
- `verify_feedback.py` executes its structural assertions, reconstructs interpolated linear/quaternion values, applies thresholds, and writes the JSON receipt (`verify_feedback.py:19-74`). Its output exists and was independently recomputed from the tar archive.
- The recomputation and tracker-hold scripts execute their calculations and emit receipts (`check_recomputation.py:32-54`; `check_tracker_hold.py:28-50`). `full_runtime_witness.py` runs parity, DDP, hold, and loop checks in sequence and explicitly says this is not full training or a paper reproduction (`baselines/reactivebfm/scripts/full_runtime_witness.py:16-39`).

The full-checkpoint 3-prompt × 3-seed evaluation path has not run because no eligible checkpoint exists. It is called by the backup loop when a milestone manifest arrives (`baselines/reactivebfm/scripts/backup_full_run.py:109-170,227-256`) and no metric from that future path is claimed now, so its lack of full-run output is a coverage limitation rather than dead-code misrepresentation.

## E. Scope Assessment: WARN

The report generally states the scope correctly, but the evidence cannot support broad reproduction or performance claims:

- Full training is incomplete: 270 updates in the frozen report, 530 in the later local observation, versus a 1,000,000-update target and 400,000-step warmup (`baselines/reactivebfm/configs/full_training_v1.json:4-5`; report `:3,9,89`). No new full-run checkpoint or new-model evaluation existed.
- The document witness covers four updates per rank. Recomputation parity covers one real-data FP32 batch. Neither establishes BF16 or whole-training equivalence (`document_witness_v1/witness.json:364-372`).
- The only generated closed-loop result uses the old single-Kungfu 101-update pilot, one prompt (`stand still`), seed 42, and 100 controls. The planned milestone matrix is three prompts × three seeds × 400 controls and has not run (`report:54-58,71`).
- Static hold proves a scripted reference can be tracked in simulation; it says nothing about generated motion quality. Closed-loop completion proves execution, while the actual height proxy shows a fall. No language alignment, agility, robustness, goal-reaching, contact, real-time, human, or real-robot result exists (`report:47-58,91-93`).
- The current training uses the released datasets as a runtime concatenation with separate normalizers because the public downloads do not contain the default premerged directory/global normalizer (`report:27`; `dataset.py:37-63,133-160`). The released normalizers do not equal a simple train-split recomputation. This prevents an unqualified claim of numerical equivalence to an unpublished premerged distribution.
- The local run uses two ranks with local batch 512, BF16, supplementary non-reentrant activation recomputation, Python 3.10/Torch 2.5.1, and per-process memory caps. Those differ from the documented eight-rank execution and can change numerical trajectories even with global batch 1,024 (`full_training_v1.json:6-14`; report `:31-41`).
- Configuration field `seed=42` is a base seed. Runtime code applies `seed + rank`, so rank RNG seeds are 42 and 43; the distributed sampler retains base seed 42 (`baselines/reactivebfm/upstream/reactivebfm/train/run_utils.py:24-28`; `data/datasets/dataset.py:177-187`).

The current report already rejects the unsupported broad claims at lines 91-93. Any downstream reuse must preserve those qualifiers.

## F. Evaluation Type: PASS

The artifacts can be classified without hidden relabelling:

| Artifact or claim | Evaluation type | Claim ceiling |
|---|---|---|
| Released-motion training loss and small dataset-target pilot checks | `real_gt` | Training/limited objective diagnostics only; no held-out paper benchmark |
| Static scripted A-pose reference | `synthetic_proxy` + `simulation_only` | Tracker/control diagnostic only |
| Recomputation gradient parity | `self_supervised_proxy` | Implementation equivalence for one deterministic FP32 batch |
| State-to-prefix feedback verification | `self_supervised_proxy` + `simulation_only` | Recorded wiring/interpolation consistency only |
| MuJoCo generated closed loop | `simulation_only` | Execution evidence; old pilot falls; no paper success metric |
| Human or real-robot evaluation | absent | No claim supported |

No artifact is presented as `human_eval`, and no model-derived reference is presented as `real_gt`.

## Operational Integrity Review: WARN

### Positive controls

- Status files and ready manifests use temporary files followed by atomic replacement (`train_full.py:43-48,104-113`).
- The worker verifies checkpoint filename, directory, stated size/hash, downloaded bytes, and GitHub-reported digest before marking a step processed (`backup_full_run.py:81-107,227-256`). Upload does not use a clobber option; a same-name/different-hash asset fails (`:85-94`).
- Evidence tarballs are frozen once, so retries reuse the same bytes (`backup_full_run.py:34-43`; `test_backup_full_run.py:70-78`).
- Old server copies are candidates only after a newer manifest and processed publication; cleanup rechecks remote path, name, size, and hash (`backup_full_run.py:193-219`). Local removal similarly checks containment and hashes (`:259-272`).
- The closed-loop archive is valid and internally consistent: 194,241 bytes, SHA-256 `ef14b3138b063766f124c5793243bcea90a77f0e18b823a3d8244cd7854c59b2`, nine safe members, no links/path traversal, valid NPZ CRCs, 12 prefixes of shape `(25,36)`, and 100 recorded states. Its hashes match `local_artifact_manifest.json:3-5`, `closed_loop_github_verified.json:3-12`, and the feedback input hash fields (`feedback_verification.json:11-16`).

### Material residual gaps

1. **Checkpoint files are not atomically published, and resume is not manifest-gated.** The wrapper calls the upstream save directly and creates the atomic ready manifest only afterward (`train_full.py:96-113`). Upstream writes model and optimizer directly to final filenames (`scheduled_forcing_loop.py:812-823`). Resume chooses the highest `model*.pt` in the save directory ahead of an explicitly supplied checkpoint and does not validate a ready manifest or file hash (`:725-733,765-783,806-810`). A crash can leave a truncated, final-named highest model; a missing optimizer file silently leaves a fresh optimizer. This is the highest-priority risk.

2. **Already-present local evaluation files can be trusted without a digest.** In the audited commit, `download(..., expected=None)` returns immediately for any existing local path (`backup_full_run.py:96-107`). Evaluation receipts, feedback JSON, and archives use this no-expected-hash path (`:127-169`). A stale or changed local file can therefore be parsed and frozen into evidence on retry. The current archive independently passed, so this is a future milestone risk rather than evidence that the audited loop result is corrupt.

3. **Deletion safety depends on Python `assert`.** Critical remote and local containment/name/hash checks use `assert` (`backup_full_run.py:206-215,268-271`), and the tests expect `AssertionError` (`test_backup_full_run.py:53-68`). These guards disappear under optimized Python. The observed worker launch used Python without `-O`, so assertions were active in this run, but permanent deletion guards should raise explicit exceptions.

4. **Completion can be inferred from an incomplete or stale rank set.** Status parse errors are silently omitted (`backup_full_run.py:69-76`), while completion requires only a nonempty list whose remaining entries all say complete (`:278-281`). The worker does not require exact ranks `{0,1}`, matching world size/source/config hashes, or fresh timestamps. One surviving completion record plus a processed final manifest could terminate the worker with the other rank absent/unreadable.

5. **Process and failure detection are weak.** Liveness matches any command containing `/code/scripts/train_full.py`, not the configured output/PID (`backup_full_run.py:78`). There is no stale-update/hang detector. Explicit `FAILED` creates a failure asset, but an absent process with incomplete statuses logs locally and stops without restart/resume (`:282-287`). If statuses are empty/corrupt and the process is absent, neither terminal branch fires. The launcher exit-code file is not inspected.

6. **Asset enumeration has a scale risk.** `assets()` assumes the release object's embedded asset list is exhaustive and implements no pagination (`backup_full_run.py:81-83`). The planned schedule can create hundreds of assets.

7. **The real checkpoint path is unexercised.** At the frozen 270-update snapshot and later 530-update observation, there was no manifest and no processed checkpoint. The recorded six tests use synthesized bytes and an in-process remote substitute (`test_backup_full_run.py:15-95`). They do not demonstrate a real model/optimizer download, immutable upload, cleanup, or resume.

8. **Long-running recovery is manual and host-dependent.** The worker runs on a local Windows machine. On `FAILED` or process absence it exits rather than orchestrating a verified resume (`backup_full_run.py:282-287`). The report accurately discloses that sleep, shutdown, disconnection, the three-hour distributed timeout, and non-bitwise resume remain limitations (`report:77`). Its sleep-inhibition wording should say the worker *requests* inhibition because the API return value is not checked (`backup_full_run.py:302-310`).

The observed worker logged source hash `4486aae1747e8f26ad85aa5f3f187d84023fe15cd25650cb6131534aa246df97` (`live_snapshot_20261022_0036/worker_v2_stdout.txt:1`), which exactly matches `backup_full_run.py` at the audited commit. This ties the operational findings to the implementation that was actually running at the observation time.

## Action Items

1. Before relying on the first checkpoint, write model and optimizer to temporary files, close and hash both, atomically rename them, and publish the ready manifest last. Resume only from a ready manifest whose two files match size/hash. Honor an explicit resume path instead of allowing an unmanifested higher filename to override it, and fail closed when the optimizer file is absent unless model-only recovery is explicitly requested.
2. On every evaluation retry, obtain the remote byte count/hash for receipts, feedback JSON, and archives; verify an existing local copy or quarantine it. Seal locally generated failure receipts and verify the seal before reuse.
3. Replace every deletion-critical `assert` with explicit exceptions and test the guards under optimized Python.
4. Require the exact configured rank set, matching world size, source/config hashes, fresh status timestamps, and a processed final manifest before declaring completion. Bind liveness to the configured run/output or recorded PID; inspect launcher exit status and add stale-progress handling.
5. Publish crash/absence evidence and implement an explicit, manifest-verified resume procedure. Record whether optimizer, RNG, and data-iterator state were restored; keep the existing statement that bitwise replay is not guaranteed.
6. Enumerate release assets with a paginated endpoint and test behavior above one page.
7. Treat the first 1,000-update checkpoint as an operational acceptance test: verify real model/optimizer hashes, immutable upload, stage-evaluation evidence, retention, cleanup safeguards, and a bounded load/resume check before trusting unattended long-term operation.
8. Tighten wording: “base seed 42; per-rank RNG seeds 42/43,” “about 10.69 GiB peak tensor allocation per rank,” “observed running at [timestamp],” and “requests Windows idle-sleep inhibition.” Reconcile the stale compute ledger.
9. Preserve the scope labels until actual checkpoints and official evaluations exist. Report each milestone's prompt/seed/control count and failed trials, and do not convert height/wiring/control-completion proxies into performance claims.

## Claim Impact

| ID | Claim | Impact |
|---|---|---|
| C1 | Official public files were acquired and the three-component full schedule was launched | **Supported**, with the released per-component normalizer and two-rank configuration qualifiers |
| C2 | The full 1M schedule or paper performance has been reproduced | **Unsupported**; also explicitly disclaimed by the report |
| C3 | Both ranks had completed 270 updates at the frozen snapshot | **Supported** as a point-in-time claim; do not sum ranks |
| C4 | Four-update DDP execution and one-batch recomputation parity passed | **Supported**, limited to implementation execution/parity |
| C5 | Static A-pose tracking remained upright for eight simulated seconds | **Supported**, `synthetic_proxy`/`simulation_only` only |
| C6 | The old 101-update model executed a real-feedback MuJoCo loop | **Supported**; it fell and supports no motion-quality success claim |
| C7 | Recorded simulator state was fed back consistently into planner prefixes | **Supported**, self-consistency/wiring only |
| C8 | Future checkpoint backup, cleanup, and recovery are fully robust | **Needs qualification / not yet demonstrated**; material resume, retry, rank-state, and liveness gaps remain |
| C9 | The preserved GitHub release assets remain available now | **Not independently verified** under the no-external-service constraint; local publication receipts are internally consistent |
| C10 | Both GPUs were specifically A6000s and unrelated jobs remained uninterrupted | **Needs qualifier**; frozen evidence does not independently establish both facts |

No current quantitative scientific claim is downgraded to fraudulent or fabricated. Claims about completion, paper-level performance, real-time behavior, real robots, and unattended recovery remain outside the evidence.

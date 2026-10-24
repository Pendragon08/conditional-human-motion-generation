# Experiment integrity audit — PhyGile completion_v1

**Overall verdict: WARN.** Acceptance is provisional; review independence is same-family. No blocking integrity defect was found. The package supports its narrowly stated engineering-interface claims, but two nonblocking issues prevent PASS: MPJVE is computed and then discarded by aggregation/promotion, and the completion raw-results archive lacks a completion-level published size/SHA entry despite the report’s checksum wording.

I read all 54 manifest files, current critical source, both executed-source archives, the raw-results archive, original pilot inputs, receipts, witnesses, checkpoint accounting, and provenance. I ran no experiment and changed no primary artifact, implementation, or threshold. All manifest byte counts/hashes matched. Both source archives matched their internal manifests; current critical source matched the recovery archive. The raw archive contained 418 files: 381 local counterparts matched byte-for-byte and 37 archive-only training-log files were present.

## A — Reference and target provenance: PASS

Dataset records are explicitly train/test, video-reconstructed and GMR-retargeted, and unpaired at results/data_provenance.json:12-14, 31-33, 93-95; source-group overlap is empty at results/data_provenance.json:1046. Curriculum output is simulation_only with a dataset-derived kinematic reference at results/completion_v1/document_run/curriculum/output/evaluation.json:5-7. The evaluator states the reference is not model output at cache/completion_audit_canonical/cache/closed_loop_evaluate.py:123-132.

Generated candidates are simulation inputs, not independent truth: FK reconstructs their targets at reimplementation/isaac_prefix_adapter.py:74-90. Recovery labels the generator untrained and zero/unconditional text at results/completion_v1/closure_recovery/output/verification.json:4-16. The accepted held-pose fixture is self_supervised_proxy and “not dataset truth or generated semantics” at the same file:109-114. The terminal pose is nominal, not author-verified, at the same file:417-420. No model-derived truth is passed as real measured ground truth.

## B — Normalization, denominators, and success: PASS

The worker saves raw post-step/pre-reset arrays and computes MPJPE in metres, MPJAE in radians, and MPJVE in radians/s at cache/completion_audit_canonical/cache/closed_loop_evaluate.py:100-132. Full-clip success requires the expected frame count and no failure termination at the same file:111-119. Aggregation uses equal weight per requested file and pairs termination-censored errors with success and episode lengths at reimplementation/curriculum_evaluator.py:46-64.

Prefix acceptance uses raw metre error, rejects incomplete or terminated executions, checks candidate identity and terminal windows, and never normalizes by its own prediction at reimplementation/physics_prefix.py:75-120. Codec normalization is train_only over 20 sequences at results/completion_v1/document_run/codec/output/verification.json:3-9; independent reconstruction reproduced mean/std exactly. No self-normalization, missing raw metric, inflated success definition, or train/test normalization leakage was found.

## C — Claims, values, statuses, and identities: WARN

Report values at reports/PhyGile课程评估与262D闭环补全报告.md:17-24 match curriculum results at results/completion_v1/document_run/curriculum/output/evaluation.json:20-23, 40-43, 57-60. Four PPO iterations and expected 80 Adam steps are scoped as an integration fixture at results/completion_v1/document_run/training_hook/output/verification.json:2-16; checkpoint optimizer state is 80 at results/completion_v1/checkpoint_accounting.json:16-18, 34-36.

All seven document-run stages completed at results/completion_v1/document_run/pipeline.json:2-101. The failed terminal receipt remains FAILED/1 at results/completion_v1/closure_final/run/receipt.json:200-204; recovery is separately COMPLETED/0 at results/completion_v1/closure_recovery/run/receipt.json:200-204. Manual stops and failures remain explicit at results/completion_v1/development_status_erratum.json:2-13. Canonical and recovery executed-source identities pass at results/completion_v1/executed_source_verification.json:5-9, 72-76.

**Warning C1:** reports/PhyGile课程评估与262D闭环补全报告.md:68 says released large files provide SHA-256, but phygile_completion_raw_results.tar.gz has no completion-level checksum/size entry. Independently computed: 9,533,668 bytes; SHA-256 1c52bd6403bd72b5399395754977e0296306124b239c0dceabc450d00b82da13. This affects documentary packaging, not scientific values or run status.

## D — Evaluation control flow: WARN

Active environments are captured inside termination computation after physics and before reset, then removed after capture at cache/completion_audit_canonical/cache/closed_loop_evaluate.py:80-108. Frame continuity and first-episode completion are asserted at the same file:108-119. Candidate rollout orders physics, failure detection, and capture without reset/resampling at reimplementation/isaac_prefix_adapter.py:101-140. Direct state-write rollback is disabled at the same file:95-99. Rejected attempts restore state, and only accepted actual execution may commit, at reimplementation/physics_prefix.py:130-179. Replay deltas are zero at results/completion_v1/replay_determinism_recovery/comparison.json:2-20.

**Warning D1:** MPJVE is computed/emitted at cache/completion_audit_canonical/cache/closed_loop_evaluate.py:117-132, but reimplementation/curriculum_evaluator.py:46-64 aggregates only MPJPE/MPJAE and its decision at lines 84-102 never reads MPJVE. It is a dead/non-gating metric. No current acceptance claim depends on it. No stale observation, reset contamination, active-mask error, bypass, or rejected-state commit was found.

## E — Actual scope, seed, budget, terminal, and rollback claims: PASS

The report explicitly calls this partial engineering validation, not reproduction of main paper conclusions, at reports/PhyGile课程评估与262D闭环补全报告.md:1-13. The document run pins checkpoint, motion, device, and seed 2026 at results/completion_v1/document_run/pipeline.json:78-98. Recovery records seed 2026, 50 requested frames, two rejected attempts, zero accepted attempts, zero committed frames, and zero PPO adaptation updates at results/completion_v1/closure_recovery/output/verification.json:18-28. Each candidate captured 1/50 frames and failed on incompleteness, termination, and MPJPE at the same file:29-63.

The committed prefix equals the initial prefix at the same file:24-28. Nominal-suffix equality is explicitly not measured standing success at results/completion_v1/closure_recovery/verification.json:2-5. Missing paired training, semantic generation, Stage I/II GMT, learned adaptation, paper-scale results, and hardware validation are excluded at reports/PhyGile课程评估与262D闭环补全报告.md:54-60. Scope and claims match the artifacts.

## F — Evaluation types and recomputation: PASS

Classification: curriculum, expert-mask, training-hook, and long-candidate execution are simulation_only; codec round-trip/FK and the held-pose fixture are self_supervised_proxy. No real_gt or human_eval exists. Generated candidates are not asserted as synthetic truth.

Raw recomputation matched: train 0/8, MPJPE 0.1672734022140503 m, MPJAE 0.3191640377044678 rad; test 3/8, 0.36546245217323303 m, 0.1520146280527115 rad; equal-file aggregate 0.26636792719364166 m, 0.23558933287858963 rad, success 0.1875. Expert-mask 0/2, 0.19820046424865723 m, 0.35598164796829224 rad; training-hook 0/2, 0.12251029163599014 m, 0.23196259140968323 rad.

All 24 codec inputs reproduced maxima 6.593780632613289e-09 m root, 4.127457264857526e-08 rad joint, and 9.62942522164667e-07 m FK at results/completion_v1/document_run/codec/output/verification.json:444-446. Recovery candidate MPJPEs 0.42900891023804066 and 0.4309579676126371 reproduced at results/completion_v1/closure_recovery/output/verification.json:29-63. Rollback/state/action/property deltas were all zero at results/completion_v1/replay_determinism_recovery/comparison.json:2-20.

## Claim impact and actions

Curriculum metrics support only the named simulation fixture. Codec claims support internal proxy consistency, not author equivalence. The untrained diffusion-to-Isaac rejection/rollback interface is supported for seed 2026; generated-motion and terminal-standing success are unsupported, with 0/2 accepted and zero committed frames. Held-pose acceptance supports only interface feedback. Full PhyGile reproduction, semantic quality, learned GMT adaptation, paper performance, and hardware performance remain unsupported and explicitly excluded.

**Blocking defects:** none.

Actions: publish a completion checksum manifest for all three completion archives; aggregate MPJVE and define whether it gates promotion, or label it diagnostic; preserve current qualifiers; obtain a different-family review before final acceptance.

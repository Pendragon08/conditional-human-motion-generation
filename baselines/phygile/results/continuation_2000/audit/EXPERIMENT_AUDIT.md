# Experiment Audit Report

**Date**: 2026-10-17  
**Auditor**: Codex fresh same-family reviewer (read-only semantic review; provisional)  
**Project**: PhyGile released-tracker continuation from 500 to 2000 PPO training iterations  
**Deterministic evidence**: `deterministic_verification.json` (SHA-256 `26936d6959379bef8d26e3c9f5bd1142d506b4d3896d583bdea9a80d4a1ca99e`)

## Overall Verdict: WARN

## Integrity Status: warn

**Blocking defects: none.** The warning is for experimental scope, not evidence integrity. The measured result is an exploratory, single-seed simulator comparison on four training clips and four test clips from one held-out source group. It does not establish a reliable general tracker or a full PhyGile reproduction.

The independent verifier passed all 24 raw NPZ files (16 historical 1/500-iteration evaluations and 8 new 2000-iteration evaluations), all saved metrics/CSVs/summaries, both survival curves, 24 evaluation receipts, the two checkpoint states, and final report hashes (`deterministic_verification.json:4-35`).

## Checks

### A. Ground Truth Provenance: PASS

The reference is transparently identified as a video-reconstructed, GMR-retargeted, FK-derived kinematic proxy rather than measured robot ground truth (`results/data_inventory.json:2`). Provenance records label the sources `video_reconstructed_and_GMR_retargeted` and unpaired (`results/data_provenance.json:13-14`), with no train/test source-group overlap (`results/data_provenance.json:1046`). The report further states that upstream processing may use models, that the reference is independent of the evaluated policy, and that it is not measured robot truth (`reports/PhyGile续训与生成核心报告.md:34`). The historical evaluator phrase “not model output” is correctly narrowed by `docs/accounting_terms_erratum.md:15` to “not output of the evaluated policy.”

### B. Score Normalization: PASS

MPJPE, MPJAE, and MPJVE are computed directly from unnormalized pre-reset reference/policy state differences in physical units (`scripts/evaluate_pilot.py:96-101`; `scripts/summarize_continuation.py:12-17`). No denominator comes from prediction maxima, minima, means, or other model-output statistics. The report discloses the units and absence of output normalization (`reports/PhyGile续训与生成核心报告.md:40`). Independent float64 recomputation differed from saved float32 metrics by at most `3.42e-7` (`deterministic_verification.json:34`).

### C. Result File Existence and Numeric Accuracy: PASS

The three-stage pipeline is recorded `COMPLETED` with zero exit codes (`results/continuation_2000/pipeline.json:2-29`), and the separately recorded state check also completed successfully. Every expected raw, metric, per-step, summary, survival, receipt, and checkpoint file exists. Direct local checkpoint hashes are `afad21ce...` at 500 iterations and `bb067f36...` at 2000 iterations; the final hash matches `results/continuation_2000/checkpoint_verification.json:4` and every final evaluation.

Independent recomputation reproduced the report's fixed-protocol test values: 500 iterations yielded 2/128 successes, 0.904375 s macro duration, and 0.1760626 m censored macro MPJPE; 2000 iterations yielded 26/128, 1.50109375 s, and 0.2750853 m (`deterministic_verification.json:62-100`). The success change is exactly +18.75 percentage points (`deterministic_verification.json:120`). The fixed-denominator survival endpoints reproduce 2/128 and 26/128 (`deterministic_verification.json:131-153`). The final report and its build manifest agree byte-for-byte (`deterministic_verification.json:276-282`).

### D. Called Code and Capture Semantics: PASS

The pipeline calls training, the fixed-checkpoint evaluator, and raw-data summarization in sequence and fails on any nonzero stage (`scripts/run_continuation_pipeline.py:9-24`). Evaluation requires the completed training receipt, selects exactly `model_1999.pt`, and evaluates only the predeclared eight clips (`scripts/evaluate_continuation.py:5-35`). All historical and new receipts bind to the same evaluator/config/termination/command/environment hashes (`deterministic_verification.json:103`).

The evaluator intercepts termination computation, copies the current reference and robot state before reset, retains the terminating row, and removes an environment only after `wrapped.step` returns (`scripts/evaluate_pilot.py:65-87`). It then enforces contiguous first-episode frames and computes all three metrics from the retained raw rows (`scripts/evaluate_pilot.py:88-101`). The summarizer independently recomputes those metrics and episode outcomes (`scripts/summarize_continuation.py:8-40`). The separately invoked state verifier checks the checkpoint iterations, finite tensors, expert changes, and optimizer counters (`scripts/verify_continuation_states.py:10-50`); its successful receipt is a report-build prerequisite (`scripts/build_continuation_report.py:26-35`). No claimed metric is dead code.

### E. Scope Assessment: WARN

The comparison has one training seed, four selected training clips, four test clips from one source group, and 32 simulator first episodes per clip. The 500-iteration test set had already been viewed. Only expert 0 among the 12 experts was active/updated, and the adapted network/reward setting differs from the paper. Two of the four final test clips still have 0/32 full-clip success.

The public report uses appropriate scope language: it calls the comparison exploratory and predeclared (`reports/PhyGile续训与生成核心报告.md:24-28`), distinguishes 32 simulator episodes from independent training seeds (`reports/PhyGile续训与生成核心报告.md:42`), says the policy is not yet a reliable general tracker (`reports/PhyGile续训与生成核心报告.md:13-17`), and rejects a full-method or main-table reproduction claim (`reports/PhyGile续训与生成核心报告.md:92-96`). `EXPERIMENT_TRACKER.md:13` marks reliable general tracking as `NOT_ESTABLISHED`.

### F. Evaluation Type: simulation_only

Classification: **simulation_only using an external reconstructed/retargeted kinematic proxy**. It is not measured-robot `real_gt`, and it is not a `synthetic_proxy` generated by the evaluated policy. This is recorded by the evaluator (`scripts/evaluate_pilot.py:101`), provenance ledger (`results/data_inventory.json:2`), erratum (`docs/accounting_terms_erratum.md:15`), and report (`reports/PhyGile续训与生成核心报告.md:34`).

## Focused Integrity Findings

- **Resume/update accounting: PASS.** The executed log contains exactly 1,500 learning iterations, indexed 500 through 1999 (`results/continuation_2000/train/stdout_stderr.log:268`, `:66224`; `deterministic_verification.json:246`). Direct CPU-only checkpoint inspection found 25 Adam counters at 10,000 in the parent and 40,000 in the final checkpoint, a 30,000-step difference. The legacy `updates` fields are correctly documented as PPO learning iterations (`docs/accounting_terms_erratum.md:5-13`).
- **Final-checkpoint selection: PASS.** The plan and suite require the fixed 2000-iteration checkpoint with no held-out-score selection (`configs/continuation_2000_plan.json:17`; `results/continuation_2000/evaluation_suite.json:4`; `scripts/evaluate_continuation.py:9-23`).
- **Pre-reset versus post-reset capture: PASS.** State is captured inside termination computation before automatic reset, with the terminating state retained (`scripts/evaluate_pilot.py:65-87`).
- **Censoring and survival: PASS with required interpretation.** Error metrics stop at first failure while the failure row remains included; survival always uses all 128 test episodes, with failures remaining failed (`scripts/build_continuation_report.py:78-100`). Because retention windows differ, the MPJPE change is not evidence of improved accuracy; the report states this explicitly (`reports/PhyGile续训与生成核心报告.md:15`, `:40`).

## Claim Impact

- **Observed fixed-protocol success/duration change from 500 to 2000 PPO iterations:** supported as a descriptive simulator result on this fixed data, seed, and protocol.
- **30,000 additional Adam steps across 1,500 added PPO iterations:** supported by direct optimizer-state counters.
- **Improved tracking accuracy from MPJPE:** unsupported; the reported MPJPE increased and uses different censoring windows. The report correctly makes no improvement claim.
- **Reliable general tracking baseline:** not established; the report and tracker say so.
- **Full PhyGile reproduction or validated language-to-motion generation:** unsupported; the report explicitly limits the generator work to numerical/interface/plumbing checks.

## Action Items

- Before making general-performance or statistical claims, add independent training seeds, additional held-out source groups, and a test set not examined during budget selection or prior analysis.
- Continue reporting survival/success beside censored errors and preserve the proxy-reference and `simulation_only` labels.
- Verify the public release page after packaging; external release availability was not used as scientific evidence in this audit.

## Review Status

The semantic verdict is **same-family and provisional**. The raw numeric, hash, receipt, and checkpoint-state checks in `deterministic_verification.json` are deterministic and accepted within their stated scope. This audit does not constitute an external-team replication.

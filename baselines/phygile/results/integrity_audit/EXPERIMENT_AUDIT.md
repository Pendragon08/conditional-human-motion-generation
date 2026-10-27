# Experiment Audit Report

**Date:** 2026-10-17  
**Project:** PhyGile released GMT tracking-code pilot  
**Audit class:** fresh same-family review, read-only, provisional  
**Evidence snapshot:** frozen local final artifacts supplied after training and the 16-run suite completed  
**Overall verdict:** **WARN**  
**Integrity status:** **warn — no fabricated, normalized-to-self, missing, or numerically inconsistent local result found**

The warning reflects the deliberately narrow one-seed pilot, metric-interpretation constraints, and untested dormant curriculum paths. It does not invalidate the reported local results. The experiment report and tracker describe these limits accurately. GitHub publication and uploaded-asset digests are a separate delivery check and are pending; they are not a blocker for this experiment audit.

## Required checks

### A. Reference provenance: PASS

The evaluated reference is a processed dataset artifact, not output from the evaluated policy. `results/data_inventory.json:2` classifies it as video-reconstructed, GMR-retargeted, and FK-derived, and explicitly says it is not a measured robot trajectory. `results/data_provenance.json:470-492` maps converted training samples to processed inputs and upstream source identifiers/hashes; `results/data_provenance.json:1046` records no train/test source-group overlap. `scripts/evaluate_pilot.py:72-76` reads the command reference and robot state into separate arrays, and `scripts/evaluate_pilot.py:101` labels the reference as dataset-derived and not model output.

The release input archive was independently hashed and inspected. All 20 training and 4 test NPZ hashes match the split manifests and inventory; audited arrays are finite; frame rate, frame count, joint/body order, quaternion norms, and empty train/test hash and source-group overlap checks pass. See `results/integrity_audit/trace/archive_verification.json`.

Boundary: the archive contains processed references and manifests, not the upstream videos/raw source NPZs. The preprocessing transformation itself was not replayed. The appropriate classification is **processed dataset-derived kinematic reference**, not measured robot ground truth.

### B. Score normalization: PASS

The three reported errors are direct raw-unit means. `scripts/evaluate_pilot.py:96-101` computes Euclidean body-position error, absolute joint-angle error, and absolute joint-velocity error without dividing by prediction statistics. `scripts/summarize_results.py:11-13` independently recomputes the same quantities and checks them against each run's JSON. The report states the same rule and the absence of self-normalization at `reports/PhyGile复现实验报告.md:69-77`. Policy observation normalization at `results/effective_configuration/agent.yaml:1-5` is unrelated to evaluation score normalization.

### C. Files, numbers, hashes, and claims: PASS for the frozen local experiment

All 16 suite entries exist and have completed exit-zero receipts. For every entry, the audit independently verified:

- the exact raw NPZ key set and finite values;
- zero-based contiguous frame prefixes for every environment;
- a single final termination or timeout flag, with no post-reset rows and no termination/timeout overlap;
- exact episode lengths, step counts, success counts, and error means;
- every per-step CSV row against its raw array values;
- one JSON-identical `PHYGILE_PILOT_RESULT` line and no traceback in each recorded log;
- suite command equality with the recorded receipt;
- exact agreement of all 16 summary rows, the CSV export, and the four group aggregates.

The recomputation is recorded in `results/integrity_audit/trace/final_deterministic_verification.json`. The group results exactly reproduce `results/summary.json:3-42`: final test success is 2/128 = 0.015625, mean duration is 0.904375 s, and macro MPJPE through failure is 0.17606261000037193 m. The displayed values at `reports/PhyGile复现实验报告.md:81-89` and the per-clip table at `reports/PhyGile复现实验报告.md:95-112` agree after stated rounding.

Training is complete rather than inferred from a running receipt: `results/pilot_train20_nostate_seed42/receipt.json:76-80` records `COMPLETED`, exit code 0 and 1327.0169 s elapsed; `results/pilot_train20_nostate_seed42/stdout_stderr.log:22200` reaches iteration 499/500 and line 22238 records 12,288,000 transitions. `results/evaluation_suite.json:1-3,358` records a completed suite, and `results/PIPELINE_DONE.json:1-3` records final pipeline completion.

The two release checkpoints rehash exactly as follows:

- 1 update / internal iteration 0: `059cf77f0c528fd727875f5df25c5ef7ae2a0b5538822c9d902d1dc134b7f158`
- 500 updates / internal iteration 499: `afad21ce7192dc4ec10e04fd2bc7684e9278f901dbffcb053cf45b8b04bda162`

These match `results/checkpoint_verification.json:4-19` and every corresponding evaluation metrics file. Both checkpoint ZIP containers pass CRC checks; their model tensor keys and 8,366,214-element model size agree, all audited tensor payloads are finite, and the expert-change pattern agrees with `results/checkpoint_verification.json:22-72`: all six expert-0 tensors changed and experts 1-11 did not.

The tracker is consistent with the evidence. `EXPERIMENT_TRACKER.md:8-12` marks training, evaluation, recomputation, and checkpoint verification DONE; `EXPERIMENT_TRACKER.md:13` marks reliable tracking `NOT_ACHIEVED` with the correct 2/128 result; `EXPERIMENT_TRACKER.md:14-17` leaves full GMT, generation, closed-loop adaptation, and hardware deployment unclaimed.

Delivery note: `reports/PhyGile复现实验报告.md:127` points to the intended GitHub release. Publication and uploaded digest verification remain pending and must be recorded separately in `results/github_delivery_verification.json`. No uploaded asset was treated as present in this audit.

### D. Metrics actually executed: PASS, with semantic cautions

The evaluated metrics were executed, serialized, printed to the recorded logs, and independently recomputed from all 16 raw bundles. `scripts/evaluate_pilot.py:67-88` captures state through the termination hook before the environment's automatic reset and stops accepting an environment after its first done. `scripts/evaluate_pilot.py:90-101` enforces the frame-prefix invariant, computes full-clip success and errors, and writes the raw and JSON outputs. `scripts/summarize_results.py:7-35` validates each raw bundle and writes the aggregate outputs. The frozen results show those paths ran.

The following interpretations are material and are correctly disclosed in the report:

1. **Automatic reset and frame indexing.** The evaluator forces frame 0 at reset (`scripts/evaluate_pilot.py:46-54`), captures before marking done (`scripts/evaluate_pilot.py:67-86`), and asserts frames `0..L-1` (`scripts/evaluate_pilot.py:90-95`). Independent raw checks found no reset-to-zero discontinuity or post-reset sample.
2. **Success and failure.** A pilot success is exactly 149 captured control intervals with no non-timeout termination (`scripts/evaluate_pilot.py:93-101`). Timeout on the final captured frame counts as success. Failure is survival under the configured anchor-height, tilt, and selected end-effector-height termination checks at `results/effective_configuration/env.yaml:931-981`; it is not a global-MPJPE threshold.
3. **World-frame metric.** Reported MPJPE uses raw world positions (`scripts/evaluate_pilot.py:72-76,96`). Training and body-position termination use the planar-position/yaw-aligned `body_pos_relative_w` (`upstream/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py:403-405,1512-1522`; `upstream/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/terminations.py:41-48`). Reset translation and yaw perturbations at `results/effective_configuration/env.yaml:1035-1075` therefore affect the reported world-frame value. It must not be compared directly with aligned training thresholds or paper metrics.
4. **Censoring and weighting.** Errors pool environment-step rows only through first failure (`scripts/evaluate_pilot.py:65-101`). Early failures contribute fewer samples, so lower error alone need not mean better tracking. `reports/PhyGile复现实验报告.md:75-77,89` retains the required warning and pairs error with success and duration.
5. **Training scalar distinction.** Upstream `Train/success` is based on non-terminated command resampling (`commands.py:423-441,1385-1395`; `my_on_policy_runner.py:400-413,1110-1114`), not the pilot's full-clip completion. `reports/PhyGile复现实验报告.md:77` states this distinction.

The saved `env.yaml` is the training snapshot. Evaluation additionally changes episode length and disables pushes and observation corruption in `scripts/evaluate_pilot.py:32-40`; receipts hash the evaluator and YAML, but `metrics.json` does not embed a resolved evaluation configuration hash. This is a reproducibility caution, not a mismatch in the frozen metrics.

Dormant curriculum code contains risks that did not execute because curriculum and subprocess curriculum evaluation are disabled at `configs/moe_pilot.yaml:77,111`. These include nonfinite tracking errors mapped to zero (`commands.py:810-825`), a subprocess branch with an unbound `min_success_file` and fabricated/ignored success (`my_on_policy_runner.py:655-709,744-746`), worsening metrics satisfying the convergence test (`my_on_policy_runner.py:947-958`), and an unused in-process evaluator that advances frames twice and ignores done (`commands.py:2006-2025`). The report appropriately labels them untested at `reports/PhyGile复现实验报告.md:123`. They do not alter the audited 16-run results, but they require repair and testing before full curriculum claims.

### E. Scope versus evidence: WARN

The evidence is intentionally small: one training seed, 20 training clips from seven source groups, four test clips from one held-out source group, 500 updates, 32 parallel first episodes per checkpoint-clip pair, and one active expert. The exact limits appear at `reports/PhyGile复现实验报告.md:45-61,65` and `results/summary.json:2`. Thirty-two parallel episodes are not independent training seeds.

The report does not inflate this scope. It calls the work a partial tracking-code pilot at `reports/PhyGile复现实验报告.md:3,7-11`, rejects full-PhyGile and paper-table use at line 9, states the generalization limits at lines 115-123, and reports reliable tracking as not achieved at line 89. The status remains WARN because the run cannot establish robust tracking, multi-expert behavior, broad action generalization, uncertainty across training seeds, full Phase A/B behavior, cross-simulator behavior, or hardware performance.

### F. Evaluation classification: PASS — simulation_only

Classification: **simulation_only with processed dataset-derived kinematic references**. `scripts/evaluate_pilot.py:101` records `evaluation_type: simulation_only`; `reports/PhyGile复现实验报告.md:7` explicitly says all results are Isaac Lab simulation and no hardware experiment was performed. It is neither measured-robot `real_gt` nor `synthetic_proxy` derived from the evaluated model.

## Claim impact

| Claim | Audit disposition |
|---|---|
| One 500-update initial-expert training run completed with 12,288,000 transitions | **Supported** |
| The predeclared 16 checkpoint-clip evaluations completed | **Supported** |
| All reported per-run and aggregate metrics match the frozen raw arrays | **Supported** |
| Final test full-clip success is 2/128 (1.5625%) | **Supported** |
| The short-run policy is not a reliable tracking baseline | **Supported** |
| YAML reward 0.8/std 0.2 was effective | **Unsupported; correctly rejected in the report.** Effective source defaults are weight 0.5/std 0.25 (`results/effective_configuration/env.yaml:833-839`), because the loader applies only command-motion overrides (`flat_env_cfg.py:29-32,59-69`). |
| Full PhyGile, two-stage GMT, broad generalization, or hardware performance | **Unsupported and correctly marked NOT_RUN / OUT_OF_SCOPE** |
| GitHub release attachment is already published and digest-verified | **Pending separate delivery verification** |

## Required follow-through

1. Preserve the world-frame, censoring, success-definition, one-seed, and single-source-group qualifiers anywhere these metrics are reused.
2. Do not promote `Train/success` to full-clip success or compare world-frame MPJPE directly with aligned training/termination metrics.
3. Repair and validate the dormant curriculum paths before enabling full Phase A/B or curriculum claims.
4. After publication, record the actual release URL, asset names, sizes, and SHA-256 digests in `results/github_delivery_verification.json`.

## Trace and acceptance boundary

The deterministic record is `results/integrity_audit/trace/final_deterministic_verification.json`; focused same-family reviews are retained beside it. The trace is available in the local delivery only and is intentionally excluded from the public Git/release package. `EXPERIMENT_AUDIT.md` and `EXPERIMENT_AUDIT.json` are the public audit artifacts. Because the reviewers are same-family, the acceptance status is **provisional** even though deterministic checks passed.

# Preliminary Experiment Integrity Audit

**Audit date:** 2026-10-17  
**Evidence snapshot:** 2026-10-17T17:37:39+08:00, with the release archive checked immediately afterward  
**Review class:** fresh same-family reviewer; provisional  
**Scope:** available PhyGile pilot artifacts only; training and the evaluation suite were still in progress  
**Overall verdict:** **FAIL (provisional)**

The failure is caused by an incomplete result set and a declared YAML reward override that the released loader did not apply. It does **not** mean that the available initial-check numbers were fabricated. The initial one-clip `model_0` metrics reproduce exactly from the saved pre-reset arrays.

No completion was inferred from a `RUNNING` receipt. This report must be superseded by a final audit after the recorded training receipt, checkpoints, effective `env.yaml`, all suite receipts/raw arrays, summaries, tracker, and generated report are available.

## Immediate findings

1. **The YAML reward declaration was not the effective reward.** `configs/moe_pilot.yaml:137-141` declares global-anchor position weight `0.8` and standard deviation `0.2`. The released environment loader reads only `commands.motion` (`upstream/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py:29-32,59-69`). The source default is weight `0.5`, standard deviation `0.25` (`upstream/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:207-210`), and the runtime reward table reports weight `0.5` (`results/eval_initial_check/stdout_stderr.log:172-178`). The current README and scope document now disclose this correctly (`README.md:42`; `docs/reproduction_scope.md:26,35`). Any statement that this run applied `0.8/0.2` is false.

2. **The saved initial-check result is numerically authentic.** The receipt is `COMPLETED` with exit code zero (`results/eval_initial_check/receipt.json:86-90`). The printed result (`results/eval_initial_check/stdout_stderr.log:201`) equals `results/eval_initial_check_metrics/metrics.json`, and all three reported means reproduce exactly from `raw_pre_reset.npz`.

3. **That check shows immediate failure, not useful tracking performance.** All 32 environments terminated after 4–7 captured samples, with zero timeouts and zero full-clip successes. Mean survival was 5.5625 frames = 0.11125 s for a 149-frame, 2.98 s reference. Reported errors are pooled only until failure and are therefore failure-censored (`results/eval_initial_check_metrics/metrics.json:14-17,51-57`). They must never be ranked without success fraction and survival duration.

4. **The documented MPJPE is a world-frame metric, distinct from the training/termination body metric.** Evaluation stores raw `mt.body_pos_w` (`scripts/evaluate_pilot.py:72-76`) and computes MPJPE from it (`scripts/evaluate_pilot.py:96-101`). Training and body-position termination use the planar/yaw-aligned `body_pos_relative_w` (`upstream/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py:403-405,1512-1522`; `upstream/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/terminations.py:41-48`). This metric is valid as world-frame, dataset-anchored recovery error, but it is not directly comparable to the aligned command metric and is sensitive to retained reset translation/yaw perturbations.

5. **The larger result set was not complete at the snapshot.** `results/pilot_train20_nostate_seed42/receipt.json:76-77` remained `RUNNING`, and the saved log ended at iteration 158/500 (`results/pilot_train20_nostate_seed42/stdout_stderr.log:7196`). `results/evaluation_suite.json`, the evaluation directory, `evaluation_summary.csv`, `summary.json`, `learning_curves.json`, and the initial-check per-step CSV were absent. `results/evaluation_suite_launcher.log` was empty. The README-linked report at `reports/PhyGile复现实验报告.md` was also not yet present (`README.md:5`). Accordingly, `README.md:46-48` described the intended suite and outputs rather than completed evidence at this snapshot.

6. **The release input archive is real and internally consistent.** `release_assets/phygile_pilot_inputs.tar.gz` exists with SHA-256 `3cac85f442411af44eebcccb259462e1a9f60f6b9558cf58325830e511cc9acd`. Direct in-archive verification found 20 train and 4 test NPZ files; every file hash matches its manifest and `results/data_inventory.json`, all audited arrays are finite, all clips contain 149 frames at 50 Hz, quaternions are normalized, and name/order metadata is consistent. Its provenance JSON is byte-identical to `results/data_provenance.json`. The upstream source files/videos were not included, so their recorded source hashes and the preprocessing transformation were not independently replayed.

## A–F checklist

### A. Reference provenance: PASS, with a verification boundary

- The reference is explicitly identified as video-reconstructed, GMR-retargeted, forward-kinematics-derived kinematics rather than a measured robot trajectory (`results/data_inventory.json:2`; `docs/data_notice.md:3-9`).
- Evaluation loads the motion NPZ independently of the policy (`scripts/evaluate_pilot.py:29-31`). No evaluated-model output is used as ground truth.
- The evaluated motion SHA-256 in `metrics.json:8` matches the first training record in `data_inventory.json:7` and its provenance mapping at `data_provenance.json:470-492`.
- Direct archive verification reproduced all 24 processed-file hashes and metadata. The 20 train and 4 test files have no identical NPZ hashes. Provenance contains seven training source groups and one test source group; the independently computed intersection is empty, agreeing with `data_provenance.json:1046`.
- Boundary: the archive begins at the processed G1 references. It does not include the upstream source videos/raw NPZs, so the recorded source hashes and preprocessing history were not independently regenerated.

### B. Score normalization: PASS

`scripts/evaluate_pilot.py:96-101` and `scripts/summarize_results.py:11-19` compute raw-unit arithmetic means from reference-versus-robot errors. No metric is divided by the prediction's maximum, minimum, mean, or another self-derived scale. `empirical_normalization: true` in `configs/moe_pilot.yaml:6` applies to policy observations, not reported scores.

### C. File, number, and claim consistency: FAIL at the preliminary snapshot

The initial-check files and numbers agree exactly, but the full result/report claims were not yet backed by files:

- Training receipt: `RUNNING` (`results/pilot_train20_nostate_seed42/receipt.json:76-77`).
- Last saved progress: iteration 158/500 (`results/pilot_train20_nostate_seed42/stdout_stderr.log:7196`).
- Suite gate requires completed training (`scripts/evaluate_suite.py:6-13`); no suite receipt existed.
- The results and report named above were absent while `README.md:19,46-48` used archived/present-tense wording.
- The release input archive does exist under `release_assets/`; the README names it as a release attachment (`README.md:17,24`) rather than a repository-root file.

### D. Metrics actually executed / dead paths: WARN

- The initial-check evaluator executed successfully and its saved raw arrays independently reproduce the metrics.
- `scripts/summarize_results.py:7-35` had not executed against a completed suite at this snapshot; none of its promised outputs existed.
- `scripts/evaluate_suite.py:17-33` calls a suite `COMPLETED` when all evaluator subprocesses return zero. That status means execution completion, not scientific success, and the script does not itself validate generated metric contents.
- The in-process curriculum evaluator at `commands.py:1945-2030` was not called by the audited path. It advances the environment and then manually advances motion time (`commands.py:2006-2018`), ignores done flags, and measures raw world-frame body error (`commands.py:2020-2025`). It is a dormant unsafe path.

### E. Scope versus evidence: WARN

The only completed policy-quality evidence was one training clip, the initial `model_0.pt` checkpoint, one evaluation seed (2026), and 32 first episodes of the same clip. All 32 failed after 4–7 samples. The 20/4 files demonstrate input inventory, not completed policy evaluation. The repository generally limits its claims appropriately (`README.md:3,40,48,52`; `docs/reproduction_scope.md:7,22-28,41`; `results/document_witness/report.md:24-33`), but the full 500-update/suite outputs were still prospective at the snapshot.

### F. Evaluation classification: PASS — `simulation_only`

The result labels itself `simulation_only` and the target as a dataset-derived kinematic reference, not model output (`results/eval_initial_check_metrics/metrics.json:2-4`). It is not measured-robot `real_gt`; it is also not a synthetic reference produced by the evaluated policy.

## Independent numerical recomputation

`results/eval_initial_check_metrics/raw_pre_reset.npz` has SHA-256 `9265317042578de99904b7d1ea641b131b81e638596afa052ed3a581f1eee13e`. It contains 178 finite rows:

| Quantity | Recomputed | Saved |
|---|---:|---:|
| MPJPE, m, until termination | 0.10472006350755692 | 0.10472006350755692 |
| MPJAE, rad, until termination | 0.23965762555599213 | 0.23965762555599213 |
| MPJVE, rad/s, until termination | 4.1565752029418945 | 4.1565752029418945 |
| Full-clip successes / trials | 0 / 32 | 0 / 32 |
| Terminal rows / timeout rows | 32 / 0 | consistent |

Per-environment lengths are `[5,7,6,5,6,5,5,6,5,5,5,5,7,5,5,5,6,5,6,5,6,6,7,5,5,4,6,6,7,6,5,6]`, summing to 178. Each environment's recorded frame sequence is exactly `0..L-1`; every environment has exactly one terminal flag on its last row. This supports the claim that terminal transitions were captured once before automatic reset and later reset samples were filtered.

## Reset, indexing, success, and censoring

- Reset is forced to frame 0 and asserted (`scripts/evaluate_pilot.py:46-54`). The control interval is asserted equal to the reference frame interval (`scripts/evaluate_pilot.py:36,45`).
- The termination hook calls the original termination computation and captures state/flags before returning (`scripts/evaluate_pilot.py:66-79`). After the wrapped step returns, done environments are removed from the active mask (`scripts/evaluate_pilot.py:80-87`). The raw monotone frame prefixes provide artifact-level support that post-reset episodes were excluded.
- Success requires exactly `nframes` captures and no failure termination (`scripts/evaluate_pilot.py:90-95`). A timeout on the nth captured sample is therefore accepted, although the `timeout` array is recorded rather than explicitly tested in the expression.
- The policy observes command frame `t`, acts for one control interval, and the post-physics state is compared with the still-current command `t` (`scripts/evaluate_pilot.py:3-4,80-84`). This one-interval hold/lag convention is explicit but differs from identical-timestamp comparison for fast motion.
- Errors are pooled across all surviving environment-steps (`scripts/evaluate_pilot.py:88,96-101`). Early failures contribute fewer samples, so lower error can result from failing before difficult late frames. Use the error, full-clip success, and time-to-failure jointly.
- Active failure terms are timeout; anchor-height difference greater than 0.3 m; projected-gravity z-component difference greater than 0.8; and selected ankle/wrist height errors greater than 0.3 m (`tracking_env_cfg.py:258-283`; functions in `terminations.py:18-58`). The orientation term is a tilt/gravity check and does not test yaw (`terminations.py:28-38`). Raw output does not retain term-level flags, so the exact cause of each initial-check failure cannot be reconstructed.

## Training semantics and dormant risks

These findings do not invalidate the current curriculum-disabled pilot by themselves, but they constrain interpretation and block stronger Phase A/B claims until fixed or explicitly bounded:

1. Training `Train/success` is the EMA of `~termination_manager.terminated` at resampling (`commands.py:423-441,1385-1395`; `my_on_policy_runner.py:400-413,1110-1114`). It does not require exact full-clip completion and is not comparable to `full_clip_success_fraction`.
2. At a motion endpoint, `_update_command` resamples (`commands.py:1504-1510`) and `_resample_command` writes a new robot root/joint state (`commands.py:1385-1415,1493-1501`) without requiring an environment done. With multi-file/jittered clips this can teleport the robot inside an episode.
3. Current success/failure totals are assigned, not accumulated (`commands.py:440-441,1394-1395`), so multiple resampling cohorts before consumption at `commands.py:1530-1538` can lose earlier counts.
4. Nonfinite training errors are converted to zero (`commands.py:810-825`); simple threshold comparisons in `terminations.py:18-58` make NaN non-failing. Nonfinite states should be hard failures, not perfect errors.
5. EMA curriculum statistics initialize unobserved files with threshold/error and 0.5-success priors (`commands.py:546-573`) while the runner does not enforce the configured minimum-attempt requirement (`my_on_policy_runner.py:839-904`).
6. The subprocess curriculum branch sets `success_ratio = 1.0` (`my_on_policy_runner.py:658-666`) and later reads `min_success_file` even though that branch did not define it (`my_on_policy_runner.py:745-746`). It will raise `UnboundLocalError` if reached.
7. Curriculum advancement reads a success threshold but bases pass/fail only on MPJPE/MPJAE (`my_on_policy_runner.py:683-709`), and explicitly logs success as ignored (`my_on_policy_runner.py:744`).
8. The convergence test computes `(old-new)/old` and declares convergence when it is below a positive threshold (`my_on_policy_runner.py:947-958`); worsening produces a negative value and therefore passes. Current pilot curriculum is disabled (`configs/moe_pilot.yaml:77`), so this is dormant here.

## Claim impact

- **Supported:** The initial one-clip `model_0` raw arrays, zero-success result, censored metric values, execution receipt, processed 20/4 reference archive, and recorded source-group separation.
- **Needs qualification:** MPJPE must be called world-frame/dataset-anchored; the source-group check applies to recorded upstream grouping; processed references are not measured robot ground truth; training EMA success is not full-clip success.
- **Unsupported at this snapshot:** Completed 500-update training, completed 16-combination suite, post-training improvement, generated per-step/summary/curve files, and the linked final report.
- **False if asserted:** The run applied YAML reward weight `0.8`/std `0.2`; runtime used source defaults `0.5`/`0.25`.

## Required final-audit evidence

1. Completed immutable training receipt with exit code, final log, exact checkpoint hashes, and effective saved `env.yaml`.
2. Suite receipt and one receipt, raw pre-reset NPZ, metrics JSON, and code/config hash set for every declared evaluation entry.
3. Recomputed per-step and aggregate summaries, with world-frame naming and paired success/time-to-failure reporting.
4. Explicit verification that the model-0 and model-499 files are distinct and mapped to 1 versus 500 optimizer updates.
5. Generated report/tracker language checked against the final files and statuses.
6. Term-level failure flags for future evaluations if exact failure attribution will be claimed.

The detailed evidence manifest, independent recomputation record, archive verification, reviewer request, and full reviewer traces are stored under `results/integrity_audit/trace/`.

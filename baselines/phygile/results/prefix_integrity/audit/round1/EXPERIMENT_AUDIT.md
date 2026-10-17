# Independent experiment-integrity review — round 1

**Review date:** 2026-10-17  
**Verdict:** **WARN**  
**Acceptance:** provisional, same-family independent-context review  
**Blocking defects:** none  
**Configured route:** `gpt-5.6-sol`, reasoning `ultra`; runtime model identity unavailable and recorded as `unknown`

## Scope and classification

This review inspected the requested implementation, evaluation scripts, original upstream source, raw NPZ captures, receipts, manifests, checkpoint, report, and dataset-lineage records directly. It did not rely on an executor summary and did not run a GPU job.

The defensible classification is:

- feedback capture and acceptance gate: **simulation-only diagnostic** using a dataset-derived kinematic reference;
- curriculum and rollback checks: **synthetic software fixtures**;
- one-PPO run: **bounded runtime diagnostic**, not a performance result;
- document witness: provenance and execution evidence, not an independent research benchmark.

The project labels these scopes correctly in `results/prefix_integrity/document_run/capture/metrics.json:2-4`, `results/prefix_integrity/document_run/feedback_gate/verification.json:2-18`, `results/prefix_integrity/document_run/regression/verification.json:2-21`, and `results/prefix_integrity/document_run/pipeline.json:2-4`.

## Integrity checks

### A. Reference provenance — PASS

The evaluated reference is a dataset input, not model output. The capture records `dataset-derived kinematic reference; not model output` and the exact motion hash at `results/prefix_integrity/document_run/capture/metrics.json:3-8`. The source inventory describes the data as video-reconstructed, GMR-retargeted, and FK-derived rather than measured robot motion at `results/data_inventory.json:2-11`; the selected sample's lineage, split, source URL, and hashes appear at `results/data_provenance.json:473-492`.

I independently extracted `data/train/sample_000_actor0.npz` from `release_assets/phygile_pilot_inputs.tar.gz`. Its SHA-256 is `a4f3aa10ad1b17ad11508b5ea3b22664beb724ce7ea9148054d137cc8da6bd1b`, with 149 frames at 50 Hz, matching the capture record. Reconstructing the candidate reference from the dataset and comparing it with the captured command produced a maximum absolute difference of `4.76837158203125e-07`, within the evaluator's `2e-5` assertion.

### B. Raw scores and normalization — PASS

The evaluator computes direct Euclidean body-position error and direct absolute joint errors at `scripts/evaluate_prefix_feedback.py:103-113`, then serializes those values at `scripts/evaluate_prefix_feedback.py:114-126`. The acceptance function explicitly uses raw metre errors at `reimplementation/physics_prefix.py:75-120`. No min-max scaling, self-normalization, score clipping, rank transform, or success-conditioned denominator occurs.

From `raw_pre_reset.npz`, I independently recomputed all 869 finite rows:

| quantity | independently recomputed | recorded |
|---|---:|---:|
| MPJPE, m | 0.18331027030944824 | 0.18331027030944824 |
| MPJAE, rad | 0.3282482922077179 | 0.3282482922077179 |
| MPJVE, recorded units/s | 1.9027197360992432 | 1.9027197360992432 |

The recorded values are at `results/prefix_integrity/document_run/capture/metrics.json:51-57`. World-coordinate and environment-local per-frame MPJPE were identical, confirming that the common translation cancels as expected.

### C. Artifacts, numbers, hashes, and receipts — PASS with one portability warning

The raw capture, initial-state NPZ, metrics, feedback metadata, gate records, five stage receipts, logs, source archive, diagnostic checkpoint, and document witness exist and are readable. The pipeline is complete and every stage records exit code zero (`results/prefix_integrity/document_run/pipeline.json:2-14`, `:17-26`, `:29-56`, `:59-79`, `:81-92`).

The raw capture contains 869 finite 71D states and matching 14-body arrays. Per-environment frames start at zero and are contiguous; lengths sum to 869; all 32 environments have exactly one final failure termination and no timeout. The saved episode lengths, success count, and scope note match `results/prefix_integrity/document_run/capture/metrics.json:14-57`.

The reported gate result is exact: 0 accepted and 32 rejected (`results/prefix_integrity/document_run/feedback_gate/verification.json:20-26`). Independently calling `assess_execution` for all 32 captured episodes reproduced every saved record and all three rejection reasons for every episode: incomplete continuation, failure termination, and MPJPE above 0.1 m.

The source archive SHA-256 is `3e041806b4e5c22b98f1d05dc8a19f758673a607b7505dd45fa2f180477464a1`. Its 153 regular files exactly match the feedback-gate receipt's recorded source set. The original transport manifest recorded 152; `results/prefix_integrity/execution_archive_verification.json:7-14` explicitly corrects the off-by-one count while preserving the original manifest. The one-PPO checkpoint exists at 67,348,310 bytes with SHA-256 `68439707ee5c166fe17a48fd141d1ed3e9b9264fe7d064f6c5ed70fc92bbe424`, matching `results/prefix_integrity/execution_archive_verification.json:3-6`.

**Round-1 warning W1:** the then-canonical compact `patches/curriculum_integrity.patch` had SHA-256 `95372e72630fcc69390bd48c22aea4176a60b112117adba9710d5421b78c50e6` and did not pass strict `git apply --check` against the pinned CRLF upstream; it applied only with whitespace-ignore. The exact generated patch had SHA-256 `1ed02041066a4c768a33530ba47cfeb9b17b238811b63bf34f8fdbccf8534e99` but showed whole-file newline churn. This did not alter the executed source or result values, because the guarded copy's original and patched hashes match the receipts (`results/prefix_integrity/document_run/regression/verification.json:22-30`), but it weakened strict patch replay and human review.

### D. Metric functions invoked — PASS

The runtime evaluator creates the environment and policy, wraps termination computation before stepping, collects pre-reset tensors, and computes the metrics from captured arrays at `scripts/evaluate_prefix_feedback.py:31-68`, `:69-103`, and `:103-126`. The offline gate reloads the capture, reconstructs dataset-derived candidates, verifies the reference match, and directly calls `assess_execution` at `scripts/diagnose_prefix_feedback.py:18-56`. The saved records therefore derive from the reviewed metric function, not from hand-entered summary values.

Upstream metric flow also invokes the relevant body and joint calculations in `upstream/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py:397-422`; accumulation/finalization occurs at `:791-830`. The guarded repair disables the two unsafe evaluation entrypoints rather than presenting their output as valid.

### E. Scope and claims — PASS

The report states that full PhyGile was not reproduced and confines the result to repairs, a synthetic rollback core, one PPO iteration, and offline acceptance of actual simulation feedback (`reports/PhyGile课程修复与物理前缀推进报告.md:3`, `:9-12`, `:39-55`). It explicitly excludes full curriculum training, generator training and quality, a verified 262D codec, a real simulator rollback adapter, online multi-window generation/execution, terminal standing, closed-loop PPO, paper-scale benchmarking, and hardware claims (`reports/PhyGile课程修复与物理前缀推进报告.md:57-75`; `results/prefix_integrity/document_witness/report.md:30`). The experiment tracker uses `DONE_REGRESSION_ONLY`, `DONE_SYNTHETIC_INTERFACE_ONLY`, and `DONE_SIMULATION_DIAGNOSTIC_ONLY` at `EXPERIMENT_TRACKER.md:24-27`.

The one-PPO checkpoint is correctly described as an engineering diagnostic rather than an improved main policy at `results/prefix_integrity/execution_archive_verification.json:3-6`. No claim exceeds the available path.

### F. Evaluation type — PASS

The feedback run is not real ground truth, not a held-out research benchmark, and not generator evaluation. It is dataset-reference replay in simulation with one training clip and randomized first episodes. This is stated at `results/prefix_integrity/document_run/feedback_gate/verification.json:3-18` and `results/prefix_integrity/document_run/capture/metrics.json:2-4`. The regression suite is correctly separated as synthetic fixtures at `results/prefix_integrity/document_run/regression/verification.json:2-21`.

## Rollback, acceptance, and patched-source lineage

`assess_execution` binds an execution to the exact candidate digest, prefix boundary, body order, schema, frame rate, timing, completeness, termination state, mean MPJPE, and optional terminal suffix (`reimplementation/physics_prefix.py:47-120`). `roll_physics_prefix` snapshots before each window, restores before retries and after exceptions/rejection exhaustion, and commits only accepted actual continuation states (`reimplementation/physics_prefix.py:130-184`). The synthetic regression suite covers rejection, rollback, retry exhaustion, wrong candidates/boundaries/timing, terminal error, unexpected exceptions, and nonfinite inputs (`results/prefix_integrity/document_run/regression/verification.json:4-16`). These establish the pure-Python contract, not a working real-simulator snapshot adapter.

The guarded bootstrap verifies both original and patched file hashes before prepending the isolated source path (`scripts/guarded_bootstrap.py:8-19`). The processes logged activation and the guarded training entrypoint, but did not record each imported module's `__file__`; this limitation is disclosed at `results/prefix_integrity/document_witness/report.md:24`. Given the import ordering and matching source archive, this is a provenance limitation rather than a contradiction.

## Round-1 disposition

The bounded scientific and engineering claims are supported. No blocking defect was found. Publication should wait for a strictly replayable canonical patch or an explicit portability receipt; after that repair, the audit may be upgraded without rerunning scientific code.

## Recorded source hashes

| artifact | SHA-256 |
|---|---|
| `reimplementation/physics_prefix.py` | `a8d08d3cd7071b65924ec4204b7ce3814dee8e650f267b26f743e779b3011057` |
| `reimplementation/test_prefix_and_curriculum.py` | `4723ad5ff14812cee19e14c9d6cf403af869bbfe5c96d089fc58d360f4bbd815` |
| `scripts/prepare_guarded_upstream.py` | `e688a87d464c0a19fbbb23c7687e26a325343039da75284339e5d492c01c067f` |
| `scripts/guarded_bootstrap.py` | `c785fbaa9c2c748ac2bf7e577e97c21a077d05f8e822c85c921bf6d03bc0b371` |
| `scripts/evaluate_prefix_feedback.py` | `b428bc1840c51786884498891ed93b29fa9cf719189b230e2749691f6261c9c` |
| `scripts/diagnose_prefix_feedback.py` | `c93290a765b76dbe12bc9d641f3d3b36c42657a8fe46ae90cebbabc1d3649965` |
| `scripts/run_prefix_recorded.py` | `c4a7624d25c261a33f8d078e5fe6e7f79f9265f2a548274151f4a1aa949aaa3` |
| `scripts/validate_prefix_milestone.py` | `6655597e554bf66a713a0c2b1c020344909770416a8e257da885860930d6634b` |
| `scripts/build_prefix_report.py` | `f6903f5142ceec142203cf0f60f0beaa8c78c948cd699a935429157a05445fdb` |
| original compact `patches/curriculum_integrity.patch` | `95372e72630fcc69390bd48c22aea4176a60b112117adba9710d5421b78c50e6` |
| `configs/prefix_diagnostic_plan.json` | `ca44f2e1e728dc6dbb34c6b7f80a07357c5c34c9f31c4b9f8d141e46918c38ca` |
| original upstream `commands.py` | `b3a35da40dfbb40f227f63e0444aa31d6dac8d2889a980b610ac3c30e30f1e4a` |
| original upstream `my_on_policy_runner.py` | `c41fa2125cd9e5c939cbd98956c601eac437d74078e0a7e9dbd3dcdf939f701c` |
| original upstream `curriculum_manager.py` | `bd4c3ad5ce65f8916ad6404338297cc56031ae08915d43967778e179043612d9` |
| patched `commands.py` | `cad7ee14922ed4d5a2ba509bfa4d455cc4b8ee0a1eafa591c4e672a87751b20b` |
| patched `my_on_policy_runner.py` | `e4b73e1e3e4bfdc79c1b23a121d2ed0cfd79afdbff1df98c3cac5cfd93be7409` |
| patched `curriculum_manager.py` | `5f1a4cbcce8c85a5b5d96c67bce73d99c4de453822eab6659de932c43c7612d9` |
| `release_assets/phygile_pilot_inputs.tar.gz` | `3cac85f4436bdbf46c4fc84ab44ab4f978e2801d1aa1088bc30e9ed015bacacd` |
| `release_assets/phygile_prefix_integrity_source.tar.gz` | `3e041806b4e5c22b98f1d05dc8a19f758673a607b7505dd45fa2f180477464a1` |
| `release_assets/prefix_execution_manifest.json` | `c34746cc67d896f33efb5e20ff3f6e91f481f60a235db2f0d28051e8b9891eda` |
| `release_assets/phygile_prefix_guard_1ppo_diagnostic_seed42.pt` | `68439707ee5c166fe17a48fd141d1ed3e9b9264fe7d064f6c5ed70fc92bbe424` |
| `capture/raw_pre_reset.npz` | `a1a5b01f96a1a890e8723ddd9c1df5764dadfc296964a7dbc838c87af3ae00c` |
| `capture/initial_state.npz` | `7a4a218e8c6e65ef82d4db72dffb5f400e220e756c16d5b9e9704de18b83c90` |
| `capture/metrics.json` | `50a1b69eaa77e955952cc1d204887c477fd499ead283c1e0acbab0bd888839` |
| `feedback_gate/verification.json` | `e91b5c0fc7f0dc034340748451c843899892e148bf3a5b3491787e97aec97a` |


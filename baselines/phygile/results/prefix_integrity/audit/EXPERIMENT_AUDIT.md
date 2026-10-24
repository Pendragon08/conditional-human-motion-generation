# Independent experiment-integrity review

**Review date:** 2026-10-17  
**Canonical verdict:** **PASS**  
**Acceptance status:** **provisional** because the reviewer is from the same model family, operating in an independent context  
**Blocking defects:** none  
**Configured reviewer:** `gpt-5.6-sol`, reasoning `ultra`  
**Runtime model identity:** `unknown`

## Review basis

I read the requested implementation, original upstream source, evaluation and report builders, dataset lineage, raw NPZ capture, receipts, logs, source archive, checkpoint metadata, document witness, and portability repair directly. I recomputed the reported metrics and acceptance records locally from the preserved arrays. No GPU job was run, no implementation was changed, and executor summaries were not treated as evidence.

Round 1 is preserved under `results/prefix_integrity/audit/round1/`. It returned WARN solely because the compact patch then published at the canonical path could not be applied strictly to the pinned CRLF source. The targeted follow-up verifies that this finding is repaired. The current canonical patch is the byte-exact patch generated and executed by the guarded-source builder, passes strict application, and leaves all scientific code and results unchanged.

## Verdict by required integrity question

| check | verdict | finding |
|---|---|---|
| A. Reference provenance | PASS | Dataset input, not model output; video-reconstructed, GMR-retargeted, FK-derived motion; not measured robot ground truth. |
| B. Raw scores / normalization | PASS | Direct metre/radian errors; independent recomputation exactly matches the saved values; no score normalization or denominator manipulation. |
| C. Artifacts / numeric values / hashes / receipts | PASS | Required files exist; raw arrays are finite; values, hashes, stage exits, archive members, and checkpoint metadata match. The 152/153 transport-manifest count has a transparent preserved erratum. |
| D. Metric invocation | PASS | The runtime capture computes metrics from saved pre-reset tensors, and the offline gate directly invokes the reviewed `assess_execution` function. |
| E. Scope and experiment classification | PASS | Claims are confined to bounded engineering validation, synthetic rollback tests, one diagnostic PPO iteration, and offline simulation feedback. |
| F. Evaluation type | PASS | Simulation-only replay of one training clip; not real ground truth, not generator evaluation, and not a paper-scale benchmark. |
| Rollback and acceptance integrity | PASS | Pure-Python transactional contract is well guarded and regression-tested; the absence of a real simulator snapshot adapter is stated. |
| Patched-source lineage | PASS | Original and patched hashes are bound, guarded activation is logged, the exact execution patch is archived, and the canonical patch now strict-applies. |

## A. Reference provenance

The capture labels the reference `dataset-derived kinematic reference; not model output` and records the motion hash at `results/prefix_integrity/document_run/capture/metrics.json:2-8`. The dataset inventory states that references are reconstructed from video, GMR-retargeted, and FK-derived rather than measured robot trajectories at `results/data_inventory.json:2-11`. The selected sample's split, kind, source identifier, URL, and hashes appear at `results/data_provenance.json:473-492`.

I independently extracted `data/train/sample_000_actor0.npz` from `release_assets/phygile_pilot_inputs.tar.gz`. Its SHA-256 is `a4f3aa10ad1b17ad11508b5ea3b22664beb724ce7ea9148054d137cc8da6bd1b`; it has 149 frames at 50 Hz. Reconstructing the candidate from this input and comparing it to the captured reference produced maximum absolute difference `4.76837158203125e-07`, within the evaluator's fixed `2e-5` consistency tolerance. The gate therefore evaluates a traceable dataset reference, not an output relabeled as ground truth.

## B. Raw metrics and absence of normalization fraud

`scripts/evaluate_prefix_feedback.py:103-113` computes Euclidean body-position error and absolute joint position/velocity errors directly from captured reference and robot arrays. `reimplementation/physics_prefix.py:75-120` likewise evaluates raw metre MPJPE, completeness, failure termination, and the optional terminal suffix. No min-max scaling, z-normalization, clipping, rank transform, or success-only denominator occurs.

Independent recomputation from `results/prefix_integrity/document_run/capture/raw_pre_reset.npz` produced:

| quantity | recomputed | saved at `capture/metrics.json:51-57` |
|---|---:|---:|
| MPJPE, m | 0.18331027030944824 | 0.18331027030944824 |
| MPJAE, rad | 0.3282482922077179 | 0.3282482922077179 |
| MPJVE, recorded units/s | 1.9027197360992432 | 1.9027197360992432 |

All 869 rows are finite. World-coordinate and environment-local per-frame MPJPE are exactly equal because the common environment translation cancels. The report also warns that errors are censored at failure and must be interpreted with episode length and success (`results/prefix_integrity/document_run/capture/metrics.json:57`).

## C. Artifacts, values, receipts, and archive integrity

The pipeline is recorded as complete with exit code zero for preparation, regression, one-PPO, capture, and gate (`results/prefix_integrity/document_run/pipeline.json:2-14`, `:17-26`, `:29-56`, `:59-79`, `:81-92`). Its stage receipts bind commands, timings, exit state, logs, and source hashes; the recorded current-source mismatch count is zero.

The raw capture contains 869 finite 71D native states with matching 14-body positions and joint arrays. Each of 32 environments has contiguous frame indices starting at zero, 21–41 frames, exactly one final failure flag, and no timeout. Episode lengths sum to 869 and reproduce `results/prefix_integrity/document_run/capture/metrics.json:14-57`.

The gate records 0 accepted and 32 rejected at the predeclared 50-frame, 0.1 m diagnostic threshold (`results/prefix_integrity/document_run/feedback_gate/verification.json:5-26`). Independently calling `assess_execution` on every episode reproduced all 32 JSON records exactly. Every episode is rejected for incomplete continuation, failure termination, and MPJPE above tolerance. No threshold was changed.

The source archive SHA-256 is `3e041806b4e5c22b98f1d05dc8a19f758673a607b7505dd45fa2f180477464a1`. Independent tar inspection finds 153 regular files, exactly matching the receipt's 153 source-hash entries. The originally transmitted manifest recorded 152 at `release_assets/prefix_execution_manifest.json:7`; a byte-identical preserved copy exists at `results/prefix_integrity/execution_transport_manifest_original.json:1-7`. The correction records 153 files and explains the counting error at `results/prefix_integrity/execution_archive_verification.json:7-14`. This is a resolved metadata count error; the archive and source set did not change.

The one-PPO checkpoint exists, is 67,348,310 bytes, and has SHA-256 `68439707ee5c166fe17a48fd141d1ed3e9b9264fe7d064f6c5ed70fc92bbe424`, matching `results/prefix_integrity/execution_archive_verification.json:3-6`. Its role is explicitly limited to an engineering diagnostic.

## D. Metric functions are actually invoked

The capture script fixes the seed and input, verifies joint/body order and frame rate, initializes the current reference, wraps termination computation before automatic reset, and steps the actual policy at `scripts/evaluate_prefix_feedback.py:31-68` and `:69-103`. It derives the metrics from the captured arrays and saves both arrays and metadata at `scripts/evaluate_prefix_feedback.py:103-126`.

The gate reloads the raw capture and initial state, reconstructs the dataset-derived candidate, asserts that the captured command matches the dataset, constructs `Execution`, and directly calls `assess_execution` at `scripts/diagnose_prefix_feedback.py:18-56`. Independent invocation produced byte-for-value-equivalent records, so the summary is not disconnected from the reviewed function.

The underlying upstream command computes body tracking error at `upstream/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py:397-422` and accumulates/finalizes metrics at `:791-830`. The guarded repair makes nonfinite values fail before statistic mutation and disables two known unsafe evaluator entrypoints instead of presenting them as valid.

## E. Scope and claims

The report says that complete PhyGile was not reproduced and limits the work to local curriculum-integrity repairs, a transactional physics-prefix core, one diagnostic PPO iteration, and offline evaluation of actual simulation feedback (`reports/PhyGile课程修复与物理前缀推进报告.md:3-12`, `:39-55`). It excludes full curriculum training, generator training and quality, a verified 262D codec, a real simulator rollback adapter, online generation–execution continuation, terminal standing, closed-loop PPO adaptation, paper-scale benchmarking, and hardware performance (`reports/PhyGile课程修复与物理前缀推进报告.md:57-75`; `results/prefix_integrity/document_witness/report.md:30`). The tracker classifies the milestones as regression-only, synthetic-interface-only, and simulation-diagnostic-only at `EXPERIMENT_TRACKER.md:24-27`.

Those limitations match the actual paths. In particular, the 0/32 figure is an acceptance-gate diagnostic for one dataset reference, fixed policy, and fixed threshold. It is not a text-generation success rate or an absolute feasibility estimate.

## F. Evaluation type

The actual feedback path is `simulation_only`; it replays one training clip across 32 randomized first episodes (`results/prefix_integrity/document_run/feedback_gate/verification.json:2-18`). The regression path is `synthetic_software_fixtures`, with no trained model, physics simulation, or full curriculum training (`results/prefix_integrity/document_run/regression/verification.json:2-21`). The one-PPO path is a runtime check of 768 transitions and is not used as a comparative score. No result is presented as real ground truth.

## Rollback and acceptance integrity

`Motion` validates dimensions, body names, finiteness, frame rate, schema, and immutable copies at `reimplementation/physics_prefix.py:21-38`; compatibility and digests are defined at `:47-63`. `assess_execution` binds the result to the exact candidate digest, committed prefix, start boundary, timing, complete frame count, termination state, raw MPJPE, and terminal suffix at `:75-120`.

`roll_physics_prefix` snapshots before each window, restores before each retry, restores after exceptions and exhausted retries, and appends only accepted actual continuation states at `reimplementation/physics_prefix.py:130-184`. The regression suite exercises rejection, rollback, retry exhaustion, wrong candidate/boundary/timing, terminal error, exceptions, nonfinite inputs, and incompatible frame rates (`results/prefix_integrity/document_run/regression/verification.json:4-16`). This supports the stated pure-Python contract. The report correctly says that a complete real-simulator snapshot/restore adapter has not been validated.

## Patched-source lineage and targeted follow-up

`scripts/prepare_guarded_upstream.py:91-113` copies the archived upstream, writes only the reviewed replacements, hashes original and patched bytes, and emits the generated patch and receipt. `scripts/guarded_bootstrap.py:8-19` checks both sides of that mapping before prepending the isolated package path. Logs contain `PHYGILE_INDEPENDENT_REPAIRS_ACTIVE`; the patched hashes in `results/prefix_integrity/document_run/regression/verification.json:22-30` match the materialized files and source archive.

Round 1 found that the compact canonical patch could not be applied strictly because its context used LF while the pinned upstream uses CRLF. That exact original file is preserved at `results/prefix_integrity/patch_portability/original_compact.patch`, SHA-256 `95372e72630fcc69390bd48c22aea4176a60b112117adba9710d5421b78c50e6`.

The current `patches/curriculum_integrity.patch` is 394,483 bytes with SHA-256 `1ed02041066a4c768a33530ba47cfeb9b17b238811b63bf34f8fdbccf8534e99`. Independent checks establish that it:

1. passes strict `git apply --check --directory=baselines/phygile/upstream` with exit code 0;
2. is byte-identical to `results/prefix_integrity/patched_source/INDEPENDENT_REPAIRS.patch`;
3. is byte-identical to a fresh in-memory regeneration from `prepare_guarded_upstream.py`;
4. matches `results/prefix_integrity/patch_portability/verification.json:2-14`.

The larger size reflects whole-file newline normalization, which reduces visual compactness but provides exact strict replay of the executed patched bytes. Because the compact semantic diff is preserved beside it, both reviewability and replayability remain available. The round-1 portability warning is resolved without rerunning or changing any experiment.

The runtime processes did not log individual imported module `__file__` values, as disclosed at `results/prefix_integrity/document_witness/report.md:24`. Guard activation, import ordering, guarded entrypoint, exact original/patched hashes, and the execution archive still establish a consistent lineage. This is a documented provenance limitation, not a contradictory or missing result.

## Remaining limits and acceptance

No actionable experiment-integrity defect remains within the bounded claims. The following limits must remain attached to publication:

- same-family review means this PASS is provisional;
- feedback is simulation-only and uses one training clip with a dataset-derived reference;
- rollback/commit behavior is validated through synthetic interfaces, not a real simulator snapshot adapter;
- full curriculum training, online generator feedback, terminal behavior, closed-loop PPO, generator quality, paper-scale comparisons, and hardware performance remain untested;
- per-module runtime `__file__` values were not recorded;
- the original transport manifest's 152-file count is preserved and superseded by the explicit 153-file verification.

The report's audit-status line can now be regenerated from this canonical JSON. No scientific rerun is required.

## Canonical recorded hashes

| artifact | SHA-256 |
|---|---|
| `reimplementation/physics_prefix.py` | `a8d08d3cd7071b65924ec4204b7ce3814dee8e650f267b26f743e779b3011057` |
| `reimplementation/test_prefix_and_curriculum.py` | `4723ad5ff14812cee19e14c9d6cf403af869bbfe5c96d089fc58d360f4bbd815` |
| `scripts/prepare_guarded_upstream.py` | `e688a87d464c0a19fbbb23c7687e26a325343039da75284339e5d492c01c067f` |
| `scripts/guarded_bootstrap.py` | `c785fbaaa6e783a43f5532281fd1cf3a4c5f5ecc56519a5a4994fd4b3330b371` |
| `scripts/evaluate_prefix_feedback.py` | `b428bc18488302178f2a35b7b29430163c1e0cc3adb4962e115bd659380a8c9c` |
| `scripts/diagnose_prefix_feedback.py` | `c93290a765b76dbe12bc9d641f3d3b36c42657a8fe46ae90cebbabc1d3649965` |
| `scripts/run_prefix_recorded.py` | `c4a7624d4cbe827a1cb3c7691bd5e8442a7024dfbf1ec5b62bf3db0a63c32aa3` |
| `scripts/validate_prefix_milestone.py` | `6655597ef83ab12dd9f8ff0fb1606a5e01c44e265c9c99683cf4f1d3ffb6634b` |
| `scripts/build_prefix_report.py` | `f6903f513cc9c702b6f489be32795fe84cdb9db1377205bc89f0c86267c45fdb` |
| `patches/curriculum_integrity.patch` | `1ed02041066a4c768a33530ba47cfeb9b17b238811b63bf34f8fdbccf8534e99` |
| `configs/prefix_diagnostic_plan.json` | `ca44f2e1e728dc6dbb34c6b7f80a07357c5c34c9f31c4b9f8d141e46918c38ca` |
| upstream `commands.py` | `b3a35da443de54c303e6b4fd3fb2b9e4153bdbeef7d91f9fb843bd5def3f1e4a` |
| upstream `my_on_policy_runner.py` | `c41fa2122f92f9413dc43c8de72a0f8dcaae518021143cc0f37e104f497f701c` |
| upstream `curriculum_manager.py` | `bd4c3ad52a16559144bb43c4c19d18a846b58a0228b515d5bea811a2909cbdd0` |
| patched `commands.py` | `cad7ee14922ed4d5a2ba509bfa4d455cc4b8ee0a1eafa591c4e672a87751b20b` |
| patched `my_on_policy_runner.py` | `e4b73e1e3e4bfdc79c1b23a121d2ed0cfd79afdbff1df98c3cac5cfd93be7409` |
| patched `curriculum_manager.py` | `5f1a4cbcce8c85a5b5d96c67bce73d99c4de453822eab6659de932c43c7612d9` |
| `release_assets/phygile_pilot_inputs.tar.gz` | `3cac85f442411af44eebcccb259462e1a9f60f6b9558cf58325830e511cc9acd` |
| `release_assets/phygile_prefix_integrity_source.tar.gz` | `3e041806b4e5c22b98f1d05dc8a19f758673a607b7505dd45fa2f180477464a1` |
| original/preserved transport manifest | `c34746cc67d896f33efb5e20ff3f6e91f481f60a235db2f0d28051e8b9891eda` |
| one-PPO diagnostic checkpoint | `68439707ee5c166fe17a48fd141d1ed3e9b9264fe7d064f6c5ed70fc92bbe424` |
| `capture/raw_pre_reset.npz` | `a1a5b01f4e96a6a717f00ab2d9968b2edec3bd6a2c532236cc45b5f5323ae00c` |
| `capture/initial_state.npz` | `7a4a218eb69f485c3f0b071c8126fc7ddfff9f52d9b2b2df51a0bbdff8183c90` |
| `capture/metrics.json` | `50a1b69e90358f4c3a0bf4bb7f735f460789909c573dadb0c6f84512fe888839` |
| `feedback_gate/verification.json` | `e91b5c0f87506f4a673a5b1e2d8b68ccfedaaf702b9d18ac535138096c2ec97a` |
| `patch_portability/original_compact.patch` | `95372e72630fcc69390bd48c22aea4176a60b112117adba9710d5421b78c50e6` |
| `patch_portability/verification.json` | `5fcc85d940bb8b4ac978ef6b59c70dc0df3ecdc6d336ee4b24ecfbb32fe15975` |

### Reviewer-record erratum

The immutable round-1 audit table contains transcription errors in several long SHA-256 strings. The canonical table above was generated by reading each file directly and is the authoritative hash record. This reviewer-record correction does not reflect any change to source, execution, or scientific output.


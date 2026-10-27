# Experiment Audit Report

**Date:** 2026-10-22  
**Auditor:** GPT-5.6-Sol, ultra reasoning, fresh same-family read-only reviewer  
**Canonical agent:** `/root/unitree_integrity_audit`  
**Project:** PhyGile Unitree Lab/Mjlab frozen-policy adaptation  
**Review independence:** `same-family`  
**Acceptance status:** `provisional`  
**Trace:** `.aris/traces/experiment-audit/2026-10-22_run01/`

## Overall Verdict: WARN

## Integrity Status: warn

The reported four-clip numbers, completion counts, checkpoint identity, dataset-derived references, and no-new-training scope all survived direct recomputation from the private artifacts. I found no fake ground truth, prediction-derived score normalization, phantom result, concealed optimizer run, or scope inflation in the published claims.

The audit remains **WARN** because the historical execution receipts do not fully bind the executed code graph. Isaac receipts bind `lab_run.py` but not its imported task/encoder/framework sources; Mjlab receipts contain no source digest. The current Mjlab evaluator differs from the archived `code_v5` copy by the documented list-versus-tuple assertion repair. The result arrays and metrics are internally consistent, but the receipts alone cannot prove every imported source byte used by the completed runs. A second, narrower provenance warning is that the environment ledger's stated canonical config digest has no documented canonicalization procedure and was not reproducible from the current JSON with standard sorted-JSON serializations.

The first reviewer route returned HTTP 403 and produced no verdict. That failed attempt is preserved separately; this resumed reviewer completed the file and array checks directly. See `results/v1/audit_route_attempt01_error.json:6-11` and the `001-route-attempt-*` trace files.

## Deterministic checks performed

- All 24 entries in `results/v1/private_raw_manifest.json:1-144` exist at their listed local paths and match both byte count and SHA-256.
- All eight local copies named by `results/v1/private_asset_manifest.json:3-63` exist and match byte count and SHA-256, including checkpoint `bb067f364de74e2da7cf1879e6f26bb961a71eb9ff64cbfbe84dd858bfae2336` and all four motion clips.
- For all eight engine/clip evaluations, the stored `reference_joint_pos`, `reference_joint_vel`, and `reference_body_pos` arrays equal the corresponding source motion arrays exactly after name-based selection (`max_abs_difference = 0`). Frames are exactly `0..N-1`; only the last captured row is terminated or timed out.
- All 24 reported MPJPE/MPJAE/MPJVE values recompute exactly from the private raw arrays (`max_abs_difference = 0`). Receipt status, duration, completion status, and the public cross-simulation summary agree. Frames are Isaac `[149, 12, 149, 62]` and Mjlab `[129, 10, 149, 14]`, yielding 2/4 and 1/4 full-clip successes.
- The four Mjlab converted motion files equal the source arrays after the receipt's name-based permutation. The 520D initial command in both engines equals source frames at offsets `[0,1,2,1,21,41,61,81]` exactly.
- Reloading the frozen checkpoint with the current portable policy reproduced every stored initial action within the fixed `rtol=atol=1e-5`; the largest fresh absolute difference was `2.682209014892578e-7`. The retained synthetic FP32/TF32 outcome also agrees with `results/v1/frozen_policy_parity_v3.json:2-50`.
- The checkpoint is 67,349,356 bytes with the stated SHA-256, `iter=1999`, 25 optimizer-state entries, and every recorded optimizer step equal to 40,000. This supports checkpoint metadata claims but is not a substitute for original training logs.
- All 14 current scripts match `results/v1/author_source_manifest.json:1-72`. Published receipts are byte-identical to their corresponding private receipts. The archived Isaac evaluator is byte-identical to current `scripts/lab_run.py` and its receipt source digest.
- The compiled Mjlab artifact contains 29 actuators; stiffness, damping, armature, and effort limits equal the contract exactly. Its own scope correctly says it is a zero-step static compilation check (`results/v1/mjlab_compiled_dynamics_v1.json:2-3,267-268`). Actual framework stepping is separately witnessed in `results/v1/doc_witness_report_v1/REPORT.md:25-42`.

## Checks

### A. Ground Truth Provenance: PASS

The tracking reference is loaded from the four private dataset motion files, not generated from policy outputs. Isaac captures command-term reference arrays and robot arrays independently (`scripts/lab_run.py:171-177`). Mjlab does the same after applying the policy joint permutation (`scripts/mjlab_evaluate.py:42-51`). The analysis computes errors directly between those two array families (`scripts/analyze_crosssim.py:18-26`). The report describes that provenance accurately (`reports/PhyGile_Unitree适配与跨仿真报告_20261022.md:48-52`). Direct array comparison found exact equality between every stored reference array and its source clip.

The original server paths in `results/v1/private_asset_manifest.json:3-63` were not remounted for this audit. The accessible local copies at `D:/具身智能/PhyGile复现/Unitree框架适配_20261022/private_assets` matched every manifest hash, so the limitation is path re-access, not an observed content mismatch.

### B. Score Normalization: PASS

MPJPE is a raw Euclidean mean; joint position and velocity errors are raw absolute means (`scripts/lab_run.py:192-197`, `scripts/mjlab_evaluate.py:59-66`, `scripts/analyze_crosssim.py:20-24`). None is divided by a prediction maximum, minimum, mean, range, or success count. The only normalization is the saved training-time observation normalizer applied before policy inference, `(obs-mean)/(std+1e-2)` (`scripts/frozen_policy.py:15-28`); it does not transform reported evaluation errors. The narrative explicitly discloses failure censoring and the absence of output-statistic normalization (`reports/PhyGile_Unitree适配与跨仿真报告_20261022.md:65`).

### C. Result Existence, Keys, Numbers, and Status: WARN

All claimed result files exist; all claimed metric keys and displayed values match. The cross-simulation artifact records seed 2026, one trial per clip and engine, zero optimizer steps, and success counts 2/4 versus 1/4 (`results/v1/crosssim_summary.json:2-13`). The tracker accurately distinguishes completed inference/pilot work from unstarted PPO and incomplete full reproduction (`EXPERIMENT_TRACKER.md:3-13`). The report's table matches the recomputed private arrays (`reports/PhyGile_Unitree适配与跨仿真报告_20261022.md:54-65`).

The warning is provenance-related:

1. Isaac writes only `sha(__file__)` into `source_sha256` (`scripts/lab_run.py:25-30`), although the run imports `lab_task.py`, the original PhyGile encoder/config, Unitree code, and Isaac Lab. The recorded `1731b0df...` matches current and archived `lab_run.py`, but it does not fingerprint the imported code graph.
2. Mjlab constructs its receipt without any source digest (`scripts/mjlab_evaluate.py:8-17`). Current `scripts/mjlab_evaluate.py` is SHA-256 `e309f901...`; the archived `results/v1/server_evidence/code_v5/scripts/mjlab_evaluate.py` is `9d464464...`. The only textual difference is line 36's `list(...)` conversion and diagnostic, matching the retained list/tuple failure described at `reports/PhyGile_Unitree适配与跨仿真报告_20261022.md:71`. This is consistent with the final run, but the final receipts do not themselves bind that repaired source.
3. `docs/COMPUTE_LEDGER.md:3-6` gives canonical digest `a43f960b...` for `configs/environment_plan_v2.json`. The current LF file digest is `639a7e9b...`, and its CRLF byte variant is the archived `1453e96c...`; line-ending normalization therefore explains the raw-byte change. However, no canonicalization algorithm is specified, and standard sorted compact JSON produced `37aa1870...`, not the stated digest. The environment values and resolved package lock were independently verified, so this affects the ledger identifier rather than the scientific metrics.

### D. Dead Metric Paths: PASS

The three reported metrics are computed in both evaluators and recomputed again in the cross-simulation analyzer (`scripts/lab_run.py:192-197`, `scripts/mjlab_evaluate.py:59-66`, `scripts/analyze_crosssim.py:18-30`). Their outputs appear in each receipt and the summary. Termination, frame-continuity, finite-value, and completion checks execute on the same path before results are saved (`scripts/lab_run.py:180-198`, `scripts/mjlab_evaluate.py:54-75`).

The training branch is live code (`scripts/lab_run.py:143-157`) but was not executed. It is not presented as a completed metric path: the tracker says `NOT_STARTED` (`EXPERIMENT_TRACKER.md:6,13`) and the report repeats that no new PPO was launched (`reports/PhyGile_Unitree适配与跨仿真报告_20261022.md:11,79`).

### E. Scope, Seed, and Claim Matching: PASS

The actual scope is four existing 149-frame clips, two simulation engines, one deterministic episode per clip/engine, seed 2026 (`results/v1/crosssim_summary.json:4-7,230`; `reports/PhyGile_Unitree适配与跨仿真报告_20261022.md:48-50`). Failures are censored at first termination and interpreted with duration/success (`reports/PhyGile_Unitree适配与跨仿真报告_20261022.md:52,65`). The documents call this a small simulation pilot and expressly reject paper-scale, equivalence, generalization, and real-robot conclusions (`README.md:32-34`; `reports/PhyGile_Unitree适配与跨仿真报告_20261022.md:7-11,79-82`). The evidence matches that limited language.

### F. Evaluation Type Classification: PASS

- Tracking errors: `real_gt` for reference provenance plus `simulation_only` for execution. Evidence: exact dataset-array equality; explicit receipt classification at `scripts/lab_run.py:192-198` and `scripts/mjlab_evaluate.py:69-75`; summary classification generated at `scripts/analyze_crosssim.py:31-43`.
- Frozen-policy numerical parity: `synthetic_proxy` implementation check, not performance. Evidence: two five-row synthetic fixtures and explicit scope at `scripts/verify_reference_v3.py:10-21` and `results/v1/frozen_policy_parity_v3.json:50`.
- Compiled dynamics: `simulation_only` static diagnostic with zero physics steps (`scripts/check_mjlab_dynamics.py:8-20`).
- Document witness: `simulation_only` bounded CPU probe; the witness explicitly limits the verdict (`results/v1/doc_witness_report_v1/REPORT.md:51-57`).
- No `human_eval` or real-robot evaluation is claimed (`reports/PhyGile_Unitree适配与跨仿真报告_20261022.md:82`).

## Frame, termination, reset, and permutation audit

Isaac forces the initial command frame to zero, realigns the relative body reference, then records inside the termination computation before the automatic reset (`scripts/lab_run.py:79-81,159-179`). It stops at the first done signal and asserts frames are contiguous (`scripts/lab_run.py:180-188`). Mjlab resets, recomputes the initial observation, refreshes derived kinematics with `sim.forward()` before termination capture, then stops at first termination/timeout (`scripts/mjlab_evaluate.py:29-63`). The installed Mjlab step implementation advances physics, increments counters, computes termination, resets, forwards, and only then advances commands (`D:/phygile_unitree_env_20261022/Lib/site-packages/mjlab/envs/manager_based_rl_env.py:368-403`); its motion command increments at `D:/phygile_unitree_env_20261022/Lib/site-packages/mjlab/tasks/tracking/mdp/commands.py:365-393`. These orders are consistent with the recorded post-physics/pre-reset reference frame `t` and the observed contiguous `0..N-1` frames.

The Mjlab adapter builds the 520D command from the eight declared offsets and selects joints by name (`scripts/mjlab_task.py:28-39`), constructs the 616D observation in policy order (`scripts/mjlab_task.py:41-48`), converts source motions by name (`scripts/mjlab_task.py:72-81`), and permutes policy actions to the framework's natural joint order (`scripts/mjlab_evaluate.py:31-39,54-57`). All four receipt mappings are 29-element bijections with valid inverses; converted arrays and initial commands were exact.

## Claim impact

- **Supported:** the adapters executed real frozen-policy simulation; success counts are Isaac 2/4 and Mjlab 1/4; the displayed failure-censored metrics are exact.
- **Supported:** the same checkpoint and saved observation normalizer were used; policy I/O parity passes the stated tolerance; TF32 sensitivity is retained rather than hidden.
- **Supported:** no new PPO continuation or new training weights were produced. Existing optimizer metadata describes the parent checkpoint only.
- **Supported with provenance qualifier:** exact execution from the published source snapshot. The numeric artifacts are coherent, but historical receipts only partially fingerprint executed source.
- **Supported only as scoped:** compiled parameter equality and bounded CPU/GPU witnesses. They do not establish solver, geometry, framework, paper-scale, or real-robot equivalence.
- **Needs correction or method disclosure:** the `a43f960b...` canonical environment-plan digest in the compute ledger.

## Action Items

1. For future runs, write a pre-execution manifest into every receipt that hashes the entry point, all project imports (`lab_task.py`, `mjlab_task.py`, `frozen_policy.py`, encoder/config), contract, framework commit, and environment lock. Bind the manifest digest to the receipt before stepping.
2. Define the exact canonicalization used for `environment_plan_v2.json` and publish a reproducer, or replace the ledger's canonical digest with a reproducible raw-file/normalized-text digest pair.
3. Keep the source-attestation limitation visible when publishing these historical runs. Do not retrofit new hashes into old receipts.
4. Treat this same-family audit as provisional. A later independent-family reviewer can increase review independence; the deterministic file/array findings do not depend on that optional step.

## Audited snapshot hashes

| Path | SHA-256 |
|---|---|
| `README.md` | `e545334180319e1d9495c88abb19bfaa27949718802af4ed7e4c8246e203bbef` |
| `reports/PhyGile_Unitree适配与跨仿真报告_20261022.md` | `5eb10fdf11d736821e05363afcaaeff90a3149a742b010004c8374c94c6f893a` |
| `EXPERIMENT_TRACKER.md` | `6dd24ab2c5b5fbbb7214c6fc5a5d05888ab2a573d0d987d0d175541fdb1fe4b6` |
| `docs/ENVIRONMENT.md` | `bba284ec60d47bd2e16aa2263f4e81dfe1b29c617c4b6449d8d881b4199d68e1` |
| `docs/COMPUTE_LEDGER.md` | `6539bc37b5fe0835eb6e50ab615023446edc5d8c8c414c128cd1762a85fdcca3` |
| `configs/environment_plan_v2.json` | `639a7e9b5e007f6f65fef2b1a8476495577a6b47154c451fb20e651958b05b79` |
| `configs/mjlab_resolved_requirements_v2.txt` | `7590bde7b6932f549229da1f1bacabf5a393a3d724b83cde3225ec6e86242303` |
| `scripts/lab_run.py` | `1731b0df2b85e178754c7ded162f34d32eebbe0b1a156a224437cfff3581e95d` |
| `scripts/lab_task.py` | `6a2df47b6705e90b6de9305a7e6c4eee91fe26f2ac07ef21a38f273a967749dd` |
| `scripts/mjlab_evaluate.py` | `e309f901ba526ad3cf0e15f79f3994b984f8e850bc18717f5c43ff34454ded29` |
| `scripts/mjlab_task.py` | `9430a1759ee774456475b4aa03bf5dccafb41f76cd90f0146ddfc15638d47882` |
| `scripts/frozen_policy.py` | `2ae2cedee9537487c99b93cae410c093845f7c1a51a33134a9c04a852390df22` |
| `scripts/analyze_crosssim.py` | `10071f3a51025aa51c1ce64dd9eeeec0c95b5f8495864db2c836bd8d30df9568` |
| `scripts/verify_reference_v3.py` | `fd11944f1881a2b95a1c35e12b422a01b8b7e7ec78d9b0654e7e885418388c4b` |
| `scripts/check_mjlab_dynamics.py` | `ec20ace08740fc9a715a88aedc3acdf94d1a1ea2d35c6fe8b8031f12f9307da4` |
| `results/v1/contract.json` | `8ab68b24aa18bda313e594b3f8807514e38bfc8d1d00e4f3e61a92e1b18a59f1` |
| `results/v1/crosssim_summary.json` | `e9cba1c5d6c6318b964cd0c19fa5165ec45252c89e5996bdca0486570d24cda9` |
| `results/v1/private_raw_manifest.json` | `c3c32c8d03013cf370478adf0be4d715ad34430198e2c62b27a760e5f646276e` |
| `results/v1/private_asset_manifest.json` | `102a73a8a7482ea4e352b3b2b7727ea374b69b18ba48ce93a2a101fe8927acf9` |
| `results/v1/frozen_policy_parity_v3.json` | `4ea6bcf2bd891c60e83eb067c52103a6e6c914154e25ef3b462befd494b808aa` |
| `results/v1/mjlab_compiled_dynamics_v1.json` | `97bd543800f263c059c2df65dcc0eef316ca5a6477b55f8538de477c54f31d97` |
| `results/v1/author_source_manifest.json` | `7eae44c157be080f1ac30cc164b9f0752c520d18918fa33feb565b1fc343b7e4` |
| `results/v1/server_evidence_manifest_v1.json` | `4184c6c8949d904514cd4daff69f4b67542ff14f5a3428044c603db53b697e0b` |
| `results/v1/doc_witness_report_v1/REPORT.md` | `9695cf7ce34786daa65d9ddfdd280cef0568ab144f87a3d12709ea2da433d43b` |
| `results/v1/audit_route_attempt01_error.json` | `cfaacd96fffa3891be3cfd8cc6add05a9646aa0bce0cf7b264f7b0172df69e17` |
| local `private_assets/parent_checkpoint.pt` | `bb067f364de74e2da7cf1879e6f26bb961a71eb9ff64cbfbe84dd858bfae2336` |

The machine-readable audit contains receipt-level hashes and structured verification results.

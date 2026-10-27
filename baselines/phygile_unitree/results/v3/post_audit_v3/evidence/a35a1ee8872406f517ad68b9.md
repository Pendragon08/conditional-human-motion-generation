# Independent V3 Experiment Integrity Supplement

**Audit cutoff:** 2026-10-22 23:00 +08:00 (15:00 UTC)  
**Skill:** `experiment-audit`  
**Requested reviewer route:** `gpt-5.6-sol`, reasoning `ultra`  
**Reviewer route verification:** requested settings recorded; the actual backend identifier and effort were not independently exposed  
**Executor self-reported label:** Codex based on GPT-6 (from runtime instructions; exact backend identifier not independently verified)  
**Review independence:** `same-family`  
**Acceptance status:** `provisional`  
**Input mode:** read-only; no training, physics rerun, source repair, or server-process operation  
**Overall verdict:** **WARN**  
**Integrity status:** `warn`

Path aliases used below:

- `P` = `C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/phygile_unitree`
- `D` = `D:/具身智能/PhyGile复现/Unitree框架适配_20261022`
- `R` = `P/../phygile/reimplementation`

The primary result is internally consistent and materially honest. The 500-iteration continuation is backed by a readable finite full checkpoint, finite initialized Adam state at step 50,000, an exact compact-policy binding, recomputable raw Isaac/MuJoCo metrics, and a current-loader execution. The negative scientific result is stated plainly: the new policy did not improve the four reused development clips, actual generated 50-frame candidates were rejected, and zero generated frames were committed.

WARN is retained for the executed-path documentary boundary in check D and for same-family review limits, rather than a fabricated result. Other bounded verification and metadata limits are recorded even where the corresponding check passes:

1. the archived four-batch production witness is for the parent `bb067f… -> 07b15d…` pair; this audit directly verified the new `30ac49… -> 28eb2d…` pair at batches 1/7/128/1024, but that combination did not have a pre-existing archived four-batch receipt;
2. the runtime compact loader enforces tensor-schema and omitted-branch safety but leaves source-checkpoint/contract/encoder digest enforcement to a separate verifier;
3. the aggregate V3 multi-label classification is correct, while the immutable new PhysX receipt itself still has singular `simulation_only` metadata and the additive classification supplement names earlier closure records;
4. the release `resume2` asset has sealed metadata and a publication receipt but no local byte copy for this audit;
5. the historical V2 trace package is preserved except that its 542-byte `trace/events.jsonl` is committed indirectly by its indexed output manifest rather than being a direct evidence-index object.

## A. Ground-truth and reference provenance — PASS

The tracking evaluations load private motion arrays as reference inputs, separately record reference and simulated robot states, and compute direct errors between them (`P/scripts/mjlab_evaluate.py:20-26,54-72`; `P/scripts/lab_run.py:197-224`). Those rows are classified as `real_gt + simulation_only` development diagnostics, not as a held-out paper benchmark (`P/results/v3/summary.json:149-341`; `P/reports/PhyGile_Unitree续训500次与闭环迁移最终报告_20261022.md:40-52`).

Generated-candidate targets are derived from candidate root/joint state by forward kinematics; observed robot positions are explicitly forbidden as desired targets (`P/scripts/unitree_prefix_adapter.py:77-92`). The PhysX receipt records an untrained generator, actual diffusion and physics, two rejected 50-frame candidates, and zero committed frames (`P/results/v3/evidence/bbcc13fda1be26a6.json:2-29,30-100`). The held-initialization fixture is explicitly a self-supervised proxy rather than dataset truth or learned semantics (`P/results/v3/evidence/bbcc13fda1be26a6.json:110-155`; `D/closure_mjlab_continue500_v1/verification.json:106-142`).

No model output is presented as dataset ground truth. The final report also rejects full reproduction, stable semantic generation, generalization, and robot-performance claims (`P/reports/PhyGile_Unitree续训500次与闭环迁移最终报告_20261022.md:17-20,70-76,98`).

## B. Raw metrics and score normalization — PASS

The Isaac and MuJoCo paths report raw metre MPJPE, radian joint-angle error, and rad/s joint-velocity error computed directly from recorded reference and robot arrays (`P/scripts/lab_run.py:212-224`; `P/scripts/mjlab_evaluate.py:65-82`). The physics-prefix acceptance logic uses fixed tolerances and rejects incomplete or terminated continuations; it does not normalize by a statistic of the candidate prediction (`R/physics_prefix.py:75-120`).

The 616-dimensional empirical policy normalizer is a serialized checkpoint input transform, applied before inference as `(obs - mean)/(std + 1e-2)`; it is not used to rescale reported quality scores (`P/scripts/frozen_policy.py:24-31`). The compact artifact's four normalizer fields are bitwise equal to the full checkpoint and have the recorded value hashes (`P/results/v3/new_policy_artifact_binding.json:8-40`). No prediction-dependent max/min/mean denominator or suspicious score inflation was found.

Independent recomputation from every available raw new-policy trajectory exactly matched all four Isaac summary rows and all eight MuJoCo receipts; all arrays were finite and their frame indices contiguous. The detailed values are preserved in `D/independent_integrity_v3/DETERMINISTIC_CHECKS.json`.

## C. Report, status, checkpoint, optimizer, evaluation, and release consistency — PASS

The report's central training claims match the sealed training receipt:

- 500 additional PPO iterations, 10,000 actual Adam calls, Adam step 40,000 to 50,000, 12,000 control calls, 21 changed unique named parameter tensors, final SHA-256 `30ac49b42a793ec8832c72fafacce5bd0a4e51abca9ed2868aebf0e5dbc528ee` (`P/results/v3/evidence/26023b2d4bf2952e.json:2-15,33-68`; `P/results/v3/summary.json:2-74`; report `:26-36`);
- the early training launcher's child exit code is explicitly `not recorded`, while the dependent evaluation/export pipeline is recorded as exit 0 (`P/EXPERIMENT_TRACKER.md:8-11,24`; report `:36`);
- four-clip completion counts and negative outcome match the summary and report (`P/results/v3/summary.json:138-341`; report `:40-52`);
- both new-policy closures report zero generated commits and an untrained generator (`P/results/v3/summary.json:343-368`; report `:68-76`).

The locally downloaded full checkpoint is 67,350,908 bytes and SHA-256 `30ac49…28ee`, matching both the independent download receipt and release manifest (`D/downloaded_full_checkpoint_verification_v3.json:2-4`; `P/results/v3/release/github_release_assets_v3.json:13-18`; `P/results/v3/release/github_weight_ppo2500_receipt.json:2-10`). Two independent hash passes were stable, the 178-member ZIP container passed integrity testing, and CPU `torch.load` succeeded.

Direct checkpoint inspection found `iter=2499`, 97 model-state tensors, and 180/180 tensors finite. Adam has one 91-parameter group; 25 active/shared/critic/gate/std entries have finite `step`, `exp_avg`, and `exp_avg_sq`, all at step 50,000. The other 66 IDs correspond exactly to inactive experts 1–11 and retain valid lazy/uninitialized Adam state, the same architectural pattern as the parent. Parent state is `iter=1999`, Adam step 40,000. There are 27 changed state-dict entries because shared tensors have duplicate aliases, corresponding to exactly 21 changed unique named parameters. The training code itself requires exact restore, finite gradients/state, and exactly the bounded Adam-call count (`P/scripts/lab_run.py:91-106,165-183`).

The compact and PPO2500 assets both match the release's byte sizes and digests (`P/results/v3/release/github_release_assets_v3.json:6-18`). The auxiliary `resume2` release record declares 67,350,908 bytes and `cf2d6fac…48fe` (`:20-25`) and has a sealed publication receipt (`P/results/v3/release/github_weight_resume2_receipt.json:2-10`), but no local byte copy was available under P or D. The report claims that the server-reported size/digest were verified, which the sealed receipt supports; it does not claim this audit independently downloaded every asset. The missing local `resume2` bytes remain a precise limitation but do not contradict a material report claim, so check C passes.

## D. Executed current loader, bindings, source snapshots, and historical limits — WARN

The compact artifact is tightly bound offline. It records the full-checkpoint, contract, and encoder hashes; retains exactly 23 full-checkpoint tensors; omits exactly 74 critic/inactive-expert tensors; embeds all four policy-normalizer fields bitwise; contains no optimizer; and says training resume is unsupported (`P/results/v3/new_policy_artifact_binding.json:2-51`). The contract independently records the 616/748 observation dimensions, full MoE configuration, full-checkpoint hash, and one active expert (`D/private_assets_continue500_v1/contract.json:355-426`). The binding verifier enforces all three source hashes, exact key sets/tensors, normalizer schema/dtype/value, and full model/optimizer finiteness (`P/scripts/verify_artifact_binding.py:10-41`).

The current production loader actually dispatches compact payloads through `load_active`, consumes the serialized policy normalizer, and installs fail-closed pre-hooks on the critic and every inactive expert (`P/scripts/frozen_policy.py:15-32`; `P/scripts/active_policy_artifact.py:10-23`). In this audit, the current full and compact policies were loaded independently and executed on identical CPU inputs at batches 1, 7, 128, and 1024. Every output was finite and bitwise identical with maximum delta 0. Deliberate calls into the critic, expert 1, and expert 11 each raised `Compact inference entered an omitted branch`; all 11 inactive experts carried hooks. The compact CPU output also matched the archived five-row GPU actions within `3.337860107421875e-6 < 1e-5`, agreeing with `P/results/v3/continue500_gpu_cpu_parity.json:2-9`.

The formal V3 document witness passed one recorded command, the same four batch sizes, source-snapshot equality, branch guards, and zero updates (`D/production_active_doc_witness_v3/witness_report.json:2-19,21-58`; `D/production_active_validation_v3/production_proof.json:17-62`). Its archived matrix is for parent full/compact hashes `bb067f…` and `07b15d…` (`production_proof.json:47-48`). The new compact `28eb2d…` was executed by all eight recorded current-loader MuJoCo runs (`D/continue500_mjlab_comparison_v1/comparison.json:2-10,106` and the corresponding later run receipts), and this audit supplies the missing direct four-batch check. Because no single pre-existing archived receipt combined the new artifact with that four-batch matrix, the documentary scope remains WARN.

The loader validates artifact kind, active expert, exact present/omitted tensor sets, and normalizer presence, but it does not enforce the embedded checkpoint/contract/encoder digests at runtime; those are enforced by the separate verifier (`P/scripts/frozen_policy.py:15-25`; `P/scripts/active_policy_artifact.py:10-19`; `P/scripts/verify_artifact_binding.py:13-15`). Stronger deployment claims should invoke that verifier or integrate equivalent hash checks before inference.

Historical limitations and negative evidence remain visible. The V2 production-loader failure is intact (`D/production_active_doc_witness_v2/witness_report.json:2-41`; `D/production_active_doc_witness_v2/full_stderr.txt:1-5`), and the later batch diagnosis explains the cross-batch FP32 rounding issue without rewriting it (`P/results/v3/production_batch_diagnosis_v2.json:1-29`). Source manifests capture loaded Python files at capture time and explicit inputs, with package versions standing in for other dependencies; they do not fully bind native binaries, all robot assets, or later lazy imports (`P/scripts/source_provenance.py:11-36`; report `:86`).

## E. Seeds, data scope, accepted horizons, and performance language — PASS

The evidence is narrow and the public language states that scope accurately:

- 20 short training clips, 12 configured experts but one active, 128 environments, and a bounded 500-iteration continuation (`P/reports/PhyGile_Unitree续训500次与闭环迁移最终报告_20261022.md:26-28`);
- four reused 149-frame development-diagnostic clips at one seed, 2026, per configuration; no unseen test motion, additional seed, confidence interval, or significance claim (`:40-52`; `P/README.md:39-41`);
- new-policy success of 1/4 in Isaac and 1/4 in official MuJoCo, with no observed improvement over the retained parent baseline (`P/results/v3/summary.json:138-341`; report `:42-52`);
- two 50-frame generated attempts in each engine, all rejected after one recorded frame, with zero generated frames committed; separately accepted one-frame held-pose windows test only feedback/rollback plumbing (`P/results/v3/evidence/bbcc13fda1be26a6.json:23-100,110-155`; `D/closure_mjlab_continue500_v1/verification.json:28-142`);
- the generator remains untrained and zero/unconditionally conditioned; no stable semantic-generation claim is made (`D/closure_mjlab_continue500_v1/verification.json:13-27`; report `:74-76`).

The report explicitly limits the conclusion to recoverable training, working framework connections, and measured degradation; it rejects complete-paper, stability, generalization, semantic-generation, and real-robot claims (`P/reports/PhyGile_Unitree续训500次与闭环迁移最终报告_20261022.md:20,52,66,76,98`). The scope wording therefore passes.

## F. Multi-label evaluation classification — PASS

The correct classifications are:

| Component | Classification | Evidence |
|---|---|---|
| Private-motion tracking | `real_gt + simulation_only`, development diagnostic | `P/scripts/mjlab_evaluate.py:20-26,54-72`; `P/results/v3/summary.json:149-341` |
| Generated-candidate tracking | `synthetic_proxy + simulation_only` | `P/scripts/unitree_prefix_adapter.py:77-92`; `P/results/v3/summary.json:343-362` |
| Held-initialization feedback/replay | `self_supervised_proxy + simulation_only` | `P/results/v3/summary.json:363-365`; PhysX receipt `:110-155`; Mjlab receipt `:106-142` |
| Full-vs-compact / GPU-vs-CPU inference parity | `synthetic_proxy` implementation check, no motion-quality claim | `P/results/v3/classification_supplement_v2.json:26-32`; `P/results/v3/continue500_gpu_cpu_parity.json:2-9` |
| Bounded continuation | training-integrity evidence, not a performance evaluation | `P/results/v3/evidence/26023b2d4bf2952e.json:2-68` |

The final summary and additive supplement contain the correct multi-label arrays (`P/results/v3/summary.json:335-368`; `P/results/v3/classification_supplement_v2.json:2-32`). The immutable new PhysX receipt has only top-level `"evaluation_type": "simulation_only"` (`P/results/v3/evidence/bbcc13fda1be26a6.json:2-5`) and the supplement's `records` field names earlier closure paths (`P/results/v3/classification_supplement_v2.json:3-11`). These are minor record-level traceability limits: the current aggregation preserves all applicable labels, the prose is explicit, and historical receipts were not silently overwritten. No mutually exclusive classification error or hidden proxy remains, so check F passes.

## Seal, index, preservation, and versioning

The public seal declares all public-root bytes sealed at 2026-10-22 14:37:20 UTC and explicitly requires later publications to use a new named seal (`P/results/v3/audit_input_seal_v3.json:2-4`). Independent rehashing found:

- **903/903** declared public-root files present and matching;
- **0** missing, mismatched, or unsealed in-scope additions under the recorded exclusions;
- **246/246** evidence-index targets present and matching, with no duplicate target;
- evidence-index SHA-256 `9815535f…0768`, exactly the hash sealed at `audit_input_seal_v3.json:97`;
- the original V2 audit Markdown/JSON byte-identical to the D originals, and historical V2 snapshots and release receipts preserved by the seal (`audit_input_seal_v3.json:867-906` and `:875-877`).

The D V2 audit directory contains 15 files. Fourteen are direct V3 evidence-index records; only `independent_integrity_v2/trace/events.jsonl` is omitted. Its 542 bytes and SHA-256 `11ec228d…b8c5` remain committed by the indexed V2 output manifest (`D/independent_integrity_v2/OUTPUT_MANIFEST.json:50-63`). This is a trace-publication completeness gap, not loss of the request/response/meta traces or alteration of the V2 verdict.

The downloaded PPO2500 checkpoint, its download verification receipt, the private Isaac arrays, and this V3 audit package postdate the 14:37 seal and/or live outside P. They are supplemental inputs/outputs and do not retroactively inherit `audit_input_seal_v3`. They should be included only by the later named final seal described in the existing seal's scope.

## Claim impact

| Claim | Impact |
|---|---|
| 500 additional PPO iterations and 10,000 actual Adam calls completed | **Supported** by sealed receipt and directly inspected full checkpoint |
| PPO2500 full checkpoint is finite and resume-capable | **Supported**; model, normalizers, and initialized optimizer moments are finite |
| New compact is bound to the full checkpoint and safe for the active inference path | **Supported with qualifier**; offline digest verifier must accompany stronger deployment identity claims |
| Current new full/compact outputs agree | **Supported** for direct CPU batches 1/7/128/1024 and saved five-row GPU tolerance; the four-batch new-policy check was created by this audit rather than a pre-existing archived witness |
| New policy improved stability or tracking | **Unsupported**, and the report correctly records no improvement |
| Generated closure produced stable learned motion | **Unsupported**; generator untrained, all real 50-frame candidates rejected, zero generated commits |
| Complete PhyGile reproduction, generalization, or robot performance | **Unsupported**, and the public narrative correctly says so |
| Every release asset was independently byte-verified locally | **Unsupported** for `resume2`; PPO2500 and compact are verified |

## Precise remaining limitations

1. Same-family review is provisional; requested `gpt-5.6-sol/ultra` settings were not independently observable.
2. Single seed and four reused development clips do not support statistical, held-out, robust, or generalization claims.
3. All generated 50-frame proposals failed; no accepted learned generated horizon exists.
4. The generator is untrained and zero/unconditionally conditioned.
5. No real-robot deployment or observation exists.
6. PhysX and MuJoCo contact/dynamics are not proven equivalent; static geometry/material comparisons do not establish dynamic equivalence.
7. The runtime compact loader delegates digest identity checks to a separate verifier.
8. No pre-existing archived witness combines the new compact with the 1/7/128/1024 matrix, although this audit directly executed it.
9. The release `resume2` bytes were unavailable locally.
10. The original training launcher's child exit code remains unrecorded; completion is supported by the final receipt/checkpoint, not by an invented exit 0.
11. Source provenance does not fully content-bind native binaries, all USD/mesh assets, or later lazy imports.
12. The existing V3 seal must not be retroactively edited; later metadata, the omitted V2 event log, this audit, and publication receipts require a separately named final seal.

## Audit artifacts

- `EXPERIMENT_AUDIT_V3.json` — machine-readable verdict, A–F statuses, claims, and limitations.
- `AUDITED_INPUT_HASHES_V3.json` — hashes for every explicit input plus eight recursively hashed D collections.
- `DETERMINISTIC_CHECKS.json` — checkpoint, optimizer, loader, raw-metric, seal, and evidence-index checks.
- `trace/` — reviewer requests, verbatim responses, per-call metadata, run metadata, and `events.jsonl`.
- `OUTPUT_MANIFEST.json` — SHA-256 and byte size of every output other than itself.

**Final verdict: WARN (same-family, provisional). No integrity FAIL was found.**


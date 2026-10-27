# Independent V2 Experiment Integrity Audit

**Audit cutoff:** 2026-10-22 20:43 +08:00 (12:43 UTC)  
**Auditor:** fresh GPT-6-based Codex reviewer, read-only  
**Review independence:** `same-family`  
**Acceptance status:** `provisional`  
**Project:** PhyGile Unitree Lab/Mjlab V2 continuation and dual-simulator closure  
**Overall verdict:** **WARN**  
**Integrity status:** `warn`

This verdict is for the exact bytes recorded in `AUDITED_INPUT_HASHES.json`. The parent reported that `EXPERIMENT_TRACKER.md`, `README.md`, and `docs/COMPUTE_LEDGER.md` changed around audit dispatch. I discarded cached interpretations, read their current bytes, and bound this audit to SHA-256 `51376c8a…`, `0dc83eab…`, and `ec5a1dfd…` respectively. No experiment was launched or retried, no server was contacted, and no source, result, or running process was modified.

The numerical and execution evidence is substantially honest. The two-iteration recovery run proves exact optimizer restoration and real finite updates. Both simulator closures executed real physics and real untrained diffusion proposals, rejected the invalid generated candidates, and committed zero generated frames. The eight contact runs support the reported 1/4 versus 1/4 negative result. The 500-iteration run is still unaccepted and must remain pending/unknown. The WARN is caused by traceability and classification gaps: the compact artifact parity proof is not an end-to-end production-loader test and its exact exporter source bytes are absent; the V2 publication snapshot does not seal its summary/index or narrative documents; the V2 report implied an archived integrity audit while the tracker still said PENDING at the audit cutoff; and the closure records omit the additional `synthetic_proxy` label for generated-reference tracking.

Path aliases used below:

- `P` = `C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/phygile_unitree`
- `D` = `D:/具身智能/PhyGile复现/Unitree框架适配_20261022`
- `R` = `P/../phygile/reimplementation`

## A. Ground-truth provenance — PASS

The contact diagnostics load a private motion array (`test_NNN.npz`) and store the reference body/joint position and velocity separately from the simulated robot outputs (`P/scripts/mjlab_evaluate.py:20,54-57`). The three metrics are direct errors between those arrays (`P/scripts/mjlab_evaluate.py:70-72`). These clips have already been used for development and are not presented as untouched test data (`P/scripts/run_contact_comparison.py:17`; `P/reports/PhyGile_无界面续训与双仿真闭环报告_20261022.md:53`).

The generated closure target is not mislabeled as dataset truth. The Unitree adapter computes every desired full-body reference from the generated candidate’s root/joint state using forward kinematics and explicitly forbids substituting observed robot positions (`P/scripts/unitree_prefix_adapter.py:77-87`). Both closure receipts say the generator is untrained and the result is engineering validation with zero committed generated frames (`D/closure_unitree_doc_witness_v3/remote/runs/closure_unitree_v3/verification.json:2-14,26`; `D/closure_mjlab_doc_witness_v1/verification.json:2-23`). The separately accepted one-frame held-initialization fixtures are explicitly `self_supervised_proxy`, not learned generation or dataset truth (`D/closure_unitree_doc_witness_v3/remote/runs/closure_unitree_v3/verification.json:111-147`; `D/closure_mjlab_doc_witness_v1/independent_artifact_verification.json:133`).

No fabricated ground truth was found. The generated-candidate tracking portions should additionally be labeled `synthetic_proxy`; that metadata issue is recorded under F rather than treated as hidden fake ground truth.

## B. Score normalization — PASS

The physics-prefix decision uses raw metre errors, requires a complete nonterminated continuation, and rejects mean MPJPE above a fixed externally supplied tolerance (`R/physics_prefix.py:75-120`). The contact evaluator reports raw MPJPE, joint-angle error, and joint-velocity error without dividing by any statistic of the prediction (`P/scripts/mjlab_evaluate.py:70-82`). Failure-censored metrics are disclosed together with evaluated duration and full-clip success (`D/contact_comparison_v4/comparison.json:116-120,341`; repeated for all eight runs).

The 262D feature normalizer is fit only from supplied finite training sequences and stores fixed training means/stds (`R/features.py:25-44`). It is used for generation inputs, not as a denominator derived from the candidate being scored. No prediction-dependent max/min/mean normalization or suspicious rescaling was found.

## C. Result existence and claim consistency — WARN

### Deterministic checks that passed

- All **287/287** `results/v2/evidence_index.json` entries exist and match their declared SHA-256; zero are missing or mismatched.
- All **404/404** files listed in `results/v2/publication_snapshot.json` exist and match; zero are missing or mismatched.
- All **29/29** files in the immutable `D/source_mj_closure_v1/source_manifest.json` match the current project bytes.
- The two-iteration smoke receipt reports `BOUNDED_TRAINING_COMPLETED`, exact policy/normalizer/optimizer restoration, 40 literal Adam calls, Adam steps `40000→40040`, 21 changed policy tensors, and finite state (`D/headless_smoke_doc_witness_v3/acceptance_report.json:42-60`). The independent checkpoint audit found final iteration 2001, 27 changed model-state tensors, and no nonfinite model or optimizer tensors (`D/headless_smoke_doc_witness_v3/independent_audit.json:50-60`). This supports restoration/training plumbing only, not performance gain.
- Every contact row in `P/results/v2/summary.json:9-98` matches `D/contact_comparison_v4/comparison.json`: official frames `[129,10,149,14]`, matched frames `[129,9,149,15]`, and 1/4 success for each. Seed 2026, zero optimizer updates, development-diagnostic status, and no test-performance selection are recorded in code (`P/scripts/run_contact_comparison.py:6-17`).
- PhysX and MuJoCo closure receipts support actual real-simulator execution, two rejected 50-frame generated proposals in each engine, only one recorded frame before failure per proposal, and zero generated commits (`D/closure_unitree_doc_witness_v3/remote/runs/closure_unitree_v3/verification.json:2-58`; `D/closure_mjlab_doc_witness_v1/verification.json:2-52`).
- The three documented failures are retained: the first headless argument failure (`D/headless_smoke_doc_witness_v2/acceptance_report.json:106-141`), invalid CPU20–23 affinity (`D/closure_unitree_doc_witness_v2/acceptance_report.json:10-18`), and missing-BOM path corruption (`D/contact_doc_witness_v3/REPORT.md:3-20,40`).
- The current status correctly says the 500-iteration run is incomplete: `P/results/v2/summary.json:2-3`, `P/EXPERIMENT_TRACKER.md:8-9`, and `P/results/v2/server_snapshot/runs/post_continue500_v1/pipeline.json:2-6`. The public training receipt is only `SIMULATION_READY` (`P/results/v2/server_snapshot/runs/headless_continue500_v1/receipt.json:2-12,33-51`). The later local read-only snapshot at `D/protected_progress_snapshot_v2.json:1` records iteration `2160/2500`, 494592 timesteps, and a still-waiting pipeline at 12:29:33 UTC. It proves progress at that instant, not completion or present liveness.

### Reasons for WARN

1. At the input cutoff, the V2 report said the integrity review was separately archived (`P/reports/PhyGile_无界面续训与双仿真闭环报告_20261022.md:71`), while the tracker said `v2_independent_integrity_audit: PENDING` (`P/EXPERIMENT_TRACKER.md:25-26`). The only pre-existing `P/EXPERIMENT_AUDIT.md:15-30` is a V1/no-new-training audit and does not bind the V2 result set. This report supplies a V2 audit now, but does not retroactively remove the contradiction in the audited input snapshot.
2. The publication collector built its file list before writing the evidence index/snapshot. Consequently the 404-file snapshot omits the substantive `results/v2/evidence_index.json` and `results/v2/summary.json` plus the intentional self-reference. It also omits the V2 report, tracker, README, compute ledger, and this audit. The listed contents are intact, but the publication root is not fully sealed.
3. The compact artifact proof reports bitwise equality on 1024 seeded rows (`D/parent_active_policy_v1/artifact_proof.json:2-8,85-95`), but the harness instantiated `compact` from the full checkpoint and replaced only the model state (`P/scripts/export_active_policy.py:23-30`). The normalizer used in the equality check therefore came from the full checkpoint rather than from reloading `active_policy.pt`; the production compact-loading branch was not executed. The report’s parity statement must be narrowed to active-model-branch parity under the full-checkpoint normalizer until an end-to-end load is witnessed.

## D. Executed code versus dead/phantom paths — WARN

The completed smoke, contact, PhysX closure, and MuJoCo closure are not phantom results. Their receipts bind commands, outcomes, source manifests, and raw metrics. The Unitree witness independently rehashed 471 unique loaded source/input files with zero mismatches (`D/closure_unitree_doc_witness_v3/witness_checks.json:2-11,22-82`). The MuJoCo closure witness reports 29 immutable project sources and 255 executed sources still matching (`D/closure_mjlab_doc_witness_v1/independent_artifact_verification.json:10-18`). Contact source drift is openly recorded: execution used `frozen_policy.py` SHA `2ae2ced…`, current is `843c374…`, and the old bytes were archived (`D/contact_doc_witness_v4/post_run_source_audit.json:3-20`).

Four relevant paths remain unexecuted or incompletely attested:

1. **Current compact loader.** `P/scripts/frozen_policy.py:19-25` can load an active artifact, but no available receipt invokes it with `active_policy.pt`. The proof records the older `frozen_policy.py` hash `2ae2ced…`, whose archived code only strict-loads full checkpoints (`D/contact_doc_witness_v4/archived_executed_sources/scripts/frozen_policy.py:15-22`).
2. **Exact compact exporter source.** The proof records exporter SHA `02f031…` (`D/parent_active_policy_v1/artifact_proof.json:91-95`), whereas the current/immutable exporter is `15a007…`. A hash search across project and D Python evidence found no retained file with `02f031…`; it is also untracked in Git. The numerical proof remains evidence of the recorded harness output, but exact source-level reproduction is unavailable.
3. **Post-training path.** `P/scripts/after_training.py:18-42` correctly requires a completed 500-iteration receipt and 10000 Adam calls before evaluation/export. Current pipeline evidence stops at `WAITING_FOR_TRAINING`, so this branch, the final 500-iteration checkpoint, evaluations, and trained export are dead/pending.
4. **Provenance boundary.** `P/scripts/source_provenance.py:14-28` hashes loaded `.py`/`.sh` files at capture time and explicit inputs, with binary dependencies represented only by package versions. It does not content-bind native binaries, robot USD/mesh assets, lazy imports after capture, or imported bytes before capture. This limitation is disclosed in the manifest scope but prevents full runtime-byte attestation.

The compact loader also checks omitted/present model keys and installs forbidden-branch hooks (`P/scripts/active_policy_artifact.py:7-22`), but it does not validate the payload’s recorded source-checkpoint, encoder, or contract hashes, and the normalizer check requires only `_mean`/`_std` keys (`P/scripts/frozen_policy.py:24-25`). Those are safeguards worth retaining, not sufficient artifact binding.

## E. Scope, seeds, data, and accepted horizons — PASS

The evidence is narrow and the narrative generally states that narrow scope accurately:

- 20 original training clips; four 149-frame, 50 Hz clips reused as development diagnostics (`P/configs/continuation_v2.json:7-15`; `P/README.md:37`).
- One seed (2026) per contact profile/clip; eight first episodes total (`P/scripts/run_contact_comparison.py:6-17`; `D/contact_doc_witness_v4/acceptance.json:369-372`).
- Two generated 50-frame proposals per simulator; every generated attempt failed after one recorded frame and zero generated frames were accepted or committed (`D/closure_unitree_doc_witness_v3/remote/runs/closure_unitree_v3/verification.json:26-58`; `D/closure_mjlab_doc_witness_v1/verification.json:23-52`).
- Two accepted one-frame held-initialization proxy windows per simulator; these are rollback/feedback fixtures, not evidence of learned motion quality (`D/closure_unitree_doc_witness_v3/remote/runs/closure_unitree_v3/verification.json:111-147`; `D/closure_mjlab_doc_witness_v1/verification.json:103-131`).
- One two-iteration/40-Adam restoration smoke; no performance comparison.
- The 500-iteration continuation, 10000 Adam updates, final checkpoint, dependent evaluations, and trained compact export were not accepted by the cutoff.

The report and README explicitly reject full-paper, robust cross-simulator, stable semantic-generation, unseen-generalization, statistical-significance, and real-robot claims (`P/reports/PhyGile_无界面续训与双仿真闭环报告_20261022.md:12,24,53,69`; `P/README.md:37`). Because the language does not exceed the evidence, the scope check passes. Any broader scientific claim would require additional seeds, held-out data, longer accepted generated horizons, and a trained generator.

## F. Evaluation-type classification — WARN

Correct audit classification is:

| Component | Classification | Evidence |
|---|---|---|
| Contact tracking against private motion arrays | `real_gt` + `simulation_only` (development diagnostic, not held-out benchmark) | `P/scripts/mjlab_evaluate.py:20,54-72`; `P/scripts/run_contact_comparison.py:17` |
| PhysX/MuJoCo generated-candidate tracking | `synthetic_proxy` + `simulation_only` | `P/scripts/unitree_prefix_adapter.py:77-87`; both closure verification files report untrained generation and rejection |
| Held-initialization acceptance/replay fixtures | `self_supervised_proxy` + `simulation_only` | PhysX verification `:111-147`; MuJoCo verification `:103-131` |
| Two-iteration continuation | bounded training-integrity diagnostic, not performance evaluation | smoke acceptance `:42-60` |
| Compact 1024-row parity | deterministic synthetic-input implementation check, not motion-quality evaluation | artifact proof `:2-8,85-95` |

The stored top-level closure records say only `simulation_only` (`D/closure_unitree_doc_witness_v3/remote/runs/closure_unitree_v3/verification.json:4`; `D/closure_mjlab_doc_witness_v1/verification.json:7`). Their held-pose blocks correctly add `self_supervised_proxy`, but the generated-reference attempt blocks lack `synthetic_proxy`. The prose is honest, so this is a metadata WARN rather than a fabricated-evaluation FAIL.

## Claim impact

| Claim | Impact |
|---|---|
| Exact restoration and real finite update in the two-iteration smoke | **Supported**, bounded to 2 PPO iterations / 40 Adam calls; no performance-gain implication |
| Matched sliding friction did not improve four-clip completion (1/4 vs 1/4) | **Supported**, single-seed development diagnostic only |
| Both frameworks completed real diffusion→physics→feedback→reject→replay wiring | **Supported with qualifier**: engineering validation using an untrained generator; zero generated commits |
| Parent compact artifact is end-to-end bitwise equivalent in the production loader | **Not fully supported**; only active-model-branch parity with the full-checkpoint normalizer is demonstrated |
| 500-iteration continuation is complete or its current process is live | **Unsupported**; only bounded progress-at-snapshot is established; completion remains pending/unknown |
| Stable semantic generation, full paper reproduction, generalization, or robot deployment | **Unsupported**, and current narrative correctly says so |

## Required action items

1. Preserve this audit and its input-hash manifest unchanged. Reconcile report line 71 and tracker line 26 with this V2-specific WARN in an append-only follow-up.
2. For the compact artifact, archive the exact exporter/loader/helper source bytes and run a fresh proof that constructs `FrozenPhyGile(active_policy.pt, contract, encoder)` directly. Bind and validate source-checkpoint, encoder, contract, normalizer schema/shape/dtype/value hash, active-expert ID, and artifact hash. Keep the full resume checkpoint separate.
3. Do not promote the 500-iteration job on `SIMULATION_READY` or a progress line. Require a supervised durable exit record, `BOUNDED_TRAINING_COMPLETED`, exactly 500 added PPO iterations and 10000 Adam calls, finite final optimizer/model tensors, final checkpoint hash, unchanged parent/source inputs, and completed dependent evaluations/export.
4. Generate the final publication manifest after summary, evidence index, report, tracker, README, compute ledger, V2 audit, and any post-training outputs exist. Use normalized POSIX relative paths and explicitly version later snapshots rather than silently mixing them into the 12:25 publication snapshot.
5. Add multi-label evaluation metadata: generated-candidate closure attempts must say `synthetic_proxy` + `simulation_only`; held-pose fixtures remain `self_supervised_proxy` + `simulation_only`; contact arrays remain `real_gt` + `simulation_only` with development-diagnostic status.
6. Expand source provenance if stronger runtime attestation is claimed: snapshot/import exact Python bytes before execution, capture lazy imports after execution, hash robot/assets and environment lockfiles, and bind native binary/package artifacts where feasible.
7. Keep all existing negative outcomes. Require multiple seeds, held-out motions, longer accepted generated horizons, and a trained generator before any performance, stability, semantic-generation, or generalization claim.

## Audit artifacts

- `EXPERIMENT_AUDIT.json` — machine-readable verdict and checks.
- `AUDITED_INPUT_HASHES.json` — 408 independently hashed inputs plus 287-entry index, 404-entry publication, and 29-source immutable-snapshot verification.
- `trace/` — full delegated reviewer prompts and verbatim responses, with same-family/provisional metadata.

No unsupported success is issued. **Overall: WARN (same-family, provisional).**
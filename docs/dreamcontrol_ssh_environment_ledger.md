# Direct SSH environment ledger

Host: `ouyanghao@10.147.17.32`; project `/data/code/ouyanghao/oyh/conditional-human-motion-generation`. Runtime is the existing standalone Isaac Sim 4.5.0 Python3.10 / Torch2.5.1+cu118 and project-local Isaac Lab overlay described by `configs/env-dreamcontrol-r3.json`. No environment rebuild or new packages in this recovery.

## 2026-10-03 official cabinet asset hydration

Dependency delta: replace a Git LFS pointer with its exact official 17599-byte cabinet USD object. SHA256 `30fddfc98008e380fe37d947e995c7875ead6113544b810d0254dc23450b6a7c`; pinned upstream commit `ec28b508313d6837de52c68247fbc8cf81fe4935`. Full task budget remains 8192 environments / 2000 updates / seed42 on GPU1; independent live renderer has its own process. Existing visualization authorization persists.

Status: staged object hash verified; actual scene/PPO readiness still requires fresh-agent diagnostic in `docs/cabinet_asset_recovery_20261003.md`. Pure interpreter `from pxr import Usd` without initializing Isaac Sim failed with ModuleNotFoundError; it is not evidence of an invalid USD or successful scene initialization. Use the existing full AppLauncher training entry for the bounded witness.

All runtime paths, reference manifests, failed initialization logs and any witness output remain on the server. No licensed assets or weights are added to GitHub. This ledger does not certify a final task score.


Validated delta spec `configs/env-dreamcontrol-cabinet-20261003.json`, canonical hash `b0234196`: fresh agent cabinet_asset_doc_witness executed the documented invocation once, exit0 and sentinel, diagnostic model/log hashes recorded in results/r3_dreamcontrol_cabinet_recovery_20261003.json. This validates one actual physics/PPO update only, not full-budget completion or Open-Door. The original scheduler was resumed after zero-update failed-attempt audit.

## 2026-10-06 execution-feedback Isaac continuation

Base spec remains canonical SHA256 `47f5eb26218254bbe270c3c43b3e7a552cca04c5caa81b6c76b4a6d2d3d2e9d4`. New invocation/resource delta `configs/env-feedback-isaac-20261006.json`, canonical hash `eec7b40a` (full `eec7b40aba36d29a484fb50977435f21cefce54c5dc55477bea436fcd8d2c22f`). Warm existing standalone Isaac Sim4.5 and project-local overlay, no dependency changes or installation. GPU0 reserved serially for retarget and Isaac frozen-policy inference; GPU1 existing independent viewer left untouched. New formal PPO not started.

Documented actual500-step capture is in `docs/feedback_isaac_execution_20261006.md`. At initial authoring the fresh-agent physical witness was pending; the completion receipt below records the later actual invocations. The existing retarget-only fresh witness passed planned arm3/3; this does not validate Isaac execution. Extra post-retarget IK excluded. Original reference library, pretrained policy and physics paths preserved. Original dataset arrays/weights remain private server assets and are not added to GitHub.


### Completed actual invocation witnesses

Stage1 capture once: exit0,500×18, unchanged original policy/library and bounded80N push. Stage2 revised smoke once: exit0,500×15; uses separate run config `configs/feedback_isaac_smoke_20261006.json` on the same environment dependency spec. Capture resource delta remains18-environment capture, not a15-environment smoke spec. Independent receipts in `results/feedback_isaac_20261006/*_doc_witness.json`. Both exposed-prefix equality and actual-feedback retarget gates FAIL; interface runtime completion does not establish causal benefit or new formal PPO readiness. Original capture hashes/outputs preserved. No dependency changes, installs, formalPPO, extraIK or repeated runs.

## 2026-10-07 fresh world-model confirmation generation

Spec `configs/env-world-confirmation-generation-20261007.json`, canonical SHA256 `20c6c394ce09869c1ccc08e5025fac58652346f401442ccaa44c337e145ecc80`. Warm read-only existing generation Python3.8/Torch1.13.1cu117 overlay and retarget Python3.10.21/JAX0.6.2/Torch2.5.1cu121 overlay; no rebuild or dependency installations. Version inspection alone is not environment readiness. GPU0 newly idle after residual development, GPU1 existing independent viewer preserved. Seeded generation/retarget CUDA backward and JAX GPU JIT witnesses PASS; fresh world_confirmation_reference_doc_witness_20261007 followed the invocation once, actual32sample inference/retarget/static selection COMPLETE, child exit0 captured immediately. 22 accepted/10 rejected, first11 selected, zero canonical near-duplicates, manifest SHA9e9da9f517a7b7d8b202fd5259c740df613e6c3174d71c1f5ffa193562f17f55. No resampling, no extraIK; docs/world_confirmation_generation_invocation_20261007.md defines the invocation.

Residual development artifacts completed: 11,796,480 transitions, 5,121 resets, zeroPPO; all3 development triages PROMISING. Fresh independent doc-follow audit verified156 file hashes, all windows/targets/OOF folds/donors and12,000 CUDA gradients. Validation chooses checkpoint, so the outcome is not a formal prediction or control gate. All original negative results preserved. Old non-residual fitter local/remote source byte discrepancy is separately retained for historical provenance; it is not called by this new queue.

Supplemental independent exit audit: all six development wrappers returned0 with complete sentinels/artifacts, but the executed wrapper captured $? after SENTINEL assignment, so actual child exit codes are unverified. Original wrapper/source/state preserved; no repeat. Data hashes/counts/gradient audits remain valid. New confirmation wrapper records actual child rc immediately after timeout.

Confirmation initial dispatch transport witness:CRLF entered tmux redirection; zero-byte log/no queue state/no model lock/no data prove scientific invocation0. Local launch_world_confirmation_transport_20261007.py revision uses fixed base64/argument dispatch and fresh doc witness; no dependency or model changes.

Actual v1 confirmation initializer blocked before loop: referenceguard concat2D vsactual MotionLib3D. No npy/steps/window/score; exact failedchild SIGTERM after >200s teardown hang, wrapper childchain returned0 but sentinelmissing caused stage/queueFAILED. Original39lock files unchanged, no formaltraining interrupted. v2 keeps originalsource and separate output/collector; only identityconcat→stack guard changes, allscience/metrics/model/reference budgetsunchanged; freshdoc-follow pending docs/world_confirmation_isaac_v2_invocation_20261007.md.

v2 later completed once: six natural child rc0 with sentinels, 2,949,120 transfers/1,948 resets. Fresh independent witness identity_v2_doc_witness verified36 output hashes and reconstructed67,584 rows exactly, bootstrap discrepancies0. All3 seeds clean/recovery prediction gates PASS; 0PPO/aux/norm updates. Actual control benefit remains unverified.

## 2026-10-07 residual PPO runtime integration

Spec configs/env-world-residual-ppo-runtime-20261007.json reuses base canonical47f5eb26 without dependency installation. GPU0 serial diagnostic, GPU1 original independent viewer preserved; shared idle viewer context allowance<1500MiB plus free compute lock, not exclusive physical GPU memory=0. New code integration is not certified by import/compile. docs/world_residual_ppo_runtime_invocation_20261007.md defines fresh-doc invocation: five256-env×2-update diagnostic arms then bounded16-env actual-scene preflight. Formal500-update×8192×3seed queue NOT_STARTED; requires runtime invariants, separate8192 capacity and new actual-scene local Gate. One aux step1024batch after each W PPO cycle is prelocked before any control score; same training data and original PPO reward/done/PD. Fresh diagnostic witness pending.

## 2026-10-07 full-size capacity and saved CUDA fixtures

User explicitly confirmed6013/live actual moving robot. New gate records direct human answer; main page failure preserved. Warm dependency spec unchanged; configs/env-world-ppo-capacity-20261007.json specifies zero-PPO/zero-sim CUDA tensors and two8192-env×2-update capacity stages. Preserves v1 five smokes, no formal auto-launch. Fresh doc invocation docs/world_ppo_capacity_invocation_20261007.md; all capacity readiness remains pending actual invocation and saved-tensor audit.

## 2026-10-07 full-size capacity and saved CUDA fixtures

User explicitly confirmed6013/live actual moving robot. New gate records direct human answer; main page failure preserved. Warm dependency spec unchanged; configs/env-world-ppo-capacity-20261007.json specifies zero-PPO/zero-sim CUDA tensors and two8192-env×2-update capacity stages. Preserves v1 five smokes, no formal auto-launch. Fresh doc invocation docs/world_ppo_capacity_invocation_20261007.md; all capacity readiness remains pending actual invocation and saved-tensor audit.

## 2026-10-07 frozen control-evaluation interface

configs/env-world-ppo-eval-interface-20261007.json is a resource delta only, no packages/rebuild. New control metrics locked in plans/world_residual_ppo_control_metrics_20261007.md before final scores. Fresh invocation docs/world_ppo_eval_invocation_20261007.md tests64×120 frozen capacity checkpoints, 0PPO/aux/norm, separately from formal500×8192. Balanced initialreference event is an explicit evaluation adaptation. Native scene traces are not videos. Actual interface readiness pending independent documented invocation.

## 2026-10-07 formal control-ablation launch contract

Warm dependencies unchanged. Runtime five arms and saved43 tensor/8192 capacity checks completed; original process-proof limitations retained. Frozen evaluator v1 failed before any control trace with its error obscured by teardown; v2 exposed directoryglob order differing from manifestorder. Failures preserved. v3 maps fixed manifestID by filename to unchanged MotionLib index. Fresh independent audit is PASS_WITH_EVIDENCE_LIMITS: 19 trace checks per arm and 11-row/macro reconstructions passed, while full force/success-event/per-frame module-hash reconstruction and natural timeout/root failure examples remain limited. configs/env-world-ppo-formal-20261007.json locks fivearms×three seeds,8192×24×500 warm-start PPO,15 real frozen2048×500 evaluations. No imagined reward, extraIK, deployment or reward/PD changes. Source SHA lock configs/world_ppo_formal_sources_20261007.json. Formal readiness is not control benefit; actual first launch still requires fresh agent-follows-doc invocation docs/world_ppo_formal_invocation_20261007.md.

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

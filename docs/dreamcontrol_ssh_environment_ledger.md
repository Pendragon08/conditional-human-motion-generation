# Direct SSH environment ledger

Host: `ouyanghao@10.147.17.32`; project `/data/code/ouyanghao/oyh/conditional-human-motion-generation`. Runtime is the existing standalone Isaac Sim 4.5.0 Python3.10 / Torch2.5.1+cu118 and project-local Isaac Lab overlay described by `configs/env-dreamcontrol-r3.json`. No environment rebuild or new packages in this recovery.

## 2026-10-03 official cabinet asset hydration

Dependency delta: replace a Git LFS pointer with its exact official 17599-byte cabinet USD object. SHA256 `30fddfc98008e380fe37d947e995c7875ead6113544b810d0254dc23450b6a7c`; pinned upstream commit `ec28b508313d6837de52c68247fbc8cf81fe4935`. Full task budget remains 8192 environments / 2000 updates / seed42 on GPU1; independent live renderer has its own process. Existing visualization authorization persists.

Status: staged object hash verified; actual scene/PPO readiness still requires fresh-agent diagnostic in `docs/cabinet_asset_recovery_20261003.md`. Pure interpreter `from pxr import Usd` without initializing Isaac Sim failed with ModuleNotFoundError; it is not evidence of an invalid USD or successful scene initialization. Use the existing full AppLauncher training entry for the bounded witness.

All runtime paths, reference manifests, failed initialization logs and any witness output remain on the server. No licensed assets or weights are added to GitHub. This ledger does not certify a final task score.


Validated delta spec `configs/env-dreamcontrol-cabinet-20261003.json`, canonical hash `b0234196`: fresh agent cabinet_asset_doc_witness executed the documented invocation once, exit0 and sentinel, diagnostic model/log hashes recorded in results/r3_dreamcontrol_cabinet_recovery_20261003.json. This validates one actual physics/PPO update only, not full-budget completion or Open-Door. The original scheduler was resumed after zero-update failed-attempt audit.

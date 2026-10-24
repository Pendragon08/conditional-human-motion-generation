# Independent recovery document witness

Status: **PASS_ENGINEERING_DOCUMENT_WITNESS**. Ran the documented command exactly once, with no implementation, threshold, or document edits and no retry. Main task history was not read.

Command: `bash scripts/run_isaac.sh scripts/validate_terminal_boundary.py --name closure_recovery`

Server: `ouyanghao@10.147.17.32:22`; directory: `/data/code/ouyanghao/oyh/phygile-reproduction-20261017`. Preflight found GPU0 RTX A6000 at 2968 / 49140 MiB, no task process and no pre-existing closure_recovery directory. Existing idle allocation was reused as documented.

Outer SSH exit: 0; recorded child exit: 0; timeout: 600 s; elapsed: 122.45 s. Completion sentinel count: 1. Both verification JSON files exist and their hash linkage matches.

The ten-frame independent replays have exactly equal state, body position, action, termination and start-state arrays (maximum differences all 0). All six initial physical-property arrays are exactly equal across all seven isolated workers. Accepted-history replay reports 0 difference, and raw next-start state equals the prior accepted continuation.

Both 50-new-frame candidates end exactly at the separately saved nominal standing target in native state and body positions. Independent state checks confirm configured joint positions, 0.76 m root height, initial XY/yaw, zero pitch/roll, and zero root/joint velocities.

Both untrained diffusion candidates were rejected; accepted generated attempts = 0, rejected = 2, committed new frames = 0. The original prefix remained committed. Two accepted one-frame held-pose windows only witness the acceptance/feedback interface. This does not demonstrate generated-motion success or stable standing. No closed-loop PPO updates occurred.

Canonical environment spec hash `fa5883c8` matches the existing `docs/environment.md` server-notes ledger. No doc-vs-reality divergence was found in the invocation or stated scope. The parent supplied the ledger/spec paths after launch; this timing is recorded in JSON.

Invocation document SHA256: `0c070dce4ee7fb26556ca66de25bfa8ce6172e1104d55b7ae7c169ffd258faad`.
Primary verification SHA256: `d218d655920a605035c65d460c927aea853db73ad2987d981a4872c19c90c3d3`.
Preserved 69 raw files (577662 bytes), all verified against remote SHA256. See `verified_raw_manifest.json`, `remote_manifest.stdout.log`, `raw_result/`, `run.command.json`, `run.stdout.log`, `run.stderr.log`, and `run.result.json`.

Failed checks: [].

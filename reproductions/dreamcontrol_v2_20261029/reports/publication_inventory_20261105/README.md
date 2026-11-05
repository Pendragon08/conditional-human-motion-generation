# Publication inventory, 2026-11-05

Latest scope update (after 11:22:53 UTC): Pick PPO 3,000 updates and Pour PPO 5,000 updates are complete, each with 0/100 success. The user cancelled later skill PPO before formal bimanual-box or push-pull training; their finals are not expected. Four audit repairs are being prepared, with no new formal training results in this publication. Eight remote archives and the Pour addendum are verified: 137/275 checkpoints are recoverable; the remaining 138 local archived checkpoints have incomplete upload groups. Four corrected reference videos are uploaded. The release stays a draft. `ARTIFACT_CATALOG.md` and `publication_summary.json` are the current coverage index; the original inventory and historical statuses below are preserved.

This directory is an inventory and upload preflight, **not evidence that the assets have already been uploaded**. Actual publication status must come from the publisher's verified GitHub asset receipts.

Target repository: [Pendragon08/conditional-human-motion-generation](https://github.com/Pendragon08/conditional-human-motion-generation). Target release ID: `434987063`. The uploader is restricted to this repository and release.

| Located artifact category | Count | Source bytes |
|---|---:|---:|
| Remote source snapshots, including three license/notice additions | 2,729 | 12,815,187 |
| Remote report/config/closed-log snapshot | 11,402 | 363,697,600 |
| Remote generated reference and execution output | 10,087 | 841,947,767 |
| Remote eligible review/result media | 328 | 61,275,063 |
| Remote project model checkpoints | 122 | 1,578,181,879 |
| Local archived project checkpoints | 138 | 4,438,255,905 |
| Server-root archived checkpoints (gzip files) | 4 | 1,138,136,074 compressed |

The local source/report/media inventory additionally locates 1,219 source snapshots (5,291,318 bytes), 993 report/config/asset files (70,768,848 bytes), and 131 media files (74,628,140 bytes). These local artifacts can overlap remote copies; the counts above are locations, not unique content counts. Videos and audit reports created after this inventory need their final verified receipts added by the publisher.

`local_inventory.json` maps archived checkpoints to actual local files or precise TAR members. All 138 listed local checkpoint locations existed at inventory time. `remote_archive_markers.json` reconciles 142 archived checkpoint markers with 138 local checkpoint routes and four server-root gzip routes. Preserved failed SuperMimic-style weights remain in that inventory.

`upload_plan.json.gz` is the executable upload plan. `remote_inventory.json.gz` is the read-only scan; `remote_verified_hashes.json.gz` contains full-content SHA256 and before/after source identities for 24,669 files. The only changed file in that hash pass was the active PPO pointer, explicitly held. The compression receipt records byte-exact gzip round-trip checks for large generated inventories; no original archive or model was removed.

The remote plan has eight streaming TAR groups, each below the GitHub per-asset ceiling. It creates no large staging archive. The first group contains code and the complete DreamControl and SuperMimic licenses plus the third-party notice. Each source file is checked by size, nanosecond mtime, inode, device, and SHA256. The uploader uses `O_NOFOLLOW`, checks the opened descriptor, computes a complete deterministic TAR SHA256 before upload, streams again while rechecking contents, and verifies GitHub's returned size and SHA256 digest. Its small progress receipt updates approximately every 30 seconds. The GitHub token is accepted only through stdin and remains in memory; it is not included in commands, logs, or output files.

The tar serializer preflight passed long-name/PAX round-trip, deterministic byte output, and a negative identity-mismatch check. These checks do not substitute for successful remote upload verification.

## Explicit holds and pending files

See `publication_holds.json` for 40 exact holds: active queue/run metadata and logs, source-video copies, source-data metadata requiring review, the third-party MDM initialization checkpoint, and the third-party HumanML evaluator assets. Raw/preprocessed AMASS, GRAB, and Omni source corpora and installed dependency environments are excluded.

Two mixed cold archives contain source-derived NPZ data alongside project model weights. Do not upload these whole: `native_and_inactive_amass/v2/payload.tar` contains 11,424 AMASS-derived NPZs plus six model weights; `parallel_archive_20261104/inactive_previews_and_diagnostic.tar` contains 4,542 Omni-derived NPZs plus one model weight. Only the listed model members belong to this upload plan.

The complete closed-log/PNG archive `training_space_archive_20261105/completed_logs_and_review_png/v1/payload.tar` (352,358,400 bytes, SHA256 `c491df125d039fe7357595176b90e576bcbc94a6f5321e8f32259344b4fdf1d4`) is eligible as a separate result asset; it is not part of the model-only stream.

At the status check, Pour PPO was at 4,465/5,000 updates, with the final checkpoint and 100-environment evaluation still pending. Bimanual-box and push/pull PPO training/evaluation remained queued. Stable numbered checkpoints may be archived without changing the running job; mutable status/log/last files must be refreshed or deferred. No queued or in-progress work is represented as a completed result.

## Final checkpoints independently rehashed

- Diffusion `step_000010000.pt`: 503,839,702 bytes; SHA256 `109fcfebf2a805831444ca8fb55d69674ef7cb19c61219e3e4ed18876d9527a9`.
- Native SuperMimic `main_resume_v1/logs/last.pt`: 218,548,321 bytes; SHA256 `571b207875610946f1065bf25bdf1dac78cb8c37f1691fc001324b1da64776ee`.
- Pick500 PPO `model_2999.pt`: 6,123,810 bytes; SHA256 `ed45d9d1f63b64ace4667964f02683280da1224b25a1ca88c2ca06487a653e31`.

No training source, active process, final model, task reference, or original cold archive was modified by this inventory.

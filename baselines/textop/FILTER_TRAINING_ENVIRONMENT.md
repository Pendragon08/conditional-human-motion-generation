# Privileged filtering policy and recorded resume

Warm tracker environment `b7a3bf0c`, no package/shared-source changes. Read
SERVER_NOTES.md and configs/filter-training-plan.json. The formal launch is
disabled until all declared candidate cohorts/exports/backups and complete-
library capacity/stability are verified. That guard is intentional.

Fresh document witness (PowerShell), exactly once unchanged:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -i 'C:/Users/Pende/.ssh/id_ed25519' ouyanghao@10.147.17.32 'python3 /data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/supervise_filter_policy.py --run filter_statistics_document_v2 --mode witness'
```

Require actual childexit0, no timeout, supervisor sentinel
`TEXTOP_FILTER_POLICY_SUPERVISED_EXIT0`, and receipt status
`TEXTOP_FILTER_STATISTICS_RESUME_WITNESS_PASSED`. The diagnostic is8envs,
2original reference motions,2completed iterations and40actual Adam updates.
It must restore exact model/Adam/two normalizers/EMA/raw attempt counters,
exclude initial resets from raw successes, and continue optimizing after
restore. Also require `resumed_physics_episodes_restarted_without_counting_reset=true`:
new simulator episodes must begin explicitly, exclude the reset itself from
counts, and count their first real completed attempts. It cannot accept
motions or start the generator. Evidence is in
tracker_validation/filter_statistics_document_v2/{receipt.json,
native_teardown_stack.txt}, logs/filter_policy_filter_statistics_document_v2.log
and receipts/filter_policy_filter_statistics_document_v2_exit.json. Copy only
those public diagnostics, not checkpoints/private data. No retry or source
inspection/repair by the independent witness.

The first author attempt failed after20 real updates when restoring an EMA
tensor allocated by upstream under inference mode. Its checkpoint, traceback,
failed receipt and actual supervised exit are retained. The local restore
operation now explicitly runs under inference mode; training/sampling math
is unchanged. Author_v2/document_v1 restored saved statistics, but a subsequent
review found that clearing the per-env attempt marker without explicitly
resetting physics could omit the first resumed episode. Author_v3/document_v2
restart physics, restore saved counts/EMA again to exclude synthetic reset
events, then mark the new episodes as genuine attempts. Author_v3 exposed the
same inference-mode constraint in the upstream command reset's metric tensors;
that failed checkpoint/trace/exit1 is retained. Author_v4 scopes the reset
itself under inference mode while loading optimizer state outside that mode.
Earlier pass evidence is retained with this narrower limitation.
`completed_learning_iterations=0` in that original failed
receipt was not flushed after the first completed update and is corrected by
the preserved checkpoint's completed1/actual20 metadata. Do not erase it.

Formal profile:8192envs,557privileged actor+critic,[2048,1024,512],24steps,
5epochs/4minibatches,lr0.001,40000iterations. Pinned PPO and EMA sampler are
reused. No DR events/observation corruption; height terminations removed for
pretraining. Default reset pose/velocity/joint perturbations from the release
remain an explicit reset distribution. Independent raw counts exclude the
initial synthetic reset; upstream EMA priors of1/1 are preserved. Checkpoints
save raw counts/EMA with reference-order fingerprint. Model and optimizer
restoration is exact; physical episodes restart after process recovery.

The available BABEL-indexed public target is8278originals;13CMUselected-export
motions and403privateauthor clips remain unavailable. This profile does not
claim all12296trackingclips used by the paper. Filter originals first and only
then retain matching mirrored copies. Do not label this full author data.

Native SDK teardown still uses the documented20s fallback only after complete
receipts/checkpoints and closed writers. Require actual supervised exit and
retain `native_close_returned=false`; fallback is not native-close success.

# Independent documented 10k filtering recovery — PASS

The documented launcher was executed exactly once with the explicit warm local Python and `PYTHONIOENCODING=utf-8`; it returned actual exit code 0. The production run `filter_policy_paper69_resume10000_filter_recovery_20261027_v1` remains running. No restart, stop signal, new numerical setting, package installation, or old diagnostic was performed by this review.

At 2026-10-27T12:25:51.122205+08:00, the actual receipt was `RUNNING_FORMAL_FILTER_TRAINING`, `resume_exact_state_verified=true`, and `resumed_physics_episodes_restarted_without_counting_reset=true`. Native zero-based iterations 10010 and 10011 both had finite value-function, surrogate, and entropy losses. Latest printed losses: {"value_function": 0.1105, "surrogate": -0.0128, "entropy": 15.0661}.

## Review and provenance

- Both local and remote copies of all 20 original frozen bindings match their original digests; the original continuation supervisor is unchanged. A line-by-line supervisor diff is retained in `supervisor_comparison.diff`.
- The new supervisor changes the run and source checkpoint and adds provenance/pause/relaxed-gate checks. PPO, 8192 environments, 557D observations, rewards, resets, solid floor, and the 40k reference budget remain frozen.
- The warm tracker environment canonical hash is `b7a3bf0c`, matching SERVER_NOTES. No environment rebuild was needed.
- The actual 10k full checkpoint is 90,799,682 bytes, SHA256 `6f949e4edc50d9a763a3634d716162cb5ddecb079bd1142d3720f907f5efab45`. Its server file and D-drive private full backup were freshly hashed, and the complete prior handoff/backup authorization chain passed.
- The prefilter generator really exited at 8100 updates. Its 513,526,637-byte full snapshot matches server and private D-drive SHA256 `2b958a5e4efbd74ddfe6bb99b8a5eb8174cec143bb782689b1f6af8934a6c2fe`. The separate accepted pause receipt is SHA-bound; the prior nonzero exits and stale receipt are retained.
- The 10k all-source relaxed summary is bound and preserved: 8278 sources, 15,887,307 attempts, zero unattempted sources, zero eligible at unchanged 0.05. No original asset or negative result was deleted.
- Only `resume_enabled` changed from false to true in the new spec. The exact disabled spec was retained remotely before dispatch; active SHA256 `f39e2103ff6e9cceb3f7442f83a5c217328cb2cb18bce9cecfabb4fd774a64b4`.

## Actual production verification

The frozen `load_recorded` function loads the source checkpoint in the real SDK process, restores model and Adam state, both normalizers, EMA vectors/totals and raw attempt counters, and asserts exact dtype/shape/value equality before the receipt flag is written. Initial episode reset is excluded from attempt statistics; subsequent episodes are marked as real attempts. RNG state and the physics trajectory are not claimed to be a bitwise continuation.

There is exactly one actual SDK Python process: 4190451. Driver 4190430, supervisor 4190431, and the SDK/wrapper are restricted to CPU 0–3. The command explicitly resumes the verified 10k checkpoint on CUDA1 in headless mode. Wrapper PID 4190443 is `/bin/bash`, not another SDK. The common lock is held by supervisor 4190431; process, supervisor, and outer exit files are all absent.

At the final snapshot, the actual SDK held 23,660 MiB on GPU1 and an auxiliary 456 MiB on GPU0. The historical 57D worker remained `T (stopped)` with 7778 MiB on GPU1; its parent was waiting. This paused process was neither signaled nor resumed. The frozen Torch allocation fraction is 0.65. SDK native threads are more than four, but all have the inherited four-CPU affinity.

Startup persistent capacity was 5,234,966,528 bytes against a 3,892,314,112-byte requirement, including 2 GiB reserve and 512 MiB log allowance. GPU1 had 40,726 MiB free and available host RAM was over 20 GiB. Five supervisor resource samples were retained by the final snapshot; regular resource and remaining-checkpoint budget checks continue in production.

The durable receipt still contains completed=0 and updates=0 because the original runtime flushes on completed round 10100. It was not rewritten. Native iteration 10011 is zero-based; the frozen callback implies completed round 10012 and cumulative optimizer updates 200240 at that log record (240 new updates). These are explicitly derived from native output and frozen callback semantics, not falsely reported as flushed receipt values.

## Limits and next review

No blocking documentation-versus-reality divergence was found. Startup display/extension warnings and the graphics shader-cache error line are preserved in early bounded snapshots; actual finite PPO updates followed them. Verification covers correct startup and continuing updates, not convergence, successful full-library filtering, semantic quality, or 40k completion.

Production was deliberately left running. Review the next actual 15k checkpoint's saved relaxed statistics and tracking quality. Keep the 5% threshold unchanged and perform strict all-survivor replay only when its input gate is real. New 15k/20k checkpoints need a new tool bound to this run; old hardcoded 5k/10k backup tools are invalid. Later generator/WBC/media capacity remains unresolved and no generator is authorized before filtering acceptance.

## Proof files

`SUMMARY.json`, `static_review.json`, `prelaunch_readonly_proof.json`, `pause_and_private_backup_verification.json`, `postlaunch_snapshot_01.json` through `postlaunch_snapshot_04.json`, corresponding bounded native excerpts, `supervisor_comparison.diff`, and the examined-source manifest preserve the review. These contain aggregate proof and code/metadata only; no full checkpoint or per-source private statistics/index arrays were copied into this report folder.

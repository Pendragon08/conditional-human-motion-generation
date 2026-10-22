# Experiment Audit Final Bounded Follow-up — Commit 50e5784

**Date:** 2026-10-22  
**Auditor:** Codex GPT-5.6-Sol ultra, same-family reviewer, read-only, provisional  
**Frozen follow-up commit:** `50e5784343187ce4e715834f5f63428c7f376f88`  
**Initial frozen commit/verdict:** `104393d48be438e923a30f3e42c466b861fb8639` — WARN  
**Prior follow-up commit/verdict:** `6850b6e404565e4716588943b03fa07f4141e085` — WARN  
**Final bounded follow-up verdict:** **WARN — material safeguards are repaired in code, but no real checkpoint or GPU recovery path has been accepted**

All file and line references below point to commit `50e5784343187ce4e715834f5f63428c7f376f88` unless another commit is named. I read the requested frozen blobs directly with `git show`; I did not substitute the changing working tree. The earlier Markdown/JSON reviews remain unchanged, and the preceding final response is preserved separately as `PRIOR_FINAL_RESPONSE.md`.

## Verdict relationship

This follow-up does not replace the initial review or the 6850b6e follow-up. The scientific A–F results remain:

| Check | Status at 50e5784 | Reason |
|---|---|---|
| A. Ground-truth provenance | **PASS** | Simulation and consistency proxies remain explicitly labeled; no prediction-derived value is presented as dataset ground truth. |
| B. Score normalization | **PASS** | Raw rollout counts/heights/errors are reported; no prediction-statistic denominator appears. |
| C. Existence and claims | **WARN** | Point-in-time numbers match the frozen snapshot, but the 1M schedule, first checkpoint, full-model diagnostics, upload/cleanup acceptance, and recovery remain incomplete. |
| D. Dead code | **PASS** | The diagnostic and feedback paths are called by the worker when an eligible checkpoint exists; the absence of new outputs is explained by the absence of any ready manifest. |
| E. Scope | **WARN** | The evidence remains a running two-rank training observation plus old-model/synthetic protocol checks, not a completed benchmark reproduction. |
| F. Evaluation type | **PASS** | Types and limits are disclosed: simulation-only and self-supervised/implementation proxies; no human, real-robot, or official real-GT benchmark result is claimed. |

The overall verdict remains **WARN** because scientific scope is incomplete and end-to-end recovery/publication is still unexercised. This is no longer a WARN for the specific `assert` bypass, pagination, rank-completion, or missing launch-time revalidation defects found at 6850b6e; those are materially repaired in code.

## Frozen inputs and integrity

The requested primary blobs were:

| Frozen artifact | SHA-256 |
|---|---|
| `baselines/reactivebfm/scripts/backup_full_run.py` | `af25cef28b4f9a00ea88de165bdde4341667f5267af3d87bf7c94f9caffe794c` |
| `baselines/reactivebfm/scripts/test_backup_full_run.py` | `a3ea6d13b66749678e626bcc09b2251092058cbbc1d68e43f1c788df44015d9b` |
| `baselines/reactivebfm/scripts/prepare_verified_resume.py` | `b5db0707c3a16310d01141a7d30c22aa2e2c4a258ad98eec56c662cfce651a23` |
| `baselines/reactivebfm/scripts/test_verified_resume.py` | `276e4957484fcc6b70f90fb9cce7a577fcd09d450755af266027563dcc450a08` |
| `baselines/reactivebfm/docs/full_run_operations.md` | `e6d162f6435ba371cd7066291b96d45d8235cb72b55e05ced7dda4221f1df618` |
| `baselines/reactivebfm/reports/ReactiveBFM完整训练与闭环进展报告_20261022.md` | `70d42dd697418234662085844d317f083faa95ea6acf95d36db88e7163def627` |
| `baselines/reactivebfm/results/full_training_20261021/backup_worker_unit_v8_stderr.txt` | `9f1c03c556bca2616c7c5fd74f7430ce85a4a3fb8bbbccb2e73adb99294f43bf` |
| `baselines/reactivebfm/results/full_training_20261021/verified_resume_unit_v4_stderr.txt` | `a78769d8a4b7e552ebc16d7f46a7b6a19ac068b02c2cf121ba1e2afe659f43b8` |
| `baselines/reactivebfm/results/full_training_20261021/backup_worker_launch_v4.json` | `b43e22fde77d26b9eaba860e88fd379a5a659030aba9c804ce7bc5df24762bce` |
| `baselines/reactivebfm/results/full_training_20261021/scoped_snapshot_v1.json` | `caade9f0d420fbc9161a681a8f9fd7a1ea42729457f5d96a91c65a91ef0191ed` |
| `baselines/reactivebfm/results/full_training_20261021/live_snapshot_v2/server_snapshot.json` | `77faf742262c8c4c4894984cfccec82e0c47cba4ef5a0f4f5904bbf5c8ee06e6` |
| `baselines/reactivebfm/README.md` | `de0e5b1dca78cd1267ead2c6c8cd7c726952b8186ec1e0309cdfea2f4d328892` |

The preserved initial/follow-up outputs still hash to:

- `EXPERIMENT_AUDIT.md`: `2edc41ba4b79996a2e9ad2684c1987dd378882d2ac87c3c916983a9206832908`
- `EXPERIMENT_AUDIT.json`: `0a85cffd6908a1779bd196e788762b0adccd7e744648287f6d20cd32dbaa15a8`
- `FOLLOWUP_6850B6E.md`: `08155443b6a079b08b154f2c6ec40482dde4603e6294bf93f4577f7a14e2814e`
- `FOLLOWUP_6850B6E.json`: `5ecdda9594b3f7a9fef8fc8ef614e305fc125797928ba7339e090a9473f546bd`

## Findings resolved since 6850b6e

### 1. Optimized-Python validation bypass: RESOLVED

The recovery helper replaces checkpoint-pair `assert` statements with an explicit `require()` that raises `ValueError` (`prepare_verified_resume.py:18-19,32-51`). Both the worker and recovery helper refuse import when `__debug__` is false (`backup_full_run.py:23-24`; `prepare_verified_resume.py:21-22`). The destructive server and local cleanup checks now use explicit `RuntimeError` branches (`backup_full_run.py:229-255,297-306`).

The recorded backup suite includes a subprocess check that both modules fail under `python -O` (`test_backup_full_run.py:124-128`), and the frozen output records 11 passing tests (`backup_worker_unit_v8_stderr.txt:1-5`). Independently, I executed each frozen module body under the local optimized interpreter: both exited nonzero and emitted the intended “Optimized Python is unsupported” guard. This closes the concrete bypass reproduced at 6850b6e.

The worker's manifest-shape checks at `backup_full_run.py:267-273` still use `assert`, but optimized execution is rejected before the worker class can run. Converting those remaining checks to explicit exceptions would improve defense in depth; they are no longer silently removable in a supported execution mode.

### 2. Plan-to-launch revalidation: RESOLVED IN CODE

Plan creation now records the original run/config paths, manifest hash, wrapper source hash, `args.json` hash, configuration hash, verified pair, fresh destination, exact command, and environment (`prepare_verified_resume.py:54-77`). `validate_recovery_plan()` reconstructs the plan from current files, re-hashes the pair through `select_verified()`, and compares the manifest, args, config, and wrapper-source hashes (`:80-86`).

The explicit Linux executor refuses an original `train_full.py` process with the same `--output`, performs that revalidation, atomically claims a previously absent destination directory, writes a `VERIFIED_RECOVERY_LAUNCH_REQUESTED` record, and only then calls the generated command (`:89-105`). The CLI prevents mixing plan-creation arguments with `--execute-plan` (`:108-118`). The tests verify pair mutation is rejected and that pure revalidation does not create the destination (`test_verified_resume.py:52-64`); the frozen output records seven passing tests (`verified_resume_unit_v4_stderr.txt:1-5`).

This closes the broad 6850b6e finding that a saved plan could be launched later without any revalidation. It does not prove the executor can load or train from a real checkpoint; the executor itself is not called by the seven tests.

### 3. Release asset pagination and ambiguity: RESOLVED IN CODE

The worker obtains the release id, enumerates 100 assets per page until the final short page, and rejects duplicate names (`backup_full_run.py:101-110`). The pagination test covers a 100-item first page and a second page (`test_backup_full_run.py:112-122`). This closes the single-page enumeration gap.

### 4. Rank completion, run-specific process identity, and abrupt-exit evidence: RESOLVED IN CODE

The remote snapshot now records status-file mtimes and filters candidate training processes by the exact configured `--output`, returning the matching PIDs (`backup_full_run.py:81-99`). The frozen scoped and live snapshots contain three matching PIDs and `training_process_running=true` (`scoped_snapshot_v1.json:173-181`; `live_snapshot_v2/server_snapshot.json:173-181`).

Completion requires exactly ranks `{0,1}`, both declaring `world_size=2`, both marked complete, both at at least 1,000,000 optimizer updates, and a processed final manifest (`backup_full_run.py:312-324`). The unit test proves one completed rank is not accepted and two exact ranks are (`test_backup_full_run.py:130-142`). Training statuses older than the configured/default 1,800 seconds produce a preserved/uploaded stale notice, explicit `FAILED` states produce a failure artifact, and process absence before accepted completion produces an absence artifact before the worker stops (`backup_full_run.py:315-332`).

These changes materially repair the initial false-completion and unscoped-liveness findings. They do not add an automatic restart; the documentation accurately says failure requires operator review (`full_run_operations.md:7-15`; report `:92`).

## Frozen execution and claim verification

### Training status

The frozen live snapshot contains exactly two statuses, ranks 0 and 1, both `TRAINING`, each with 810 optimizer updates and official step 809 (`live_snapshot_v2/server_snapshot.json:2-6,80-86,88-92,165-171`). The target and warmup are 1,000,000 and 400,000 in both recorded configurations (`:11-12,96-97`). The manifest list is empty (`:174`), and the scoped process check reports three matching processes and a running task (`:175-181`).

The report therefore correctly says both ranks reached 810 updates, updates must not be summed across ranks, and the first 1,000-update checkpoint does not yet exist (`report:3-9`). It also correctly says no full-model checkpoint upload, cleanup, diagnostic, or recovery acceptance has occurred (`:7,96`).

I independently recomputed the report's rate from the frozen rank-0 snapshot: `3480.15072529763 / 810 = 4.2964823769` seconds/update. The linear remaining-time extrapolation is `49.6875` days, matching the stated 4.30 seconds/update and about 49.7 days (`report:94`). Peak allocated memory is 10.687/10.691 GiB for the two ranks, matching the report's 10.69 GiB-per-rank qualifier (`snapshot:82-84,167-169`; report `:94`).

### Protocol-test evidence

The backup test source defines 11 test methods and the recovery source defines seven. Both files compile from the frozen bytes. Their stderr receipts record `Ran 11 tests ... OK` and `Ran 7 tests ... OK` (`backup_worker_unit_v8_stderr.txt:1-5`; `verified_resume_unit_v4_stderr.txt:1-5`). The test module docstrings explicitly say they use dummy bytes and are not robot/model performance results (`test_backup_full_run.py:1`; `test_verified_resume.py:1`).

The report preserves that boundary: it labels the tests virtual-byte protocol tests, says metadata download checking is not a real-checkpoint substitute, and says no real GPU recovery was run (`report:78-84`). The operations guide repeats the same limit (`full_run_operations.md:19-32`). The recorded test receipts support code-path behavior only.

### Worker launch evidence

`backup_worker_launch_v4.json:1-6` records a start time, PID 25492, and distinct stdout/stderr paths. It does not record the command, configuration hash, worker source hash, process inspection, or a subsequent heartbeat. The worker code would log its own source hash at startup (`backup_full_run.py:339-356`), but that v4 log/state evidence is not among the frozen artifacts requested here. The launch JSON therefore proves a launch record was written, not that PID 25492 continued running the intended code.

## A–F details

### A. Ground-truth provenance: PASS

No new scientific metric or ground-truth source is introduced by 50e5784. The report explicitly calls height, static-reference, and state-prefix checks diagnostic proxies rather than paper success rate or dataset truth (`report:48-57`). Future checkpoint diagnostics are described as simulation cases with fixed prompts/seeds and receipt verification (`:64-72`), and the operations guide restricts `PHYSICAL_CLOSED_LOOP_EXECUTED` and `FEEDBACK_CONSISTENCY_PASSED` to execution/wiring meanings (`full_run_operations.md:34-36`). There is no unlabelled model-derived ground truth.

### B. Score normalization: PASS

The worker reports raw counts, minimum root height, and a Boolean height proxy (`backup_full_run.py:208-225`); it does not divide a metric by prediction maxima, minima, or means. The report publishes raw rollout counts/heights/errors and labels them proxies (`report:48-57`). No suspicious near-perfect performance score is asserted; the old model is explicitly reported as falling.

### C. Result existence and claim matching: WARN

Every new numerical claim checked in this bounded revision matches the frozen snapshot or test receipt: 810 updates per rank, official step 809, 3480.2 seconds, 4.30 seconds/update, about 49.7 days, approximately 10.69 GiB per rank, 11 backup tests, and seven recovery tests. The report also correctly records zero ready manifests and the absence of any real checkpoint/recovery acceptance (`report:7,80-84,94-96`; snapshot `:174`).

The status remains WARN because 810/1,000,000 is an in-progress observation, not completion, and no first checkpoint/result exists. `backup_worker_launch_v4.json` is launch metadata rather than worker-liveness proof. There is no phantom success claim, but no evidence supports upgrading publication, cleanup, or recovery to DONE.

### D. Dead code: PASS

The checkpoint worker calls the simulation routine for eligible update counts, captures/freeze-seals failure or physical receipts, calls feedback verification for physical runs, records the returned results, produces a report/package, and uploads them (`backup_full_run.py:141-227,257-294`). There are no such full-run outputs yet because the frozen manifest list is empty. The new pagination, optimized-mode, exact-rank, pair-change, and read-only-revalidation paths have matching unit tests (`test_backup_full_run.py:112-142`; `test_verified_resume.py:52-64`).

The new `execute_plan()` branch itself has no direct unit or real-system execution evidence. That is an operational test gap rather than a dead scientific metric.

### E. Scope: WARN

The current evidence covers one two-rank run at 810 updates, an old 101-update model's single recorded closed loop, implementation/feedback proxies, and dummy-byte protocol tests. It does not cover the 1M schedule, the planned three prompts × three seeds at a new checkpoint, the unavailable goal/reaching data, the official full evaluation, TensorRT/real-time behavior, or a real robot (`report:42-74,90-96`; README `:73-84`). The report avoids “completed,” “paper performance reproduced,” and other scope inflation, so this is an incompleteness WARN rather than a misrepresentation FAIL.

### F. Evaluation-type classification: PASS

- Static A-pose tracker check: **simulation_only**, with a handcrafted/static reference.
- Old 101-update generated-motion closed loop: **simulation_only**.
- State-to-prefix consistency: **self_supervised_proxy + simulation_only**.
- Activation-recomputation comparison and document witness: **self_supervised_proxy / implementation check**.
- Dummy-byte backup and recovery suites: **protocol tests**, not a scientific evaluation.
- Planned full-checkpoint prompt/seed diagnostics: **simulation_only**.
- Official dataset-ground-truth benchmark, human evaluation, and real-robot evaluation: **not executed**.

The report's labels are consistent with these classifications (`report:42-57,64-72,78-84,96`).

## Residual limits

### 1. No real checkpoint has exercised the hardened path: MATERIAL EVIDENCE GAP

The live snapshot has no ready manifest at 810 updates (`live_snapshot_v2/server_snapshot.json:174`), and the report explicitly says the first checkpoint and all corresponding acceptance work remain absent (`report:7,96`). Thus none of these have been demonstrated with a real full-run pair:

- model/optimizer deserialization;
- upload and server-reported digest acceptance;
- nine diagnostic attempts and feedback checks;
- immutable evidence/report publication;
- local or remote retention cleanup;
- verified plan creation for this run;
- executor launch, optimizer restoration, or a resumed optimizer update.

The 11/7 passing protocol tests cannot be promoted into any of those claims.

### 2. The checkpoint writer still writes final filenames non-atomically: DISCLOSED WARN

The frozen training wrapper still calls the upstream saver, which writes model and optimizer final names before the wrapper hashes both and atomically publishes a ready manifest (initial frozen `train_full.py:96-113`; upstream `scheduled_forcing_loop.py:812-823`). Commit 50e5784 does not change that writer. The worker and recovery helper now consume only a fully validated ready manifest, so a partial write is fail-closed on the documented path. A crash can still leave partial final-named files for investigation, and an operator who bypasses the helper can still select them. The operations guide and report disclose this (`full_run_operations.md:19-21`; report `:84`).

### 3. Recovery validates byte identity, not checkpoint semantics: MATERIAL ACCEPTANCE GAP

`select_verified()` checks manifest shape, path containment, names, sizes, and SHA-256 (`prepare_verified_resume.py:32-51`). It never calls `torch.load`, checks expected state-dict structure, compares optimizer counters with the manifest, or verifies that the pair is compatible with the installed model/runtime. The dummy tests deliberately use arbitrary short byte strings (`test_verified_resume.py:21-26`), proving that passing selection is not semantic checkpoint acceptance.

The first real recovery acceptance must separately demonstrate deserialization, model/EMA and optimizer-state loading, the expected resume step, and at least one successful resumed optimizer update. No such inference is allowed from these protocol tests.

### 4. “Training source revalidation” covers only the wrapper file: QUALIFIER REQUIRED

The plan records and compares SHA-256 for `scripts/train_full.py` only (`prepare_verified_resume.py:66,80-86`). The running status likewise records only `Path(__file__)` for that wrapper (initial frozen `train_full.py:37-41`). It does not re-hash the upstream training modules, the 226-file vendor tree, Python packages, environment, data, or normalizers at launch.

Accordingly, report lines 40 and 82 and operations-guide line 30 should describe this as **wrapper-script** revalidation unless a pinned Git tree/upstream manifest/environment receipt is also checked. The current hash is useful but cannot prove the whole training source/runtime is unchanged.

### 5. `args.json` has temporal consistency, not trusted provenance

The helper requires `args.json`, records its current hash, and rejects a change between plan and launch (`prepare_verified_resume.py:58-67,80-86`). The ready manifest does not contain an expected `args.json` hash, and the recovery command is built from the separately supplied config rather than reading that file. The “args revalidated” claim is supported only as unchanged-since-plan, not as authenticity relative to the original checkpoint.

### 6. A narrow revalidation-to-spawn race remains

The executor checks the old process before revalidation, returns from revalidation, creates the destination/record, and then starts a child from filesystem paths (`prepare_verified_resume.py:89-105`). A concurrent mutation or original-process restart in those intervals can change the bytes or process state after the final check. Immutable staging, held file descriptors, or a child-side verification immediately before deserialization would close this window. This is much narrower than the 6850b6e plan-to-launch gap and does not negate the code-level remediation.

### 7. Launch-request evidence is not completion evidence

The executor writes `VERIFIED_RECOVERY_LAUNCH_REQUESTED` before calling `subprocess.run` and returns the child code without updating the record with spawn success, exit code, stdout/stderr paths, loaded checkpoint identity, or first resumed update (`prepare_verified_resume.py:99-105`). A launch failure can therefore leave only a correctly labeled request record. The documentation properly disclaims successful recovery, but future acceptance needs a post-launch receipt.

The executor also relies on the operator to recheck GPU resources and create a fresh backup config/local output/evaluation root/release tag (`full_run_operations.md:30-32`; plan requirements at `prepare_verified_resume.py:74-76`). Those steps are not enforced by the executor.

### 8. Long-running supervision remains manual and point-in-time

The worker records stale/failed/absent states and stops safely, but it does not restart itself or training (`backup_full_run.py:315-337`; report `:92`). Its Windows power request return value is not checked (`backup_full_run.py:343-355`), which the report discloses. The v4 launch receipt alone does not establish continued worker health. A service supervisor or independent heartbeat/alert is still required for unattended operation.

## Action items

1. At the first 1,000-update ready manifest, preserve the only server pair and run a bounded acceptance: re-hash both files, deserialize model/EMA and optimizer on CPU if feasible, verify optimizer/update counters, upload and verify remote digests, archive all nine diagnostic attempts, and do not exercise destructive cleanup until independent copies are confirmed.
2. Add direct CPU/mocked tests of `execute_plan()` covering an active original process, changed wrapper/config/args/manifest/pair, destination races, subprocess spawn failure, and child nonzero exit. Keep these separate from scientific/GPU claims.
3. For a real recovery rehearsal, record the exact pair/manifest hash, full source commit/tree and environment, command, PID, stdout/stderr, return code, loaded optimizer step, first resumed update, new worker namespace, and resulting ready manifest.
4. Extend recovery provenance to verify the pinned Git tree/upstream manifest and environment receipt; add `args.json` to the original ready manifest if it is meant to be authenticated.
5. Close the final-check-to-spawn race with immutable staging or child-side digest verification, and write a terminal launch/result receipt after the child starts/exits.
6. Convert `backup_full_run.py:267-273` manifest assertions to explicit exceptions for defense in depth, even though optimized mode is already refused.
7. Preserve the current claim ceiling until real evidence exists: running at a frozen observation, protocol safeguards tested with dummy bytes, no full training completion, no new-model performance, no real checkpoint acceptance, and no demonstrated GPU recovery.

## Claim impact at 50e5784

| Claim | Impact |
|---|---|
| Both ranks had 810 updates at the frozen observation | **Supported** by `live_snapshot_v2/server_snapshot.json`; point-in-time only |
| The full 1M schedule completed | **Unsupported and explicitly disclaimed** |
| A real full-run checkpoint was accepted/uploaded/cleaned | **Unsupported; zero ready manifests** |
| The optimized-interpreter assertion bypass remains | **Resolved**; both modules reject `-O`, and recovery/deletion checks use explicit exceptions |
| Release assets are fully enumerated | **Supported in code/unit test** through pagination; no large real release integration test was reviewed |
| Completion requires both exact ranks and 1M updates | **Supported in code/unit test** |
| Process liveness is scoped to this run | **Supported in code and snapshot** by exact `--output` matching; future continuity is not implied |
| Recovery revalidates the plan before launch | **Supported in code** for pair/manifest/config/args/wrapper bytes; executor path itself remains unexecuted |
| “Training source” as a whole is revalidated | **Needs qualifier**; only the wrapper script is hashed |
| A validated pair is a loadable/compatible checkpoint | **Unsupported until real deserialization/load evidence exists** |
| Real GPU recovery succeeded | **Unsupported and explicitly disclaimed** |
| Paper benchmark performance was reproduced | **Unsupported and explicitly disclaimed** |

## Final conclusion

Commit `50e5784` materially repairs the concrete control-plane defects identified at `6850b6e`: optimized-mode bypass, deletion guards, pagination, exact-rank completion, run-scoped process detection, stale/absent failure evidence, and launch-time plan revalidation. The frozen report is candid about the remaining scientific and operational boundary.

The correct final status is still **WARN**, not FAIL: I found no fake ground truth, prediction-based score normalization, phantom completed result, or mismatched reported number. It is also not PASS: the snapshot ends at 810/1,000,000 with no ready manifest, all new recovery evidence uses dummy bytes, the executor is untested, byte hashes do not prove semantic loadability, and no real checkpoint publication, cleanup, or resumed GPU optimization has occurred.

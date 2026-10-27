# Fresh agent-follows-doc witness v3

Result: PASS_ENGINEERING_WITH_DOC_PATH_AMBIGUITY.

The exact command in CLOSED_LOOP_V2_INVOCATION.before.md was executed once with code_v11 / closure_unitree_v3. There was no source repair, environment change, experiment retry, training launch, external notification, or automation. Original V2 evidence was not modified.

## Launch and completion

- Submitted at 2026-10-22T12:05:04.920399+00:00; submit command exited 0.
- Supervisor PID 3160058; supervised child PID 3160059.
- Binding: validated CPU16-19, cuda:1, headless.
- Prelaunch free GPU memory: GPU0 12087 MiB; GPU1 13248 MiB.
- Actual child completion: EXITED / exit_code 0 at 2026-10-22T12:07:59.356604+00:00.
- The actual completion record is preserved at remote/logs/closure_unitree_v3_exit.json.
- Read-only poll observations: 12:06:10Z, 12:07:09Z, 12:08:10Z. All intervals were at least 30 seconds; each explicit wait was 30 seconds; run monitoring finished within 15 minutes.

## Evidence and scope

verification.json reports COMPLETED_ENGINEERING_VALIDATION / REJECTED_WITHOUT_INVALID_COMMIT. All seven actual Unitree-G1-PhyGile-v0 Isaac physics workers completed with exit 0. Frozen GMT and generator parameters remained unchanged.

The same-seed fresh-process replay repeated ten frames with state and body maximum delta 0.0 (tolerance 0.0005). The separate held-initialization-pose proxy accepted two one-frame windows, at MPJPE 0.0162544111 m and 0.0247734247 m. The accepted history replay maximum state difference was 0.0.

Actual untrained, unconditional diffusion produced two attempted 50-frame windows. Both recorded one frame before failure and were rejected for incomplete_continuation, failure_termination and mpjpe_above_tolerance; MPJPE was 0.3931637644 m and 0.3922078545 m. No generated frames were committed; initial and committed prefix hashes match. This is bounded engineering evidence for replay, feedback and rejection. It does not establish stable or semantic motion generation, trained generation quality, a motion benchmark, or full PhyGile reproduction. No PPO updates occurred.

All eight copied loaded-source manifests match their verification hashes. A separate read-only remote rehash checked 471 unique loaded source/input files against these manifests: zero mismatches. The environment spec canonical SHA256 matches the ledger: a43f960bd3bcdcd3398655bffe291138b73de5ca7f30901765172ff9821e1954.

## Document/reality divergence

The invocation itself worked verbatim. The documentation's logs/... references are ambiguous about their base directory. Reading logs under the returned runs/closure_unitree_v3 directory failed; the supervisor arguments revealed that the logs are under /data/code/ouyanghao/oyh/phygile-unitree-20261022/logs/. The witness then read those actual locations without changing source, command, or run. The initial failed reads are retained in poll_001.json. This is a documentation path ambiguity, not a simulator failure.

## Preserved artifacts

- exact_invocation.txt, launch_result.json and launch_output.txt preserve the exact command and launch output.
- Input documentation snapshots and input_document_hashes.json preserve what the fresh agent read.
- poll_001.json through poll_003.json preserve the bounded observations; poll_003 display was truncated, so complete stdout and metadata were separately copied.
- remote/runs/closure_unitree_v3/verification.json is the complete untruncated verification.
- remote/logs/closure_unitree_v3_stdout.txt, closure_unitree_v3_stderr.txt and closure_unitree_v3_exit.json are the actual full run logs and completion record.
- remote/runs/closure_unitree_v3/executed_sources.json and each simulator/invocation_*/executed_sources.json preserve the loaded source manifests.
- worker_result.json, request.json and worker.log are retained for every physics invocation.
- witness_checks.json, loaded_source_hashes.json and source_hash_recheck.json preserve independent evidence checks.
- copied_metadata_manifest.json identifies all 33 copied metadata files by SHA256 and size.
- remote_metadata.tar.gz SHA256: 96f14fc1bf88e196781332edf82a75e24c86159d27e76febe2585438f90d7215.

Only JSON, log and text metadata were copied. Raw private arrays, datasets and checkpoints remain on the server. ReactiveBFM processes, code, environment and inputs were not operated on.


# Corrected recorded-resume independent document witness v2

Status: v2 documented witness passed, with the documented native-close fallback. All 25 recorded checks are true; no divergence between the updated procedure and this run was found. No formal training, physical filtering, dataset acceptance or generator claim is made.

## Preserved v1 limitation

The v1 saved-value restoration was correct, but clearing the per-environment attempt marker without explicitly restarting physics could omit the first real resumed episode. Earlier v1 counting evidence is narrower than a complete resumed-episode accounting guarantee. All v1 evidence was preserved unchanged.

The updated procedure explains that resumed physics must restart explicitly, then saved EMA/raw counts must be restored again to exclude the synthetic reset and genuine attempts marked active. It also records the preserved author_v3 inference-mode reset failure and author_v4's scoped reset fix. This witness did not inspect or repair source code, inspect those prior failures, or overwrite any earlier evidence. The limitation is recorded here without changing the immutable v1 report.

## Exact new single invocation

Reread only FILTER_TRAINING_ENVIRONMENT.md, SERVER_NOTES.md and the exact referenced configs/filter-training-plan.json. The run-experiment skill had already been read for this agent. Executed the updated documented SSH invocation once unchanged for filter_statistics_document_v2, mode witness. The exact command is in documented-command.ps1.

Local session 21725 returned actual exit 0 and TEXTOP_FILTER_POLICY_SUPERVISED_EXIT0. Supervisor child PID 3493736, child_exit_code=0, timed_out=false, elapsed_seconds=41.536639554426074. The numerical receipt, supervisor and log all report TEXTOP_FILTER_STATISTICS_RESUME_WITNESS_PASSED.

Only the four whitelisted public remote files were read/copied: numerical receipt, supervisor exit receipt, log and teardown stack. No retries, edits, source inspection/repair, other process action, private checkpoint/data copies, publication or messages to other chats occurred.

## Required corrected-runtime checks

- mode=witness, num_envs=8, reference_sources=2. The plan/procedure specifies originals; no private motion names or arrays were inspected.
- completed_learning_iterations=2, target_learning_iterations=2, optimizer_updates=40.
- completed_1.pt public metadata: one completed iteration and 20 actual optimizer updates.
- completed_2.pt public metadata: two completed iterations and 40 actual optimizer updates.
- resume_exact_state_verified=true.
- resumed_physics_episodes_restarted_without_counting_reset=true.
- initial_resets_excluded_from_raw_counts=true.
- raw_completed_attempts=9.
- Log contains learning iteration 0/1 and resumed learning iteration 1/2; public update metadata advances from 20 to 40.
- output_writers_closed=true.

Per the procedure, the exact-state aggregate flag covers model, Adam, both normalizers, EMA and raw counters. The newly required aggregate restart flag records that new physical episodes were explicitly started without counting the reset and that their real attempts are counted. Verification uses these public flags, the success marker and continued-optimization metadata. Private state arrays were not independently reconstructed. Exact saved state is not bit-exact simulator continuation.

## Hash evidence

The exact local authorized plan byte hash independently matches receipt plan_sha256:
10c9ef10b7408c0023414c503dc4eef6a03415a2afc680befaee9a00fdcd6338

Source/runtime hashes are recorded from the public receipt only, without source inspection:
- source_sha256: 3929b9d8df0575778bfa5104a86d4515d834b20a8f1e54252964128ac0e16a53
- runtime_sha256: ad3bb92f5836e5213450420e8f33deffd43443da7d451796adef3d6193ef6962
- reference_fingerprint: 9d9ada9fcb17014efe0a663e28baa94f43c7a9ba57af0098862c0ed27fc80470

Private checkpoints remain on server. Public receipt metadata records both as 90600818 bytes, with hashes:
- completed_1.pt: 8ab2d121e750eb1e081ddb31da27b3072e5205e910231590beddf1f4a2264426
- completed_2.pt: 2a755bd0a59c013465e0bd0efd75b3be52c440714cce53ab3385437e5aaf7459

These checkpoint hashes were not independently recomputed; no checkpoint was opened or copied.

## Shutdown fallback

native_close_returned=false, teardown_fallback_used=true, shutdown_grace_seconds=20, shutdown_completion_verified=true. Receipts/checkpoints were durable and writers closed before the documented fallback. The stack blocks in SimulationContext.stop through the stop callback/render/physics-device chain; it does not prove SimulationApp.close was reached. Actual supervised exit 0 is verified separately. Native-close success is not claimed.

## Resources and original diagnostics

The log identifies headless cuda:1. The plan requests CPU4 and Torch allocator fraction 0.65; this public receipt does not separately expose measured affinity, thread or fraction fields. Peak Torch allocated=352427520 bytes (0.328 GiB).

Supervisor whole-device memory includes other processes: GPU1 starts at 7823 MiB and sampled peak is 10864 MiB of 49140 MiB; GPU0 starts at 12223 MiB and sampled peak is 12739 MiB. These samples cannot be attributed exclusively to this witness.

The original log preserves both optimization segments: value-function loss 1.5343/1.1541, mean reward -2.48/-3.32, body-position metric 0.0514/0.3377 and landing-force metric 28.7463/321.3867. These are small diagnostic-run observations, not convergence or tracking-quality evidence. Original display/GLFW/OmniHub/shader-cache/Git-metadata warnings remain retained without repairs.

formal_filter_training_started=false, full_generator_training_started=false, physical_filter_executed=false, dataset_accepted=false. The authorized plan keeps formal_launch_enabled=false. The limited BABEL-indexed public profile is not the full author corpus. No 8192-environment/40,000-iteration formal training, accepted motion filtering, full-corpus readiness or generator training follows from this witness.

## Local evidence

- documented-command.ps1
- stdout.log
- local-execution-receipts.jsonl
- filter_policy_filter_statistics_document_v2_exit.json
- receipt.json
- filter_policy_filter_statistics_document_v2.log
- native_teardown_stack.txt
- verification.json
- evidence-readback-receipts.jsonl

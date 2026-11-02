# Independent recorded filter-statistics/resume document witness

Status: the documented witness passed, with the documented native-close fallback. All 23 recorded checks are true; no documentation-to-reality divergence was found. This certifies the small recorded-statistics/resume diagnostic only.

## Scope and one exact invocation

Read only SERVER_NOTES.md, FILTER_TRAINING_ENVIRONMENT.md and its exact referenced configs/filter-training-plan.json. The run-experiment skill had already been read for this agent. Executed the documented PowerShell SSH invocation once unchanged for filter_statistics_document_v1, mode witness. The exact command is in documented-command.ps1.

Local session 14197 returned actual exit 0 and TEXTOP_FILTER_POLICY_SUPERVISED_EXIT0. Supervisor child PID 3487578 returned child_exit_code=0, timed_out=false, elapsed_seconds=41.51786145195365. The receipt, supervisor and original log report TEXTOP_FILTER_STATISTICS_RESUME_WITNESS_PASSED.

The only remote evidence read/copied was the four procedure-whitelisted public files: the numerical receipt, supervisor exit receipt, diagnostic log and teardown stack. No source inspection, retry, repair, source/document edit, other process action, private checkpoint/data copy or publication occurred. The previous author failure was not changed.

## Required result and restoration evidence

- mode=witness, num_envs=8, reference_sources=2. The document and plan define this selection as original reference trajectories; private reference names/arrays were not inspected.
- completed_learning_iterations=2, target_learning_iterations=2, optimizer_updates=40.
- completed_1.pt public metadata: one completed iteration, 20 actual optimizer updates, 90600818 bytes.
- completed_2.pt public metadata: two completed iterations, 40 actual optimizer updates, 90600818 bytes.
- resume_exact_state_verified=true. Per the procedure, this aggregate witness assertion covers exact model, Adam, both normalizers, EMA and independent raw-attempt counters.
- initial_resets_excluded_from_raw_counts=true; raw_completed_attempts=5.
- The original log contains learning iteration 0/1, a reconstructed model/critic, then learning iteration 1/2. The checkpoint metadata advances from 20 to 40 actual updates, confirming continued optimization after restore.
- output_writers_closed=true.

Restoration and reset-exclusion conclusions use the required public witness flags, program success marker and public checkpoint/update metadata. Private state tensors and counter arrays were not read or copied. The raw count is a diagnostic count, not a motion-acceptance result. Exact saved optimization/statistical state does not imply bit-exact simulator continuation: physical episodes restart after recovery, as the plan states.

## Hash evidence boundaries

The exact authorized local filter-training-plan.json byte hash independently matches receipt plan_sha256:
10c9ef10b7408c0023414c503dc4eef6a03415a2afc680befaee9a00fdcd6338

Public receipt records source_sha256:
2f8fe7756eb948cfa4196a62db2b6b6cc086687f8ae00b768ec5bb3764972bd9

Public receipt records runtime_sha256:
4d3fcb8662bfca49ee1c2709b0509bb932cbc28d12a05422e9ac63671a1cd0f9

Public receipt reference_fingerprint:
9d9ada9fcb17014efe0a663e28baa94f43c7a9ba57af0098862c0ed27fc80470

Checkpoint SHA256 values below are recorded from the public receipt only, not independently recomputed; checkpoints were neither opened nor copied:
- completed_1.pt: 8ab2d121e750eb1e081ddb31da27b3072e5205e910231590beddf1f4a2264426
- completed_2.pt: bb04ae41b5f8ae0888628ef03970364055b5f7649f7bed9418806f92895bf8b8

## Native shutdown fallback

native_close_returned=false, teardown_fallback_used=true, shutdown_grace_seconds=20 and shutdown_completion_verified=true. Writers were closed and completed receipts/checkpoints saved before the bounded fallback. The retained stack blocks in SimulationContext.stop via the stop callback/render/physics-device chain, and does not prove SimulationApp.close was reached. Actual supervised exit 0 is verified separately. This is not native-close success.

## Resources, original diagnostics and limitations

The log confirms GPU1/cuda:1 and headless IsaacLab. The plan requests CPU4 and Torch fraction 0.65; this receipt does not separately expose measured CPU affinity/Torch thread or fraction fields. Peak Torch allocated=352427520 bytes (0.328 GiB).

Supervisor device-memory samples include other processes. GPU1 starts at 7823 MiB and sampled peak is 10866 MiB of 49140 MiB; GPU0 starts at 12223 MiB and sampled peak is 12739 MiB. These are whole-device sampled totals.

The original log retains both short optimization segments, including value-function loss 1.5343/1.3430, mean reward -2.48/-2.24, body-position metric 0.0514/0.5962 and landing-force metric 28.7463/185.5016. These are diagnostic outputs, not evidence of convergence or tracking quality. Headless display/GLFW warnings, inaccessible OmniHub, DriverShaderCacheManager initialization error and missing rsl_rl Git metadata remain preserved without repairs.

formal_filter_training_started=false, full_generator_training_started=false, physical_filter_executed=false and dataset_accepted=false. The plan's formal_launch_enabled remains false. No 8192-environment/40,000-iteration formal training, motion filtering/acceptance or generator training was launched by this witness.

The available BABEL-indexed public profile is not the full 12,296 author tracking clips; unavailable selected exports/private clips and complete-library capacity/stability remain explicit limitations. This witness cannot establish final corpus readiness.

## Local evidence

- documented-command.ps1
- stdout.log
- local-execution-receipts.jsonl
- filter_policy_filter_statistics_document_v1_exit.json
- receipt.json
- filter_policy_filter_statistics_document_v1.log
- native_teardown_stack.txt
- verification.json
- evidence-readback-receipts.jsonl

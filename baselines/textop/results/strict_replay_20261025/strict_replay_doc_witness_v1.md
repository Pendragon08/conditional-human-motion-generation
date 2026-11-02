# Independent strict-replay documented witness

Status: infrastructure diagnostic passed with the documented native-close fallback. All 31 procedure checks are true; no documentation-to-reality divergence was found. Both motion trials failed, complete_trials=0 and completion_branch_exercised=false. No natural-completion, tracking-quality or accepted-motion pass is claimed.

## Scope and exact single execution

Read only SERVER_NOTES.md, STRICT_REPLAY_ENVIRONMENT.md and its exact referenced configs/strict-replay-spec.json. The run-experiment skill had already been read for this agent. Executed the documented PowerShell SSH command exactly once unchanged for strict_replay_document_v1, mode witness. The exact command is in documented-command.ps1.

Local session 52300 returned actual exit 0 and TEXTOP_STRICT_REPLAY_SUPERVISED_EXIT0. Supervisor child PID 3526179, child_exit_code=0, timed_out=false, elapsed_seconds=36.56384531036019. Receipt, supervisor and original log report TEXTOP_STRICT_REPLAY_DIAGNOSTIC_PASSED.

Read/copied only the four permitted public receipts/log/stack. As explicitly required, source and spec were remotely SHA256-hashed without inspecting source content. No retries, edits, source inspection/repair, other process actions, private checkpoint/per-motion/reference copies or publication occurred.

## Actual dynamic execution and limits

- reference_sources=2; eligible_original_sources=2; num_envs=2.
- optimizer_updates=0.
- policy_checkpoint_completed_iterations=2.
- finished_trials=2, failed_trials=2, complete_trials=0.
- actual_vectorized_control_calls=16; actual_scene_physics_steps=64; max_contiguous_control_steps=16.
- diagnostic_dynamic_rollouts_executed=true.
- completion_branch_exercised=false; motion_complete termination events=0.

The two failures are permissible for this two-iteration policy diagnostic. They are failed motion trials within a completed executor test, not successful tracking. This run did not exercise genuine SDK natural completion. The documentation's pure accounting/schema coverage cannot be substituted for a dynamically completed SDK motion, and this witness makes no such claim.

## Reset alignment and strict conditions

reset_reference_alignment_refreshed_without_time_advance=true and deterministic_reset=true. Initial maximum tracked-body discrepancy=0.026968445628881454 m.

The receipt preserves pelvis-height threshold 0.25 m and end-effector-height threshold 0.5 m. The original termination-manager log shows anchor_pos, anchor_ori, ee_body_pos and motion_complete, with motion_complete as timeout. The spec enables exact motion-end replacement of the default timeout. This independently checks the public thresholds/active-term evidence; source logic was not inspected.

Observed termination_term_event_counts:
- anchor_pos: 1
- anchor_ori: 0
- ee_body_pos: 1
- motion_complete: 0

The updated procedure records that author_v1's first-step height failures were contaminated by stale pinned alignment buffers, and that author_v2 refreshes the released alignment transform after reset without advancing time. That earlier source/log evidence was left untouched. The required corrected alignment flag is true in this witness. It does not establish overall tracking success.

## Independent source/spec hash checks

Remote sha256sum matches receipt source_sha256:
db5d04ee87603f88034eb24eb84fd80d69294a53f4a53fa2a1ae1627c3429535

Remote sha256sum and the exact authorized local spec hash both match receipt replay_spec_sha256:
e5f95521cb54084f550362053e9d130595b51eb9895569ff18cc600aef86188f

The receipt-reported checkpoint SHA matches the exact SHA pinned in the spec:
2a755bd0a59c013465e0bd0efd75b3be52c440714cce53ab3385437e5aaf7459
The checkpoint file was not independently read, copied or rehashed by this witness.

Private relaxed/strict log hashes are retained in the public receipt only; per-motion JSON and trajectories stayed on server.

## Shutdown fallback

output_writers_closed=true, native_close_returned=false, teardown_fallback_used=true, shutdown_grace_seconds=20 and shutdown_completion_verified=true. The numerical outputs were saved and writers closed before fallback.

The retained stack is blocked in SimulationContext.stop through callback/render/physics-device queries; it does not prove SimulationApp.close was reached. Actual child exit 0 is independently preserved by the supervisor. This is the documented bounded process-exit fallback, not native-close success.

## Resources, warnings and acceptance boundary

Log identifies headless cuda:1; receipt CPU affinity is [0,1,2,3]. The spec requests four Torch threads and allocator fraction 0.65; the public receipt does not separately measure those two settings. Peak Torch allocated=130169344 bytes (0.121 GiB).

Whole-device memory samples include other processes. GPU1 starts at 7823 MiB, sampled peak 10628 MiB and ends at 7823 MiB of 49140 MiB. GPU0 starts at 12223 MiB, sampled peak 12739 MiB and ends at 12223 MiB. These are sampled driver totals, not memory attributable exclusively to the diagnostic.

Original headless display/GLFW/OmniHub/shader-cache warnings remain in the saved log without repair or suppression.

formal_filter_training_started=false, physical_filter_executed=false, dataset_accepted=false and full_generator_training_started=false. Formal evaluation remains disabled in the spec pending a real 40,000-iteration policy and complete reference library. The two-iteration checkpoint cannot certify accepted motions, final corpus readiness or learned tracking quality.

## Local evidence

- documented-command.ps1
- stdout.log
- local-execution-receipts.jsonl
- strict_replay_strict_replay_document_v1_exit.json
- receipt.json
- strict_replay_strict_replay_document_v1.log
- native_teardown_stack.txt
- source_and_spec_remote_sha256.txt
- verification.json
- evidence-readback-receipts.jsonl

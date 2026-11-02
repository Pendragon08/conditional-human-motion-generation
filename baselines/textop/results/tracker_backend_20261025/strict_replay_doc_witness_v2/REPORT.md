# Independent strict-replay prefix-fixture document witness v2

Status: documented infrastructure witness passed with native-close fallback; all 44 recorded checks are true and match the updated procedure. Both FULL original motions still failed, with zero full completions. Two shortened two-frame fixtures completed, verifying only SDK end-signal handling. No full-motion, tracking-quality or acceptance pass is claimed.

## Exact single run and evidence scope

Reread only STRICT_REPLAY_ENVIRONMENT.md, SERVER_NOTES.md and its exact referenced configs/strict-replay-spec.json. The run-experiment skill had already been read for this agent. Executed the updated SSH command once unchanged for strict_replay_document_v2, mode witness; see documented-command.ps1.

Local session 99662 returned actual exit 0 and TEXTOP_STRICT_REPLAY_SUPERVISED_EXIT0. Supervisor child PID 3529810, child_exit_code=0, timed_out=false, elapsed_seconds=36.86169198527932. Supervisor, numerical receipt and original log report TEXTOP_STRICT_REPLAY_DIAGNOSTIC_PASSED.

Only the permitted public receipts/log/stack and requested source/spec SHA256 output were collected. No source content was inspected. No retry, source/procedure edit, repair, other process action, private checkpoint/per-motion/reference copy or publication occurred. All v1 evidence was preserved unchanged.

## Full original motions remain failures

The documentation identifies the original references as 85 and 90 frames. Public receipt records:
- reference_sources=2; eligible_original_sources=2; num_envs=2.
- policy_checkpoint_completed_iterations=2; optimizer_updates=0.
- finished_trials=2; failed_trials=2; complete_trials=0.
- completion_branch_exercised=false.
- termination_term_event_counts: anchor_pos=1, anchor_ori=0, ee_body_pos=1, motion_complete=0.
- max_contiguous_control_steps=16.
- diagnostic_dynamic_rollouts_executed=true.

Both private original-trial log hashes in the public receipt match the previously collected v1 public receipt. This supports unchanged recorded original-trial results without copying the private logs. It does not independently inspect reference arrays.

No FULL original completed naturally, so genuine full-motion natural completion and successful learned tracking remain unvalidated.

## Separate two-frame SDK end-handling fixture

diagnostic_prefix_probe is explicitly labeled:
- kind=two_frame_prefix_control_flow_fixture_only.
- fixture_envs=2; reference_frames=2.
- control_calls=1.
- complete_fixture_trials=2.
- sdk_motion_end_branch_verified=true.
- full_original_motion_passes=0.
- original_reference_arrays_unchanged=true.

The fixture deliberately shortens the diagnostic duration to two frames and uses one real vectorized SDK control step for its two environments. These two fixture completions exercise motion-end signal handling only. They do not turn either original motion into a passing full trial and do not produce accepted motion data.

The public unchanged-array flag is verified as true; arrays were not independently read or copied. Full original completion_branch_exercised=false remains distinct from the prefix probe's sdk_motion_end_branch_verified=true.

Overall observed work: actual_vectorized_control_calls=17 and actual_scene_physics_steps=68, including the added fixture step. These match the documented latest author expectation. No numeric differences were observed in the expected summary counts.

## Reset and strict-condition evidence

reset_reference_alignment_refreshed_without_time_advance=true; deterministic_reset=true; initial_max_tracked_body_error_m=0.026968445628881454 m.

Public thresholds remain pelvis height 0.25 m and end-effector height 0.5 m. Termination-manager log includes anchor_pos, anchor_ori, ee_body_pos and motion_complete, with motion_complete as timeout. The spec retains exact T-1 motion-end replacement of the default timeout. No threshold relaxation is evidenced or claimed by this witness, and source code was not inspected.

The documented stale-buffer first-step issue in historical author_v1 and its retained source/logs were not changed by this witness.

## Independent source/spec hashes

Remote source sha256sum independently matches receipt source_sha256:
5c8e0bbec3486580596916eb8fcc823d9c47f46880c8b39068b7655b4ecc4ce0

Remote spec sha256sum and the exact authorized local spec hash independently match receipt replay_spec_sha256:
e5f95521cb54084f550362053e9d130595b51eb9895569ff18cc600aef86188f

Receipt policy checkpoint SHA matches the exact spec pin:
2a755bd0a59c013465e0bd0efd75b3be52c440714cce53ab3385437e5aaf7459

The checkpoint itself was not opened, copied or independently rehashed.

## Native-close fallback and resources

output_writers_closed=true, native_close_returned=false, teardown_fallback_used=true, shutdown_grace_seconds=20, shutdown_completion_verified=true. Durable outputs and closed writers precede fallback. The retained stack blocks during SimulationContext.stop's callback in get_timeline_interface; it does not prove native SimulationApp.close was reached. Actual exit 0 is separate from native-close success.

Log confirms headless cuda:1 and receipt affinity [0,1,2,3]. The spec requests four Torch threads/allocator fraction 0.65; the public receipt does not separately measure those settings. Peak Torch allocated=130169344 bytes (0.121 GiB).

Supervisor whole-device memory includes other processes. GPU1 starts/ends at 7823 MiB, sampled peak 10628 MiB of 49140 MiB. GPU0 starts/ends at 12223 MiB, sampled peak 12739 MiB. Original headless display/GLFW/OmniHub/shader-cache warnings remain retained.

## Acceptance boundary and evidence files

formal_filter_training_started=false, physical_filter_executed=false, dataset_accepted=false, full_generator_training_started=false. formal_evaluation_enabled=false in the spec. This two-iteration policy and shortened control-flow fixture do not certify 40,000-iteration training, full corpus readiness, trained tracking quality or accepted motions.

Saved evidence:
- documented-command.ps1
- stdout.log
- local-execution-receipts.jsonl
- strict_replay_strict_replay_document_v2_exit.json
- receipt.json
- strict_replay_strict_replay_document_v2.log
- native_teardown_stack.txt
- source_and_spec_remote_sha256.txt
- verification.json
- evidence-readback-receipts.jsonl

# Experiment status — V2 snapshot, 2026-10-22

- framework_inference: DONE
- cross_simulation_four_clip_pilot: DONE_WITH_FAILURES
- fresh_document_cpu_witness: DONE
- headless_user_authorization: EXPLICIT_CURRENT_TASK_EXCEPTION
- restored_ppo_two_iteration_witness: DONE_40_ACTUAL_ADAM_UPDATES
- continued_ppo_500_iterations: IN_PROGRESS
- dependent_isaac_evaluation_export: WAITING_FOR_TRAINING
- physx_prefix_closure_migration: DONE_ENGINEERING_VALIDATION
- mujoco_prefix_closure_migration: DONE_ENGINEERING_VALIDATION
- generated_motion_accepted_frames: 0
- semantic_motion_generation: NOT_TRAINED_OR_VALIDATED
- matched_sliding_friction_diagnostic: DONE_NO_IMPROVEMENT_1_OF_4_BOTH
- full_paper_reproduction: NOT_COMPLETED
- github_publication: PUBLISHED_CORE_ARTIFACTS_VERIFIED
- github_v2_publication: PREPARING
- datasets_deleted: False
- checkpoints_deleted: False

V1's unverified local viewer condition and no-training status remain in its historical report. The later explicit user authorization permits headless continuation for this task. The completed two PPO iterations are distinct from the ongoing 500-iteration run; submission or inference does not count as completed training.

Both closure paths ran actual untrained diffusion and real physics. Two one-frame held-pose proxy windows were accepted in each; actual generated 50-frame proposals were rejected and zero generated frames committed. Engineering validation does not establish stable semantic generation. ReactiveBFM was not stopped or restarted.

- independent_integrity_audit: WARN (same-family/provisional; historical source attestation limitation retained)
- v2_independent_integrity_audit: PENDING

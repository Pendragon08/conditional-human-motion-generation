# Experiment status — V3 completed bounded run, 2026-10-22

- framework_inference: DONE
- cross_simulation_four_clip_pilot: DONE_WITH_FAILURES
- fresh_document_cpu_witness: DONE
- headless_user_authorization: EXPLICIT_CURRENT_TASK_EXCEPTION
- restored_ppo_two_iteration_witness: DONE_40_ACTUAL_ADAM_UPDATES
- continued_ppo_500_iterations: COMPLETED_10000_ACTUAL_ADAM_UPDATES
- training_child_exit_code: NOT_RECORDED_BY_EARLY_LAUNCHER
- dependent_isaac_evaluation_export: COMPLETED_SUPERVISED_EXIT_0
- continued_policy_dual_simulator_evaluation: COMPLETED_NO_IMPROVEMENT
- retained_tracking_baseline: ORIGINAL_2000_ITERATION_PARENT
- physx_prefix_closure_migration: DONE_ENGINEERING_VALIDATION
- mujoco_prefix_closure_migration: DONE_ENGINEERING_VALIDATION
- generated_motion_accepted_frames: 0
- semantic_motion_generation: NOT_TRAINED_OR_VALIDATED
- matched_sliding_friction_diagnostic: DONE_NO_IMPROVEMENT_1_OF_4_BOTH
- full_paper_reproduction: NOT_COMPLETED
- github_publication: PUBLISHED_CORE_ARTIFACTS_VERIFIED
- github_v3_publication: PREPARING
- datasets_deleted: False
- checkpoints_deleted: False

V1's viewer/no-training condition and V2's pending500 state remain in their historical reports and results/v3/history_v2. Later user authorization permits headless continuation. V3 completion is based on final training receipt, 10000 actual Adam calls, preserved full finite checkpoint and executed-source hashes; no exit0 is invented for the early unsupervised training launcher. Its dependent evaluation/export pipeline and new-policy PhysX closure have supervised actual exits0.

Both closure paths ran actual untrained diffusion and real physics. Two one-frame held-pose proxy windows were accepted in each; actual generated 50-frame proposals were rejected and zero generated frames committed. Engineering validation does not establish stable semantic generation. ReactiveBFM was not stopped or restarted.

- independent_integrity_audit: WARN (same-family/provisional; historical source attestation limitation retained)
- v2_independent_integrity_audit: WARN_SAME_FAMILY_PROVISIONAL_PRESERVED
- v3_independent_integrity_supplement: PENDING

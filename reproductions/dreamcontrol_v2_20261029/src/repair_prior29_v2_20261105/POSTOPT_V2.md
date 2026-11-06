# Motion-preserving postoptimization v2

This is a new project implementation responding to the 2026-11-05 audit, not released DreamControl-v2 author code. Existing sources, references, checkpoints, and evaluation results remain unchanged.

## Input and coordinate contract

The sampler supplies `canonical_points (1,T,22,3)`, `target_points (T,22,3)`, `target_mask (T,22)`, `target_points_world (T,22,3)`, `desired_wrist_quat_wxyz (T,2,4)`, `planned_hand_closed (T,2)`, and `desired_object_positions (T,3)`. T is 120 for Pick and 180 for the other tasks, at 20 Hz.

Metadata must include explicit forward and inverse transforms, following `world = canonical @ canonical_to_world_rotation.T + canonical_to_world_translation`. The independently saved world targets must match the round trip within 1e-6 m and their float32 content SHA. `coordinate_provenance` must use schema `training_floor_canonical_v2`, record `ground_z`, `height_origin_world_m`, `calibration_sha256`, and the compact-JSON SHA of the actual 29 joint names. The postoptimizer does not derive a new floor or add a second root-height offset.

Both wrist orientations and binary hand plans come from the sampler. Pick uses saved `grasp_frame`/`grasp_time_s`, including randomized frames 40–50; it is not overwritten with a hidden frame-40 schedule.

## Two stages and rejection

1. **A: realize the decoded motion.** Fit all 22 points and their temporal changes to the generated world-space trajectory, subject to joint-limit and geometry penalties. Initialize root position/facing from each decoded frame. A small joint-only gauge prior resolves point-unobservable twists. No constant neutral-root target, fixed foot XY target, or waist lock is used.
2. **B: limited task correction.** Start from A and keep A as the reference anchor. Add active wrist position/orientation objectives and body/root/joint trust-region penalties. Try task penalty multipliers 1, 2, and 4, always restarting from A to avoid cumulative drift. Accept the first result satisfying every independent CPU gate; retain all attempted raw states.

The solver uses penalty residuals, not an exact constrained optimizer. Its objective value is never treated as acceptance. NumPy URDF FK and all collision-mesh vertices independently recompute the final checks after a strictly sub-1mrad joint-limit projection.

Preserved physical gates are wrist error <3 cm, wrist angle <0.20 rad, floor penetration ≤5 mm, foot sole <2 cm above the physical floor, at most 10% foot intervals exceeding 0.15 m/s, and root height strictly between 0.3 and 1.2 m. The 3 cm/0.20 rad limits apply to repaired Pick as well as the three other tasks and are stricter than some historical early Pick checks. Table vertex penetration also must not exceed 5 mm. This is a geometric check, not exact contact dynamics or a full triangle-intersection/self-collision proof; native admission remains required.

New explicit project quality gates prevent a precise wrist fit from replacing the motion: decoded-point vector RMSE ≤5 cm and peak ≤15 cm; decoded-velocity vector RMSE ≤0.20 m/s; B-to-A nonwrist RMSE ≤3 cm and peak ≤8 cm; root shift ≤5 cm; joint shift ≤0.5 rad; joint per-frame step ≤0.30 rad; wrist per-frame angular change ≤0.25 rad. A raw wrist target error >12 cm rejects the candidate before fitting instead of rewriting a grossly uncontrolled sample. These are proposed project quality constraints, not paper-reported thresholds, and their production pass rate has not been established.

All thresholds and objective scales are versioned in `postopt_v2.py`. Failure does not change a target, lower a gate, or restart an old job. Fresh output directories preserve stage A, each B attempt, source hashes, metrics, and any traceback. Successful geometry produces `robot29.npz` with the old downstream key layout plus provenance. Its status is only `kinematic_candidate_ready_for_native_probe`; `eligible_for_calibrated_scene_reference` remains false until independent native admission. Binary `hand_closed`, planned object paths, and fitted wrist orientations are never described as measured palm/finger contact or physical task success.

## Integration and validation

Import `MotionPreservingSolverV2` once per frame count and reuse its compiled solver for a cohort. Its `process(sample_path, fresh_output_dir, expected_sha256)` method handles one immutable candidate. The command-line entry point accepts `--sample`, `--sample-sha256`, `--output-dir`, `--urdf`, and an optional `--fk-module`; the original `g1_fk.py` is loaded read-only and hash-recorded.

Nine CPU-only tests cover explicit height transforms, old zero-translation rejection, inverse and source-target hash checks, unchanged strict physical boundaries, rejection of precise wrists with excessive body changes, temporal gates, vector-RMSE definitions, randomized Pick grasp-time preservation, and NaN rejection. These tests import NumPy and the validation helpers only; they do not import JAX or execute optimization. The initial attempt using base Python lacked NumPy; tests subsequently passed in the already installed `pytorch_gpu` environment without installation.

JAX compilation and actual solver convergence remain untested by this CPU contract suite. The root controller must perform its authorized isolated runtime validation before processing a new cohort; no successful fitting, native admission, or task performance is claimed here.

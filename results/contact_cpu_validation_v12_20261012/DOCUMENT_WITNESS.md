# V12 independent document-following witness

Status: **PASS_RECORDING_ONLY**

Fresh agent: contact_cpu_full_validation_v12_document_witness. Date: 2026-10-12. Read the run-experiment skill, its compute-environment contract, the specified CLoSD baseline ledger, and docs/contact_cpu_validation_v12_20261012.md before execution. The exact documented command was executed once, without edits, retries, --run, or extra simulations:

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_contact_cpu_validation_v12_20261012.py --check
```

Observed exit code: 0. Actual stdout:

```text
CONTACT_SAMPLING_BATCH_PASS val_planned_s11
CPU_EXPORT_VALIDATION_DOCUMENT_CHECK_PASS
```

The command produced the planned/seed11 recording: 72 first attempts from 24 distinct original sources, with exactly offsets -0.15, 0.0, +0.15 metres for each source. Arrays have 120 time slots and 72 cases; valid prefixes include the two retained early terminations. Actual completion is 70/72 (97.22%); termination is 2/72 (2.78%). Case 010623:-0.15 terminates with 95 valid frames, and 010623:0.15 with 109. No predetermined success count was imposed.

The saved reference clock is exactly 0..119. All arrays in trajectories.npz, initial_states.npz, and sensor_diagnostic.npz are finite, including padded slots. Valid masks are contiguous prefixes. All seven bootstrap/first-action state arrays are exactly equal (maximum difference 0.0), with equal state hashes, zero progress and start times, and motion IDs 0..71. Initial reset gate passes.

Thirty frozen code, external-input and recorded-output SHA256 comparisons pass, as do request/lineage and configuration/manifest/threshold/version bindings. Upstream commit matches d8950bdb78312fcd2844f336b3fc079e7368c87f. Actual generator and controller files match their frozen hashes. The actual controller location is established from the batch log's loading messages; resolved_config.yaml contains an inherited default model path, so the executed loading log was used for this audit. No execution-document divergence was observed for the required invocation and recording interface.

Actual recorded frozen-threshold counts are condition 23/72 (31.94%), physical 0/72 (0%), tracking 53/72 (73.61%), and joint 0/72 (0%). These are descriptive results from this single planned/seed11 batch. The saved numeric comparison reports PASS_ROUNDOFF_ONLY, 22 changed floating values, maximum absolute difference 4.440892098500626e-16, identical classifications, unchanged thresholds and unchanged saved scores.

Elapsed recorder receipt: 45.40392518043518 seconds. Trajectory SHA256: 45486979e705ce28f3e5e10976814ce901f3008ea902e5e23d326a5c0a4311bd.

Licensed raw NPZ archives were inspected on the server and remain server-local. Only aggregate checks and hashes appear here. The witness certifies saved recording integrity for this invocation only; it does not establish efficacy, full measurement validity, crossroute invariance, independent semantic validity, known-good humanoid quality calibration, downstream utility or robot utility. Root owns the remaining five batches.

## Machine-readable aggregate audit

```json
{
  "status": "PASS_RECORDING_ONLY",
  "checks": {
    "case_count_72": true,
    "manifest_24_unique_sources": true,
    "exact_three_offsets_per_source": true,
    "case_sources_equal_manifest": true,
    "shapes_120x72": true,
    "reference_clock_exact_0_to_119": true,
    "all_saved_arrays_finite": true,
    "all_terminal_cases_retained": true,
    "valid_prefix_contiguous": true,
    "reset_states_identical": true,
    "progress_zero_before_action": true,
    "motion_ids_and_clock_initial": true,
    "frozen_code_external_inputs_and_output_hashes_match": true,
    "request_lineage_identical": true,
    "request_hashes_match": true,
    "frozen_generator_controller_hashes_match": true,
    "upstream_commit_matches": true,
    "frozen_models_metadata": true
  },
  "counts": {
    "completed": 70,
    "terminated": 2,
    "condition_pass": 23,
    "physical_pass": 0,
    "tracking_pass": 53,
    "joint_pass": 0
  },
  "rates": {
    "completed": 0.9722222222222222,
    "terminated": 0.027777777777777776,
    "condition_pass": 0.3194444444444444,
    "physical_pass": 0,
    "tracking_pass": 0.7361111111111112,
    "joint_pass": 0
  },
  "early_terminal_cases": [
    {
      "index": 12,
      "case_id": "010623:-0.15",
      "valid_length": 95,
      "terminated": true,
      "saved_metric_terminated": true
    },
    {
      "index": 14,
      "case_id": "010623:0.15",
      "valid_length": 109,
      "terminated": true,
      "saved_metric_terminated": true
    }
  ],
  "length_range": [
    95,
    120
  ],
  "finite_arrays": {
    "trajectories.npz": {
      "actual": true,
      "reference": true,
      "quaternion": true,
      "linvel": true,
      "angvel": true,
      "contact_force": true,
      "valid": true,
      "terminated": true,
      "reference_frame": true
    },
    "initial_states.npz": {
      "bootstrap__humanoid_root_states": true,
      "bootstrap__dof_pos": true,
      "bootstrap__dof_vel": true,
      "bootstrap__rigid_body_pos": true,
      "bootstrap__rigid_body_rot": true,
      "bootstrap__rigid_body_vel": true,
      "bootstrap__rigid_body_ang_vel": true,
      "first_action__humanoid_root_states": true,
      "first_action__dof_pos": true,
      "first_action__dof_vel": true,
      "first_action__rigid_body_pos": true,
      "first_action__rigid_body_rot": true,
      "first_action__rigid_body_vel": true,
      "first_action__rigid_body_ang_vel": true,
      "first_action_contact_forces": true
    },
    "sensor_diagnostic.npz": {
      "sensor": true,
      "root": true,
      "dof_pos": true,
      "dof_vel": true,
      "action": true
    }
  },
  "reset_actual_max_differences": {
    "_humanoid_root_states": 0,
    "_dof_pos": 0,
    "_dof_vel": 0,
    "_rigid_body_pos": 0,
    "_rigid_body_rot": 0,
    "_rigid_body_vel": 0,
    "_rigid_body_ang_vel": 0
  },
  "models": {
    "generator": {
      "path": "/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/CLoSD/closd/diffusion_planner/save/DiP_multi-target_10steps_context20_predict40/model000300000.pt",
      "actual_sha256": "3ca68ae54975946b3fe5c6cd64a6eafc439b5fc43c166bc9a98d1c43aeef6618",
      "expected_sha256": "3ca68ae54975946b3fe5c6cd64a6eafc439b5fc43c166bc9a98d1c43aeef6618",
      "match": true
    },
    "controller": {
      "path": "/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/CLoSD/output/CLoSD/CLoSD_multitask_finetune/Humanoid.pth",
      "actual_sha256": "7ead3778b95516d8b91b7c5939674e45dd0ad10fcc22da60f9f57d204695f6ac",
      "expected_sha256": "7ead3778b95516d8b91b7c5939674e45dd0ad10fcc22da60f9f57d204695f6ac",
      "match": true,
      "path_evidence": "val_planned_s11.log explicitly loads this controller twice"
    }
  },
  "upstream_commit": "d8950bdb78312fcd2844f336b3fc079e7368c87f",
  "hash_check_count": 30,
  "hash_failures": [],
  "trajectory_sha256": "45486979e705ce28f3e5e10976814ce901f3008ea902e5e23d326a5c0a4311bd",
  "receipt": {
    "status": "complete",
    "cases": 72,
    "completed": 70,
    "joint_pass": 0,
    "elapsed_seconds": 45.40392518043518,
    "trajectories_sha256": "45486979e705ce28f3e5e10976814ce901f3008ea902e5e23d326a5c0a4311bd"
  },
  "numeric_recomputation_summary": {
    "status": "PASS_ROUNDOFF_ONLY",
    "changed_floating_values": 22,
    "maximum_absolute_difference": 4.440892098500626e-16,
    "classifications_identical": true,
    "thresholds_changed": false,
    "saved_scores_unchanged": true
  },
  "scope": "Saved recording integrity only. No efficacy, full measurement validity, crossroute invariance, independent semantic validity, known-good quality calibration, downstream or robot utility claim."
}
```


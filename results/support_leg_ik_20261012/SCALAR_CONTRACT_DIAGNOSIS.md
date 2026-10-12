# Separate scalar-contract diagnosis

Status: PASS_DOCUMENTED_PREPARATION_INVARIANTS

Original DOCUMENT_WITNESS FAIL is untouched. No --check rerun, simulator, gate patch, source modification, threshold change or efficacy conclusion.

Bitwise equality of logged float64 accepted_shift_m is not required by the frozen document. All numerical evidence and individual scalar discrepancies are retained in JSON.

{
  "status": "PASS_DOCUMENTED_PREPARATION_INVARIANTS",
  "original_witness_status_unchanged": "FAIL",
  "checks": {
    "frozen_prepare_cartesian_exact": true,
    "frozen_prepare_ik_exact": true,
    "original_audit_cartesian_exact": true,
    "original_audit_ik_exact": true,
    "cartesian_prefix61": true,
    "cartesian_endpose": true,
    "cartesian_unchanged_anchors": true,
    "cartesian_unchanged_targets": true,
    "cartesian_finite": true,
    "cartesian_10cm": true,
    "cartesian_L_same_float32_foot_translation": true,
    "cartesian_R_same_float32_foot_translation": true,
    "two_link_ik_prefix61": true,
    "two_link_ik_endpose": true,
    "two_link_ik_unchanged_anchors": true,
    "two_link_ik_unchanged_targets": true,
    "two_link_ik_finite": true,
    "two_link_ik_10cm": true,
    "two_link_ik_L_same_float32_foot_translation": true,
    "L_L_Hip_1e5_segment_error": true,
    "L_L_Knee_1e5_segment_error": true,
    "two_link_ik_R_same_float32_foot_translation": true,
    "R_R_Hip_1e5_segment_error": true,
    "R_R_Knee_1e5_segment_error": true,
    "original_all_non_scalar_checks_pass": true,
    "all_failures_only_extra_exact_scalar_criterion": true,
    "original_files_unchanged": true
  },
  "documented_coordinate_invariants": [
    {
      "arm": "cartesian",
      "max_joint_shift_m": 0.0782050916595791,
      "max_segment_error_m": 0.05798189353482597
    },
    {
      "arm": "two_link_ik",
      "max_joint_shift_m": 0.0782050916595791,
      "max_segment_error_m": 8.361849629379137e-08
    }
  ],
  "dtypes": {
    "baseline": "float32",
    "solver_velocity": "float64",
    "translation": "float64",
    "IK": "float64",
    "accepted_shift_norm": "float64 -> Python float",
    "stored_reference": "float32",
    "prepare_bone_error": "float32 norm -> Python float",
    "audit_geometric_error": "float64 norm"
  },
  "array_layout": {
    "baseline_strides": [
      20736,
      288,
      12,
      4
    ],
    "baseline_c_contiguous": true,
    "baseline_f_contiguous": false,
    "prepare_copy_strides": [
      20736,
      288,
      12,
      4
    ]
  },
  "runtime": {
    "python": "3.8.20 (default, Oct  3 2024, 15:24:27) \n[GCC 11.2.0]",
    "numpy": "1.23.5",
    "openblas_threads": "2",
    "omp_threads": "2"
  },
  "bindings": {
    "original_files": {
      "DOCUMENT_WITNESS.json": "725192069fbfb02249643efcc9df985945f6114c8fc6d91d2f032afec7b7aa4e",
      "DOCUMENT_WITNESS.md": "026fb9db591c1540a5b7dd485ef42ceefe4d45ad4427cd5ed7cb184b382929c9",
      "preflight.json": "8b6bf591e4884886b712a11926a74af40a38de5943799c1dafe08bcd60e61750",
      "preparation.json": "67bfcd7cfc7ea23f4265126df73306eb141b7aedc61b1acce58238b4b3f400f1",
      "versions.json": "40f9c9ae71efd12177add7df9864817e5cbf06364928a41b1b39c3ce3c7ab5c1"
    },
    "preparation_source_sha256": "4978f7250dd78ce1bc70bc61014a3f8e6540ca3b4ba657d9364265d20a7b4e7c",
    "original_audit_source_sha256": "ba1ba329cdf8520188658b6651a853659a4e69215e2bb9a576323e641db85c92",
    "document_sha256": "d8a39aef7892d829fe629fc9c15635d72097e30df4bbea4366e1d86911ed4490",
    "diagnostic_source_sha256": "f4c7e287654be9909fccb9dfba29bde5a291618492c468d5df025745dfc9f7d1",
    "source_trajectories_sha256": "e96cb97ad5d12d4c54a3fbf57625d9ab4491e5c38febd800d78ae2a1baecfaaf",
    "references": {
      "baseline": "14dcb4ea715f485547c9abeba294ad3ead67dabee9807a83ebe6c98b2ea4b207",
      "cartesian": "74d9bd83981712d153f463ad8002cf92b61fce339da045c42dc7ecb2b35b7cfd",
      "two_link_ik": "1e6696de8f073a2d1c4b2c2d82caeed6747d75ce68ec557f32deb071002119b1"
    }
  },
  "frozen_prepare_scalar_summary": {
    "compared_leg_frames": 8352,
    "mismatched_legs": 117,
    "mismatched_fields": 5648,
    "mismatch_counts": {
      "accepted_shift_m": 5648
    },
    "max_absolute_difference": {
      "frame": 0.0,
      "scale": 0.0,
      "accepted_shift_m": 6.938893903907228e-17,
      "bone_error_m": 0.0
    }
  },
  "original_follower_scalar_summary": {
    "compared_leg_frames": 8352,
    "mismatched_legs": 117,
    "mismatched_fields": 5648,
    "mismatch_counts": {
      "accepted_shift_m": 5648
    },
    "max_absolute_difference": {
      "frame": 0.0,
      "scale": 0.0,
      "accepted_shift_m": 6.938893903907228e-17
    }
  }
}

This diagnosis emits no simulator authorization.

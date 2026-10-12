# Independent V63 document witness

Status: FAIL

Executed the documented --check once. Exit 0; SUPPORT_REFERENCE_V63_DOCUMENT_CHECK_PASS observed. No prepare, --run, retry, source repair, API, training or Overleaf write.

Observed launcher PID 2881291, collector PID 2881314, child PID 2881344; launcher/collector began 2026-10-12 02:43:39 UTC. Live process observation precedes complete receipt and execution-time hash binding.

The JSON contains every actual check, source/reference/output binding, denominator, reset/interface/clock rule, and charged budget. Source construction cost is explicitly retained as nonzero.

This witnesses execution/data integrity only. Tracking and surface metrics concern simulated synthetic proxies, not MoCap truth; no efficacy, novelty, downstream benefit or G1 conclusion.

Failure: AssertionError('applied_reference_canonical_exact')

Retained failure detail: {"check": "applied_reference_canonical_exact", "required_max_error_m": 0.0, "actual_max_error_m": 1.866893570624082e-16, "raw_input_max_error_m": 5.960464477539063e-08, "max_error_index": [3, 63, 23, 1], "frozen_reference_value": 0.0, "canonical_reference_value": 0.0, "actual_applied_reference_value": -1.866893570624082e-16, "different_scalar_count": 1, "total_scalars": 622080, "initial_failure_json_sha256": "1c22b6291edd93bde408d18b5c41f546b1f002debb98dbd8839534b8ec30216a", "initial_failure_md_sha256": "548d57ea478815175734f15868a5476017f61816e898beac3c0fc9d484a8f787", "no_tolerance_relaxation": true, "no_rerun_or_source_fix": true}

Denominators: {"source_cases": 72, "source_clusters": 24, "offsets_per_source": 3, "source_complete_inputs": 72, "source_incomplete_inputs": 0, "source_completed": 72, "baseline_cases": 72, "baseline_sources": 24, "baseline_completed": 72, "baseline_terminated": 0, "baseline_joint_pass": 0, "all_cases_retained": true}

Budget: {"charged_baseline_ledger": [{"label": "val_baseline_s11", "stage": "measurement", "seconds": 50.4387053065002, "exit_code": 0, "request_sha256": "a076366a7bdf104d60cb45d9224cf2dd53e2007df7e55bf7a9a5e46e54b905e9"}], "shared_source_native_generation_and_first_build_ledger": [{"label": "val_shared_s11_r0", "stage": "measurement", "seconds": 49.190580459311604, "exit_code": 0, "request_sha256": "78a25989a0d0887f5504086409332d1dd66c0c0c4924e73c29dc12129492b82d"}], "shared_source_generation_batch_calls": 1, "shared_source_cost_reused_not_zero": true, "own_charged_seconds": 50.4387053065002, "own_remaining_seconds": 549.5612946934998, "global_charged_seconds": 16227.973373092711, "global_cap_seconds": 259200, "tracked_bytes_plus_archive": 8817661547, "disk_cap_bytes": 16106127360, "new_generation_calls": 0, "new_simulator_batches_observed": 1, "remaining_documented_batches_not_authorized_after_FAIL": 2}

Remaining failed checks: ["applied_reference_canonical_exact", "reference_velocity_successive_exact"]

Actual launcher runtime witness hash (does not certify PASS): b88031939051092821ca9592229c27163cf316888aae71bce672093cfa8872aa

At action k reference k and observation k are recorded before action; actual k is recorded after that action completes. Next observation requests k+1. Same transition index is verified, not simultaneous pre-action reference/actual states.

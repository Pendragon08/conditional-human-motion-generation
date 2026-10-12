# Independent frozen numerical contract addendum

Status: PASS

Original FAIL witness remains unchanged: c8f320671fba8203c74e91fa46080d3318f678adaf046ceff405ba063076a5f5

Actual runtime witness: b88031939051092821ca9592229c27163cf316888aae71bce672093cfa8872aa

No baseline rerun, source/threshold repair, repair-arm simulation, generation, training, API or Overleaf activity. Saved-state CUDA arithmetic is separately charged.

All input hashes, commands, full original protocol checks and actual numerical comparisons appear in JSON. This is execution-integrity adjudication only; no efficacy or innovation conclusion.

root_mapping: {"status": "PASS", "canonical_expression": "(selected_ref-selected_ref[:,:,0:1,:])+selected_ref[:,:,0:1,:]", "absolute_tolerance_m": 1e-12, "relative_tolerance": 0, "criterion_source": "scripts/physics_guided_com_lineage_v24_20261012.py:94-96", "actual_max_error_m": 1.866893570624082e-16, "raw_input_difference_m": 5.960464477539063e-08, "new_tolerance_added": false, "explanation": "The pre-existing verifier models float32 root-centering before coordinate conversion. One scalar residual after coordinate mapping is 1.866893570624082e-16 m; the original exact-zero audit criterion was not the frozen contract."}

velocity: {"status": "PASS", "expression": "(current-previous)/task.s_dt", "task_s_dt": 0.03333333333333333, "task_s_dt_assignment": "third_party/CLoSD/closd/env/tasks/closd.py:64", "rule_source": "scripts/reference_velocity_collect_v43_20261012.py:15-35", "torch_version": "1.13.1+cu117", "device_name": "NVIDIA RTX A6000", "dtype": "torch.float32", "tested_scalars": 622080, "exact_same_cuda_arithmetic": true, "max_abs_error_m_s": 0.0, "first_frame_zero": true, "previous_frame_exact": true, "trace_applied_reference_exact": true, "original_numpy_comparison_denominator": 0.03333333507180214, "original_numpy_vs_actual_max_error_m_s": 9.5367431640625e-07, "explanation": "Original NumPy audit divided by float32 serialized task.dt=0.0333333350718021, rather than the executed CUDA rule using Python scalar task.s_dt=1/30. Saved runtime velocities are tested against precisely the frozen CUDA expression; no experimental tolerance introduced."}

charged_budget: {"saved_array_cuda_seconds": 0.9037324842065573, "baseline_plus_addendum_seconds": 51.342437790706754, "own_cap_seconds": 600, "global_prior_seconds": 16227.973373092711, "global_with_addendum_seconds": 16228.877105576918, "global_cap_seconds": 259200, "separate_ledger": "numerical_adjudication_budget_ledger.json", "original_budget_ledger_unchanged": true}

Machine-readable bindings: {"status": "PASS", "original_witness_sha256": "c8f320671fba8203c74e91fa46080d3318f678adaf046ceff405ba063076a5f5", "runtime_witness_sha256": "b88031939051092821ca9592229c27163cf316888aae71bce672093cfa8872aa", "versions_sha256": "aad9d6d0707717f6e851032ab078317cf10849e17dbfd2e2670c6cfd58421fbb", "simulation_rerun": false, "source_changed": false, "experimental_thresholds_changed": false}

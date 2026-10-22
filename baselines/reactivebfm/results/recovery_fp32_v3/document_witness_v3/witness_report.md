# Fresh agent-follows-doc witness: recovery FP32 v3

**Result: PASS.** All 321 saved mechanical checks passed. No doc-versus-reality discrepancy was found in the documented eight-update acceptance checks.

I read only the run-experiment skill, its compute environment contract, recovery_provider_ledger_v3.md and recovery_fp32_v3.md before invoking the recovery. I executed the documented PowerShell SSH command verbatim exactly once, with --probe-updates 8. The saved command was mechanically compared with the PowerShell block in the raw copied documentation. SSH and supervisor both exited 0. The local SSH stream was empty; the supervisor's raw output was copied after exit.

| Rank | Status | Initial | Final | Resumed | Attempts | Skipped | Calls / validated | EMA validated | Trace rows |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| 0 | PROBE_COMPLETED | 10021 | 10029 | 8 | 8 | 0 | 10029 / 10029 | true | 8 |
| 1 | PROBE_COMPLETED | 10021 | 10029 | 8 | 8 | 0 | 10029 / 10029 | true | 8 |

Both final health files contain terminal metadata: process_status=PROBE_COMPLETED, supervisor_exit_code=0, process_running=false. Every per-attempt row has sequential counters, step_applied=true, step_validated=true, ema_validated=true and finite pre-clip, post-clip and parameter norms. Final shared norms are 1.8784774920903131, 0.9999994389585694 and 205.91820761800216 respectively. Rank elapsed times were 69.9234 s and 69.8009 s.

Supervisor stdout explicitly records loading the checkpoint model, both model and model_avg, and optimizer state. The source-pin artifact reports six recovery entrypoint hashes and a 117-file upstream Python tree. These hashes match the configuration pins captured in both rank status files; the entrypoint and guard hashes also match the status/health metadata. The same artifact reports checkpoint CPU inspection passed: 388 finite model tensors, 388 finite EMA tensors, 365 finite optimizer states and optimizer step 10021. This witness inspected these emitted pin/inspection records; it did not read or modify recovery source code.

The captured configuration reports FP32 (use_bf16=false), 0.65 GPU memory fraction and 4 CPU threads per rank. The actual checkpoint save argument points into the fresh remote document_witness_v3 output. The embedded base configuration retains its full-continuation run_output field; this does not contradict the documented --output override and did not cause an acceptance-check discrepancy.

Raw stderr contains two upstream torch.load FutureWarning messages. They did not change the exit result. No environment rebuild, package installation, code/config/document repair, output reuse, full-training launch or external notification was performed. full_training_completed=false and physics_evaluated=false remain explicit in both rank status files.

Evidence saved beside this report:

- command.ps1: exact invocation.
- invocation_result.json and local_ssh_combined_output.txt: original tool completion/poll records and raw local stream.
- raw_inputs/: byte copies of the four permitted inputs.
- raw_outputs/: byte copies of status_rank0/1.json, health_rank0/1.json, health_rank0/1.jsonl, supervisor_exit.json, source_pins_verified.json and supervisor_stdout/stderr.txt.
- artifact_copy_result.json: successful copy result.
- raw_output_inventory.json: file sizes and SHA256 values of the copied artifacts.
- check_evidence.ps1 and checks.json: reproducible saved mechanical checks and all results.

The remote witness output is /data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/repair_20261022/document_witness_v3. This PASS certifies the documented bounded probe only; the full continuation was not started by this witness.
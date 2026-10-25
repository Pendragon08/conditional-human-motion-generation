# Fresh document-following PPO witness v1

Verdict: the documented invocation and numerical/resume contract passed in one execution. No numerical/status/hash divergence. Resource enforcement has the evidence limits described below; this is not certification of full training, policy quality, physical acceptance, or 8,192-environment capacity.

## Procedure

Read the run-experiment skill, SERVER_NOTES.md and TRACKER_PPO_ENVIRONMENT.md before execution. Ran the PowerShell SSH invocation verbatim exactly once; command.ps1 retains it. Captured stdout.txt, local timestamps and local-exit-code.txt. No repairs, retries, package changes, shared source edits or process stopping were performed. Post-run SSH/SCP actions only read public text evidence, calculate hashes, and inspect process/GPU state. No private arrays, references, USDs or checkpoints were copied locally. Nothing was published or sent to another chat.

## Exit and numerical proof

- Direct PowerShell/SSH exit: 0; stdout marker TEXTOP_TRACKER_JOB_EXIT0.
- Supervisor child PID 3438393; child_exit_code=0, timed_out=false, elapsed_seconds=43.03257471136749.
- Public numerical status TEXTOP_TRACKER_PPO_RESUME_WITNESS_PASSED.
- optimizer_updates=40; completed_learning_iterations=2; num_envs=8; environment_transitions=384; policy_observation_dim=557; reference_trajectories=4.
- model_restored_exactly=true; adam_restored_exactly=true; both_normalizers_restored_exactly=true. Receipt compares exact values/dtypes/shapes and explicitly permits Adam scalar step CPU residency.
- Log shows learning iteration 0/1, checkpoint/load, and resumed iteration 1/2. The completed-count resume value is 1 while the original first-run last-index is 0, matching the documented counter convention.
- full_training_started=false, dataset_accepted=false, physical_filter_executed=false, full_filter_training_ready=false, output_writers_closed=true.
- All 16 documented numerical/hash/shutdown expectations match in doc-field-verification.json.

## Independent server-side hashes

The source and diagnostic spec were hashed as bytes without reading their implementation as witness instructions.

| Item | Bytes | SHA256 | Receipt match |
|---|---:|---|---|
| scripts/tracker_ppo_resume_witness.py | 6,979 | cfff854196ebf022377b9a1abcf0698a038e45023d3abf3c84c66232d3068e32 | true |
| configs/tracker-ppo-diagnostic.json | 724 | c978be1ace1d759f0020f946a6211c7ce94cda81e6ef0f3413399161c191681a | true |
| completed_iteration_1.pt | 90,600,722 | de1fa2448399f948e981eb87ec695c66289b4fca6513933de64a765d29c3deed | true |
| completed_iteration_2.pt | 90,600,722 | 6f7dd581611140a2a94d1f6b92ed59e2a2190d26315c55b1610d6b54d80b2e87 | true |

Checkpoint bytes remained on the server; only sizes and SHA256 values were collected. Checkpoint hashes need not equal the author's per the procedure.

## Actual resource evidence and limit

The AppLauncher and simulation log both report cuda:1. The public receipt records peak_torch_allocated_bytes=443603456 (423.05 MiB). This is a Torch allocation measurement, not a measurement of all PhysX/Vulkan memory. The Isaac startup table enumerates both Vulkan GPUs as active; it does not supply per-process live renderer allocation evidence.

The authorized/documented contract is CPU4 and a 65% Torch allocator cap. Neither the public numerical receipt nor supervisor proof persists actual live CPU thread/affinity and allocator-limit fields, and the process had already exited before collection. Therefore CPU4/cap65 enforcement could not be independently observed from these run artifacts and is not certified by this witness. No second execution was made to fill this evidence gap.

At 2026-10-25T05:11:11Z both launcher PID3438393 and numerical Python PID3438399 had no /proc entries and neither appeared among GPU compute applications. GPU1 held 7,823 MiB total with existing PID3237858 using 7,778 MiB; GPU0 held 12,223 MiB total with PID3010094 using 12,190 MiB. These are post-run shared-device observations. No other process was stopped or changed.

## Shutdown fallback, explicitly retained

native_close_returned=false; teardown_fallback_used=true; shutdown_grace_seconds=20; shutdown_completion_verified=true; output_writers_closed=true. The documented watchdog fallback was actually used after durable results. Direct exit0 is therefore not evidence that native SDK close returned normally.

native_teardown_stack.txt shows the main thread inside SimulationContext.stop, then IsaacLab's on-stop callback and render, then SimulationManager.get_physics_sim_device / is_gpu_dynamics_enabled. The ten-second stack dump and twenty-second bounded process exit are distinct. The supervisor timed_out flag is false because its outer bound was not reached. Both diagnostic processes are now absent, independently corroborating process completion.

## Exact doc-versus-reality outcome

Invocation, exit/status, numerical fields, checkpoint sizes and source/spec hashes: no divergence. Fallback teardown occurred exactly as documented and must be reported as fallback, not native close success. Remaining evidence gap: the documentation states CPU4/Torch65%, but the persisted public run proof does not expose their actual enforcement. Numerical pass stands; full resource-contract certification remains incomplete.

Evidence: command.ps1; stdout.txt; tracker_job_ppo_document_v1_exit.json; receipt.json; tracker_job_ppo_document_v1.log; native_teardown_stack.txt; public-evidence.json; public-evidence-file-list.json; hash-and-resource-verification.json; doc-field-verification.json; local timestamp/exit files. All evidence is local to this witness folder, with private artifacts retained server-side.

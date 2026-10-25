# Fresh TextOp tracker document witness v2

Executed the single documented PowerShell SSH invocation verbatim exactly once. The direct command exited **0**, printed **TEXTOP_TRACKER_SUPERVISED_EXIT0**, and completed within the 280-second bound. No repair, retry, flag change, training launch, shared environment mutation, process stop, or publication was performed.

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -i 'C:/Users/Pende/.ssh/id_ed25519' ouyanghao@10.147.17.32 'python3 /data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/supervise_tracker_kernel.py --run document_v2'
```

## Direct execution and supervisor

- Supervisor child PID: 3429693.
- Actual child exit code: 0; timed_out: false; elapsed: 30.866338981315494 seconds.
- Scope: zero-update numerical diagnostic only; full_training_started: false.
- Console numerical status: TEXTOP_TRACKER_PHYSICS_KERNEL_PASSED. The persisted supervisor exit receipt omits a numerical_status key; the separate numerical receipt and child log retain the numerical pass.

## Document comparison

Every expected numerical receipt field, both code hashes, and the shared URDF receipt hash match the document.

| Field | Document expected | Observed | Comparison |
|---|---|---|---|
| status | TEXTOP_TRACKER_PHYSICS_KERNEL_PASSED | TEXTOP_TRACKER_PHYSICS_KERNEL_PASSED | match |
| dof | 29 | 29 | match |
| num_envs | 8 | 8 | match |
| physics_steps | 32 | 32 | match |
| dt | 0.005 | 0.005 | match |
| body_count | 30 | 30 | match |
| optimizer_updates | 0 | 0 | match |
| physical_filter_executed | false | false | match |
| full_filter_training_ready | false | false | match |
| ground_present | false | false | match |
| tracking_success_evaluated | false | false | match |
| output_writers_closed | true | true | match |
| spec_canonical_hash | b7a3bf0c | b7a3bf0c | match |
| seeded_paper_MLP_forward_backward_passed | true | true | match |
| MLP_input_dim | 557 | 557 | match |
| source_sha256 | a224ffba40f9189e882121397b9f4702bc43f6973b6f1f064952d4bf69eec1f2 | a224ffba40f9189e882121397b9f4702bc43f6973b6f1f064952d4bf69eec1f2 | match |
| shutdown_helper_sha256 | f42904e6aa8517156c64505155ff1cabd62ebeaf2bda10ed3ff67c803ed317c9 | f42904e6aa8517156c64505155ff1cabd62ebeaf2bda10ed3ff67c803ed317c9 | match |
| urdf_sha256 | dbc8471004e9b0b85c27a4ebc1f0d90a727c1cf7ca9ef507b805596f2004d64b | dbc8471004e9b0b85c27a4ebc1f0d90a727c1cf7ca9ef507b805596f2004d64b | match |
| shutdown_grace_seconds | 20 | 20 | match |
| shutdown_completion_verified | true | true | match |

The direct remote SHA256 command independently verified both code files and the spec file. Canonical sorted compact JSON hashing independently returned b7a3bf0c. The spec byte hash is 57992eea2e2f6aa879493409a7bc2b4d320451e3f24090ed731f1d580a058a93, matching the numerical receipt; canonical and byte hashes cover different serializations.

GPU1 and the headless IsaacLab experience are directly recorded in the child log. The existing runtime inventory records Python3.10.15, Torch2.5.1+cu118, NumPy1.26.0, Gymnasium0.29.1, RSL-RL2.3.1 and IsaacLab distribution0.36.21, all matching the documented versions. That inventory is historical (spec e65409d5), rather than a newly generated current-run inventory. IsaacSim4.5 is visible in the current log paths; the exact IsaacLab project VERSION2.1.0 was not independently re-read. Current numerical receipts do not independently expose CPU affinity/library-thread count4, Torch cap0.65, MLP hidden widths2048/1024/512, name-mapped FK implementation, callback clearing, no camera/video writers, or shared package/asset immutability. These claims are not contradicted by observed evidence, but this bounded witness does not independently re-certify them beyond matching the documented code hashes. There is no optimizer update, physical filtering, ground or tracking-success evaluation.

## Explicit bounded shutdown outcome

- teardown_fallback_used: **true**.
- native_close_returned: **false**.
- shutdown_grace_seconds: **20**.
- output_writers_closed: **true**.
- shutdown_completion_verified: **true**.

This is the documented allowed process-exit fallback. It is **not native SDK close success**.

The native_teardown_stack.txt snapshot says Timeout (0:00:10) and shows the main thread blocked in SimulationContext.stop, through IsaacLab's _app_control_on_stop_handle_fn, render and get_physics_sim_device. It therefore does not establish that execution reached SimulationApp.close. The document's narrower suggestion that native close itself hangs is not established by this run; the observed problem is a blocked native teardown path, before the close call. No repair was attempted.

## Warnings and limits

The child log retains display/GLFW warnings and a gpu.foundation DriverShaderCacheManager::init() called without a shutdown() error. Numerical validation still passed and both actual exits were0. No Replicator import failure appears in this log.

No receipt contradicts the required pass fields. The material qualification is the teardown location described above; historical inventory and the unexposed implementation/resource claims are explicitly bounded as unverified here.

This certifies the documented zero-update backend diagnostic with the explicit fallback only. It does not establish full Tracker MDP behavior,8192-env capacity,40k filter-policy readiness, exact TextOp author-asset equivalence, final corpus acceptance, or generator training readiness.

## Retained evidence

- direct_command.txt and direct_execution.json
- tracker_kernel_document_v2_exit.json
- receipt.json
- tracker_kernel_document_v2.log
- native_teardown_stack.txt
- remote_sha256.txt and canonical_spec_hash.txt
- tracker_runtime_inventory.historical.json

Private reference arrays and generated USDs remain on the server and were not copied.


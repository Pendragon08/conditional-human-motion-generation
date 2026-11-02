# Independent initialization document witness

Status: documented diagnostic completed, with the documented native-close fallback. All 20 recorded procedure checks are true; no documentation-to-reality divergence was found. No tracking-quality or motion-acceptance pass is claimed.

## Exact single invocation

Read only the required SERVER_NOTES.md, TRACKER_INITIALIZATION_AUDIT.md and its exact referenced configs/tracker-initialization-audit.json. The run-experiment skill had already been read for this fresh document-following agent. No source content or wider project data was inspected; source bytes were hashed remotely only to verify the public receipt.

Executed the procedure's PowerShell SSH command exactly once unchanged for initialization_document_v1. The exact command is in documented-command.ps1. Local session 58432 returned actual exit 0 and TEXTOP_INITIALIZATION_SUPERVISED_EXIT0. Supervisor child PID 3482881, child_exit_code=0, timed_out=false, elapsed_seconds=57.345872946083546. The numerical receipt and log both contain TEXTOP_INITIALIZATION_AUDIT_FINISHED.

The receipt records reference_trajectories=1286. The log records 1286 environments. All 20 PD steps are present and sequentially numbered 1 through 20. optimizer_updates=0. This is reference-position PD control with deterministic frame-0 assignment and no reset perturbations, not a trained controller.

No retries, procedure/source edits, repairs, other process actions, private-data copies, publication or messages to other chats occurred.

## Required root-position check

Initial root-position discrepancy count=1286 and maximum=1.1920928955078125e-7 m, strictly below 1e-4 m. Median=0 m; p95=p99=5.960464477539063e-8 m. This satisfies the procedure's root-position check only.

## Diagnostic observations, without quality thresholds

- Initial joint clipping affects 466/1286 environments (36.24%). Maximum clipping difference=0.08726763725280762 rad (approximately 5 degrees).
- Initial tracked-body error: p99=0.023538471907377245 m; maximum=0.07813523709774017 m.
- Initial root speed maximum=8.536494255065918 m/s; initial joint-speed maximum=59.34614181518555 rad/s. No reset perturbations does not mean reference velocities are zero.
- Contact-force maximum over the 20 reported steps=14025.572265625 N, at step 1.
- Body-position-error maximum over the 20 reported steps=1.4364484548568726 m, at step 4.
- At step 20, body-position error median=0.14528917521238327 m, p95=0.6679853498935691 m, p99=0.8387845307588611 m, maximum=1.312088131904602 m.
- Maximum PD-step joint speed=48.179359436035156 rad/s.
- Sum of reported termination events=37; timeout events=0. This is an event sum across steps, not a count of unique trajectories.

All per-step percentiles, rewards and termination counts remain in receipt.json. diagnostic-summary.json records a concise summary. No diagnostic spike was suppressed and none of these observations was assigned a tracking pass threshold.

## Hash verification

Source public-receipt SHA256 independently matches remote sha256sum:
59b677d7437819dec825e7559dc41b24923f07b7625ea99b2b51eedaa7ffbca6

Audit spec public-receipt SHA256 independently matches both remote sha256sum and the exact authorized local spec:
3207d0fda974ca0c4ec82275df7eda427d1be8504428879809bcd93cc9f25b88

Public receipt private-manifest SHA256:
5601a109e150f9ef991f4dda0b1f618b743c278db3b64c509642d36f2eca6158
No private manifest, raw trajectories, index, checkpoint or USD file was copied.

## Resources and original warnings

Log confirms cuda:1 and headless mode; public receipt observes CPU affinity [0,1,2,3]. The authorized audit spec requests four Torch threads, four CPUs and Torch allocator fraction 0.65. The public receipt does not separately expose observed Torch-thread or allocator-fraction fields, so those settings are not claimed as independently measured by this witness.

Peak Torch allocated bytes=2373836288 (2.211 GiB). Supervisor whole-device samples include other processes: GPU1 starts at 7823 MiB and sampled peak is 13444 MiB of 49140 MiB; GPU0 starts at 12223 MiB and sampled peak is 12739 MiB. These are sampled total-device values and cannot be attributed exclusively to this diagnostic.

The original log retains headless display/GLFW warnings, inaccessible OmniHub and DriverShaderCacheManager::init() called without shutdown(). None was repaired or hidden.

## Shutdown and claim boundaries

output_writers_closed=true, native_close_returned=false, teardown_fallback_used=true, shutdown_grace_seconds=20 and shutdown_completion_verified=true. The retained stack blocks in SimulationContext.stop via stop-callback/render/physics-device calls. It does not prove SimulationApp.close was reached. This is the documented exit fallback after completed durable receipts and closed writers, not native-close success.

dataset_accepted=false, physical_filter_executed=false, formal_filter_training_started=false and full_generator_training_started=false. Arbitrary reference-position PD cannot replace a learned tracking controller. Motion acceptance, trained tracking success, final full-corpus readiness and paper-scale training completion are not established.

## Evidence files

- documented-command.ps1
- stdout.log
- local-execution-receipts.jsonl
- tracker_initialization_initialization_document_v1_exit.json
- receipt.json
- tracker_initialization_initialization_document_v1.log
- native_teardown_stack.txt
- source_and_spec_remote_sha256.txt
- local_initialization_spec_sha256.json
- verification.json
- diagnostic-summary.json
- evidence-readback-receipts.jsonl

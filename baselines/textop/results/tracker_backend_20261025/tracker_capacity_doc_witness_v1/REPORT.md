# Tracker capacity independent document witness

Status: PASS for the documented capacity diagnostic, with the documented native shutdown fallback. No documentation-to-reality divergence was found after the input-scope clarification. This is not evidence of tracking quality, a physically accepted dataset, final full-corpus capacity, or 40,000-iteration training readiness.

## Scope and execution

Read C:/Users/Pende/.codex/skills/run-experiment/SKILL.md, SERVER_NOTES.md, TRACKER_CAPACITY_ENVIRONMENT.md and the exact referenced configs/tracker-capacity-spec.json. Initially the delegated scope named only the ledger/procedure, whereas the procedure required the spec. The parent explicitly authorized that exact spec as an additional input before execution. This was a scope clarification; no procedure or source was repaired and no run had started.

The verbatim documented PowerShell SSH command in documented-command.ps1 was executed exactly once, unchanged, for capacity_document_v1. Local shell session 91733 returned actual exit code 0 and TEXTOP_8192_CAPACITY_SUPERVISED_EXIT0. Supervisor child PID 3476900 returned 0, timed_out=false, elapsed 109.72530910745263 seconds. Numerical status and the saved log both report TEXTOP_8192_CAPACITY_PROBE_PASSED. Command/stdout and local execution receipts are retained.

No retries, package changes, source edits, process stopping, private arrays/checkpoints/USD copies, publishing, or messages to other chats were performed. The checkpoint stayed on the server; only its hash was returned.

## Verified numerical and reference shape

- num_envs=8192; one completed learning iteration; 24 rollout steps per environment; environment_transitions=196608; optimizer_updates=20.
- policy_observation_dim=557 and critic_observation_dim=557.
- reference_source_sequences=643; reference_trajectories=1286; reference_frames=946700.
- actor and critic hidden dimensions in the log are 2048, 1024, 512.
- parameter_max_change=0.0045095644891262054, finite and positive.
- The program's numerical success marker reports completion of its finite-gradient/memory assertions. Public receipts do not expose per-gradient values, so no stronger independent per-gradient reconstruction is claimed.
- output_writers_closed=true.
- dataset_accepted, physical_filter_executed, formal_filter_training_started, full_generator_training_started, full_corpus_capacity_passed are all false.

## Resource evidence

The environment log identifies cuda:1 and headless IsaacLab. Receipt cpu_affinity=[0,1,2,3], torch_threads=4 and torch_allocator_fraction_configured=0.65 match the authorized resources. GPU name is NVIDIA RTX A6000. Before environment construction the receipt reports 42,333,437,952 bytes free of 50,907,250,688 bytes, above the 24 GiB prerequisite. Free memory after environment construction is 30,897,012,736 bytes.

Peak Torch allocated=10,292,943,872 bytes (9.586 GiB), reserved=11,414,798,336 bytes (10.631 GiB). These counters are below the configured Torch allocator fraction. They do not represent total PhysX/driver memory.

Supervisor samples include other processes. GPU1 starts at 7,823 MiB and its sampled peak is 25,406 MiB of 49,140 MiB. GPU0 starts at 12,223 MiB and its sampled peak is 12,741 MiB. These are sampled whole-device totals and do not establish exclusive attribution to this job.

## Hashes

Source SHA256, confirmed independently by remote sha256sum:
9b579a2c5ab6bb427e88f81fc9aac12fa3b0f2d3c4511b7e0070fd2c8b7a9e7a

Capacity spec SHA256, confirmed by remote sha256sum and the exact authorized local spec:
d805fbb7c472a68392f68ad705302accf33370e47307b6aedfd54802b549a5bb

Private reference manifest SHA256 in the public receipt matches the documented expected value:
5601a109e150f9ef991f4dda0b1f618b743c278db3b64c509642d36f2eca6158
The private manifest and arrays were not copied.

Checkpoint capacity_completed_iteration_1.pt is 90,601,506 bytes. Remote sha256sum independently matches the numerical receipt:
a74fd431de4e0323de5eda30e46fc1c89c14160e4c84fb06abc5dcea7a073903

## Native shutdown fallback

native_close_returned=false, teardown_fallback_used=true, shutdown_grace_seconds=20, shutdown_completion_verified=true. Writers and durable checkpoint/receipt were saved first. The retained stack is blocked in SimulationContext.stop through the app-control callback/render/physics-device chain; it does not prove native SimulationApp.close was reached. This was the documented bounded fallback, not native-close success. Supervisor preserved actual exit 0.

## Untrained-policy limitations and retained warnings

The random-policy log reports mean value-function loss 4,087,196.5409, mean reward -2.41, soft_landing reward -1,248.5947, joint_limit -42.5461, feet_slide -9.4180 and landing_force_mean 1,109,027.1250. These large loss and contact-related spikes remain part of the result. Finite optimization and memory success do not establish training stability or successful motion tracking.

The original log also retains headless display/GLFW warnings, DriverShaderCacheManager::init() called without shutdown(), unavailable OmniHub and missing rsl_rl Git metadata. None was repaired or hidden.

The 643-source library is incomplete relative to the intended final corpus. The complete future library needs another memory check; no full filter or generator training gate is enabled by this witness.

## Local evidence

- documented-command.ps1
- stdout.log
- local-execution-receipts.jsonl
- tracker_capacity_capacity_document_v1_exit.json
- receipt.json
- tracker_capacity_capacity_document_v1.log
- native_teardown_stack.txt
- checkpoint_remote_sha256.txt
- source_and_spec_remote_sha256.txt
- local_capacity_spec_sha256.json
- public-artifact-listings.txt
- evidence-readback-receipts.jsonl
- verification.json (all 26 recorded checks true; no failures)

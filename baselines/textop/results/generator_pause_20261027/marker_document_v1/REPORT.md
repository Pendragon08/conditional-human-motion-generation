# Independent marker document witness and safety review

Verdict: the signal-free approach is suitable for one parent-controlled production attempt after fresh identity, source, ownership, space and authorization preflight. No blocking defect was found in the marker-triggered save path for the currently running, already-initialized trainer. This is conditional acceptance of an attempt, not proof that production has paused or that a neural checkpoint is recoverable. The independent agent created no production marker, sent no process signals, changed no active source or flags, and used no GPU/SDK/package installation.

## Actual documented execution

Executed exactly once:

```
taskset -c 0-3 python3 /data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/validate_closed_boundary_marker.py --run marker_document_v1
```

Actual SSH/command exit **0**; owned fixture PID **4183020**, actual child exit **1**, expected because the deliberate marker makes receipt writes raise `IsADirectoryError`. All **6/6** fixture checks passed. The saved simulation is step **20**, `boundary=true`, `simulation_only=true`. No neural checkpoint was made. Receipt SHA256: `565a45cd7eaa69c843e8fcbb4975bbfa095b7918b336bb3daa62f0101a9e37e9`.

The original fixture log independently shows the first traceback at fixture line9 (normal periodic save), then the second traceback at line14 (receipt save inside the exception handler). The simulated emergency file exists, `finally_closed` contains `yes`, and original receipt bytes exactly match the before-pause copy. The marker is absent and the original fixture PID has disappeared. These raw files and directory ownership listing are retained under `fixture_proofs/` and `remote_after.json`.

## Frozen production semantics reviewed

- `run_generator8278_vae.py`: sets `at_closed_boundary=False` before training; completes all four primitives and pending validation before setting it true; normal periodic receipt save is line70. Its outer `except BaseException` invokes the existing emergency save only at that complete boundary. Its `finally` closes the runtime metrics stream. In the active initialized training stage, the earlier initialization/quality receipt writes have already completed.
- `generator_training_state.save_manager`: serializes model, optimizer, EMA map, primitive clock, manager fields and Python/NumPy/Torch/CUDA/named-generator RNG state. It requires a multiple-of-four step and aligned clock, creates a new checkpoint `.partial` exclusively, flushes and fsyncs it, renames it to the final checkpoint, and fsyncs the containing directory. The marker lives at `receipt.partial`, a different path; no signal interrupts serialization. Existing checkpoint names are not overwritten.
- `checkpoint_storage_guard.check`: validates resolved personal path, UID1013 ownership through the allowed root, persistent filesystem, writability, bounded remaining saves, reserve and log allowance. Concurrent disk consumption can still make a later save fail; this remains an explicit failure, not a guaranteed save.
- `closed_boundary_marker.request_boundary_save`: validates resolved run/evidence directories and ownership; binds a pidfd to the original PID/start time and checks full identity; atomically creates an absent empty directory without overwriting an existing receipt temporary file. A busy file is allowed to finish; a pre-existing directory is rejected. It retains the original receipt copy. Only after the original pidfd proves exit does it check inode, device, UID, directory type, no symlink and emptiness before removing that exact marker. Imports expose signal helpers, but this call path invokes no signal operation.
- `pause_generator8278_with_marker.py`: binds source SHA, PID identity, expected argv, pre-existing RUNNING receipt, empty emergency/partial slots, storage guard, own controller lock, explicit authorization and two fixture receipts. It requires one bounded emergency checkpoint and real child-reaping exit1, checks the first traceback against the normal boundary line and exact marker path, tests common-lock release, and fully decodes the gzip metrics. Its final state explicitly remains `USER_PAUSED_GENERATOR_CHECKPOINT_PRIVATE_BACKUP_PENDING`; it neither changes the original stale receipt nor claims backup/CPU audit completion.

## Failure and timeout behavior

No signal is used even on timeout. After 600 seconds without proven worker exit, the controller raises, retains the marker and records inspection-required status; the marker may still trigger a later normal save. It must not be retried or removed while the original worker is live. A natural unrelated training error is not relabeled as a controlled pause; production must verify the FIRST traceback origin and full checkpoint independently. A failed emergency save is not rescued or fabricated by this controller. Evidence-write failure after marker creation can leave a pending marker and incomplete journal, but cannot interrupt checkpoint serialization or leave a SIGSTOP-held process. Parent inspection must use actual file/process state. Timeout and evidence-write-failure branches were reviewed statically, not executed in extra fixtures.

## Required production completion evidence

Before claiming safely saved pause, parent must retain and verify: first-traceback origin at frozen line70; actual full emergency checkpoint and its exact SHA; successful CPU model/Adam/RNG/accounting reload audit; original worker/supervisor/outer exit records; complete gzip decoding and released common lock. Then copy the exact generated emergency file to a private D-drive destination and verify source/destination SHA equality, retaining the server original and old stale RUNNING receipt. The marker wrapper intentionally does not perform that backup, and this marker document does not name the exact D destination; parent must record the exact source/destination and copy proof in its final handoff. CPU audit preserves CUDA RNG structurally and does not claim an actual GPU resume.

## Preserved pre-fixture collection failure

The initial independent read-only collector exited1 on an additional local-vs-remote byte-equality assertion before any fixture invocation. Only `configs/generator8278-marker-pause-spec.json` differed: local CRLF bytes SHA `a1959a5b36e0d50da1237667d04d4b94965785a21122e53d6a58113f0b9c3a85`; remote LF bytes SHA `83f975ca3c8f3ae4b88f99c6bded605d7bd60b6efe32cee3a91bc9cd9f8d4c1d`. Normalized text was identical. All executable bytes, documentary marker hashes, and every frozen remote source hash matched. No fix was applied; this harmless serialization difference was reported to parent and both exact identities preserved. The subsequently executed documented fixture was the first and only fixture attempt.

## Bounded evidence

`SUMMARY.json` contains machine-readable checks and scope. `remote_before.json` retains reviewed source text and actual remote SHA values. `remote_after.json` retains fixture files, hashes, ownership listing and after-check source hashes. `fixture_stdout.txt`, `fixture_stderr.txt`, `command_exit.json`, `invocation.json`, `pre_fixture_failure.txt` and `local_remote_divergence.json` retain actual command/failure evidence. Production remained active with stale snapshot step7700, authorization false, and no production marker at this review's read-only snapshot. That is a historical observation, not a current status assertion after parent proceeds.

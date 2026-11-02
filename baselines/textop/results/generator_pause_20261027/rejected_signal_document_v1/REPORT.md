# Independent generator-pause document review v1

Result: **normal CPU fixture passed; production pause review blocked**.

This fresh reviewer read the run-experiment skill, compute environment contract, existing SERVER_NOTES provider ledger, and GENERATOR_PAUSE_ENVIRONMENT.md. It reviewed the exact documented control source, frozen trainer/runtime/state/storage sources, and the parent's disabled production wrapper. No production wrapper was executed. No training, SDK, GPU, package installation, checkpoint change, or production signal was performed.

## One documented execution

The following command was executed exactly once through the existing provider_tools/session.py SSH endpoint, using the documented absolute local Python and PYTHONIOENCODING=utf-8:

```
taskset -c 0-3 python3 /data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/validate_closed_boundary_interrupt.py --run boundary_document_v1
```

- SSH command exit: 0; stderr empty.
- Owned fixture exit: 41; all six receipt checks passed.
- Simulated saved boundary: true; step 128, divisible by four; ext4; simulation_only=true.
- Exactly one recorded SIGINT targeted original fixture PID 4179306. The only other signaled process was its exact findmnt child PID 4179338.
- Signal sequence recorded: child SIGSTOP, fixture SIGSTOP, fixture SIGINT, fixture SIGCONT. The child was already gone when optional child SIGCONT was considered, consistent with Python subprocess cleanup.
- Both fixture and findmnt /proc entries were absent during read-only postflight.
- The control, validator, pidfd helper, and frozen trainer source hashes matched before and after execution.
- Receipt SHA256: `32293de2f12783b2620202ea4f85a323df51f5b960f29d9219fbcb359ac82502`.
- Journal SHA256: `55d7d10ad90363704ac1eb2556c9b6bf73c971c3f3c14e97f9dadaac67dc9be0`.

Small exact remote proof files are retained under proof/. Read-only local and remote source snapshots and their hashes are retained separately. The pause spec differed only in byte formatting between local and remote copies; parsed objects were identical and both had pause_authorized=false and no CPU witness bindings.

## Verified normal-path properties

The frozen launcher run_generator8278_vae.py, SHA256 `7e4ff4b7c427a884deb9708ab77c8fb4a62304ec505fb5329dd83758f650fe45`, clears at_closed_boundary before each training batch (line 62), executes all four primitives, drains pending native validation (lines 63-64), then sets the flag true (line 65). Its regular 100-step storage guard runs afterward. checkpoint_storage_guard.py issues exactly the documented findmnt argument vector.

Holding that still-live exact direct child prevents normal subprocess completion. The controller binds pidfds and verifies PID/start ticks, parent PID, session, UID, full argv and resolved executable before holding the worker, then verifies stopped identities. The journal uses exclusive creation, preventing a repeat invocation from replaying the same journal. Signals target pidfds, never a process group. The normal CPU run demonstrated this sequence.

The frozen emergency save helper contains the model, optimizer, EMA, primitive clock, manager/evaluation fields, binding and RNG/generator states. It asserts a four-primitive boundary, writes and fsyncs a partial file, renames it and fsyncs the directory. This static review did not create or restore an actual neural checkpoint.

## Production blockers

### B1: Journal errors can leave a controller-stopped process suspended

In the reviewed control SHA256 `15bb7ecb9601ce01ec78d627520cd19d95468db3c1470b8f689972183997a318`, send() performs the signal syscall at line 79 and then a journal write/flush/fsync at line 80. The stopped_child and stopped_worker flags are set only after send() returns (lines 113 and 119). A post-SIGSTOP journal exception therefore reaches finally with the stopped flag still false; no recovery SIGCONT is attempted. Even when a stopped flag is already true, both the inner cleanup and finally cleanup use send(), whose pre-signal journal write/fsync can fail before SIGCONT. Repeated controller SIGTERM/SIGINT also remain active while finally is running and can interrupt cleanup.

This is a concrete control-flow defect, not a failure observed in the successful normal fixture. It contradicts the unconditional handled-error cleanup claim. This blocker was reported immediately, before the CPU execution. The parent acknowledged it and explicitly requested completing the one unchanged v4 normal-path fixture, then retaining the blocked review. This reviewer made no fix or second run.

### B2: An identical child can belong to an already-running emergency handler

The frozen launcher calls the same check(dest, ROOT, policy, remaining_saves=1) inside the emergency-save handler (line 89) as it uses at ordinary closed boundaries. The failed status is first assigned only in memory (line 84); receipt.json is not saved until after the emergency attempt (line 95). Therefore the parent's preflight can still read RUNNING while the worker is already inside that emergency guard.

The exact direct-child command and identity do not distinguish these two Python call sites. If the controller captures the emergency handler's findmnt and sends SIGINT there, the handler's inner except BaseException (line 93) records emergency_save_refused_or_failed, and the original error is reraised without producing the intended emergency checkpoint. The current fixture models only the normal outer-try path and cannot rule out this case.

Thus an exact held findmnt establishes a storage-call boundary, but it alone does not establish that the existing outer handler will receive this first exception and perform a save. This second blocker was reported to the parent immediately after identifying the duplicate call site. The line reference in that early message was subsequently checked against the retained snapshot: the emergency check is line 89, not line 99.

## Disabled production wrapper assessment

The reviewed wrapper SHA256 `39a3084d8854192a3af0f3f4725372cf40fc11cb442a6d0aed339f8cb7d881b7` correctly leaves execution disabled in the supplied spec. It binds frozen sources and exact worker identity, requires the running receipt, checks storage capacity, uses a control lock and exclusive output directory, and does not rewrite the original trainer failure receipt.

On a successful intended pause it requires the original FAILED_GENERATOR8278_VAE / KeyboardInterrupt receipt, an emergency file with matching size/SHA/step/update counters, actual worker exit, original process/supervisor/driver exit evidence, common-lock release, and full gzip metrics decoding. It explicitly leaves private backup and CPU checkpoint restoration audit pending. These are useful postconditions; they do not repair B1 or distinguish the call sites in B2. Production use of this reviewed combination is therefore not approved by this witness.

## Scope and retained limitations

This result validates one normal CPU signal/control execution only. It is not a neural checkpoint, safe production pause completion, filtered corpus acceptance, model-quality result, or authorization to stop the real trainer. All pre-existing failures were retained. No fault-injection commands, repeat fixture commands, fixes, production signals or launches were performed by this reviewer.

The parent may revise only its separately owned control environment and obtain a new author/fresh review. This v1 evidence must remain immutable and must not be relabeled as production approval.

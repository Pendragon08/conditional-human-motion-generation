# REJECTED — do not activate this signal-based controller

Fresh review found B1 journal-failure cleanup and B2 ambiguity with an existing emergency save. No real trainer was signalled. Retained as failed historical evidence only. The actual completed pause used GENERATOR_PAUSE_MARKER_ENVIRONMENT.md, with a real8100 fullstate/privatebackup receipt. Do not repeat either pause.

# User-authorized generator pause at an existing closed-batch save boundary

User2026-10-27 explicitly changed back to paper order: safely preserve and pause
the currently running pre-filter generator, complete library physics filtering,
then train the filtered generator. Do not launch LDM or modify activeVAE sources.
Server ledger is SERVER_NOTES.md; system Python3.8 and ownUID1013 CPU-only control,
CPU0-3, no packages/SDK/GPU calls. Current VAE retains its commonGPU lock until exit.

The immutable run_generator8278_vae.py sets at_closed_boundary=True after all4
trainingprimitives and native pendingvalidation, then calls checkpoint_storage_guard
every100 steps. This guard invokes exactly findmnt -T <checkpoint_dir> -n -o FSTYPE.
Its KeyboardInterrupt handler writes a complete emergency state only when that
boolean is true. A blind SIGINT during training could lose unsaved progress.

closed_boundary_interrupt.py observes only the exact worker's directfindmnt child,
binds process identity/startticks/UID/argv/exe to pidfds, briefly holds that child,
then holds the originalworker. After stopped identities are rechecked, one SIGINT
and SIGCONT reaches the originalworker at the verified storage-call boundary.
The child's SIGCONT/normal subprocesscleanup releases it; existing trainer handles
full-state save and exit. No sourceedit/injected Python/mathchange/group-wide signal,
and no forcedtrainer TERM/KILL. UtilitySTOP intent is flushed then fsynced afterSTOP
to avoid the short utility expiring; worker SIGINT intent is fsynced beforehand.
Exclusive newjournal prevents repeating a possibly pending signal sequence.

Authorv4 realCPUfixture exited41 as expected; controllerexit0 and6 checks passed.
Oldv1/v2 observed argv becomingempty duringkernel exit, v3 missedshort children;
allfailedfilesretained, no realtraining signals. Cached sameverifiedLinux syscall
numbers and deferred utilitySTOP fsync let v4 capture a live exactfindmnt.
This CPU fixture is not a savedneuralcheckpoint; actual trainerhandoff remainspending.

Current sourceSHA:
- scripts/closed_boundary_interrupt.py:15bb7ecb9601ce01ec78d627520cd19d95468db3c1470b8f689972183997a318
- scripts/validate_closed_boundary_interrupt.py:294f4a0ec2538ff74eedbfe4f7d36c2cb3526c48bb72ae8df2249985e48884da
- scripts/quick_pidfd.py:df180bffb3e35dd77c8bb4f8ca9b143879a4184f6f8aab98188e1139de6538f2
- activeVAEentry:7e4ff4b7c427a884deb9708ab77c8fb4a62304ec505fb5329dd83758f650fe45
- activeVAEspec:ec16b1468543e7a5347595d94ecc915502bb775fe885f97318ff53454ebe3fdc

## Independent fresh document execution

Use localworkspace C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成,
localPython C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe,
PYTHONIOENCODING=utf-8 and BASE/provider_tools/session.py SSH helper. ROOT is
/data/code/ouyanghao/oyh/textop-reproduction-20261024. Read files/hashes and review
the real closed-boundary proof and PID-reuse/cleanup safeguards. Never signal the
production worker or alter its files. Execute exactly once on the server:

```
taskset -c 0-3 python3 /data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/validate_closed_boundary_interrupt.py --run boundary_document_v1
```

This creates and signals ONLY its newCPUfixture and thatfixture's ownfindmnt.
Expected commandexit0, fixtureexit41,6checks, simulated_emergency withboundarytrue
andmultiple-of4step, actualoneSIGINT tofixture, noactualneuralfile/GPU/trainingSignal.
Read exact journal and originalsource to assess whether productionguard call is
indeed atclosed boundary. Anyblocker report immediately, do notfix/retry/launchtrain.
Save REPORT.md/SUMMARY.json andsmallCPUproof under newlocal
work/textop_reproduction_20261024/generator_pause_document_v1. All oldfailures kept.
Parent alone will implement the productionbinding/resource/backupwrapper and,
only after thisactualreview passes, perform one user-authorized realpause.

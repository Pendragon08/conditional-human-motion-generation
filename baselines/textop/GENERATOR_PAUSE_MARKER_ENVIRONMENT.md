# Signal-free controlled generator save/pause

User2026-10-27 explicitly requested safely save/pause the current prefilter VAE,
then all-library physics screening before generator training. Existing source/spec
must remain frozen and all weights/data/failed attempts retained. Warm system
Python3.8 ownUID1013 CPU0-3; no package install, GPU, SDK, modelupdate in control.
SERVER_NOTES.md remains the provider ledger.

The earlier findmnt/pidfd SIGINT approach was independently REJECTED for journal
failure cleanup and ambiguity with an already-running emergency save. It sent
zero realtrainer signals. Do not activate configs/generator8278-pause-spec.json
or run pause_generator8278_at_boundary.py. Preserve all v1..v4 and review failures.

This distinct approach sends NO signals and does not change executable code.
The active launcher writes receipt.partial only at the known completed4primitive
and validation boundary during normal training. closed_boundary_marker.py creates
an own EMPTY directory at that absent ephemeral name using atomic mkdir (never
overwrites an existing file). The next periodic write raises IsADirectoryError;
the frozen outer exception handler writes/fsyncs/renames its full emergency state
before trying to update the receipt again. That second receipt write fails due
to the marker; the existing finally closes metrics and the worker exits1.
The marker is removed only after the original pidfd proves exit, with exact inode,
device, UID, no-symlink, empty-directory checks. It is our ownnew emptytemporary,
not an asset or checkpoint. Existing original receipt remains staleRUNNING and
must NEVER be overwritten to claim success; a separatehandoff explains the stop.

Unlike SIGINT on findmnt, this does not interrupt any in-progress emergency save.
If an unrelated error happens first, retain it; production acceptance must verify
the FIRST traceback is IsADirectoryError at the normal periodic save line, plus
actual fullstate/realexit/gzip closure/lock release. Otherwise no successfulpause
claim, even if the marker is present. Natural midbatch failure did not support a
checkpoint in the original launcher either; never fabricate one.

Review scripts/closed_boundary_marker.py, validate_closed_boundary_marker.py,
pause_generator8278_with_marker.py, frozen run_generator8278_vae.py and its
save_manager/checkpoint guard. Static review must include path ownership,
nonoverwriting creation, no checkpointinterruption, exception ordering, oldreceipt
preservation, exact privatebackup requirement and handled timeout behavior.
No real production marker may be created by the independent agent.

Author marker_author_v1 CPU actualcommandexit0/fixtureexit1/6checks, step20 at
closedboundary, simulatedemergency saved, finallyclosed, oldreceiptretained,
markerremoved; zero signals/GPU/actualneuralcheckpoint.
Source SHA closed_boundary_marker.py=0b4b0a2150d0a341ba896ae89b6bba1373b99c45ba76cf3ba733729fb017f0d2;
validate_closed_boundary_marker.py=373b0a3ef84233b17bf27d12141466aeacfe13dfd94aadcf10235feff9a9e138.

## Fresh actual document command, one CPU fixture only

Localworkspace C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成;
localPython C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe;
PYTHONIOENCODING=utf-8. Use BASE/provider_tools/session.py remote/SSH; ROOT is
/data/code/ouyanghao/oyh/textop-reproduction-20261024. Read actual source hashes,
then invoke exactly once:
```
taskset -c 0-3 python3 /data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/validate_closed_boundary_marker.py --run marker_document_v1
```
Expect wrapperexit0, ownedfixtureexit1 caused by deliberateIsADirectoryError,
6checks, stepmultiple20, simulated emergencyfile and closedfinally. Check original
fixturelog and directories to verify ordering; not a real neuralcheckpoint.
Do not run production controller, modify files, retry oldtests, installpackages
or send any process signal. Report blockers immediately without improvising fixes.
Save REPORT.md/SUMMARY.json andsmallproof in newlocal
work/textop_reproduction_20261024/generator_pause_marker_document_v1.
Parent can enable ONE production marker only after actualfresh validation and
ownership/space/activePID/source/preflight. CPUfullstate reload, Dprivatebackup
and SHAverification are separately required after actualtrainer exit.

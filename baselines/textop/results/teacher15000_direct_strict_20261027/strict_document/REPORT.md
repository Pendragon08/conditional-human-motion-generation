# Independent frozen15000 strict replay witness

PASS: the documented invocation ran exactly once. Actual physics completed all 8 original-source trials naturally, with 0 failures and 0 new optimizer updates. Collector document_v3 validated closed traces and actual process/outer exit 0; no worker remained and the common SDK lock was free.

Run: `teacher15000_smoke_document_v1`. Server supervision took 117.26973501592875 seconds. This result is a smoke witness only, with 8270 originals unattempted.

## Prerequisite and source inspection

Before launch, the local author progress showed completed_execution_verified=true, common_lock_free=true, actual outer exit 0 and no workers. Read the new document, spec, all four new runtime files, launcher, collector, strict coverage implementation and the existing b7a3bf0c environment ledger. All seven bound local source hashes matched the spec before launch and remained unchanged afterwards. No applicable AGENTS.md was found along the task subtree; the provided document and existing server ledger supplied the environment contract.

The evaluator loads the hashed saved15000 checkpoint for inference with load_optimizer=False and blocks optimizer.step. It retains strict .25m pelvis/.5m end-effector thresholds, orientation termination, deterministic source assignment, full T-1 transitions, failure precedence and pre-autoreset pose capture. The supervisor checks the common lock, bound files, GPU/RAM/disk availability and owns actual exit/resource receipts.

## Exact commands and actual exits

Commands ran from `C:\Users\Pende\Documents\ChatGPT\机器人格斗动作捕捉数据生成` in PowerShell. Launch was invoked once; subsequent calls collected evidence only. The prescribed 45-second initial wait and 30-second follow-up waits were used.

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' 'work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop/provider_tools/launch_teacher15000_strict.py' --stage smoke --run teacher15000_smoke_document_v1
```

Actual local and SSH client exit: 0; evidence `dispatch.json`.

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' 'work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop/provider_tools/collect_teacher15000_strict.py' --run teacher15000_smoke_document_v1 --evidence document_v1
```

Actual local and SSH client exit: 0; evidence `document_v1_client.json`.

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' 'work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop/provider_tools/collect_teacher15000_strict.py' --run teacher15000_smoke_document_v1 --evidence document_v2
```

Actual local and SSH client exit: 0; evidence `document_v2_client.json`.

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' 'work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop/provider_tools/collect_teacher15000_strict.py' --run teacher15000_smoke_document_v1 --evidence document_v3
```

Actual local and SSH client exit: 0; evidence `document_v3_client.json`.

The first collection was pending during setup. The second had completed numerical trials but no outer exit and a held lock. The third provided complete closure. No launch was retried, no old task was modified, no external signal or notification was sent.

## Actual checks

- 8 original sources; 8 natural completions, 0 failures; completion branch exercised.
- 63 vectorized control calls and 252 physical steps, exactly 4 steps per call.
- 8 closed private traces, 458 total poses, 65,952 bytes of finite 36D float32 qpos.
- Root quaternions normalized within the collector bounds; samples equal control_steps+1; byte counts and binary SHA256 match.
- Trial journal and final strict trials identical; metadata and trace index identities match.
- Child reaped with exit 0, outer exit 0, timeout false, supervision passed, no resource sampling errors; 10 resource samples ending after_exit.
- Common lock released and workers empty. All output writers closed before process completion.
- Saved15000 checkpoint and execution settings preserved. Author/document strict log and trial journal hashes are identical.

Native close did not return. The existing 20-second completed-job fallback was used: teardown_fallback_used=true and shutdown_completion_verified=true. This is actual exit success, not a claim of native SDK close success.

## Evidence SHA256

| Evidence | SHA256 |
|---|---|
| Environment document | `8b98898f7660ef34798acd8dd85b0974451a350dc08238e347d3e5a66e421dcb` |
| Strict spec | `6c3c12ca245bdae70dc86d099c28de9b925931d7c4543779679a87bc87617e0d` |
| Execution settings | `4e91e529ebdd92b0098670fa083b22bf97f105eb40d4c9e22d127d1deb5a942b` |
| Frozen checkpoint | `66e87c9471d2860f1b43a18a41e938c12647c93212f1fe44c8be1f1b27febc88` |
| Remote receipt | `b5daf420a82f1989404fae75090aaccfca2ff711f5889505db99bda72c363193` |
| Remote driver | `87f77890cdec1230f29bc9362ec7b2b2d87233d5f06c585de8e55b5204f02a06` |
| Remote process | `d415c4465b07d3a37a3ce4cf55ab4dd8fe7c848d21f33a0ae710557bcb62edce` |
| Remote exit | `89ab328ebb1b739995dffcec2aab8b67b041610c78f544673be2ee2f47723aa1` |
| Remote outer | `4ce907daf27d679864b769f47613c84e61f6e35e48a697680126aa3a388e2dc0` |
| Remote resources | `6a830407f2b366f118bc6c5adc554c58b15a1bccbd2a5d3398627ad332f3a72f` |
| private_trial_journal_sha256 | `91724c7e7a211d88b2eb9306600e46cb1c79fadf99047f9d56da020e75825690` |
| private_strict_log_sha256 | `abf4e01a934ccc47d8bb0a113d48da1c0c52e1921e13e94df47f4a4d2f754e2d` |
| private_executed_trace_index_sha256 | `ee748c4e135f8e6bf54fc5ba667b06e51447433566a0c5cd6d6058f1429c15a0` |
| Local dispatch.json | `24eaa82aa311c0782234908d6cb49d0800782e7c1f426f6ea05c76ff0b5c6de2` |
| Local document_v1.json | `c0e179ea4e1ee121ed77288befec37f88009bc7c237d9455d1ac716a0a71f3dc` |
| Local document_v2.json | `541f8fc958bd5c5a6a9dc33231d92543e4ab50d439cd769bed815c3f95c57dbf` |
| Local document_v3.json | `a308945fa5d342a017828d1a9cbdf74d132bec3dcc44db7a0d9f6e1cc5fcd5fa` |
| Local progress.json | `a308945fa5d342a017828d1a9cbdf74d132bec3dcc44db7a0d9f6e1cc5fcd5fa` |

All local evidence files and all seven bound source hashes are listed in SUMMARY.json. Evidence remains under `C:\Users\Pende\Documents\ChatGPT\机器人格斗动作捕捉数据生成\work\phygile_reproduction_20261017\conditional-human-motion-generation\baselines\textop\results\teacher15000_direct_strict_20261027\teacher15000_smoke_document_v1`.

## Divergence and limits

No material invocation or executable-behavior divergence was found. Minor documentation count ambiguity: the instruction says "three new Python sources" while the inventory lists four; all four were read. The pending first/second collections were handled by the documented unique-evidence sequence. No runtime/spec edits or command improvisation were needed.

- Only 8 selected shortest original motions were replayed; 8270 originals remain unattempted. This is not full-library filtering or corpus acceptance.
- Historical EMA<=5% candidates remain 0; this user-directed strict route bypasses that selection and does not establish the paper two-stage filtering claim.
- Current-target errors do not establish independent final-reference endpoint, semantic quality, or downstream generator readiness.
- All 8 trials naturally completed; physical failure execution was not exercised by this smoke. Failure precedence was inspected in StrictCoverage; synthetic fixtures are not physical evidence.
- Native SDK close did not return; the existing 20-second bounded completed-job fallback was used, with actual exit 0 and shutdown_completion_verified=true.

Full scope was not activated or launched. Independent corpus/text/split/endpoint/semantic/capacity acceptance remains the parent task responsibility after full execution.

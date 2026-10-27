# CURRENT v4 — direct stream files and guarded metadata mirror

V3 fresh passed its9 Windows checks,4 transport cases and pending10400 read-only
preflight, with22 production/8 server/4 local source bindings unchanged. Its
remaining findings were a D metadata mirror write bypassing guard_local, and
Windows reader-thread buffers yielding empty TimeoutExpired streams. Both are
corrected only in the new local provider; the v2 server code and production
sources remain unchanged. Every v1-v3 report and exact source snapshot remains.

The D mirror now goes through the same current-user ownership/no-link guard on
every persist. Every SSH/SCP/GitHub client's stdout and stderr are redirected
**directly to exclusive retained files**, not PIPE reader threads. Timeout
snapshots read those files; failure to read a snapshot is recorded without
interrupting cleanup. The files remain even if a second wait times out. Such a
client is explicitly unreaped, streams_final=false/output_may_grow=true, with
remote outcome unknown. No client success is substituted for the remote CPU
proof. Only emitted bytes can be preserved; unflushed bytes in a killed child
are not claimed. Updated transport_author_v4 actual four cases passed, including
nonempty first-timeout stdout from its retained direct file.

## Fresh v4 document invocation — exactly two commands

Do not rerun the old47 metadata,10 server filesystem,9 Windows path, or old
supervisor suites. The current transport changed and needs its own new run:

```powershell
$env:PYTHONIOENCODING='utf-8'
C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop/provider_tools/validate_filter_recovery_transport.py --run transport_document_v4
```

Then run the real current provider read-only under the new bindings:

```powershell
$env:PYTHONIOENCODING='utf-8'
C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop/provider_tools/backup_filter_recovery_checkpoint.py --completed 15000 --preflight-only
```

Expect4 synthetic transport cases and pending readiness/no model loads. Check
real raw stdout/stderr files versus fixture receipts, initial timeout text,
explicit nonfinal unreaped branch and guarded D mirror statically. Review prior
findings for regressions, compare new source/spec bindings and unchanged22
production files, confirm the same existing SDK teacher and no audit worker.
No real backup/model load, SDK/GPU job, production edit or signal. If15k now
exists, only report it. Do not fix discrepancies. Reports/raw command exits:
work/textop_reproduction_20261024/filter_recovery_checkpoint_document_v4.

backup_enabled is still false until this final preparation review; actual new
checkpoint audit/backup/statistics remain pending and are NOT inferred from
these fixtures. Current disabled specSHA
001869e422e5d8a71bbfdf093899de91503506eea356d0fc8eca0668173526f3.

---

# Historical v3 — local ownership and complete client evidence

Fresh v2 completed server10 filesystem checks, local4 transport cases and real
pending preflight. It retained two residual blockers: local D owner/partial-file
checks, and remaining unrecorded client failure paths. The server ancestry and
CPU child supervision fixes were accepted. All v1/v2 reviews/bytes remain.

Current local provider validates every existing D destination ancestor against
the **current Windows user's SID**, rejects all reparse points/aliases, and checks
final and .partial paths before and after transfer. The D backup mutex path is
also checked. Native Windows PowerShell5.1 uses only its own built-in module
path in that child process. No package, shared dependency or user-wide setting
was changed. The first windows_author_v3 fixture failed before its first check
because inherited PSModulePath could not load Get-Acl; its directory/source and
failure record remain. Corrected windows_author_v4 passed9 checks, including a
real NTFS junction rejection and real current-user ownership; foreign-owner and
reparse metadata rejection cases are explicitly simulated.

All readiness/audit/final-hash SSH calls, SCP transfers and GitHub clients now
write immutable intents and full client exit/streams. Timeouts preserve both
available stream snapshots and explicit unreaped/unknown states. Client exit0
is never substituted for the server CPU supervisor's actual proof. A failed
or uncertain stage must be inspected before continuation; do not restart a CPU
child when its server intent or partial audit already exists. Four updated
synthetic transport_author_v3 cases passed; no SSH/model/GPU was used by those
fixtures. The v2 server code and original22 production files are unchanged.

## Historical v3 invocation — completed, DO NOT repeat

Run the new native-Windows path check:

```powershell
$env:PYTHONIOENCODING='utf-8'
C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop/provider_tools/validate_filter_recovery_windows_paths.py --run windows_document_v3
```

Run the updated synthetic client transport check:

```powershell
$env:PYTHONIOENCODING='utf-8'
C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop/provider_tools/validate_filter_recovery_transport.py --run transport_document_v3
```

Run the current actual provider in preflight-only mode:

```powershell
$env:PYTHONIOENCODING='utf-8'
C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop/provider_tools/backup_filter_recovery_checkpoint.py --completed 15000 --preflight-only
```

Expect9 Windows checks,4 client cases and pending readiness (zero checkpoint
loads/exports/SDK). Preflight now preserves tiny client observation metadata;
it still creates no checkpoint/audit output. If15k unexpectedly exists, report
it without running a positive audit. Independently review both residual v2
findings, all bound local/server sources/spec and unchanged22 training bindings,
no new audit worker and the one existing Python SDK teacher. No source edits,
old47/old10 fixtures, actual checkpoint loads, SDK jobs or trainer signals.
Keep REPORT.md/SUMMARY.json and raw exits under
work/textop_reproduction_20261024/filter_recovery_checkpoint_document_v3.

The spec still has backup_enabled=false pending this fresh review. A later flag
activation only enables the future real15k workflow below; it does not claim a
checkpoint backup/strict evaluation/corpus or convergence pass.

---

# Historical v2 — destination and failure-evidence corrections

Fresh v1 ran its two commands once (exit0,47 metadata checks, pending10100,
no model load); all22 active and6 backup bindings matched. It BLOCKED future
positive backup on two static findings: output ancestry was not verified and
failure paths could omit the CPU/SSH journal. Its report and all v1 bytes are
retained in results/filter_recovery_checkpoint_tools_20261027.

New v2 adds an owned/non-link/persistent/same-volume output guard before and
after the unchanged serializer. Actual server filesystem author fixtures10pass.
The future CPU child now uses the unchanged, previously tested
contact_process_supervision.py and resource sampler (active bindings already
pin both), plus a durable pre-dispatch intent. Only that new child's process
group may receive timeout termination. It cannot signal any existing teacher.
The local provider retains full SSH client outcomes without session.remote's
truncation. Four real local synthetic transport fixtures (normal,exit7,timeout,
spawn error) pass, with all intents/streams/exits preserved. They use no SSH,
models or GPU. Local backup destinations also reject links/aliases.

No positive real15k checkpoint has been loaded, exported, or backed up.
backup_enabled remains false pending this NEW fresh review. No active training
file or existing core supervisor/serializer has changed. Do not repeat the47
metadata suite, v1 review commands, old7 supervisor tests, or any GPU diagnostic.

## Historical v2 document invocation — completed, DO NOT repeat

From the Windows workspace, preserve stdout/stderr/actual exit. First the new
server filesystem fixtures, which retain small owned fixtures in their own folder:

```powershell
$env:PYTHONIOENCODING='utf-8'
ssh -o BatchMode=yes -o ConnectTimeout=15 -i C:/Users/Pende/.ssh/id_ed25519 ouyanghao@10.147.17.32 "CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 taskset -c 0-3 python3 /data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/validate_filter_recovery_destination.py --run destination_document_v2"
```

Then the local transport fixtures (zero SSH; their processes are newly created
synthetic Python children only):

```powershell
$env:PYTHONIOENCODING='utf-8'
C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop/provider_tools/validate_filter_recovery_transport.py --run transport_document_v2
```

Finally the same real read-only future-checkpoint entry, under the corrected spec:

```powershell
$env:PYTHONIOENCODING='utf-8'
C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop/provider_tools/backup_filter_recovery_checkpoint.py --completed 15000 --preflight-only
```

Expect10 filesystem checks,4 transport cases, and pending readiness with no
real checkpoint load. Independently verify the two v1 findings are addressed,
local/server new-source/spec equality, unchanged22 active files, singleton
existing Python SDK teacher, no new audit output/worker/signals. Do not run
future positive backup; if15k already exists, report it. Do not fix discrepancies.
Write REPORT.md/SUMMARY.json under
work/textop_reproduction_20261024/filter_recovery_checkpoint_document_v2.
No stage is a physical-filter or convergence pass.

---

# Recovery-run checkpoint and relaxed-statistics preparation — 2026-10-27

This is an independent CPU-only tool for future **15000..40000** snapshots of
`filter_policy_paper69_resume10000_filter_recovery_20261027_v1`. The live teacher
continues, and the pre-filter VAE stays paused. This tool never signals, pauses,
resumes or launches a trainer/SDK. No new real checkpoint is available yet.

## Scope and environment

- Server root: `/data/code/ouyanghao/oyh/textop-reproduction-20261024` (UID1013).
- Use system `python3` for metadata fixtures/preflight (no Torch/model load).
  Later positive audits reuse `ROOT/env/bin/python`, the existing Python3.8 /
  Torch1.13.1 CPU trusted-file serializer. No packages are installed or changed.
- CPU affinity0–3; CUDA hidden; Torch/BLAS threads1 during the future audit.
  Original8192 GPU physics training and its22 frozen files remain unchanged.
- `configs/filter-recovery-backup-spec.json` binds the active resume spec,
  original receipt/code/reference/contact fingerprints and new audit code.
  Read the current `SERVER_NOTES.md`/`CONTINUATION.md` top, not older entries.
- Current local/server synthetic metadata and saved-counter fixtures each pass47
  checks. These use four invented source records, not real motion experiments.
  An actual read-only preflight returns `PENDING_NO_NEW_CHECKPOINT`, with no
  checkpoint loads/exports/SDK dispatches. Do not repeat author_v1.
- The backup flag starts false. Enable only after the fresh documentation review;
  this does not waive the future stable-checkpoint/CPU audit/private backup gates.

## Historical v1 document invocation — already completed, DO NOT repeat

On Windows from the task workspace, use the existing Python3.12 runtime. The
following two commands are the documented independent pass. Neither may dispatch
SDK/GPU work or load an old checkpoint. Keep stdout/stderr/real return codes.
First execute the **new** server fixture path once through SSH:

```powershell
$env:PYTHONIOENCODING='utf-8'
ssh -o BatchMode=yes -o ConnectTimeout=15 -i C:/Users/Pende/.ssh/id_ed25519 ouyanghao@10.147.17.32 "CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 taskset -c 0-3 python3 /data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/validate_filter_recovery_backup.py --output /data/code/ouyanghao/oyh/textop-reproduction-20261024/filter_recovery_checkpoint_tools_20261027/document_v1.json"
```

Then run the real local provider's read-only entry. Its Windows lock is the same
backup mutex as previous teacher backups, but it creates no checkpoint output.

```powershell
$env:PYTHONIOENCODING='utf-8'
C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop/provider_tools/backup_filter_recovery_checkpoint.py --completed 15000 --preflight-only
```

Expect fixture exit0/47 checks and pending readiness exit0/checkpoint_loads0.
Confirm server/local source/spec SHA agreement, all22 active bindings unchanged,
one existing production SDK only, no round15000 output/audit/backup worker,
and no training signals or original-checkpoint re-audit. Inspect new source for
provenance/privacy/failure preservation; report any divergence without fixing it.
Do not call any non-preflight backup command in this document pass. If the real
checkpoint unexpectedly exists, report that state rather than starting its audit.
Write `REPORT.md` and `SUMMARY.json` beneath
`work/textop_reproduction_20261024/filter_recovery_checkpoint_document_v1`;
only these aggregate reports may be copied to the public repository.

## Future actual checkpoint workflow (NOT the document fixture)

After confirming no existing backup worker and a new real latest snapshot:

```powershell
$env:PYTHONIOENCODING='utf-8'
C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop/provider_tools/backup_filter_recovery_checkpoint.py --completed 15000
```

The gate requires exact new run/provenance/solid-floor profile, actual completed
and optimizer counts, latest checkpoint name/size/SHA, owned non-link path and
stability greater than30s. Initial zero receipt counters mean pending first
100-round flush. A newer latest checkpoint causes rejection; do not weaken the
rule to recover a missed older version. For40000, actual closed writer and
process/supervisor/outer successful exits are additionally required.

The unchanged serializer checks trusted CPU state fields, finite tensors,
all Adam steps, 557D→29D shapes and exact network/normalizer inference reload.
It is not an actual SDK resume or optimizer-step convergence test. The new
adapter separately reads all8278 saved per-motion counters using SHA-bound FK
manifests and the original ordered fingerprint. The5% EMA gate is unchanged;
unattempted/invalid/above-threshold records remain in the private decisions.
Historical EMA is not absolute physical infeasibility. Candidates still need
complete strict physical replay; corpus/semantic acceptance stay false.

All full checkpoints and per-source decisions remain server+D private. Only a
separate network/normalizer export without Adam or per-motion arrays goes to the
new GitHub prerelease `textop-filter-recovery29d-20261027`, with asset digest
verification. No release is created during preparation. The D transfer keeps
2GiB free; the server serializer retains its3GiB free-space gate, and the active
trainer retains its own save-budget checks. No files are deleted. If space,
network, latest-receipt, parsing, CPU child, or digest checks fail, preserve the
exact incomplete stage and files; inspect before any continuation. The reused CPU supervisor signals only its newly spawned owned child group on
timeout and records its actual exit; it never discovers or signals the trainer.
An intent journal precedes dispatch. Full SSH client exit/stdout/stderr/timeout
are preserved locally; SSH failure leaves remote outcome unknown, never success. Original round10000/5000 backups and private assets stay put.

Public preparation evidence:
`results/filter_recovery_checkpoint_tools_20261027`.
Future actual backup evidence:
`results/filter_recovery_checkpoints_20261027/round_15000` (not yet created).
Future private decisions:
`D:/具身智能/TextOp复现_20261024/datasets/filter_recovery_statistics_private_20261027/round_15000`.

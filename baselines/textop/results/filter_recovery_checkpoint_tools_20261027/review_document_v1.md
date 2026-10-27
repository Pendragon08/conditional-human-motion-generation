# Independent filter-recovery checkpoint document review

**Outcome: both documented metadata commands passed. Future positive checkpoint audit/backup remains blocked by F1 and F2.** No actual checkpoint backup, model load, SDK dispatch or production change was performed.

The fresh reviewer read the run-experiment skill, compute environment contract, current provider ledger entry, checkpoint environment document and referenced source/spec files. The two Fresh document invocation blocks were saved unchanged, executed exactly once each from the workspace, and independently compared byte-for-byte with the document. An outer PowerShell wrapper only propagated the native `$LASTEXITCODE`; raw stdout and stderr were saved separately. No failed command was retried or fixed.

| Documented command | Actual result |
| --- | --- |
| Server synthetic fixture | Exit 0, 47 checks, empty stderr; saved server JSON independently matches stdout |
| Local provider `--completed 15000 --preflight-only` | Exit 0, `PENDING_NO_NEW_CHECKPOINT`, observed 10100, checkpoint loads 0, exports 0, SDK dispatches 0; empty stderr |

Execution times were 2026-10-27 04:43:35–04:43:38 UTC. Independent process/path/hash verification was sampled at 2026-10-27T04:44:52.720772+00:00. Local runtime is Python 3.12.14; the remote metadata runtime is Python 3.8.10, with CUDA hidden and CPU affinity 0–3 for the independent verification.

All 22 active production bindings matched locally and on the server: 20 frozen bindings, the resume spec and recovery supervisor. All six backup-spec-bound files and the backup spec itself also matched. The backup spec byte SHA256 is `844c442c8ac07fd1a6f5afce7b16428937db4b00d717ef30382ac2e893f2547b`; canonical JSON hash is `826eabac505cbda47e1d9af94c0ff4328d9d2e1b73edb4bdd9c4afbbe8b62bed` (ID `826eabac`). Checked local source/spec bytes remained identical when this report was written.

There was one actual Python SDK teacher, running, with one separate bash launcher; the shell is not a second SDK. No remote checkpoint-audit worker or local backup worker was found. The new 15000 checkpoint, server round15000 audit directory/journal, local round15000 evidence directory and D-drive round15000 statistics directory were absent. The running receipt independently reported 10100 completed iterations and 202000 updates. No checkpoint file bytes were read by this pass.

## Findings requiring correction before positive use

### F1 — High: destination ancestry is not checked before writes

The core serializer checks the source checkpoint, then constructs `checkpoint_backups/round_N/audit_run`, checks only that the final destination does not exist, and calls `mkdir(parents=True)` (core lines 58–62). An existing linked or wrongly owned `checkpoint_backups` or `round_N` parent is not rejected. The new adapter calls the core before adding any destination checks; the provider wrapper likewise checks absence only. This can redirect the receipt, public export and private per-source decisions outside the intended private destination. The D-drive transfer paths have a corresponding parent reparse/ownership-check gap.

The current existing server `checkpoint_backups` is UID1013, resolves to itself, is not a link and has mode0775; round15000 does not exist. No redirection or disclosure was observed. The blocker concerns the future write path. Validate ancestry, ownership and containment and safely create/verify new destination directories before positive loading/export. Do not change the production teacher to address this.

References: `scripts/inspect_formal_filter_checkpoint_v2.py:58–62`; `scripts/inspect_filter_recovery_checkpoint.py:35–36`; `provider_tools/backup_filter_recovery_checkpoint.py:81–87,120–138`.

### F2 — Medium: exact failure journals are not guaranteed on all CPU execution paths

The CPU execution receipt is opened only after `Popen` and `communicate` return. A spawn failure or second timeout during the post-kill reap escapes without that receipt. A transport timeout/failure can also leave the remote outcome unknown to the caller. The SSH helper raises a truncated tail of stdout/stderr for a nonzero remote exit and discards the actual return code; local error persistence then saves a traceback rather than a complete stage record. This falls short of the document's broad failure-preservation promise for future CPU/network failures.

The ordinary nonzero-child case and the first timeout followed by successful reap do preserve stdout/stderr and actual exit, and partial audit/output files are retained. The issue is the uncovered paths; no failure injection or actual CPU checkpoint audit was performed. Create an exclusive durable start/stage record before spawn, finalize it through exception/reap paths, preserve full streams and real return codes independently of JSON parsing, and explicitly mark uncertain outcomes without relaunching.

References: `provider_tools/backup_filter_recovery_checkpoint.py:82–94,112,186–189`; `provider_tools/session.py:6–9`.

## Provenance, privacy and acceptance boundaries

The existing code gates new-run metadata, 8192 environments, 8278 references, exact completed/update counts, latest-only checkpoint identity, contact profile, stable file SHA and the required final exits. Synthetic fixtures exercise those metadata rules and four invented saved-counter records. They do not establish a positive real-checkpoint audit or filesystem failure handling.

The static export allowlist includes network/normalizer state and aggregate iteration information, while excluding Adam and per-motion arrays. The adapter keeps per-source decisions in a private-named file; strict/semantic/corpus acceptance remains false. No raw motion arrays, source identities, manifests, checkpoints or private decisions were copied into these aggregate reports. The destination checks in F1 must be completed before relying on the private storage claim.

The local provider script is intentionally absent from the server; it is a Windows entrypoint, so that absence is not a source-sync failure. Its current SHA256 is `ebe795afdd675cd4222b10340e2eab9367cf7831cbe61a784d24deefbc8d0a17`; helper SHA256 is `a340b39c5432ae4764ecde1b0d66ea1a02939b94fbdab3263dae564c8bd26d8d`. Neither is in `bound_backup_files`; this review records their current bytes, and the six required server-bound files match. The broader independent hash inventory therefore contains an expected absent local-only provider path, not a mismatch in any declared server binding.

The current provider ledger says the preflight and fresh pass are pending, while the environment document already describes the earlier author preflight. That status text was stale at review time; the execution evidence above is authoritative for this pass. No ledger/document edits were made by this reviewer.

`backup_enabled` remains false. Do not mark the future positive backup ready from this metadata success. No GPU kernel witness or new environment build is claimed. Exact private raw evidence is retained as `server_fixture.stdout`, `server_fixture.stderr`, `local_preflight.stdout`, `local_preflight.stderr`, their execution JSON receipts and independent verification artifacts. Only `REPORT.md` and `SUMMARY.json` are suitable for public copying.

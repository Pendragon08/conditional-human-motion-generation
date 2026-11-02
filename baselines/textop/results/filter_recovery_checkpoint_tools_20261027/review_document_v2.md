# Independent filter-recovery checkpoint document review v2

**Outcome: all three CURRENT v2 document commands passed once. Future positive checkpoint audit/backup remains blocked by F1-remaining and F2-remaining.** The server destination and CPU-child journal parts of v1 are corrected; the complete destination/failure-preservation promises are not yet satisfied. No production file was changed and no actual checkpoint audit, backup, SDK or GPU task was launched.

The fresh reviewer read the run-experiment skill, compute environment contract, current SERVER_NOTES.md entry, CURRENT v2 document, referenced v1 review and source/spec files. No AGENTS.md was found along the workspace ancestry or beneath work. The three current document blocks were extracted without editing, saved as scripts and run exactly once; an outer PowerShell wrapper only propagated the native LASTEXITCODE. Each raw stdout/stderr and real exit is retained, and the saved commands were compared back to the current document. No historical v1/47-check/supervisor suite, positive backup or GPU diagnostic was run.

| Current documented command | Actual outcome |
| --- | --- |
| Server destination fixtures | Exit 0; 10 real filesystem checks; empty stderr; server saved summary independently matches stdout |
| Local transport fixtures | Exit 0; 4 synthetic cases; empty stderr; zero SSH calls; every intent and result retained |
| Real provider --completed 15000 --preflight-only | Exit 0; PENDING_NO_NEW_CHECKPOINT; observed 10200; zero checkpoint loads, exports and SDK dispatches |

Commands ran from 2026-10-27T04:54:53.250307+00:00 to 2026-10-27T04:54:57.319157+00:00. Independent read-only state/hash verification sampled 2026-10-27T04:56:03.097879+00:00. The later receipt naturally advanced to 10300 completed iterations / 206000 optimizer updates, with latest_checkpoint still null. This is not a command/doc divergence.

## F1-remaining — High: local private output paths are not fully guarded

The server correction is present and effective for the tested cases: scripts/filter_recovery_destination.py:7-33 checks containment, normalized/non-link paths, root ownership, persistent filesystem, each existing descendant's directory/owner/device and writability, and output absence. The adapter calls it before and after the unchanged serializer (scripts/inspect_filter_recovery_checkpoint.py:31-39), and the future provider calls it before CPU dispatch. All ten prescribed real server fixtures passed and are retained.

The Windows D transfer branch checks resolve equality only for D, dest and target (provider_tools/backup_filter_recovery_checkpoint.py:169-193). There is no owner check for task directories before or after mkdir. More directly, the actual transfer destination is a separate .partial path; only partial.exists() is checked. A dangling link passes that absence check, and this exact destination is not checked for resolution, reparse/link state, ownership or containment before it is passed to SCP. The server fixtures do not exercise these Windows paths. Thus v1 F1 is only partially closed.

Read-only ACL inspection found the existing task-private ancestors owned by the current user with Directory attributes and no ReparsePoint; the D volume root itself is SYSTEM-owned. No round15000 recovery destination exists and no redirection/disclosure was observed. A correction should validate the task private root, descendants, final file and exact temporary transfer path, with an explicit trusted-volume-root policy, while retaining incomplete files. This reviewer made no correction and injected no extra path fixture.

## F2-remaining — Medium: some transport failures still lose exact evidence

The CPU-specific correction is sound in the reviewed path: provider_tools/backup_filter_recovery_checkpoint.py:129-144 creates and fsyncs an exclusive intent before invoking the unchanged contact_process_supervision.py. The supervisor captures child output directly to a retained log, records resource samples, and writes exit/error/cleanup proof in its finally path. Signals can target only its newly spawned owned session/group. No current teacher discovery or signal path is added. The CPU journal correction was reviewed statically; no real checkpoint child was launched.

The new recorded_remote helper also retains durable client intents and final outcomes for all four prescribed synthetic cases: normal exit0, nonzero exit7 with both streams, timeout exit1/reaped with pre-timeout output, and spawn error with null PID/exit. The remote outcome remains explicitly UNKNOWN_UNTIL_SERVER_PROOF. These results establish the exercised client branches, not every network failure.

Remaining gaps in the documented broad preservation claim:

- The initial readiness call still uses session.remote at provider line97, before any local backup record is created. Final source verification also uses it at lines222-225. session.py:6-9 truncates nonzero output to the last6000 characters of each stream and omits actual exit; a timeout has no durable per-call journal. This review's outer capture preserves this successful preflight, but the future tool itself does not preserve those failed transport stages.
- SCP lines188-190 write the transfer record only after subprocess.run returns. Spawn/timeout failures skip it, leaving an outer traceback rather than full transport streams and actual/unknown outcome.
- In recorded_remote lines56-65, if the second communicate also fails, stream fallback reads the first TimeoutExpired object instead of the later cleanup exception. Later partial stream bytes can be omitted. The prescribed single-timeout fixture did not exercise this branch.

Use durable transport stage records for each SSH/SCP call and retain the newest available partial streams with explicit incomplete/reap/remote-unknown status. Preserve uncertainty without relaunching. These are static findings; no such real failure was observed or injected during this pass.

## Provenance, live state and privacy boundaries

All22 active production bindings matched their expected bytes both locally and on the server:20 original frozen files plus resume spec and recovery supervisor. All8 backup-spec-bound files and the current backup spec also matched. The existing serializer and supervisor/resource sampler remained unchanged. Current backup-spec byte SHA256 is `af8d20a80ad8f7ba00c7c43da2e9b80c7517813bec6318b597e2d5afad198716`; canonical JSON SHA256 is `5e14b21d05d760120bbc1dc713bc1b8bf988576970fc17789e6deb1c2955725a` (ID `5e14b21d`). All checked local source/spec/document bytes still matched when this report was written. Local-only provider/helper source hashes are recorded in SUMMARY.json; these local entrypoints are not server bindings.

There is one actual Python SDK teacher with the same PID and argv as in v1, plus its existing shell/supervision chain. No new remote audit worker or local backup/fixture worker remained. The round15000 checkpoint, server audit output, CPU execution journal/intent/log/resource file, local round evidence and D private output were all absent. This pass read no checkpoint bytes. Shared GPU authorization was respected; no empty-GPU requirement, GPU probe, package installation or environment change was introduced.

The unchanged protocol still gates the exact new run/resume/launch/spec/source/reference/contact profile,8192 environments,8278 sources, exact completed/update counts, latest-only checkpoint identity, owned stable file/hash, and successful final closure/exits at40000. The server and local source checks support those static gates; no positive15k checkpoint is available to validate the load/export path.

The public export allowlist remains network and normalizers plus aggregate iteration metadata; optimizer and per-motion recording state remain excluded. The saved-counter adapter keeps the original5% EMA rule, SHA-bound ordered reference fingerprint, and separate private per-source decisions; strict/semantic/corpus acceptance remain false. No raw motion/source identity/manifest/checkpoint/private decision was copied into these aggregate reports. F1-remaining must be closed before relying on local private-output guarantees.

backup_enabled remains false. This is preparation validation only, not a real15k audit, successful backup, physical-filter pass or convergence claim. Raw command streams/exits, synthetic transport records, exact verification output, process arguments and ACL evidence are retained locally; only REPORT.md and SUMMARY.json may be copied to the public repository. No source/spec/document changes, uploads, releases, deletions or existing-process signals were made. The only termination was the documented newly created synthetic timeout child.

# Independent document-following recovery witness v2

Outcome: the exact documented invocation executed once and exited 0. The eight-update execution and numerical acceptance criteria passed on both ranks. Full training was not launched.

Inputs read: run-experiment/SKILL.md, shared-references/compute-env-contract.md, recovery_provider_ledger.md, and recovery_fp32_v2.md only. No code/configuration/doc changes, installation, hidden environment setup, or repair were performed. A read-only GPU availability query preceded the invocation.

## Acceptance evidence

| Field | Rank 0 | Rank 1 |
|---|---:|---:|
| status_rank status | PROBE_COMPLETED | PROBE_COMPLETED |
| optimizer_updates | 10009 | 10009 |
| initial_optimizer_updates (health) | 10001 | 10001 |
| resumed_updates | 8 | 8 |
| attempts_since_resume | 8 | 8 |
| skipped_attempts | 0 | 0 |
| consecutive_skips | 0 | 0 |
| step_applied | true | true |
| pre-clip FP64 gradient norm | 2.4528279596224762 | 2.4528279596224762 |
| post-clip FP64 gradient norm | 0.9999996170872744 | 0.9999996170872744 |
| FP64 parameter norm | 205.9739373038983 | 205.9739373038983 |
| rank exit code | 0 | 0 |

Both offending_parameters lists are empty. Both status receipts report full_training_completed=false and physics_evaluated=false. Supervisor stdout contains "loading both model and model_avg" twice and "loading optimizer state from checkpoint: .../opt000010000.pt". Stderr contains torch.load FutureWarning text but no traceback or fatal error.

## Documentation divergence and metadata observations

No missing dependency, undocumented setup, invocation flag mismatch, path resolution failure, or numeric acceptance mismatch was encountered.

The document says "Both ranks must show PROBE_COMPLETED". Both status_rank files do so. The final health_rank files retain status="TRAINING" from the last applied step, while recording optimizer_updates=10009 and resumed_updates=8. Thus completion is established by status_rank files; if the document intended every health file to carry terminal PROBE_COMPLETED, that stronger interpretation is not satisfied. This ambiguity is reported, not repaired.

The configuration snapshot's run_output retains the full-continuation default ending repaired_fp32_v2, while official_arguments.save_dir correctly points to recovery_doc_witness_v2/checkpoints. The invocation's output override is reflected in the actual receipt destination. No receipt or original file was edited.

The existing-destination refusal was not tested: rerunning into the witness directory is expressly prohibited. No second execution occurred.

## Evidence

- invocation.ps1: verbatim documented local PowerShell SSH command.
- invocation_result.json: timestamps, exit code, all terminal output chunks.
- invocation_output.txt: combined terminal output (empty; launcher wrote supervisor files).
- preflight.json: read-only GPU query evidence.
- status_rank0.json and status_rank1.json: unparsed SSH cat output.
- health_rank0.json and health_rank1.json: unparsed SSH cat output.
- supervisor_stdout.txt and supervisor_stderr.txt: unparsed SSH cat output.
- receipt_reads.json: exact read-only commands and result envelopes.
- acceptance_checks.json: parsed mechanical acceptance checks.


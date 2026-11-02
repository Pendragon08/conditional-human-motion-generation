# Independent checkpoint storage guard document witness

Date: 2026-10-26 (Asia/Shanghai)
Result: PASS. The documented command was executed exactly once with `--run document_v1`; the direct command exit code was 0. No retry or implementation repair was performed.

## Invocation

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -i 'C:/Users/Pende/.ssh/id_ed25519' ouyanghao@10.147.17.32 'python3 /data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/audit_checkpoint_storage_guard.py --run document_v1'
```

## Independent observations

- Read CHECKPOINT_STORAGE_GUARD.md before other materials, then read the run-experiment skill. Implementation source was not read.
- The command returned CHECKPOINT_STORAGE_GUARD_WITNESS_PASSED and 11 checks passed; the checks list also contains 11 entries.
- Read the aggregate server receipt at `/data/code/ouyanghao/oyh/textop-reproduction-20261024/receipts/checkpoint_storage_guard_document_v1/receipt.json` after the run. Its parsed contents exactly match stdout.
- Receipt flags: `cpu_only=true`, `new_checkpoints_written=0`, `training_started=false`, `output_writers_closed=true`.
- The live destination is an owned, persistent, writable ext4 directory. Observed free space was 8783667200 bytes against 3892314112 required bytes.
- Both hashes match the author's local `baselines/textop/results/checkpoint_storage_guard_20261026/author_v1/receipt.json`:
  - source_sha256: `bff54d2b4fb7f29fdc18f6cc46d86626ca14b299faba49161c1c10897897788c`
  - witness_sha256: `11dfa9dc8b36c1c1619eb183ba7b57a96fe8e6dbb1851b5d94ad8aa9c5952b94`
- No GPU query, checkpoint creation, training launch, code change, package change, data operation, worker intervention, Git commit, or publication was performed by this verifier. Only this local evidence bundle was written; the witness reports its own small receipt/test-link output as documented.

## Documentation versus execution

No functional discrepancy was observed. The documented PowerShell/SSH command ran unchanged, returned the expected status, passed all 11 checks, and produced the aggregate receipt in the documented run directory. The author and document receipts differ only in their run-specific destination and observed free-space value (author 8783675392 bytes; document 8783667200 bytes). Their source and witness hashes are identical.

The prose contains joined words/numbers such as `nine128MiB`, `directexit0`, and `validate8192`. These are readability issues, not execution differences; no documentation edits were made.

## Scope

This is a CPU standard-library storage-condition witness. It does not establish model or simulation correctness, physical acceptance, 8192-environment GPU capacity, checkpoint tensor serialization, or full-training completion. Existing restoration evidence remains separate.

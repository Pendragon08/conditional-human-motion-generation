# Formal filtering-policy checkpoints: private recovery and public inference

The first real checkpoint is completed_1000.pt from
filter_policy_paper69_solid_floor_20261026_v1. This is 1000 completed learning
iterations and20000 actual optimizer updates, not a complete40000-round policy.
Training continues unchanged; no simulator, optimizer update or resume job is
started by the backup tools.

The full90799598-byte checkpoint remains on the server and in the private local
D:/具身智能/TextOp复现_20261024/checkpoints/private/ directory. Its SHA256 is
76b3cecff93807b90b61202907cc62e5b40c6678d0da2f4258d3a5541ce91b08.
It includes model, optimizer, normalizers and per-motion adaptive-sampling state.
Per-motion state is not published. Raw motion arrays/text indexes are not uploaded.

The separate30208218-byte public export contains the same model/normalizer tensors
and aggregate iteration metadata. It omits optimizer and per-motion sampling state,
and is explicitly marked inference_only_export=true. It cannot replace the full
private checkpoint for training recovery or the later statistics-based filtering.
Its SHA256 is aaa262a1a8af05bcdba3292f17f5ffda24d3fefea14bda05b3979e3ed7878f28,
verified against server, localD and GitHub asset digest/size:

https://github.com/Pendragon08/conditional-human-motion-generation/releases/tag/textop-filter-policy29d-20261026

## Actual checks and preserved failure

CPU audit checked the actual1000/20000 counters, every Adam state's20000 steps,
80 finite tensors,557D actor/critic inputs and29D actor output. A freshly reloaded
public export matched all25 model/normalizer tensors exactly. The original full
file's hash matched before and after audit, download and publication. All16 active
trainer files remained byte-identical; nothing was deleted or stopped.

The first audit failed in the existing CPU environment's Torch1.13.1 weights-only
parser on FLOAT opcode71. Its source, failed execution and output directory are
retained. This was a backup-reader failure, not a policy-training failure.
inspect_formal_filter_checkpoint_v2.py uses the legacy loader only on the own
checkpoint already SHA-bound to the actual trainer receipt. CPU Torch1.13.1+cu117
remains CUDA-uninitialized; no package was changed. Actual training continues in
the unchanged pinned SDK Torch2.5.1 environment. This is serialization/state
integrity verification, not an actual SDK recovery or tracking-quality test.

Evidence: results/formal_filter_checkpoints_20261026/round_1000, including the
retained v1 failure, v2 actual execution, SCP statuses and final backup receipt.
audit_runtime_bytes.json preserves exact own v1/v2 source bytes across Git newline
conversion. Do not rerun round1000 or restore over active files.

## Later stable checkpoints

Next scheduled checkpoint is5000, then10000..40000. First read the live trainer
receipt and verify the new file is complete/stable. Inspect real local/server
backup-worker commands. The old checkpoint_sync_local.py watches only the paused
57D MVAE run and is not this policy's backup; do not repurpose its deletion logic.

Run only for a new latest stable checkpoint, from the repository root:

```
C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe baselines/textop/provider_tools/backup_formal_filter_checkpoint.py --completed 5000
```

The script has a distinct local backup lock and CPU audit lock, checks all frozen
runtime hashes, at least3GiB server free and1GiB local free beyond each copy, and
refuses to replace differing files or an incomplete audit directory. It retains
server checkpoints and all versions onD; GitHub gets inference-only files, never
the full private file. No automatic checkpoint deletion exists in this tool.
An interrupted transfer or audit must be inspected at its exact stage; keep all
failure evidence and use a new audit version if repairs require re-execution.

The tool presently requires latest_checkpoint in the trainer receipt to match the
requested round. If a scheduled snapshot was missed and a newer one exists,
prepare explicit provenance verification for the older file rather than weakening
that check. Do not claim a GitHub inference export is a full recovery backup.

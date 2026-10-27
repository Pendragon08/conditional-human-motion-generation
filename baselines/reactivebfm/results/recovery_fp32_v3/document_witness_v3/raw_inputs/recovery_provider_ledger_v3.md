# Recovery v3 provider ledger

### env: reactivebfm@5d436660

Existing direct-SSH Python3.10/Torch2.5.1+cu121 overlay; no dependency rebuild.
GPU: two shared RTX A6000, allocation cap65%, CPU4/rank. Read-only original
vendor source reused through code_v3/upstream. Config source pins cover the
recovery entrypoints and full upstream Python tree.

The10021-update model/EMA/AdamW pair was SHA/size verified and CPU-deserialized;
all388 model tensors,388 EMA tensors and365 optimizer states were finite, with
optimizer step10021 throughout. V3 adds post-step/EMA validation and explicit
called-versus-validated counters. Numerical unit and distributed injection tests
are recorded in v3_tests_launcher.json. Tier3 requires the exact fresh command
in recovery_fp32_v3.md; not yet independently certified.

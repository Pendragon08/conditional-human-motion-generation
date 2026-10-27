# Recovery provider ledger

### env: reactivebfm@5d436660

Provider: existing direct SSH server and dependency overlay documented in recovery_fp32_v2.md. Warm reuse: env_spec.json unchanged; no installation or rebuild. Existing torch2.5.1+cu121, Python3.10, RTX A6000x2.

Recovery code/config is a new variant, FP32 model computation, FP64 norm reductions, global batch1024, shared GPU allocation cap65%, CPU4/rank. Dataset/weights paths and original attempt preserved.

Tier2 prechecks: real10,001-update model/EMA/AdamW pair size/SHA and CPU finite checks passed; all365 AdamW step counters10001. Three numerical regression tests passed. Two-process Gloo one-rank-Inf test passed on both ranks without parameter/optimizer mutation.

Tier3: fresh agent must execute the exact eight-update command in recovery_fp32_v2.md and inspect both rank receipts. Not yet certified.

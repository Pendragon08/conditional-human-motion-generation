# ReactiveBFM bounded numerical recovery witness (2026-10-23)

Warm reuse of the existing server environment: Python 3.10.21, PyTorch
2.5.1+cu121, two RTX A6000 GPUs, provider ledger spec hash prefix `5d436660`.
No dependency or driver update. Official Python source tree SHA256 remains
`cc0d1e8bbdd605962a29437dbdda6c9d73275bb18e239b6cdf93ea1f96485e3f`.

The failed v3 job has exited autonomously. Restore its verified model, EMA and
AdamW at 10041 updates; the later unsaved 16871 state is unavailable. The new
compatibility layer computes only official DiT self/cross SDPA in FP64 and casts
outputs back to FP32. Parameters and AdamW remain FP32. No Q/K normalization,
logit clipping, loss change, learning-rate change or architecture change.
This is an independently implemented numerical experiment, not author code or a
claim that the long-run fault has been fixed.

The 20-update probe retains global batch 1024 (512 per GPU), LR 1e-4, teacher
forcing, the original dataset, 400k/1M schedule configuration, activation
recomputation, four CPU threads per rank and 65% GPU allocator caps. It saves no
new checkpoint. The script rejects an existing training process or insufficient
GPU headroom. All old data, checkpoints and logs remain intact.

## Exact invocation for a fresh reader

Execute once in PowerShell on the user's local computer:

```powershell
$env:PYTHONUTF8='1'
& 'C:\Users\Pende\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'C:\Users\Pende\Documents\ChatGPT\机器人格斗动作捕捉数据生成\work\phygile_reproduction_20261017\conditional-human-motion-generation\baselines\paper_faithful_reproduction\scripts\recovery_probe_document_witness.py' --output 'D:\具身智能\PhyGile与ReactiveBFM_论文对齐复现_20261022\recovery_document_witness_v4'
```

Expected: both ranks report `PROBE_COMPLETED`, 10041 → 10061 optimizer updates,
zero supervisor exit code. The local output contains the exact launch, status,
raw logs and result. Runtime is several minutes. Existing output paths reject a
repeat; inspect them rather than launching another copy. A successful probe
does not certify long-run convergence, physical performance or paper reproduction.

Deployment for other machines: copy the pinned official repository to an
`upstream/` child of `recovery_v4/`, map dataset/text-model/checkpoint paths in
the configurations to locally authorized assets, and regenerate configuration
source hashes after any code edit. The exact machine invocation above is the
audited path for this experiment, not a portable installer.

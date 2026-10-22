# Environment and bounded invocation

2026-10-23 update: training v3 exited on a numerical watchdog. The separate
[v4 numerical recovery environment and bounded invocation](recovery_v4/ENVIRONMENT.md)
records the new precision experiment. The asynchronous simulation witness below
remains the previously executed 10041-update checkpoint experiment; its success
does not certify the subsequently failed training run.

No shared environment is changed. Server reuses the ReactiveBFM Python3.10.21,
Torch2.5.1+cu121, MuJoCo3.3.7 overlay described in
the separately checked-out ReactiveBFM branch, at the existing local file
`C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/reactivebfm_reproduction_20261021/repository/baselines/reactivebfm/docs/recovery_provider_ledger_v3.md`.
Local rendering reuses `D:/phygile_unitree_env_20261022/Scripts/python.exe`:
Python3.12.14, Torch2.14.1+cpu, MuJoCo3.5.0, mjlab1.2.0. The existing
Unitree local environment spec canonical SHA is
`a43f960bd3bcdcd3398655bffe291138b73de5ca7f30901765172ff9821e1954`.

The user's headless and shared-GPU permissions remain in force. Do not stop
ReactiveBFM training. Inference alone uses GPU1, four CPU threads and a16%
PyTorch allocation limit. Inductor cache goes to the user's home directory.
No TensorRT installation, driver upgrade, new RL training, or deletion occurs.

The compiled runner fixes Torch2.5 CUDA-graph output lifetime incompatibilities:
CFG outputs are cloned outside the compiled forward; graph replay is disabled
because official DiT text KV tensors persist across calls. Inductor kernels,
official model weights, objective, sampling steps and CFG arithmetic remain.
First actual CFG invocation is compared with eager execution at atol=rtol=2e-4.
This tolerance is an implementation witness, not a paper metric.

## Fresh reader witness (run once, exact invocation)

From PowerShell, execute:

```powershell
$env:PYTHONUTF8='1'
& 'C:\Users\Pende\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'C:\Users\Pende\Documents\ChatGPT\机器人格斗动作捕捉数据生成\work\phygile_reproduction_20261017\conditional-human-motion-generation\baselines\paper_faithful_reproduction\scripts\document_witness.py' --output 'D:\具身智能\PhyGile与ReactiveBFM_论文对齐复现_20261022\document_witness_v1'
```

Expected:200 actual physics control calls, at least one `planner_activated`
event, finite states, unchanged sources/checkpoint, exit0. Raw stdout/stderr,
full receipt and before/after training health are retained. An existing output
directory deliberately stops a repeated launch. This certifies bounded
simulation execution only; it does not certify real-time speed, semantic quality,
paper success rate, full data coverage or completed training.

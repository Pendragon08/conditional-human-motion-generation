# Clock-consistent PhyGile pairs: bounded diagnostic witness

Reuse the unchanged local Unitree/Mjlab environment documented in
`../phygile_unitree/docs/COMPUTE_LEDGER.md` and
`../phygile_unitree/configs/environment_plan_v2.json` (canonical SHA256
`a43f960bd3bcdcd3398655bffe291138b73de5ca7f30901765172ff9821e1954`).
Python3.12.14, Torch2.14.1+cpu, MuJoCo3.5.0, mjlab1.2.0; no install,
server process, GPU allocation, optimizer, or deletion. Use four CPU threads.

This is one actual first-episode rollout of a **decoded existing 262D pair**,
not generated motion. The frozen compact inference artifact comes from the
2,500-iteration pilot policy; its network/training scale differs from the paper.
This witness must not be described as the paper's physical filtering or
curriculum labeling. Inputs and licensed derived trajectories remain private.

From the workspace PowerShell, run this exact command once. The output path
must not already exist. Save stdout, stderr, real exit code, UTC start/end and
this document's SHA256 before/after in a separate fresh witness directory.
Do not repair or retry on failure; report the failure and preserve all files.

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
& 'D:/phygile_unitree_env_20261022/Scripts/python.exe' 'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/paper_faithful_reproduction/scripts/screen_phygile_pairs_mjlab.py' --manifest 'D:/具身智能/PhyGile与ReactiveBFM_论文对齐复现_20261022/phygile_clock_gate_v1/manifest_clock_consistent.json' --phygile-code 'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/phygile' --unitree-scripts 'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/phygile_unitree/scripts' --unitree-checkout 'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_unitree_20261022/vendor_unitree_rl_mjlab' --assets 'D:/具身智能/PhyGile复现/Unitree框架适配_20261022/private_assets_continue500_v1' --checkpoint 'D:/具身智能/PhyGile复现/Unitree框架适配_20261022/private_assets_continue500_v1/parent_checkpoint.pt' --output 'D:/具身智能/PhyGile与ReactiveBFM_论文对齐复现_20261022/phygile_pair_screen_witness_v1' --limit 1
```

Expected: exit0 and `summary.json` status `DIAGNOSTIC_SCREEN_COMPLETED`,
one record with finite actual pre-reset states and a complete first-episode
termination record, unchanged input hashes, no extrapolation or time stretching,
zero training updates and zero curriculum labels. A robot tracking failure is a
valid diagnostic outcome and must be retained; it is not an invocation failure.
The last source sample can lie between control ticks: the omitted tail is
reported, bounded below 0.02s, and captions are not moved. Rollout errors are
censored at first termination and cannot be read without episode duration.
This is a CPU execution witness, not a real-time or GPU-throughput benchmark.

The full batch uses the same script and options, omitting `--limit 1` and using
a new output directory. No original pair is deleted, filtered out of the source
manifest, or assigned an author difficulty label by this script.

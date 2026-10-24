# Independent agent-follows-doc witness

- Witness date: 2026-10-22.
- Verdict: **PASS for the two documented bounded CPU witness invocations.**
- Witness agent: fresh delegated agent, separate from the author of the environment documentation.
- Shell: `C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe`, `login:false`.
- Invocation time window from generated receipts: 2026-10-22 09:46:27–09:46:47 UTC.
- Both documented invocations were executed verbatim, separately, exactly once. No retry, repair, source or dependency change, output deletion, RL launch, external notification, or server process action was performed.

## Initial inputs

Only these four inputs were read before the witness invocations:

1. C:/Users/Pende/.codex/skills/run-experiment/SKILL.md
2. C:/Users/Pende/.codex/skills/shared-references/compute-env-contract.md
3. C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/phygile_unitree/docs/COMPUTE_LEDGER.md
4. C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/phygile_unitree/docs/ENVIRONMENT.md

After execution, only the generated kernel JSON and G1 receipt JSON were read for acceptance checks. The new report directory was then created, with an explicit failure if it already existed.

## Acceptance evidence

| Check | Documented expectation | Observed | Result |
|---|---|---|---|
| Kernel invocation exit | 0 | 0 | PASS |
| Kernel sentinel | MJLAB_CPU_KERNEL_WITNESS_PASSED | Same in stdout and generated JSON | PASS |
| Kernel device | CPU | cpu | PASS |
| Seeded Torch work | Seeded backward witness | torch_seed=42, torch_loss=21.18722915649414; sentinel passed | PASS |
| MuJoCo-Warp kernel physics | 2 physical steps | physics_steps=2; z 1.0 to 0.9998822808265686; CPU kernel dispatch printed | PASS |
| G1 invocation exit | 0 | 0 | PASS |
| G1 receipt status | BOUNDED_PHYSICS_PROBE_COMPLETED | Same | PASS |
| G1 control cycles | frames_evaluated=3 | 3 | PASS |
| G1 physical steps | physics_steps=12 | 12 | PASS |
| Action contract | action_scale_offset_exact=true | true | PASS |
| No policy optimization | optimizer_steps=0 | 0 | PASS |
| Unitree Mjlab commit | 1425b15f73bd4095f0df53709d7c389c3eb9e790 | Same in receipt | PASS |
| Checkpoint provenance | bb067f364de74e2da7cf1879e6f26bb961a71eb9ff64cbfbe84dd858bfae2336 | Same in receipt | PASS |
| Bounded-probe interpretation | Cannot call this full-clip success | full_clip_success=false, paper_benchmark_reproduced=false, 3/149 frames evaluated | CONSISTENT |

The kernel receipt reports package metadata versions torch 2.14.1, torchvision 0.29.1, tensordict 0.15.2, mjlab 1.2.0, mujoco 3.5.0, mujoco-warp 3.5.0, warp-lang 1.12.0, and rsl-rl-lib 5.0.1. These match the documented base versions. This witness did not separately audit Python, NumPy, SciPy, the Torch local build suffix, source integrity, the environment spec hash, or the complete package freeze.

The G1 receipt records seed=2026, device=cpu, control_dt=0.02, physics_dt=0.005, terminated=false, timeout=false, and post_physics_forward_before_termination=true. CPU dispatch logs were printed for the actual MuJoCo-Warp modules.

## Doc-versus-reality divergences

- No mismatch was found in any documented command, path resolution, exit-code expectation, status, or numeric acceptance field.
- One non-blocking extra diagnostic appeared on G1 startup: Torch emits a FutureWarning that `torch.jit.script` is deprecated and suggests `torch.compile` or `torch.export`. The warning does not change the documented invocation or its successful result.
- Wording note: the document calls the destinations “输出目录”; the first invocation actually targets a JSON file, while the second targets a directory. Both exact invocations succeeded. No manual output replacement or deletion was attempted.
- No independent pre-execution existence check was added for the two prescribed output destinations, to preserve the exact documented invocations. Successful execution was observed; their pre-execution existence is therefore not separately attested by this witness.

## Scope of verdict

This passes the fresh-agent documentation-following gate for the two bounded local CPU probes only. It does not establish full motion success, full benchmark reproduction, PPO continuation readiness, server environment readiness, or local live GUI viewing. No such action or verification was attempted.

## Evidence files

- `command_output_trace.txt`: exact commands, shell, exit codes, complete captured stdout/stderr.
- `command_output_trace.json`: machine-readable invocation records and generated-receipt readback.
- Kernel artifact: D:/具身智能/PhyGile复现/Unitree框架适配_20261022/doc_witness_kernel_v1.json
- G1 artifact: D:/具身智能/PhyGile复现/Unitree框架适配_20261022/doc_witness_g1_v1/receipt.json


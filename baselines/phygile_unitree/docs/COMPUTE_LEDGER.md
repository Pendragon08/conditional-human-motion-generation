# Compute provider ledger

### env: unitree-local@a43f960b

spec: configs/environment_plan_v2.json
canonical_sha256: a43f960bd3bcdcd3398655bffe291138b73de5ca7f30901765172ff9821e1954
canonicalization: Parse JSON, then Python json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True); hash UTF-8 bytes without a newline. ASCII escaping of Chinese path values is explicit and required.
reproducer: python scripts/canonical_env_hash.py configs/environment_plan_v2.json (from baselines/phygile_unitree)
how: isolated Windows venv at D:/具身智能/PhyGile复现/Unitree框架适配_20261022/mjlab_env; ASCII junction D:/phygile_unitree_env_20261022 points to the same directory.
tier: {cpus: 4, device: cpu, new_rl_training: false}
weights: private_assets/parent_checkpoint.pt; original checkpoint SHA in ENVIRONMENT.md. Inputs are read only.
validated: author seeded Torch/MuJoCo-Warp kernel dispatch and actual G1 three-step dispatch passed; fresh doc witness pending at this ledger entry.
gotcha: initial PyTorch 2.7.1 request was upgraded by transitive torchvision/tensordict resolution to 2.14.1+cpu in this isolated venv. Revised spec v2 and exact freeze retain provenance. SciPy runtime import was undeclared by upstream packaging; explicitly added 1.16.2. Native MuJoCo Windows file parsing fails on Chinese package paths; use ASCII junction interpreter; no installed source patches.

### env: unitree-isaac-existing

how: warm read-only reuse of server Isaac Sim 4.5.0 / Isaac Lab 2.1.0 / Torch 2.5.1+cu118 / RSL-RL 2.3.1; custom Unitree task in new version directories.
tier: {cpus: 4, policy_physics_gpu: 1, native_gui_render_gpu: 0, torch_memory_fraction: 0.20}
validated: actual native GUI frame plus 100 frozen-policy control calls completed; exact model and normalizer load, original MDP semantic equality confirmed. This does not validate PPO continuation.
gotcha: GPU1 cannot present on Xorg :10. Render on GPU0 while policy/physics use GPU1. Cold shader startup was slow. Local viewing connection remains unverified; no new PPO launch.

### Fresh document witness 2026-10-22

Two exact invocations passed independently at 09:46:27–09:46:47 UTC. Evidence: results/v1/doc_witness_report_v1/. Validation is limited to the local bounded CPU path; no new PPO or local GUI viewing claim.

### V2 explicit headless exception and unchanged environments

The user's later instruction explicitly authorizes headless continuation for this task, recorded verbatim in configs/headless_authorization_20261022.json. It supersedes the earlier local-viewer gate for this scope; historical policy and receipts remain unchanged. Dependency specs and the local canonical hash a43f960b are unchanged: warm reuse, no shared-environment upgrade.

The actual server two-iteration continuation completed 11:58:00–11:58:49 UTC, 40 Adam calls, exact optimizer restoration, finite changed policy, parent and 446 executed sources/inputs unchanged. Independent document witness passed; source code_v9 and output headless_smoke_v3 are immutable. A separate 128-environment, 500-iteration continuation is IN_PROGRESS; completion must be verified from its final receipt.

Actual Unitree PhysX and Unitree MuJoCo-Warp closure document witnesses passed. Both preserve explicit untrained/unconditional generator labels and rejected candidates. CPU sliding-friction comparison completed eight first episodes; both profiles completed1/4 full clips. These are development diagnostics. Raw licensed motion arrays remain private.

### V3 bounded continuation and transfer completed

The separate500-iteration run completed13:28:50 UTC:10000 actual Adam calls, step40000→50000, parent and captured sources unchanged, finite full model/optimizer. Its early launcher has no durable child exit code; this remains unknown. The dependent four-clip Isaac evaluation/export completed13:31:40 UTC with supervised exit0. New policy Isaac completes1/4 and MuJoCo1/4, so the parent remains the current baseline. New-policy closure also executed in both engines; untrained generated candidates remain rejected with zero commits. Current production compact-loader document witness passes same-batch bitwise parity and preserves the earlier cross-batch oracle failure. No environment or dependency rebuild occurred.

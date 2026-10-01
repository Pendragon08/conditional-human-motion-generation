# Frozen Kick/Punch robustness and collidable-pad execution

This is inference only. No policy learning, reference changes, extra IK, or asset download. Warm-reuse `configs/mujoco_sim2sim_env.json` (canonical hash `91455383`) and `.aris/compute/ssh-mujoco.md`. The original adapter is unchanged. MuJoCo 3.2.3 physics runs on CPU; Torch inference and EGL rendering use NVIDIA GPUs.

Server: `ouyanghao@10.147.17.32`; Windows SSH source address `10.147.17.250`, key `C:/Users/Pende/.ssh/id_ed25519`. Root: `/data/code/ouyanghao/oyh/conditional-human-motion-generation`. Runtime: `/data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python`. No package installation is needed.

Run these exact commands from local PowerShell:

```powershell
ssh -b 10.147.17.250 -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 'bash /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/r3_mujoco_robust_contact.sh --mode witness --task Kick --device cuda:0'
ssh -b 10.147.17.250 -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 'bash /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/r3_mujoco_robust_contact.sh --mode witness --task Punch --device cuda:1'
```

Both must exit 0 and print `ROBUST_CONTACT_ENV_WITNESS`. Witness verifies original actor/normalizer tensors, Isaac observation/action parity, 30 moving no-pad steps identical to the original transfer, engineered positive collision fixture, collision-disabled negative fixture, and nonempty NVIDIA EGL rendering. The deliberately overlapping positive fixture is solely a detector test and is never counted as a strike success.

Protocol: `configs/mujoco_robustness_contact_protocol.json`, fixed before evaluation. Four shards (two per task), 14 conditions × (66 Kick + 100 Punch) = 2,324 single-start episodes. Each runs at most 500 control steps (10 s), stops at a fall, has no autoreset, and retains failed initializations in the denominator. Reference 0/1 videos are selected before testing, not by outcome. Workers have distinct output files.

V2 corrects a V1 configuration diagnostic: equal-priority MuJoCo contact friction takes the maximum of two geom values. Floor-only 0.5 left effective floor/foot friction at 1.0. V2 sets both floor/robot sliding coefficients, records every observed floor-contact friction, and asserts the intended value. Subtree-mass caches are restored between conditions as well. Original V1 source, raw outcomes and videos remain under server `artifacts/mujoco_robust_contact_diagnostic_v1`; they are not final robustness evidence. Policies, strike thresholds and test counts are unchanged.

Nominal no-pad physics preserves the prior adapter exactly. Runtime mass changes require `mj_setConst`, which also refreshes cached inverse weights after armature assignment. A separate `nominal_mass_constants_refresh` control applies the same refresh; compare mass ±10% against that control rather than attributing any refresh difference to mass alone. Other disturbances compare against `nominal`.

The pad is a fixed collidable mocap box with a 24 × 24 cm face, thickness 5 cm. It is not a dynamic opponent or a calibrated punching bag. Its front face is at the original virtual goal; lateral ±5 cm tests keep the frozen policy's observation/reference unchanged. Normal contact forces come from `mj_contactForce`, impulses sum force × physics dt. Contact timing uses the solver step time (within one 5 ms integration step). The approach-speed filter uses the pre-step nominal striking point as a proxy, not exact surface impact velocity.

Report any contact, correct-effector contact, attack-window contact, force thresholds 5/10/20 N, front-side/approach-qualified strike and strike plus full-horizon standing separately. Original Kick foot collision geometry consists of small sole spheres; hand geometry is also the upstream model, not a newly fitted high-fidelity collider. Force peaks depend on solver/model settings and cannot be interpreted as real-world strike force. No hardware or universal robustness claims follow from this bounded test.

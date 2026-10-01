# Frozen Kick/Punch transfer: documented environment witness

This is inference and CPU physical simulation, not RL training. Read
`configs/mujoco_sim2sim_env.json` before running. Reuse the specified existing
server Python without installing/upgrading packages. All new output stays in
the conditional-human-motion-generation repository. The old runtime is only a
Python executable and a read-only source of the original Kick policy.

From local Windows PowerShell, the following commands are the exact witness
invocations (Kick and Punch separately):

```powershell
ssh -b 10.147.17.250 -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 "bash /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/r3_sim2sim_mujoco.sh witness Kick"
ssh -b 10.147.17.250 -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 "bash /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/r3_sim2sim_mujoco.sh witness Punch"
```

Both must exit 0 and print `INTERFACE_PARITY_PASS` and
`SIM2SIM_ENV_WITNESS`. The first compares against four real Isaac Lab snapshots:
626 policy inputs and 29 outputs, max error below 0.0002. The second executes
seeded CUDA policy inference, advances floating-base MuJoCo dynamics, and
renders a nonempty frame using an NVIDIA EGL renderer. JSON witnesses are
written under `results/sim2sim_mujoco/`. Import success alone is insufficient.

Evaluation invocation on the server, once the witness is validated:

```bash
cd /data/code/ouyanghao/oyh/conditional-human-motion-generation
bash scripts/r3_sim2sim_mujoco.sh eval Kick
bash scripts/r3_sim2sim_mujoco.sh eval Punch
```

Each uses the existing frozen whole-body policy with its exported observation
normalizer. No fitting, retraining, extra IK, or changing the references occurs.
One episode starts at time zero for each original reference, up to 500 control
steps at 50 Hz. Falling ends that trial and remains in the denominator. This
differs from the official Isaac 1000-environment auto-reset evaluation, so do
not present the two rates as identical protocols. The target marker has no
collision; goal reach is not a physical hit against an object.

The official MuJoCo floating-base 27-body-joint plus 14-hand-joint model is
used. Recorded runtime overrides match training gravity, friction, gains,
armature and effort limits. The explicit torque PD follows official Sim2Real
deployment, but differs from implicit Isaac drives. Model inertia/collider
parity is still an outstanding limitation. Videos show new MuJoCo physics,
not reference playback. Results do not establish successful transfer merely
because the environment witness passed.

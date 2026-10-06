# New feedback/world-model module pilot — exact invocation

This uses the unchanged environment `configs/mujoco_sim2sim_env.json`, hash
`91455383`, as recorded in `.aris/compute/ssh-mujoco.md`. No package install,
baseline rerun, PPO update or post-retarget robot IK is allowed. All new arrays
and model weights stay in ignored artifacts of the new project.

Independent witness, from local Windows PowerShell, verbatim:

```powershell
ssh -b 10.147.17.250 -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 "bash /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/r3_feedback_world_model_pilot.sh witness"
```

Must exit 0 and print `FEEDBACK_WORLD_MODEL_WITNESS_PASS`: both original frozen
Kick/Punch actors and normalizers match checkpoints, four Isaac snapshots
pass observation/action parity, seeded CPU physics actually moves, NVIDIA EGL
renders a nonempty image, and CUDA prediction forward/backward is finite.
Output: `results/feedback_world_model_20261006/environment_witness.json`.
This witness is not an ablation outcome or local live training confirmation.

After independent witness approval, each stage is run once, sequentially:

```bash
bash scripts/r3_feedback_world_model_pilot.sh probe
bash scripts/r3_feedback_world_model_pilot.sh collect
bash scripts/r3_feedback_world_model_pilot.sh fit
```

Probe is 20 short frozen-policy episodes used to measure throughput. Collection
uses frozen policies and real MuJoCo physics, 66 reference IDs per task, up to
500 steps each; falls remain recorded, not replaced with successful episodes.
Reference ranges [0,36), [36,46), [46,66) define train/validation/test. All
variants of a reference stay in one split. Test-only lateral force is 80 N for
0.2 s at 1 s. External force is not a world-model input; thus this set tests an
unmodeled disturbance, not a claim that the model predicts unseen force.

The fit uses the separately locked `configs/feedback_world_model_fit_v2_20261006.json`
review corrections before the first fit. It compares persistence, constant
velocity, ridge, single-step MLP, five-step MLP, an identical-size MLP with its
action input zeroed, and a task-bit-free MLP. Only the27 physical PD actions are
used; the last2 policy outputs are zeroed in every dynamics feature. All neural
arms use the same data, update count, sampled windows and seeds42/43/44. Loss
uses only training normalization; checkpoints are selected on validation.
Evaluation uses recorded future actions: it is replay prediction, not learned
MPC or improved robot task success. A useful prediction loss is insufficient
to pass a control-benefit or full feedback innovation gate.

After the review corrections, a fresh independent agent must run this exact
additional kernel witness before the first fit:

```powershell
ssh -b 10.147.17.250 -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 "bash /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/r3_feedback_world_model_pilot.sh fit_witness"
```

Must exit0 and print `FIT_KERNEL_WITNESS_PASS`, exercising5-step prediction and
backpropagation on actual probe transitions, finite gradients for all three
feature variants, identical parameter counts, and zero final2 action features.
No fitted model is retained by this witness. Main predictive gate isH10 body
site FK RMSE, reference/condition macroaveraged with reference-cluster bootstrap;
independent neural seeds define repetitions. Seed-averaged predictions are
secondary and do not define the predictive gate. Force phases and censored
trial durations/falls are retained. No validated uncertainty claim is made.

## Actual execution deviation (preserved, not a retrospective pre-gate)

In the first bounded pilot the queued fitter started10:39:50UTC, while the fresh
additional fit witness ran10:42:25UTC. The latter passed but is a post-start
kernel check. A shell wrapper was also edited in place while bash waited for
Python; all scientific outputs were computed and13 model hashes verified,
but the returning launcher encountered a changed read offset and exited2.
The original FAILED queue state/log are retained. Do not retrain this pilot
or label its launcher exit normal. See `execution_audit.json` and report21.
Future invocations must fix wrapper/source/config before launch and must not
edit or deploy over any active script. The new fit guard requires a matching
fit-kernel witness; it did not retroactively apply to the already-running fit.

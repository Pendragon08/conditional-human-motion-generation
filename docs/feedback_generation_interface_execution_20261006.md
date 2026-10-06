# Sparse actual-execution feedback: bounded generation interface pilot

This is inference with frozen released OmniControl, not PPO training. It does
not implement complete263D HumanML inpainting or prove control improvement.
Kick only, three predeclared reference IDs46/47/48, matching source samples
44/76/88 verified by exact original joint tensors. Original human arrays are
SHA-pinned existing local assets copied to ignored new-project artifacts.

The origin/delta sparse calibrationv2 is exploratory after v1 failure. It is
not a confirmatory inverse-retargeting result. The actual feedback uses
`human_reference + executed-minus-reference residual`, with exact root units
and calibrated relative ankle residuals. Equal executed/reference states
return the original human hint. Current/previous torso+ankle spatial hints are
not a complete actual movement prefix. Original future joint11 Kick goal is
bitwise unchanged, and no right-wrist task trigger is introduced. No extra
post-retarget robot IK is used.

Reuse the existing Python3.8/Torch1.13 environment and existing OmniControl
checkpoint. The overlay contains only pre-existing CLoSD data-loader/CLIP
dependencies. No installation or upstream source changes. Seed42 and full
official diffusion sampling are fixed for all three arms: predicted context,
actual context and cyclically mismatched actual context.

Fresh independent documentation witness must execute this exact local
PowerShell command once. The predicted arm is ALSO the first recorded pilot
arm; do not rerun it as a duplicate after witnessing.

```powershell
ssh -b 10.147.17.250 -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 "bash /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/r3_feedback_generation_interface_20261006.sh predicted"
```

Must exit0, produce three finite196-frame human motions and nonempty
`artifacts/feedback_world_model_20261006/generation_v3/predicted/results.npy`,
and print `FEEDBACK_GENERATION_ARM_COMPLETE arm=predicted`. Saved actual hints
must agree with the planned input to tolerance1e-5. A successful call means
the frozen sampler interface works, not that execution feedback is effective.

After that witness, run `actual` and `shuffled` exactly once through the same
wrapper, then analyze with the existing MuJoCo Python:

```bash
bash scripts/r3_feedback_generation_interface_20261006.sh actual
bash scripts/r3_feedback_generation_interface_20261006.sh shuffled
MUJOCO_GL=egl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python scripts/r3_feedback_generation_interface_pilot.py --stage analyze
```

The original goal and hint-fit errors, wall time, counts and hashes are saved
under `results/feedback_world_model_20261006`. These are generation-interface
metrics only. No claims about regenerated G1 dynamics, online2-second CLoSD
replanning, PPO improvement or verified full innovation are permitted.

Actual witness: the first predicted arm exited0 and checked finite motion shape.
Saved-hint comparison belongs to the subsequent analyze stage, not its wrapper.
All three arms later exited0 and analyze confirmed hints, exact future-goal
input preservation, source-hash consistency and independent residual fixtures.
Report21 preserves the finite-sample, nominal-context and latency limitations.

# Full-scene contact/state diagnostic

The actual three-state baseline contact_baseline_author_v1 exited0,24substeps,
but did NOT reproduce the original first-step explosion. Initial restoration
matched, yet first joint velocity was13.25rad/s versus3772.65 in the original.
Native merged masses and inertia eigenvalues were positive. Do not claim a fix
or run small-scene contact interventions to infer the original cause. Their
specification remains historical/unexecuted. Near-root distances alone are not
collision evidence. All first-replay output and private arrays are retained.

Next profile keeps the full8278-library/8192-scene initialization and same seeded
policy collection as the original numerical probe. It adds native PhysX versus
cached root/joint state comparison, native mass/inertia checks across clones,
and verifies the three selected first actions against the saved original.
There are24control/96physics steps and ZERO optimizer updates. A policy is
initialized only to reproduce the original sampled first action; no training
quality or filtering claim is permitted. Dependencies b7a3bf0c, GPU1/CPU0–3,
Torch0.65 and the original private input hash bindings remain fixed.

Spec configs/full-contact-spec.json was declared before execution. Baseline
changes only read-only audit and skips PPO updates after the rollout. Separate
optional no_self_collision and no_floor cases are diagnostic interventions;
they may be run serially only after reviewing actual baseline evidence. Never
adopt such a setting for training merely because it removes extreme numbers.

After resource/mutual-exclusion and unused-run checks, the one-time baseline is:

```
ssh -o BatchMode=yes -o ConnectTimeout=15 -i C:/Users/Pende/.ssh/id_ed25519 ouyanghao@10.147.17.32 "python3 /data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/supervise_full_contact.py --run full_contact_baseline_v1 --case baseline"
```

The supervisor bounds its own group to1800seconds and retains exit/resources.
Require actualexit0,96physics/24rewards,zero optimizerupdates and unchanged model
parameters to call this diagnostic complete. Evaluate original-state/action
equivalence separately. No checkpoint, dataset rejection or formal training.
Keep full_contact_private/ and numerical_private/ arrays and mappings private.
Only aggregate receipts/trace/resource evidence and report may be published.
Retain native-close versus bounded-exit distinction and any failed trial.

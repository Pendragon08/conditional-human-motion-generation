# Flat solid-floor counterfactual, not yet a training fix

Full-scene baseline reproduced the first-step anomaly with zero optimizer
updates. Disabling self-collision retained the first-step spike (3714rad/s);
removing the floor removed it (first-step max37.0002rad/s). Original8192 root,
joint states and selected first actions match exactly in all three cases.
This implicates floor contact in the trigger, but does not prove a specific
PhysX/asset/initialization root cause. Groundless operation is not a valid fix.

The declared new spec is configs/solid-floor-diagnostic-spec.json. It retains
self-collisions, flat topz0, friction1, restitution0 and multiply combination,
but uses a static4000×4000×0.2m box collider beneath the same visual plane,
instead of the generated infinite Plane collider. All available sources,
initialization, policy sampling, SDK versions and resource limits are unchanged.
This is a controlled collider-representation intervention, not paper training.

Invoke once after previous jobs exited and fresh resource/unused-path checks:

```
ssh -o BatchMode=yes -o ConnectTimeout=15 -i C:/Users/Pende/.ssh/id_ed25519 ouyanghao@10.147.17.32 "python3 /data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/supervise_solid_floor.py --run full_solid_floor_author_v1 --case solid_floor"
```

The same mutual-exclusion/1800s supervisor,96physics/24reward samples and zero
optimizerupdates apply. Positive completion means diagnostic trace collected,
not stability or dataset acceptance. Preserve native close/fallback distinction.
Keep all private snapshots local/server only. If successful, broader actual
training stability and a fresh documented witness are still required before
adopting a profile or starting40k training. Do not relax physical thresholds,
remove data, or silently overwrite the production floor/configuration.

Metadata note: original full-contact no_self/no_floor numerical_summary inherits
the old helper's hardcoded simulation_or_reward_changes=false. That field is
incorrect for those counterfactuals; case/spec and the retained comparison
correction state the true change. Original receipts are immutable. This new
probe sets the field true and records the intervention before closing writers.

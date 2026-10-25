# Three-outlier contact replay diagnostic

This investigates numerical_trace_author_v1 first-step explosions. It does not
train a policy, accept motions, or amend paper settings. Warm b7a3bf0c SDK and
the pinned G1 asset are reused unchanged. GPU1, CPU0–3, Torch0.65; no new package.
Read configs/contact-replay-spec.json before dispatch. Formal training remains
disabled. Fresh small-scene equivalence is UNVALIDATED before actual execution.

Each case uses the three exact saved WORLD root/joint states and their recorded
first actions. The first action is held for24physics substeps; policy inference,
reward/termination/reset and optimizer work are deliberately absent. This differs
from a training episode after the first substep and is explicitly a controlled
physical diagnostic. The reduction8192→3 can change contact warm starts or solver
order. Measure initial restoration and first executed state against the original;
do not assume bitwise or physical equivalence merely because inputs match.

The baseline retains contact settings. Only after the baseline actually exits,
the no_self_collision case disables self-collisions, and no_floor omits terrain
while retaining self-collisions. Each uses a separate fresh SDK process. These
counterfactuals are not fixes adopted for training. Keep all results, including
failed restoration or lack of reproduction. Contact readings are diagnostics;
never discard the corresponding motion or clip a reward to force a pass.

Exact one-time baseline invocation (first verify unused output name and resources):

```
ssh -o BatchMode=yes -o ConnectTimeout=15 -i C:/Users/Pende/.ssh/id_ed25519 ouyanghao@10.147.17.32 "python3 /data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/supervise_contact_replay.py --run contact_baseline_author_v1 --case baseline"
```

Subsequent unique cases are contact_no_self_author_v1 / no_self_collision and
contact_no_floor_author_v1 / no_floor. Run serially, never while another SDK job
holds the common lock. The supervisor bounds only its own group to600seconds,
retains logs/resource samples/exit status, and rejects existing outputs. Require
actualexit0 and24physics samples to call a trace complete, not stable. There are
0optimizerupdates,0checkpoints, no dataset acceptance. Native SDK close and
bounded fallback remain separately reported.

Only aggregate receipt and resource evidence may be published. replay_private/
initial/first-step arrays and all source mappings stay in personal server/private
local backup. Diagnose first-step equivalence and the contact intervention
results before proposing a fix. Any amended training profile needs a fresh
document-following actual witness before being marked ready.

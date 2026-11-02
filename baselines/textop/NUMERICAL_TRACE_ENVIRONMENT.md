# Full-library numerical anomaly trace

This is a targeted investigation of the existing 3.304791e27 value loss. It is
not a repeated capacity acceptance or formal training. Read SERVER_NOTES.md,
FULL_LIBRARY_CAPACITY_ENVIRONMENT.md and configs/full-library-numerical-spec.json.
Reuse the unchanged warm b7a3bf0c SDK environment, GPU1, CPU0–3, Torch fraction0.65.
No package, data, model, reset, reward, contact parameter, or PPO update changes.

The new probe is derived from the SHA-bound existing capacity probe and adds
numerical_trace.py after environment/runner construction. State is observed
after each of 96 physics substeps. Contact reads are only made after the existing
reward manager consumed the sensor. The 24 weighted reward vectors are summarized
before the environment resets. The initial simulator state and first extreme
environment/source mappings are private evidence, not reusable checkpoints.
Diagnostic snapshot thresholds (velocity1000, force1e6, reward1e4) are NOT motion
acceptance criteria. They cause no termination, rejection or clipping.

Completed2026-10-26: numerical_trace_author_v1 actualexit0/195.599s,96physics
samples/24reward samples/20updates. Value loss equals original; first state
outlier occurs atphysicsstep1. This run is immutable; DO NOT repeat the command
below. It is retained only as the exact executed invocation:

```
ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -i C:/Users/Pende/.ssh/id_ed25519 ouyanghao@10.147.17.32 "python3 /data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/supervise_full_library_numerical.py --run numerical_trace_author_v1"
```

The supervisor uses the same SDK flock, bounds its own process to1800seconds,
samples resources, retains its SDK log and distinguishes native close from the
existing bounded fallback. Require actual exit0 plus96physics/24reward samples
and20optimizer updates for a complete trace. This says nothing about convergence.
If a source or algorithm issue is found, preserve this run and document any fix
separately, including a fresh documented witness before declaring the corrected
training profile ready. Never silently retry a run name or relax a threshold.

Publish only receipt, exit/resource evidence, numerical_summary.json and the
aggregate numerical_trace.jsonl. Keep numerical_private/ and all source-level
indices on the personal server/private local backup. Existing datasets,
checkpoints, paused57D/PhyGile/ReactiveBFM processes and shared dependencies stay
unchanged. Formal filter and generator launch gates remain false.

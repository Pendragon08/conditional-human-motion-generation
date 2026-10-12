# Strict original-reference cache reuse certificate

V33 is complete. Its bound source, document, frozen thresholds and historical
scores remain unchanged. The independent audit identified missing witness
digest checks on future cache reuse, without observing replaced current data.
V36 adds a fail-closed, read-only certificate: it verifies the committed witness,
runtime witness and terminal summary anchors, all witness inputs and sources,
external motion/model inputs, C1 manifest/threshold equality, both batch lineage
requests/outputs, reset agreement, reference clock and original target binding.
Every stored metric row is recomputed from the licensed raw trajectories using
the frozen NumPy runtime and unchanged V23 metric.

Run this before trusting or reusing the completed V33 cache. A new data version
requires a new witness and anchors, not editing the hard-coded old digests.
Failure aborts verification; no old launcher is executed and no data is repaired
or overwritten automatically. An isolated self-check rejects modified and
missing fixtures, leaving experimental files untouched.

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 env PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/verify_mocap_cache_binding_v36_20261012.py
```

Expected exit 0 and JSON status PASS. This invocation is CPU-only, creates a
new timestamped certificate, and invokes neither a model nor a simulator. It
does not amend C1's negative result or resolve the semantic question of whether
closest approach is the intended arrival-time event. It is not an innovation
efficacy test. No result is synchronized to Overleaf.

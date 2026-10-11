# Reset-state consistency repair v4 — reproducible invocation

Remote: ouyanghao@10.147.17.32. Warm Python runtime and cached licensed assets; no installs or model changes.

The collector now removes its redundant refresh AFTER the official reset. Official HumanoidAMP reset restores four rigid-body caches once; another refresh without a physics step reads the previous simulation snapshot. Root and DOF tensors do not show that corruption. No fabricated physics measurements, new IK, or policy changes are introduced.

## Independent document-following witness

Execute exactly once from this computer:

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_reset_alignment_checks_v4.py --check
```

Expected exit 0 and both sentinels `RESET_V4_BATCH_PASS planned_s11`, `RESET_V4_DOCUMENT_CHECK_PASS`. This creates 12 first attempts, not a full measurement or efficacy claim. Read result receipt, metadata, initial state gate, and request/output hashes. Raw `initial_states.npz` and trajectories remain server-local under existing licensing. Report any divergence; do not fix it, repeat invocation, or run --run.

## Full measurement after the witness

Same invocation with `--run`. Completed batches are reused only if bound hashes agree. Six seed/mode batches (12 cases each) plus two all-validation batches (72 each), 216 total attempts. Each must have identical seven exported bootstrap/first-action state tensors; reference XY <1mm; immutable original targets and planned prefixes equal across seeds/modes; exact reference/action clocks; world offsets ±15cm; geometry and source provenance retained. The analyzer also reports individual quality gates without changing thresholds. A zero joint-pass validation floor is NOT an efficacy failure, and does not authorize confirmatory continuation.

Existing viewer GPU contexts are preserved. Select GPU0 only while utilization≤5% and at least24GiB VRAM free; wait at most60 seconds between finite checks, preserving a resource-stop status if not available. Before every batch require≥20GiB /data space,≤15GiB round outputs, cumulative≤4 measurement GPUh /72 totalGPUh including previous failed attempts. No old output is deleted. Each child≤600seconds and charged elapsed time; no unpaid/VLM APIs, environment builds, or real-robot tests.

Current results are synchronized only to local reports/GitHub. Overleaf experiment-results updates are disabled by the user for this round.

# Read-only diagnosis of the frozen slip proxy

Warm environment: .aris/compute/closd-baseline-20261010.md. No installs, model
calls, simulations, new GPU time or changes to frozen thresholds. Data remain
server-local. Run all432 V12 attempts; no result-based selection.

Fresh document-following agent runs this command exactly once. Do not retry:

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/contact_slip_diagnostic_v13_20261012.py
```

Expect exit0 and SLIP_PROXY_OFFLINE_DIAGNOSTIC_PASS. Inspect actual diagnostic
JSON, source and input bindings. Check that the frozen proxy separates already
recorded stationary box, frictionless slide and airborne controls. Report actual
counts. The slowest near-lowest point is an optimistic diagnostic lower bound,
not the true contact point and not a replacement metric. Finite-difference
velocity disagreement at30Hz is not enough to infer an SDK velocity-frame bug.

This does not validate humanoid quality discrimination, any candidate innovation
or downstream training. Preserve errors and arrays. Save independent report to
results/contact_slip_diagnostic_v13_20261012/DOCUMENT_WITNESS.md locally; do not
publish licensed motion arrays or write experimental results to Overleaf.

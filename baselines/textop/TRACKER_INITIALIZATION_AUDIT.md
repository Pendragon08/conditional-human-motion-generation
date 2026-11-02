# Deterministic reset/contact diagnostic

This procedure reuses warm tracker environment `b7a3bf0c` from SERVER_NOTES.md.
No dependency installation, shared-source change, training or motion acceptance.
Read configs/tracker-initialization-audit.json. It deliberately uses1286envs,
one per exported trajectory, zero reset perturbations, starts at frame0 and
commands the current reference joint positions through the installed PD action.
Those are diagnostic choices, not changed paper training hyperparameters.

Fresh document execution, exactly once, in PowerShell:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -i 'C:/Users/Pende/.ssh/id_ed25519' ouyanghao@10.147.17.32 'python3 /data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/supervise_tracker_initialization.py --run initialization_document_v1'
```

Require actual exit0, `TEXTOP_INITIALIZATION_SUPERVISED_EXIT0`, a matching source
and spec hash,1286 trajectories,20 PD steps and0 optimizer updates. Evidence is
in tracker_validation/initialization_document_v1/receipt.json and receipts/
tracker_initialization_initialization_document_v1_exit.json. Copy only these
public receipts, logs and teardown stack, never raw trajectories or private
indexes. Runs are immutable: no retries or edits under the same run name.

The first-frame root-position discrepancy must be below1e-4m. Joint clipping,
body errors, contact forces and termination events are reported without a
pass threshold or a tracking-quality claim. Arbitrary reference-PD control
cannot replace a trained tracking controller. All original spike evidence is
retained. SDK native-close hang remains: only after completed receipts and
closed writers can the documented20s exit fallback fire; inspect actual child
exit and `native_close_returned=false` separately.

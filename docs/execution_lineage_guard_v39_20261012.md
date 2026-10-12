# Execution-time lineage guard after the completed V28 run

The original resumable queue can accept a receipt without a contemporaneous
lineage and retrospectively certify it. Independent review found this reachable
control weakness. It was not exercised in the 74 completed V28 batches: every
lineage is bound_request_at_execution. The original source, protocol and negative
result are retained unchanged so that the audit remains reproducible.

V39 is a strict read-only guard for current evidence and a callable guard for
future queues. It refuses missing or retrospective lineage, changed execution
requests, incomplete output bindings, changed output bytes or inconsistent
receipts. It does not call the old certifier, create a lineage, amend a request,
recompute a success score, or invoke a simulator. Missing evidence requires a new
explicit attempt; a verifier must not repair its absence retrospectively.

Current V28 witness, independent interface witness, versions and terminal summary
are anchored by committed digests. Source and request inputs are checked, all
74 batches must have charged ledger labels, and V36's external-input bindings
are checked again. Functional tests reject missing, retrospective, mismatched,
incomplete and tampered isolated fixtures. They are not efficacy experiments.

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 env PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/verify_execution_lineage_v39_20261012.py
```

Expected exit0, status PASS and checked_batches 74. No GPU, model generation or
Overleaf write. A timestamped certificate is created without replacing results.

Do not resume the terminal V28/V33 launchers. Any future C2/C3 runner must call
guard_batch before trusting an existing receipt, pin its own witness digests,
and undergo a fresh document-following execution check. This guard's existence
does not by itself integrate that protection into the old launchers or change
the original audit verdict. The endpoint event-definition issue is separate and
remains unamended in the historical metric.

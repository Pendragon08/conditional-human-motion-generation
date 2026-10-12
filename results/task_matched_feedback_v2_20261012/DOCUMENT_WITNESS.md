# Independent document-following witness
Status: **PASS**. Same-family/provisional.

The fresh witness agent executed the documented v2 check exactly once via SSH. Actual exec session `85146`, remote PID `2929067`, terminal chunk `b6ee2b`, actual exit **0**, exact stdout sentinel `TASK_MATCHED_FEEDBACK_CHECK_PASS`. No prepare/run invocation, retry, installation, source or threshold change occurred.

```bash
cd /data/code/ouyanghao/oyh/conditional-human-motion-generation
PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python scripts/task_matched_feedback_v2_20261012.py --check
```

All three actual GPU batches retained 72 raw cases in the preregistered case/source/text/goal order. Charged exits were 0/0/0; charged seconds were 92.96196814067662, 91.94964390061796, and 90.88670927286148 (total 275.79832131415606). Original batch logs, requests, execution lineages, receipts and archives are preserved.

Read-only independent verification confirmed four actually consumed dataset-file byte bindings in each batch and both actually loaded published weight-byte hashes. The zero-correction object and exact feedback-parent output bytes match its execution request. All 28 exported arrays across initial_states.npz, observation_trace.npz and trajectories.npz are exactly equal between the same-input generation runs. Fixed selected-reference motion content is bound to replay; each reference_frame array is exactly 0..119; replay versus the frozen float32 canonical reference has maximum error 0.0.

- versions_sha256: `812e1c56258da2d736205d60f1a2b4f5263b8734b8dc23dde89999ee14042c53`
- runtime_witness_sha256: `2ff3b4c0c1af6b7d7f0442724e458d87a6009193ba50fcaf8b00423f46aaa7fd`
- raw_verification_sha256: `bd7f3efb0b0364da2e9d797db1289e27e4a64aaeeaaa92b0ce969345b3542394`
- Full trace and read-only verifier: `.aris/traces/task_matched_feedback_v2_20261012/document_witness/`
- Local complete remote-output snapshot: `remote_result_snapshot/` within that trace.

This same-family/provisional witness certifies the documented invocation and the listed exported measurements/bindings. Hidden PhysX identity, semantic correctness, candidate efficacy and innovation are not certified. No Overleaf experiment write was performed.

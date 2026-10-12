# Controller state diagnosis v2 document witness

Status: PASS

Actual documented --check invoked once. Terminal exit: 0. Successful documented sentinel: True.

Review independence: same-family. Acceptance: provisional.

Runtime witness SHA256: 81f2d42777926ef927f35b548c7500e72779684b1449bb79b52616047b4e6e93

Versions SHA256: e94a000d3260e0993a20151b8b1fff877c4632964febd66c75f72790722da196

Fresh document-following same-family witness of internal validation fixed replay using released multitask CLoSD training state. Simulated human proxy measurements, not candidate efficacy, innovation, G1, downstream learning, independent final test, or hidden PhysX identity.

```bash
cd /data/code/ouyanghao/oyh/conditional-human-motion-generation
PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python scripts/controller_state_diagnosis_v2_20261012.py --check
```

Detailed runtime bindings, denominator, reset gates, command/tool evidence and hashes are recorded in DOCUMENT_WITNESS.json. Full trace is under .aris/traces/controller_state_diagnosis_v2_20261012/document_witness/.

Artifact verifier correction: the initial verifier used an incorrect extra axis0 case-count assumption. It is preserved in trace as artifact_verifier_attempt1_DOCUMENT_WITNESS.json/.md with its source and transcript. Collector and frozen evaluator use (time,cases,bodies,xyz), observed (120,72,24,3). The corrected read-only examination uses axis1. The documented check was invoked once and passed; runtime files and source were unchanged.

# Document-following runtime witness: FAIL

The documented --check command was executed exactly once in the existing warm runtime. SSH and the invoked Python process terminated with exit code 1. The required CONTROLLER_STATE_DOCUMENT_CHECK_PASS sentinel was absent. No retry and no --run invocation occurred.

Failure: scripts/controller_state_diagnosis_20261012.py:68 raised KeyError: 'upstream_commit'. The prepared versions.json does not contain this required top-level field. This is a preparation/runtime contract divergence, not a controller-quality observation.

The failure occurred in validate() before batch() or any collector subprocess. Actual raw/ is empty; raw/multitask, requests/multitask.json, runtime_witness.json, loaded_input_binding.json, loaded_checkpoint_binding.json, metrics and initial-state exports were not produced. The budget ledger remains []. Planned 24 sources/72 rows are preserved as the intended denominator; observed rows are 0 and no 72-row or reset-gate verification is possible. Generator calls and simulation attempts for this invocation were 0 because execution stopped before the collector. No gate to --run is satisfied.

Read-only inspection observed actual upstream HEAD d8950bdb78312fcd2844f336b3fc079e7368c87f and an empty tracked closd diff. No state.json exists: initial validate() is outside the main exception handler that would record FAILED_RETAINED. The original traceback and authoritative terminal exit are retained in the trace directory.

Review independence: same-family. Acceptance status: provisional. The witness did not edit implementation or documentation. Hidden PhysX identity, checkpoint efficacy, robot utility, semantic truth and innovation efficacy are not established.

Exact executed command:

```bash
cd /data/code/ouyanghao/oyh/conditional-human-motion-generation
PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python scripts/controller_state_diagnosis_20261012.py --check
```

Authoritative execution tool evidence: exec_command chunk e6b669, exit_code 1, terminal marker AUTHORITATIVE_SSH_EXIT=1. Read-only artifact inspection: chunk 173f28, exit_code 0. Trace: .aris/traces/controller_state_diagnosis_20261012/document_witness/.

runtime_witness_sha256: null (file absent).
versions_sha256: f3a849d62b541896f81d139ab503ba17efa08d3be9e8f3cbd1d089c15f46ab9c.

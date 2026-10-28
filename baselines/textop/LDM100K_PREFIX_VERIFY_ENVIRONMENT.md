# CURRENT — independent CPU verification sources deployed DISABLED, no real input yet

VAE100k is complete; the original LDM still trains toward the user100k endpoint.
The original LDM and endpoint observer source/spec remain frozen. The new CPU
export also remains disabled. Do not load any model or run either CPU entry
while the original model is training, and do not rerun old preparation fixtures.

New `scripts/ldm100k_export_acceptance.py`,
`verify_ldm100k_prefix_export.py`, `supervise_ldm100k_prefix_verify.py` and local
activation/dispatch/collection bridge implement a separate, read-only verifier.
Spec `configs/ldm100k-prefix-verify-spec.json` is disabled with
`verify_authorized=false`, `export_binding=null`; SHA
82c0f2ac228362f8484af7ac9df3f383476562944068bdef9a5087f79a2100bf.
Actual deployment, Python3.8 parsing and18 new isolated metadata negative cases
are recorded in `results/ldm100k_user_handoff_20261028/prefix_verify_preparation_v1.json`.
The fixture child exited0, was reaped and had final streams. Production model/
checkpoint reads, GPU contexts, signals and actual verification jobs remain0.
These fixtures establish neither real endpoint acceptance nor numerical quality.

## Parent sequence after the real100k stop and actual CPU export

First follow LDM100K_USER_ENDPOINT_ENVIRONMENT.md and
LDM100K_PREFIX_EXPORT_ENVIRONMENT.md: require the real100k checkpoint and
closed endpoint controller, original worker exits and common-lock release,
then the actual `user_ldm100k_prefix_export_v1` CPU export with process,
supervisor and outer0. Collect its actual result. Only then execute once:

```
provider_tools/activate_ldm100k_prefix_verify.py
```

Activation binds the real export receipt/spec/exit/inference bytes, rechecks
the stopped original source, actual closed CPU export and immutable fullstate,
archives disabled spec bytes, and returns exact activated bytes with SHA. It
does not dispatch a job or load a model. It requires16GiB available RAM,
3GiBdata and2GiB HOME reserve. Do not run repeated pending activation attempts.

After that actual activation, use one fresh agent to execute the documented
invocation below, rather than the author rerunning or fabricating a SUMMARY.
The training/export sources and specs remain unchanged during its execution.
No question to the user is needed; this is part of the authorized handoff.

## Fresh document invocation, only with genuinely enabled input

Use the existing bundled local Python and explicit PYTHONIOENCODING=utf-8.
Run exactly once:

```
provider_tools/launch_ldm100k_prefix_verify.py --run user_ldm100k_prefix_document_v1
provider_tools/collect_ldm100k_prefix_verify.py --run user_ldm100k_prefix_document_v1 --evidence NEW_SHORT_ASCII
```

Existing intent/driver/worker or unknown SSH result means inspect the original
operation, never dispatch again. Collection is metadata/log-tail/resource-only;
use new short evidence names and wait for the actual child and outer result.
Local progress is under results/ldm100k_user_handoff_20261028/
user_ldm100k_prefix_document_v1/progress.json. The actual CPU verifier is bounded
900s with CPU0-3, no visible CUDA,15s resource logging and the common GPU lock.
It cannot overlap SDK/model generation/evaluation jobs.

The independent CPU process loads the immutable100k fullstate and exported
inference separately, requires finite CPU tensors, full exact denoiser equality,
strict loading in two newly instantiated CPU denoisers, actual saved Adam/clock/
stage/history counters, exact CPU Adam state restoration with0optimizer steps,
and unchanged fullstate/inference SHA/owner/no-symlink/fingerprints. It preserves
the real native300k schedule provenance while asserting the selected user100k
prefix and no300k-completion claim. A deterministic nonzero synthetic history/
text/noise probe checks both conditional branches and exact CPU forward equality.
It neither restores CUDA RNG nor uses heldout arrays. These synthetic probes
are integrity checks, not text semantics, generated-motion quality or physics.

Require actual `USER_LDM100K_PREFIX_FRESH_CPU_VERIFIED`,
`closed_execution_verified=true`, process/supervisor/outer0, final resources
after_exit and no workers/common-lock free. Check source/spec hashes before
and after and inspect all actual discrepancies. Write a real local SUMMARY:

```
status: USER_LDM100K_PREFIX_FRESH_DOCUMENT_VERIFIED
run: user_ldm100k_prefix_document_v1
actual_exit_code: 0
findings: []
executor_sha256: actual receipt source_sha256
spec_sha256: actual receipt spec_sha256
selected_checkpoint_sha256: actual selected_checkpoint.sha256
inference_sha256: actual inference.sha256
export_receipt_sha256: actual receipt export_receipt_sha256
actual_metadata_audit: actual receipt metadata_audit
actual_closure: actual collector closure
```

An empty findings list is permissible only when the documented integrity
execution really passes; do not invent success or omit mismatches. Actual
training-history findings in metadata_audit remain visible even if the CPU
artifact identity passes. Zero history counters cannot authorize further
training beyond100k. Do not rewrite the original interrupted nonzero exits.

After genuine CPU verification, continue implementing/running heldout multi-seed
text versus unconditional controls, full five-step sampling with explicit CFG5,
TMR metrics, FK/physical endpoints, WBC/online and licensed films. Their execution
entries, inputs and actual capacities are still not accepted. No staged source
is a complete reproduction. The full LDM state and optimizer stay private and
must receive a separately verified private byte backup with actual capacity;
do not apply a VAE-only copy tool. All originals, failed outputs and partials
remain intact, each filesystem keeps its2GiB floor, no new training is authorized.

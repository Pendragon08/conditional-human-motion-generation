The documented command was passed to SSH verbatim **once**, after an explicit remote `OUTPUT_ABSENT` check and a check for another witness/training process for this project. Exit code: **0**. Final receipt: **`OFFICIAL_DIT_KERNEL_AND_BACKWARD_PASSED`**. Exact command is in `exact_command.txt`; command arguments, UTC times, exit code, stdout and stderr are preserved in `documented_run.*`.

Only the three requested setup inputs were read to obtain setup and invocation: `docs/environment.md`, `.aris/compute/reactivebfm-20261021.md`, and `configs/env_spec.json`. No packages, code, datasets or documents were changed, and no other process was stopped.

The receipt reports prediction shape `[1,36,1,50]`, nonzero finite gradients, zero optimizer updates, the expected 107,045,924 trainable parameters, NVIDIA RTX A6000, source commit `5a79d03cab824eb5917271ba720294e8a1b9d374`, and all eight required package versions matching the spec. It reports 226 official source files verified, a real Kungfu training batch, and peak allocated/reserved memory of 1,170,842,112/1,283,457,024 bytes. Stderr is empty. The diagnostic finite loss is not a quality result.

Local SSH UTC: `2026-10-21T14:47:35.1706938Z` to `2026-10-21T14:47:40.7797363Z`. Remote receipt UTC: `2026-10-21T14:47:39.672453+00:00` to `2026-10-21T14:47:42.351333+00:00`. Both clocks are preserved separately: the remote receipt end is about 1.57 seconds later than the local SSH end, so synchronized clocks are not established.

The byte-preserving receipt copy has matching remote/local SHA256 `9d94fa996d5b2cb8fe6c880494304c8fc3fcdc76ad9818d391108b63107f273f`. Its witness source SHA256 `43a01fd0dea2ac11ad03f017806adb99f3a63f6070a59528e7ae9b760303445e` matches the independent source inventory. Source inventories cover 231 regular files beneath remote `code/`, excluding `.git`, `__pycache__`, and `*.pyc`; their before/after SHA256 is identical: `f31d2c2a7962197848c6d279ed0e2ede13909d83b8a0dc6d67552a3732ee8081`. The 231-file inventory has broader scope than the receipt's 226 official files. All three document hashes and byte snapshots are preserved before/after and match.

Deviations and evidence limits:

- There was no deviation from the documented command. The prescribed compatibility pilot uses Python3.10.21/Torch2.5.1+cu121 rather than the upstream recommendation Python3.12/Torch2.11+cu128. Python's runtime version is not separately reported by the receipt.
- The allowed setup inputs do not state an expected Kungfu SHA256 value. The receipt reports `2fddadcd7361d3333f18cdcc556bcb0fc2d1ee30cc2fe047e191e23382a25eb5`; this witness preserves that value without asserting an independent expected-data digest comparison.
- Shared GPU1 and four CPU-library threads are explicit in the invocation. The 25% PyTorch allocation cap is documented as enforced by the witness, but its configured fraction is not exposed in the final receipt; no additional probe was run.

This is selected-runtime, CUDA-kernel and official-model forward/backward evidence. It makes no claim of generator training, MuJoCo control, motion quality, paper performance, target-reaching or hardware validation. `witness.json` contains the structured evidence; `trace_manifest.sha256` binds all saved trace files. The collection script must not be rerun into this completed directory.

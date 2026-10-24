# Fresh agent document witness — 2026-10-21

The fresh verifier read only the run-experiment skill, compute environment contract, provider ledger, and the designated environment document before running its documented CPU invocation verbatim through the existing SSH endpoint and key.

- Documented invocation: command.sh
- Raw combined command output: raw-output.txt
- Public execution metadata and exit code: status.json
- Remote output path: /data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/ssm_20261021/environment_extension_v1/cpu_witness_document.json

## Result

The documented invocation exited 0 and emitted status CPU_WITNESS_PASSED with spec_hash 141547bf, seed 42, and device cpu. It reported 16 FK poses, maximum position absolute error 4.5578329702089526e-07 m, and maximum rotation matrix absolute error 1.4077799777162392e-06. Both errors satisfy the documented threshold of 1e-5. No command, package, environment, script, parameter, or document was changed.

Doc-versus-reality assessment: the documented success contract matches the observed exit code, status, spec hash, CPU device, pose count, and tolerances. The command additionally emitted an optional xrobotoolkit_sdk-not-found message, which was not listed in the document; it did not block this CPU witness. No functional divergence was observed.

The first tool attempt did not execute because automatic permission approval review timed out. The tool explicitly permitted one retry. The same SSH invocation was retried once and completed successfully.

## Verification boundary

This verification does not prove that human body model weights have been obtained or that real human motion has been retargeted. The witness explicitly reports human_model_forward_pass_executed=false, gmr_retargeting_executed=false, and environment_fully_ready_for_humanml_processing=false. It does not validate SMPL-X human-model inference, GMR human-motion retargeting, filtering, new 262D data generation, training, complete AMASS download, or full dataset disk capacity.

# V51 independent document witness — PASS

Fresh agent read the run-experiment skill, compute contract, provider ledger and V51 invocation document. The exact documented `--check` was executed once, exit 0, with sentinel `PAIRED_FEEDBACK_VALIDATION_DOCUMENT_CHECK_PASS`. No `--run`, retry, experiment-source repair or package installation was performed.

Local execution session 71584; remote parent PID 2849663, collector PID 2849686 and child PID 2849719. All recorded processes exited. Terminal state is `WITNESS_COMPLETE_PENDING_INDEPENDENT_REVIEW`. GPU0 started at 699/49140 MiB and 0% utilization. Existing cuda:1 viewer/evaluation processes had incidental GPU0 allocations and were preserved; the documented provider guard explicitly permits their contexts with at least 24 GiB free and utilization at most 5%.

The only charged batch is `val_shared_s11_r0`: validation partition, seed11, ordinal0, 72 inputs from 24 sources, 49.190580459311604 seconds, charged to measurement. Audited aggregate totals: global15014.283680932596 seconds (<72h), measurement4235.553678927943 seconds (<4h), V5149.190580459311604 seconds (<1h). Data disk has38201565184 bytes free (>20GiB); tracked outputs plus archive total7417669525 bytes (<15GiB).

Independent read-only remote audit confirmed all28 arrays have identical keys, shapes, dtypes and values against V46 native `val_native_s11`. The three archives are also byte-identical:

| Archive | SHA256 |
|---|---|
| trajectories.npz | e96cb97ad5d12d4c54a3fbf57625d9ab4491e5c38febd800d78ae2a1baecfaaf |
| observation_trace.npz | 482a3c2bebcf5cc4fff1d071fb6d4aa3d7f7edfbf5a0a082a6d89e9c2627f338 |
| initial_states.npz | b25404dd3cfb63710078bf7dbb824e3749b1cdebabd68e7719c2f6b3b12617fe |

The execution request, execution-bound lineage outputs, receipt, initial archive, observation trace and charged ledger match their hashes. Frozen source files, upstream CLoSD commit and unchanged tracked closd sources, generator/controller/evaluator weights, external motion/semantic inputs, V43 witness and V52 calibration witness/summary were checked. Python3.8.20 and NumPy1.23.5 were read in the audit; the simulation log records Torch1.13.1+cu117. Local source/document and copied small-artifact hashes match remote bytes. Licensed NPZ files remain remote.

Runtime witness SHA256: `6ad22044c8392a8cc679d03a56b99185dd55ab1be77846c1498fb756ebe379f1`.

The result certifies only this documented interface/recording batch and provisional same-family calibration. It does not certify real contact points, feedback efficacy, semantic or diversity gains, independent testing, G1 utility or innovation. No prior score or threshold was changed. V51 versions.json retains some historical V28 batch/test-count fields; its explicit V51 fields and document define24 validation sources,15 construction batches and6 replay batches. Those inherited historical fields are not evidence of V51 test completion.

Failures were preserved: the optional read-only Torch-version command had a shell-quoting failure before Python started and was not retried (`ancillary_read_failure.json`). Torch version was established from the completed simulation log. The missing project AGENTS.md and absent prelaunch V51 output directory were read-only observations; the provider ledger supplied the explicitly instructed environment. No document-versus-command divergence was found.

Evidence: DOCUMENT_WITNESS.json, command_evidence.json, independent_readonly_audit.json, original val_shared_s11_r0.log, runtime_witness.json, versions.json, budget_ledger.json, requests/val_shared_s11_r0.json and raw JSON sidecars.

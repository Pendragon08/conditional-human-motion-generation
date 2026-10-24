# Fresh agent document witness v3

Verdict: **FAIL — witness harness encoding failed before any physics execution.**

The exact single command line captured from CONTACT_V3_INVOCATION.md was launched once at 2026-10-22T12:07:54.4685871Z through a saved PowerShell harness. The child exited with actual exit code **2** at 2026-10-22T12:07:55.2321887Z. No retry, repair, package installation, source edit, server operation, private asset write, or policy update was performed by this witness.

The saved harness was created by PowerShell Core 7.6.5 using UTF-8 without BOM, then run with Windows PowerShell 5.1. The missing BOM allowed legacy script decoding to corrupt Chinese path arguments. Python reported that it could not open the script path, with `[Errno 22] Invalid argument`. This was an avoidable witness harness error. It does not establish a defect in the documented invocation, warm environment, policy, or physics.

## Acceptance

| Requirement | Actual | Outcome |
|---|---|---|
| Canonical environment hash matches ledger | a43f960bd3bcdcd3398655bffe291138b73de5ca7f30901765172ff9821e1954 | PASS |
| Documented invocation attempted once | 1 saved command line launched; runtime Unicode arguments corrupted | ATTEMPTED, INVALID WITNESS |
| Actual process exit 0 | 2 | FAIL |
| EIGHT_PAIRED_DIAGNOSTICS_COMPLETED | Comparison directory not created | NOT REACHED |
| Eight first episodes / child exit codes 0 | 0 episodes / no child receipts | NOT REACHED |
| Finite first-episode metrics | No metrics | NOT EVALUATED |
| Compiled foot sliding coefficients 0.6 and 1.0 | No compiled model | NOT EVALUATED |
| Executed source integrity | 366 project/vendor/mjlab source and model files hashed before and after; 0 changes | PASS for snapshotted files |
| Frozen policy and development-diagnostic scope | No policy execution reached | No optimization or quality claim |

The four clips are development diagnostics. Improvement would not be an acceptance condition. Neither full motion success nor paper benchmark reproduction is claimed.

## Evidence

- CONTACT_V3_INVOCATION.md: immutable captured invocation document from before launch.
- command.txt: exact command line captured from the document.
- execute_once.ps1: original saved wrapper, retained without encoding repair.
- launch_and_capture.ps1: original process capture harness.
- command_receipt.json: exact command, paths, process ID 15564, start/finish, actual exit code.
- stdout.txt: full captured child stdout (empty).
- stderr.txt: full captured child stderr.
- launcher_console.txt: launcher console output.
- source_hashes_before.json and source_hashes_after.json: 366 file hashes.
- acceptance.json: machine-readable acceptance and harness diagnosis.
- preflight.json: inputs read and canonical hash result.
- evidence_hashes.json: hashes of evidence files.

The parent reported a later v4 document revision with a fresh output destination and explicit UTF-8-with-BOM harness requirement. This agent did not execute the revised invocation. The captured v3 failure is preserved as-is.

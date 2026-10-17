# Terminal-boundary document-following witness

**Result: FAILED. Exactly one documented invocation was launched; no rerun or repair was performed.**

The witness read only the three permitted initial files before launch. The initial invocation document omitted SSH connection parameters. No experiment was started at that point. The parent corrected that document, after which the witness re-read the same three files. Initial and corrected SHA-256 hashes are preserved in `document_hashes.json` and `corrected_document_hashes.json`.

## Invocation and prerequisites

Environment spec: `fa5883c8`. Original server: `ouyanghao@10.147.17.32`, port 22, existing `C:/Users/Pende/.ssh/id_ed25519`, `BatchMode=yes`, `ConnectTimeout=15`.

The output directory `results/completion_v1/closure_final/` did not exist before launch. The prior `document_run` verification statuses were `COMPLETED_ENGINEERING_VALIDATION`, `COMPLETED`, `PASS`, and `COMPLETED`. GPU 0 had 2968/49140 MiB allocated; the environment ledger explicitly allows shared existing idle allocations. No process was stopped.

The two documented commands were sent once, with an outer `timeout --signal=TERM 600 bash -s` transport wrapper:

```sh
cd /data/code/ouyanghao/oyh/phygile-reproduction-20261017
bash scripts/run_isaac.sh scripts/validate_terminal_boundary.py
```

## Exact failure

The recorded child returned exit 1 after 49.73612189292908 seconds, without reaching the timeout. Its last stages were `runtime_ready`, `snapshot_ready`, `first_rollback_rollout`, and `second_rollback_rollout`. The generated main log ended with:

```text
AssertionError: ('Exposed-state rollback witness failed', 4.238388895988464, 0.0018104314804077148)
```

A separate read-only comparison of the preserved rollout NPZ files reproduced state max absolute difference `4.238388895988464` and body-position max absolute difference `0.0018104314804077148` m, exceeding the recorded `0.0005` gate. Action max absolute difference was `0.23255288414657116`. The witness did not investigate or modify implementation.

A separate transport defect is preserved: the PowerShell native pipeline appended a CR-only blank line after the two documented commands. Bash reported `bash: line 3: $'\r': command not found`, so the SSH transport exit was 127. This extra blank line occurred after the experiment's independently recorded exit 1 and does not explain away that scientific failure. No corrected launch was attempted.

## Primary-artifact inspection

`output/verification.json` is absent. `PHYGILE_TERMINAL_BOUNDARY_COMPLETE` is absent from captured invocation stdout/stderr and the generated main log. Therefore the document's completion conditions are not met.

The partial remote output is preserved. Eleven NPZ files were opened using a standard-library ZIP/NPY reader, with key, dtype, shape, SHA-256, and floating-point finiteness checks saved in `primary_artifact_inspection.json`. The nominal target exists with states shape `(1,71)`, body positions `(1,14,3)`, root height `0.76` m, and zero root/joint velocities. The two rollout raw files each have states `(10,71)`, body positions `(10,14,3)`, and actions `(10,29)`. All inspected floating-point arrays were finite. These partial artifacts do not establish terminal-boundary completion, candidate acceptance/rejection, dynamic standing stability, or semantic generation.

## Evidence

- `invocation.stdout.txt`, `invocation.stderr.txt`, `transport_exit.txt`: captured single invocation and transport result.
- `preflight.stdout.txt`, `preflight.stderr.txt`, `preflight.transport_exit.txt`: prerequisite evidence.
- `primary_artifact_inspection.json`: all generated JSON, NPZ structure/statistics, file hashes, and generated log tails, including the complete 12-line main log.
- `rollback_npz_comparison.json`: independent comparison of preserved generated arrays.
- `report.json`: structured result, document hashes reference, and completion-condition checks.
- `evidence_manifest.json`: local witness evidence hashes.

No code, installation, thresholds, or scientific evidence was changed. The existing partial `closure_final` directory was not renamed or overwritten, and the invocation was not repeated.

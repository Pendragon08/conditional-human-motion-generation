# Headless continuation fresh document witness v2

Result: **FAIL — documented invocation did not reach bounded PPO completion.**

The documented PowerShell command was executed exactly once with the legacy Windows PowerShell shell and matched the invocation document byte-for-byte after trimming its code-block boundary. The submit command returned exit code 0 and remote PID 3156839 at 2026-10-22T11:50:53.255351+00:00. The launcher reported GPU1 had 16330 MiB free before submission.

The first read-only status poll found that PID absent. Training stdout was:

```text
There was an error running python
```

Training stderr ended with this exact failure:

```text
lab_run.py: error: argument --headless-authorization: expected one argument
```

The complete stderr, including argument usage, is preserved in `remote_files/headless_smoke_v2_stderr.txt`. The detached training exit code is unavailable; submit exit code 0 does not establish training completion.

Neither `receipt.json` nor `executed_sources.json` existed. Consequently the required completion status, headless flag, exact optimizer restore, two additional PPO iterations, 40 Adam calls, changed policy tensors, unchanged sources, unchanged parent, and receipt-to-source SHA256 binding are all unverified. No quality-improvement claim is supported.

The command, separate submit streams, full remote streams, launch record, read-only poll transport record, document snapshots, and SHA256 manifest are preserved in this directory. No repair, retry, source/dependency change, process interruption, deletion, or external notification was performed.

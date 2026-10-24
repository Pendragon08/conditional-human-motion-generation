# Independent paired-screen document witness

Status: PASS_WITH_ALLOWED_EARLY_TERMINATION. The documented PowerShell block was extracted without alteration and executed exactly once from the specified workspace. The actual native exit code was 0; the child episode exit was also 0. No repair, retry, package installation, environment/source change, server operation, or dataset change was performed by this witness.

UTC execution: 2026-10-23T01:38:05.4482587Z through 2026-10-23T01:38:37.4361916Z (31.9486503 seconds). Summary status: DIAGNOSTIC_SCREEN_COMPLETED. The invocation document's before and after SHA256 was identical: 9efc260e73e2071bf77b95c397bf288891363f4465d3d5afc484e02394e501ce. The environment plan canonical SHA256 independently matches the ledger: a43f960bd3bcdcd3398655bffe291138b73de5ca7f30901765172ff9821e1954.

Selected one train sequence, motion 003852 interval 0, not mirrored: 50 source frames at 20 Hz produce 123 requested control frames at 50 Hz. Only 14 control steps (0.28 simulated seconds, 56 physics steps) were observed before termination. The last observed row (index 13) is the only terminated row; no timeout occurred. Full-clip success is false (0/1). This robot tracking failure is explicitly allowed by the invocation and is retained; it must not be called full-clip success.

All raw pre-reset arrays are nonempty and finite. The receipt declares post-physics forward before termination. Independent hashes match all 10 captured inputs and 246 captured executed source/input files; receipt, raw-state archive and executed-source manifest hashes match their recorded hashes. Source last sample is 2.45 s, last control sample 2.44 s, omitted tail 0.010000000000000231 s (< 0.02 s), with no time stretching or extrapolation. Zero optimizer updates and zero curriculum labels were recorded. CPU device is confirmed in the receipt.

Termination-censored metrics: MPJPE 0.052472926676273346 m; MPJAE 0.1801006942987442 rad; MPJVE 1.483838438987732 rad/s. These metrics cover only the 0.28 s first episode, not the full clip.

No invocation or documented acceptance discrepancy was found. Episode stderr contains a 197-byte torch.jit.script deprecation FutureWarning; outer stderr is empty. The termination condition name is not present in the receipt, so cause-specific diagnosis is unsupported. Runtime package metadata lists Torch 2.14.1 (without a build suffix); CPU execution is explicitly recorded. These observations do not alter the documented acceptance result.

Evidence directory: D:/具身智能/PhyGile与ReactiveBFM_论文对齐复现_20261022/paired_screen_document_witness_v1

Run output: D:/具身智能/PhyGile与ReactiveBFM_论文对齐复现_20261022/phygile_pair_screen_witness_v1

The evidence directory preserves invocation_verbatim.ps1, source document snapshots, execution_receipt.json, stdout.log, stderr.log, acceptance_checks.json, and the read-only verification script. The run output preserves summary.json, input_hashes.json, episode request/stdout/stderr, receipt.json, executed_sources.json, and private arrays. Licensed arrays were inspected locally only for finite state/shape and termination metadata; their values were not copied into this report or exposed.

Scope remains one decoded existing 262D pair and the undertrained 2,500-iteration pilot frozen policy. This is neither generated motion nor a reproduction of paper physical filtering/curriculum labeling; no training or GPU-throughput claim is supported.
# Independent V26 document-following witness

Verdict: **PASS_RUNTIME_REPAIR_ONLY**. The exact documented check completed successfully once. This certifies the verifier runtime repair and reuse of the original 72-input cache. It does not establish physical quality, actual rolling-contact ground truth, semantic compliance, innovation efficacy or downstream readiness.

## Exact invocation and terminal evidence

Executed once from local PowerShell, without retry:

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_contact_com_runtime_v26_20261012.py --check
```

Terminal exit: **0**. Output contained exactly the expected success lines:

```text
CPU_EXPORT_VALIDATION_DOCUMENT_CHECK_PASS
CONTACT_COM_RUNTIME_DOCUMENT_CHECK_PASS
```

No document-versus-runtime divergence was found. No new simulation, package installation, runtime repair, source edit, threshold widening or Overleaf write was performed in the witness role.

## Independent evidence checks

The separately executed read-only SHA audit verified all 23 files declared in original V24 versions, both V26 wrapper/document bindings, and all 11 original outputs listed in the runtime receipt. Local V26 document and source hashes match the remote bindings. The original V23 metric SHA is `b05b6ed8053ac037432471dabdf8608da383c76d4d2375ce10823e003f232c69`; the original V24 comparator SHA is `fa84c42cc2c894adb4458413ddbac16abf760a4ded71449ee467c1c8451cc8b5`.

The preserved snapshot remains `FAILED_RETAINED`, with the original error `Metric differs beyond float64 roundoff: metrics[0].surface_slip_proxy_mean_m_s`. Snapshot/current V24 versions are byte-identical, SHA256 `07bcf4822997ea17cb8b9ce3bfeb9a20116d28916104a687df7f38414e6ff97a`. The existing live V24 state advances to `WITNESS_PASS` as documented; this does not rewrite the preserved failed snapshot or the original negative witness.

The original request equals the execution lineage and independently reconstructed request exactly: val/planned/seed 11/round 0/noise round 0, identical config, manifest, thresholds and versions bindings. Lineage remains `bound_request_at_execution`. Original metadata, metrics, trajectory, sensor, native-contact, receipt, request and lineage hashes match the recorded preserved-output hashes.

The budget ledger and preserved original ledger are byte-identical, SHA256 `bbb0c9d2c0c7c5de0ec5d1f625e8bf1582cf561e03762f2a6da2a1dbf0884f52`. The receipt and independent audit show **1 before, 1 after**, with the single original 79.80916035920382-second measurement batch and no new charge. Cache contains 72 cases and 72 saved metric rows.

`runtime_check.json` records NumPy **1.23.5** from `/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages/numpy/__init__.py`. A separate direct import under that overlay independently prints the same version/path. An initial read-only audit assertion assumed `version.py` contains a literal version string; inspection showed it delegates to `_version`. That audit assumption was corrected through direct import, without rerunning the documented check or changing runtime files.

The numeric certificate `raw/val_planned_s11/metric_numerics.json` is `PASS_ROUNDOFF_ONLY`: zero changed floating values, zero maximum absolute difference, empty difference list and identical classifications. Saved metrics SHA matches the original metrics file; saved and recomputed canonical object hashes are equal (`69f6d00c8fbcfa0f67d5b0b48e0880f20a1f50d30cda26e87fae2cf4a40535f1`). The original rule remains `32 * float64_epsilon * max(1, abs(saved), abs(recomputed))`; thresholds and saved scores are unchanged.

The saved classifications remain: 70 completed, 23 condition pass, **0 physical pass**, 53 tracking pass, **0 joint pass**. A passing runtime certificate does not improve those scores.

## Original negative witness retention

Local V24 `DOCUMENT_WITNESS.md` remains the original FAIL_RETAINED report; SHA256 `4ec3f370552c8042fa5ff54f62f9359ae7c8aa8bf2ce550cfd5c63964b951c9b`. Its independent JSON remains negative (exit 1, absent expected sentinel); SHA256 `3e1864e0eea2942bcd15e3207a6321045bbe049e25d5fb3d123e6b83286ad5c8`. Neither was edited or reclassified.

Compact verdict is `DOCUMENT_WITNESS.json`; full independent read-only hash/certificate audit is `document_witness_audit.json`. This witness performed no authorized full-run continuation.

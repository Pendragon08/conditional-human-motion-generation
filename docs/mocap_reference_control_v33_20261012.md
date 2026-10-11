# Original MoCap-derived reference tracking diagnostic

Purpose: determine whether the frozen tracker and measurement pipeline also produce a quality floor on original HumanML3D references, rather than attributing every failure to the generator. This is a diagnostic control, not a fourth innovation, a new physical ground-truth benchmark, or a replacement for C1. It cannot by itself distinguish controller limitations from representation conversion or proxy limitations.

No launch while V28 is running. After V28 is terminal, use the same frozen runtime, source manifest, models, COM-corrected metric, thresholds and original goals. No generated continuation is sampled; the entire 120-frame reference comes from the original cached HumanML3D sequence through the frozen representation handler. The dataset-derived input reference is separately saved and hash-bound; it is not constructed from actual execution. The original ±15cm goals are retained, so these offset variants need not satisfy the condition predicate. Report the zero-offset condition result separately when interpreting it; the diagnostic does not relabel offsets or drop them from totals.

## Fresh document-following check, exactly once after V28 completion

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_mocap_reference_control_v33_20261012.py --check
```

Expected exit0 and `MOCAP_REFERENCE_CONTROL_DOCUMENT_CHECK_PASS`. The check executes one 72-input validation replay, with no generation calls. Inspect all original targets, exact 0–119 reference clock, canonical input-to-applied-reference equality, dataset-reference checksum, strict cache guards, reset/fence/COM/device evidence, frozen source/config/weights bindings and a charged measurement ledger entry. Human physical pass count is an observed diagnostic, **not a required positive value for witness PASS**. Witness PASS certifies the interface, not the data quality. Do not retry, fix or improvise; preserve any failed outputs.

Save independent witness under the result directory as `DOCUMENT_WITNESS.md` and `DOCUMENT_WITNESS_AUDIT.json` with top-level `verdict: PASS` only if all checks pass. After independent PASS, the executor may run the same command with `--run`: it reuses the bound validation batch, then performs one 144-input test replay charged to the reserve budget. It does not train policies or run G1. Cumulative time/output/data-space limits are unchanged. No paid API, hardware, generator optimization or Overleaf update.

If original-reference physical quality is also poor, record that the fixed tracker/representation/proxy combination is a confound requiring diagnosis; do not conclude that all synthesized references are inherently infeasible. If originals pass while generated references fail, this supports a more specific generated-reference quality issue in this setting, still without controller-independent truth. Do not use this control to loosen the original C1 thresholds or turn its failed joint gate into a success.

Current status: prepared, not yet independently witnessed or executed.

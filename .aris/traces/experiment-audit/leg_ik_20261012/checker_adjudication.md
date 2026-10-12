# Checker adjudication

This note preserves every automated failure label. It does not rerun the simulator, training, generation, or either historical witness. It distinguishes failed extra assertions from the documented contracts and keeps the real execution-binding warnings.

## Retained checker flags

| Checker flag | Adjudication | Effect on the frozen result |
|---|---|---|
| `request_reference_hash=false` in all three arms | The audit checker compared `request.replay_motion` with the NPZ file-byte SHA. The request uses the canonical NPZ-content hash defined in `physics_guided_com_lineage_v24_20261012.py:19-25`. All recomputed content hashes match. | No current reference mismatch. The separate child-side TOCTOU warning remains because `physics_guided_collect.py:289-299` does not recompute that content hash at execution. |
| Source construction `request_scope=false` | The checker expected replay mode. The fixed V51 source construction is native `val`, seed 11, round 0; its actual request and lineage match. | No source-order or source-count mismatch. |
| Raw reference bitwise equality failed | The documented upstream canonical transform uses `atol=1e-12`; raw-to-canonical error is `1.866893570624082e-16 m`. The `5.960464477539063e-08 m` selected/raw difference is float32 root subtract/re-add roundoff. | The documented replay invariant passes. |
| Cross-runtime velocity bitwise equality failed | Positions, previous-position recurrence, reference frames, and first zero velocity are exact. NumPy float32 re-evaluation differs by at most `9.5367431640625e-07 m/s`; retained same-arithmetic CUDA evidence matches 622,080 scalars exactly. | No metric gate or class changes. |
| Cross-runtime reductions retain differences of `2.7755575615628914e-17` in one analysis mean, up to `2.7989e-11 m/s` in component weighted means, and `6.54324e-10 m/s` in one descriptive slip-speed mean | Every difference is retained as cross-runtime reduction roundoff. Support/slip counts, slip fractions, all 216 classifications, and the frozen decision match. | No outcome change. |
| Original document witness has 117 scalar failures | Retained. Those checks required exact equality between float64 logged norms and norms from saved float32 vectors. The independent scalar contract verifies the saved arrays and documented geometry. | No retroactive PASS; use the independent addendum for the actual invariant. |
| Continuation witness has one ledger failure | Retained. It compared a historical baseline-only snapshot with a later append-only three-row ledger. The addendum reconstructs both independently. | No cost mismatch. |

## Warnings that remain

The child collector validates mode, seed, round, manifest, config, and thresholds before writing `bound_request_at_execution`, but it does not recompute the loaded replay file’s content hash or query the actual upstream CLoSD commit and dirty state. Current files and applied raw references agree, and a later read-only check found the expected clean upstream checkout; that is retrospective evidence, not execution-time certification.

`EXECUTOR_TERMINAL_OBSERVATION.json` accurately records SSH exit 0 and the completion sentinel, but it is explicitly an external tool observation. `pipeline_finally_receipt.json` is a process-finally receipt and does not claim an autonomous server parent-exit observation.

`summary.json.original_baseline_budget_ledger_sha256` points to the current append-only parent ledger (`c1a19…`), while the historical baseline-only snapshot is `d450a…`. The costs and rows are correct; the field name should be clarified prospectively without rewriting this run.

